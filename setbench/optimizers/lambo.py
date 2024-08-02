import hydra
import wandb
import pandas as pd
import time
import numpy as np
import torch
import random

from torch.nn import functional as F

# from pymoo.factory import get_performance_indicator
from pymoo.indicators.hv import HV

from setbench.models.mlm import sample_tokens, evaluate_windows
from setbench.models.shared_elements import check_early_stopping
from setbench.models.lanmt import corrupt_tok_idxs
from setbench.tools.data_op import DataSplit, update_splits, str_to_tokens, tokens_to_str, safe_np_cat
from setbench.tools.sample_op import weighted_resampling
from setbench.tools.pareto_op import pareto_frontier
from setbench.tools.misc_op import Normalizer
from setbench.tools.file_op import write_pkl
from setbench.optimizers.active_set_helper import ActiveSetHelper
from setbench.optimizers.multi_simple.lambo_loop import LaMBOLoop

class LaMBO(object):
    def __init__(self, bb_task, tokenizer, encoder, surrogate, acquisition, num_rounds, num_gens,
                 lr, num_opt_steps, patience, mask_ratio, resampling_weight,
                 encoder_obj, optimize_latent, position_sampler, entropy_penalty,
                 window_size, **kwargs):

        self.tokenizer = tokenizer
        self.num_rounds = num_rounds
        self._hv_ref = None

        self.hydra_configs = {
            'bb_task': bb_task,
            'encoder': encoder,
            'surrogate': surrogate,
            'acquisition': acquisition,
        }

        self.bb_task = hydra.utils.instantiate(bb_task, tokenizer=tokenizer, candidate_pool=[])

        self.encoder_config = encoder
        self.encoder = hydra.utils.instantiate(encoder, tokenizer=tokenizer)
        self.encoder_obj = encoder_obj

        self.surrogate_config = surrogate
        self.surrogate_model = hydra.utils.instantiate(surrogate, tokenizer=self.encoder.tokenizer,
                                                       encoder=self.encoder)
        self.acquisition = acquisition

        self.resampling_weight = resampling_weight

        self.algorithm = LaMBOLoop(
            bb_task=self.bb_task,
            tokenizer=self.encoder.tokenizer,
            encoder=self.encoder,
            num_gens=num_gens,
            lr=lr,
            num_opt_steps=num_opt_steps,
            patience=patience,
            resampling_weight=self.resampling_weight,
            encoder_obj=self.encoder_obj,
            optimize_latent=optimize_latent,
            position_sampler=position_sampler,
            entropy_penalty=entropy_penalty,
            window_size=window_size,
        )

        self.train_split = DataSplit()
        self.val_split = DataSplit()
        self.test_split = DataSplit()

    def optimize(self, candidate_pool, pool_targets, all_seqs, all_targets, log_prefix='', seed=None):
        self.batch_size = self.bb_task.batch_size
        ash = ActiveSetHelper(self.bb_task, self.tokenizer, candidate_pool, pool_targets, all_seqs, all_targets, self.resampling_weight)
        # logging setup
        round_idx = 0
        self._log_candidates(ash.pareto_candidates, ash.pareto_targets, round_idx, log_prefix)
        metrics = self._log_optimizer_metrics(ash.norm_pareto_targets, ash._ref_point, round_idx, ash.total_bb_evals, ash.start_time, log_prefix)

        print('\n best candidates')
        obj_vals = {f'obj_val_{i}': ash.pareto_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
        print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

        # Active Pool
        pool_candidates, pool_targets, pool_seqs = ash.active_candidates, ash.active_targets, ash.active_seqs
        # New Candidates
        new_seqs, new_targets = ash.all_seqs.copy(), ash.all_targets.copy()
        for round_idx in range(1, self.num_rounds + 1):
            metrics = {}

            ash.update_active_set(pool_candidates, pool_targets, pool_seqs)

            print('\n---- fitting surrogate model ----')
            # acquisition fns assume maximization so we normalize and negate targets here
            z_score_transform = Normalizer(ash.all_targets.mean(0), ash.all_targets.std(0))

            tgt_transform = lambda x: -z_score_transform(x)
            transformed_ref_point = tgt_transform(ash.rescaled_ref_point)

            new_split = DataSplit(new_seqs, new_targets)
            holdout_ratio = self.surrogate_model.holdout_ratio
            all_splits = update_splits(
                self.train_split, self.val_split, self.test_split, new_split, holdout_ratio,
            )
            self.train_split, self.val_split, self.test_split = all_splits

            X_train, Y_train = self.train_split.inputs, tgt_transform(self.train_split.targets)
            X_val, Y_val = self.val_split.inputs, tgt_transform(self.val_split.targets)
            X_test, Y_test = self.test_split.inputs, tgt_transform(self.test_split.targets)

            records = self.surrogate_model.fit(
                X_train, Y_train, X_val, Y_val, X_test, Y_test,
                encoder_obj=self.encoder_obj, resampling_temp=None
            )

            # # Caching for first-round experiment
            # if round_idx == 1:
            #     save_dict = {
            #         'round_idx': round_idx,
            #         'records': records,
            #         # 'model_state_dict': self.surrogate_model.state_dict(),
            #         'all_splits': all_splits,
            #         'all_seqs': ash.all_seqs,
            #         'all_targets': ash.all_targets,
            #         'hv_ref': self._hv_ref,
            #         'hypercube_transform': ash.hypercube_transform,
            #         'z_score_transform': z_score_transform,
            #         'ref_point': ash._ref_point,
            #         'transformed_ref_point': transformed_ref_point,
            #         'rescaled_ref_point': ash.rescaled_ref_point,
            #         'pareto_candidates': ash.pareto_candidates,
            #         'pareto_targets': ash.pareto_targets,
            #         'pareto_seqs': ash.pareto_seqs,
            #         'norm_pareto_targets': ash.norm_pareto_targets,
            #         'active_candidates': ash.active_candidates,
            #         'active_targets': ash.active_targets,
            #         'hv': self.recent_hypervol_abs,
            #         'hv_rel': self.recent_hypervol_rel,
            #         'hydra_configs': self.hydra_configs,
            #     }
            #     import os
            #     os.makedirs(f'/storage/deokjae/lambo_data/{self.acquisition._target_.split(".")[-1]}/', exist_ok=True)
            #     write_pkl(save_dict, f'/storage/deokjae/lambo_data/{self.acquisition._target_.split(".")[-1]}/{self.bb_task.task_name}_s{seed}_r{round_idx}.pkl')

            # log result
            last_entry = {key.split('/')[-1]: val for key, val in records[-1].items()}
            best_idx = last_entry['best_epoch']
            best_entry = {key.split('/')[-1]: val for key, val in records[best_idx].items()}
            print(pd.DataFrame([best_entry]).to_markdown(floatfmt='.4f'))
            metrics.update(dict(
                test_rmse=best_entry['test_rmse'],
                test_nll=best_entry['test_nll'],
                test_s_rho=best_entry['test_s_rho'],
                test_ece=best_entry['test_ece'],
                test_post_var=best_entry['test_post_var'],
                # test_perplexity=best_entry['test_perplexity'],
                round_idx=round_idx,
                num_bb_evals=ash.total_bb_evals,
                num_train=X_train.shape[0],
                time_elapsed=time.time() - ash.start_time,
            ))
            metrics = {
                '/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()
            }
            wandb.log(metrics)

            baseline_seqs = np.array([cand.mutant_residue_seq for cand in ash.active_candidates])
            baseline_targets = ash.active_targets
            baseline_seqs, baseline_targets = pareto_frontier(baseline_seqs, baseline_targets)
            baseline_targets = tgt_transform(baseline_targets)
            acq_fn = hydra.utils.instantiate(
                self.acquisition,
                X_baseline=baseline_seqs,
                known_targets=torch.tensor(baseline_targets).to(self.surrogate_model.device),
                surrogate=self.surrogate_model,
                ref_point=torch.tensor(transformed_ref_point).to(self.surrogate_model.device),
                obj_dim=self.bb_task.obj_dim,
            )

            print('\n---- optimizing candidates ----')
            _, assets = self.algorithm.optimize(self.surrogate_model, acq_fn, ash.active_candidates, ash.active_targets)
            base_cand_batches, new_seq_batches, new_seq_scores, batch_entropy = assets

            ash.total_bb_evals += self.batch_size
            # Update history and log
            new_candids_whole = ash.get_new_candids(base_cand_batches, new_seq_batches, new_seq_scores, batch_entropy, round_idx, log_prefix)
            if new_candids_whole is None:
                new_candidates, new_seqs, new_targets = np.empty(0), np.empty(0), np.empty((0,self.bb_task.obj_dim))
            else:
                new_candidates, new_seqs, new_targets, base_candidates, base_seqs = new_candids_whole
            ash.update_new_candids(new_candidates, new_seqs, new_targets, base_candidates, base_seqs)
            self._log_candidates(new_candidates, new_targets, round_idx, log_prefix)
            metrics = self._log_optimizer_metrics(
                ash.norm_pareto_targets, ash._ref_point, round_idx, ash.total_bb_evals, ash.start_time, log_prefix
            )
            pool_candidates = np.concatenate((pool_candidates, new_candidates))
            pool_targets = np.concatenate((pool_targets, new_targets))
            pool_seqs = np.concatenate((pool_seqs, new_seqs))

        return metrics

    def sample_mutation_window(self, window_mask_idxs, window_entropy, temp=1.):
        # selected_features = []
        selected_mask_idxs = []
        for seq_idx, entropies in window_entropy.items():
            mask_idxs = window_mask_idxs[seq_idx]
            assert len(mask_idxs) == len(entropies)
            window_idxs = np.arange(len(mask_idxs)).astype(int)
            entropies = torch.tensor(entropies)
            weights = F.softmax(entropies / temp).cpu().numpy()
            selected_window = np.random.choice(window_idxs, 1, p=weights).item()
            selected_mask_idxs.append(mask_idxs[selected_window])
        return np.concatenate(selected_mask_idxs)

    def _log_candidates(self, candidates, targets, round_idx, log_prefix):
        table_cols = ['round_idx', 'cand_uuid', 'cand_ancestor', 'cand_seq']
        table_cols.extend([f'obj_val_{idx}' for idx in range(self.bb_task.obj_dim)])
        for cand, obj in zip(candidates, targets):
            new_row = [round_idx, cand.uuid, cand.wild_name, cand.mutant_residue_seq]
            new_row.extend([elem for elem in obj])
            record = {'/'.join((log_prefix, 'candidates', key)): val for key, val in zip(table_cols, new_row)}
            wandb.log(record)

    def _log_optimizer_metrics(self, normed_targets, _ref_point, round_idx, num_bb_evals, start_time, log_prefix):
        hv_indicator = HV(ref_point=_ref_point)
        new_hypervol = hv_indicator.do(normed_targets)
        self._hv_ref = new_hypervol if self._hv_ref is None else self._hv_ref
        metrics = dict(
            round_idx=round_idx,
            hypervol_abs=new_hypervol,
            hypervol_rel=new_hypervol / max(1e-6, self._hv_ref),
            num_bb_evals=num_bb_evals,
            time_elapsed=time.time() - start_time,
        )
        self.recent_hypervol_abs = new_hypervol
        self.recent_hypervol_rel = new_hypervol / max(1e-6, self._hv_ref)
        print(pd.DataFrame([metrics]).to_markdown(floatfmt='.4f'))
        metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
        wandb.log(metrics)
        return metrics