import hydra
import wandb
import pandas as pd
import time
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.nn import functional as F
from torch.distributions import Categorical

from pymoo.indicators.hv import HV

from setbench.tools.pareto_op import pareto_frontier, thermometer
from setbench.tools.misc_op import Normalizer
from setbench.tools.data_op import DataSplit, update_splits, str_to_tokens, tokens_to_str
from setbench.tools.sample_op import generate_simplex
from setbench.optimizers.active_set_helper import ActiveSetHelper
from setbench.optimizers.multi_simple.setrl_loop import SetRLSeqLoop

class SetRL(object):
    '''
    Here instead of generating sequence from scratch, we generate modifications for the current pareto_front
    '''
    def __init__(self, bb_task, tokenizer, encoder, surrogate, acquisition, num_rounds, 
                 num_opt_steps, resampling_weight, encoder_obj, model, reinit_model, **kwargs):

        self.tokenizer = tokenizer
        self.num_rounds = num_rounds
        self._hv_ref = None
        self._ref_point = np.array([1] * bb_task.obj_dim)
        self.obj_dim = bb_task.obj_dim

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        self.bb_task = hydra.utils.instantiate(bb_task, tokenizer=tokenizer, candidate_pool=[])

        self.encoder_config = encoder
        self.encoder = hydra.utils.instantiate(encoder, tokenizer=tokenizer)
        self.encoder_obj = encoder_obj

        self.surrogate_config = surrogate
        self.surrogate_model = hydra.utils.instantiate(surrogate, tokenizer=self.encoder.tokenizer,
                                                       encoder=self.encoder)
        self.acquisition = acquisition
        self.resampling_weight = resampling_weight

        self.algorithm = SetRLSeqLoop(
            bb_task=self.bb_task,
            tokenizer=self.encoder.tokenizer,
            num_opt_steps=num_opt_steps,
            model=model,
            **kwargs
        )   

        self.reinit_model = reinit_model

        self.train_split = DataSplit()
        self.val_split = DataSplit()
        self.test_split = DataSplit()

    def optimize(self, candidate_pool, pool_targets, all_seqs, all_targets, log_prefix='', seed=None):
        self.batch_size = self.bb_task.batch_size
        ash = ActiveSetHelper(self.bb_task, self.tokenizer, candidate_pool, pool_targets, all_seqs, all_targets, self.resampling_weight)
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
            if self.reinit_model:
                self.algorithm.reinit_model()
            _, assets = self.algorithm.optimize(self.surrogate_model, acq_fn, ash.active_candidates, ash.active_targets)
            base_candidates, new_seqs = assets
            # logging
            metrics = dict(
                round_idx=round_idx,
                num_bb_evals=ash.total_bb_evals,
                time_elapsed=time.time() - ash.start_time,
            )
            print(pd.DataFrame([metrics]).to_markdown(floatfmt='.4f'))
            metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
            wandb.log(metrics)

            print('\n---- querying objective function ----')
            new_candidates = self.bb_task.make_new_candidates(base_candidates, new_seqs)
            new_targets = self.bb_task.score(new_candidates)
            print("remaining new_candidates", len(new_candidates))
            # Update history and log
            ash.total_bb_evals += self.batch_size
            new_candids_whole = ash.filter_new_candids(new_candidates, new_seqs, new_targets, None, None)
            if new_candids_whole is None:
                new_candidates, new_seqs, new_targets = np.empty(0), np.empty(0), np.empty((0,self.bb_task.obj_dim))
            else:
                new_candidates, new_seqs, new_targets, _, _ = new_candids_whole
            ash.update_new_candids(new_candidates, new_seqs, new_targets, None, None)
            self._log_candidates(new_candidates, new_targets, round_idx, log_prefix)
            metrics = self._log_optimizer_metrics(
                ash.norm_pareto_targets, ash._ref_point, round_idx, ash.total_bb_evals, ash.start_time, log_prefix
            )
            pool_candidates = np.concatenate((pool_candidates, new_candidates))
            pool_targets = np.concatenate((pool_targets, new_targets))
            pool_seqs = np.concatenate((pool_seqs, new_seqs))

        return metrics
    
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
        print(pd.DataFrame([metrics]).to_markdown(floatfmt='.4f'))
        metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
        wandb.log(metrics)
        return metrics