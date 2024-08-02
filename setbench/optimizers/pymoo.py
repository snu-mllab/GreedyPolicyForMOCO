import numpy as np
import torch
import hydra
import wandb
import time
import pandas as pd


from pymoo.termination import get_termination
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

from setbench.models.lm_elements import LanguageModel
from setbench.tasks.surrogate_task import SurrogateTask
from setbench.tools.data_op import DataSplit, update_splits
from setbench.tools.sample_op import weighted_resampling
from setbench.tools.pareto_op import pareto_frontier
from setbench.tools.misc_op import Normalizer
from setbench.optimizers.active_set_helper import ActiveSetHelper
from pymoo.util.ref_dirs import get_reference_directions
from setbench.tasks.surrogate_task import UCBSurrogateTask
import random

class SequentialGeneticOptimizer(object):
    def __init__(self, bb_task, algorithm, tokenizer, num_rounds, num_gens, seed,
                 residue_sampler='uniform', resampling_weight=1., **kwargs):
        self.task_config = bb_task
        self.bb_task = hydra.utils.instantiate(bb_task, tokenizer=tokenizer, candidate_pool=[])
        self.algorithm = algorithm
        print("algorithm", algorithm)
        
        self.num_rounds = num_rounds
        self.num_gens = num_gens
        self.term_fn = get_termination("n_gen", num_gens)
        self.seed = seed
        self.residue_sampler = residue_sampler

        tokenizer.set_sampling_vocab(None, bb_task.max_ngram_size)
        self.tokenizer = tokenizer

        self.encoder = None

        self._hv_ref = None
        self._ref_point = np.array([1] * self.bb_task.obj_dim)

        self.resampling_weight = resampling_weight

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


        # set up encoder which may also be a masked language model (MLM)
        encoder = None if self.encoder is None else hydra.utils.instantiate(
            self.encoder, tokenizer=self.tokenizer
        )

        if self.residue_sampler == 'uniform':
            mlm_obj = None
        else:
            raise ValueError
        
        n_seq_init = len(pool_seqs)
        for round_idx in range(1, self.num_rounds + 1):
            metrics = {}

            ash.update_active_set(pool_candidates, pool_targets, pool_seqs)

            if self.resampling_weight is None:
                active_weights = None # np.ones(ash.active_targets.shape[0]) / ash.active_targets.shape[0]
            else:
                _, active_weights, _ = weighted_resampling(ash.active_targets, k=self.resampling_weight)

            # prepare the inner task
            # z_score_transform = Normalizer(self.all_targets.mean(0), self.all_targets.std(0))
            z_score_transform = Normalizer(ash.all_targets.mean(0), ash.all_targets.std(0))

            # algorithm setup
            if 'nsga3' in self.algorithm._target_:
                if self.bb_task.obj_dim <= 3:
                    n_partitions = 100
                else:
                    raise NotImplementedError
                ref_dirs = get_reference_directions("das-dennis", self.bb_task.obj_dim, n_partitions=n_partitions)
                ref_dirs = ref_dirs[[(r > 0).all() for r in ref_dirs]]

                ref_dirs = ref_dirs[random.sample(range(len(ref_dirs)), self.algorithm.pop_size)]
                print("ref_dirs", ref_dirs.shape)
                algorithm = hydra.utils.instantiate(self.algorithm, ref_dirs=ref_dirs)
            else:
                algorithm = hydra.utils.instantiate(self.algorithm)
            algorithm.initialization.sampling.tokenizer = self.tokenizer
            algorithm.mating.mutation.tokenizer = self.tokenizer

            problem = self._create_inner_task(
                candidate_pool=ash.active_candidates,
                candidate_weights=active_weights,
                input_data=new_seqs,
                target_data=new_targets,
                transform=z_score_transform,
                ref_point=ash.rescaled_ref_point,
                encoder=encoder,
                round_idx=round_idx,
                num_bb_evals=ash.total_bb_evals,
                start_time=ash.start_time,
                log_prefix=log_prefix,
                ash=ash,
            )

            print('---- optimizing candidates ----')
            res = minimize(
                problem,
                algorithm,
                self.term_fn,
                save_history=False,
                verbose=False,
            )

            # query outer task, append data
            print('\n---- querying objective function ----')
            new_candidates, new_seqs, new_targets, bb_evals = self._evaluate_result(
                res, ash.active_candidates, z_score_transform, round_idx, ash.total_bb_evals, ash.start_time, log_prefix
            )
            ash.total_bb_evals += bb_evals
            # Update history and log
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

    def _evaluate_result(self, *args, **kwargs):
        raise NotImplementedError

    def _create_inner_task(self, *args, **kwargs):
        raise NotImplementedError

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


class ModelFreeGeneticOptimizer(SequentialGeneticOptimizer):
    def _create_inner_task(
            self, candidate_pool, input_data, target_data, transform, candidate_weights, *args, **kwargs):
        inner_task = hydra.utils.instantiate(
            self.task_config,
            candidate_pool=candidate_pool,
            transform=transform,
            tokenizer=self.tokenizer,
            batch_size=1,
            candidate_weights=candidate_weights,
        )
        return inner_task

    def _evaluate_result(self, result, candidate_pool, transform, *args, **kwargs):
        new_candidates = result.pop.get('X_cand').reshape(-1)
        new_seqs = result.pop.get('X_seq').reshape(-1)
        new_targets = transform.inv_transform(result.pop.get('F'))
        bb_evals = self.num_gens * self.algorithm.pop_size
        return new_candidates, new_seqs, new_targets, bb_evals


class ModelBasedGeneticOptimizer(SequentialGeneticOptimizer):
    def __init__(
            self, bb_task, surrogate, algorithm, acquisition, encoder, tokenizer, num_rounds, num_gens, seed,
            encoder_obj, **kwargs
    ):
        super().__init__(
            bb_task=bb_task,
            algorithm=algorithm,
            tokenizer=tokenizer,
            num_rounds=num_rounds,
            num_gens=num_gens,
            seed=seed,
            **kwargs
        )
        self.encoder = encoder
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.train_split = DataSplit()
        self.val_split = DataSplit()
        self.test_split = DataSplit()
        self.encoder_obj = encoder_obj

    def _create_inner_task(self, candidate_pool, candidate_weights, input_data, target_data, transform, ref_point,
                           encoder, round_idx, num_bb_evals, start_time, log_prefix, ash):

        self.surrogate_model = hydra.utils.instantiate(self.surrogate, encoder=encoder, tokenizer=encoder.tokenizer,
                                                        alphabet=self.tokenizer.non_special_vocab)

        # prepare surrogate dataset
        tgt_transform = lambda x: -transform(x)
        transformed_ref_point = tgt_transform(ref_point)

        new_split = DataSplit(input_data, target_data)
        holdout_ratio = self.surrogate.holdout_ratio
        all_splits = update_splits(
            self.train_split, self.val_split, self.test_split, new_split, holdout_ratio,
        )
        self.train_split, self.val_split, self.test_split = all_splits

        X_train, Y_train = self.train_split.inputs, tgt_transform(self.train_split.targets)
        X_val, Y_val = self.val_split.inputs, tgt_transform(self.val_split.targets)
        X_test, Y_test = self.test_split.inputs, tgt_transform(self.test_split.targets)

        # train surrogate
        records = self.surrogate_model.fit(
            X_train, Y_train, X_val, Y_val, X_test, Y_test, resampling_temp=None,
            encoder_obj=self.encoder_obj
        )
        # log result
        last_entry = {key.split('/')[-1]: val for key, val in records[-1].items()}
        best_idx = last_entry['best_epoch']
        best_entry = {key.split('/')[-1]: val for key, val in records[best_idx].items()}
        print(pd.DataFrame([best_entry]).to_markdown())
        metrics = dict(
            test_rmse=best_entry['test_rmse'],
            test_nll=best_entry['test_nll'],
            test_s_rho=best_entry['test_s_rho'],
            test_ece=best_entry['test_ece'],
            test_post_var=best_entry['test_post_var'],
            round_idx=round_idx,
            num_bb_evals=num_bb_evals,
            num_train=self.train_split.inputs.shape[0],
            time_elapsed=time.time() - start_time,
        )
        metrics = {
            '/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()
        }
        wandb.log(metrics)

        # complete task setup
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
        if 'nsga3' in self.algorithm._target_ or 'nsga2' in self.algorithm._target_:
            inner_task = UCBSurrogateTask(self.tokenizer, candidate_pool, acq_fn, batch_size=1, max_len=None, candidate_weights=candidate_weights,
                            obj_dim=self.bb_task.obj_dim, allow_len_change=self.bb_task.allow_len_change, max_num_edits=self.bb_task.max_num_edits)
        elif self.algorithm._target_=='pymoo.algorithms.soo.nonconvex.ga.GA':
            inner_task = SurrogateTask(self.tokenizer, candidate_pool, acq_fn, batch_size=acq_fn.batch_size, max_len=None, candidate_weights=candidate_weights,
                                    allow_len_change=self.bb_task.allow_len_change, max_num_edits=self.bb_task.max_num_edits)
        else:
            raise NotImplementedError
        return inner_task

    def _evaluate_result(self, result, candidate_pool, transform, round_idx, num_bb_evals, start_time, log_prefix,
                         *args, **kwargs):
        all_x = result.pop.get('X')
        all_acq_vals = result.pop.get('F')

        cand_batches = result.problem.x_to_query_batches(all_x)
        query_points = cand_batches[0]
        query_acq_vals = all_acq_vals[0]

        batch_idx = 1
        while query_points.shape[0] < self.bb_task.batch_size:
            query_points = np.concatenate((query_points, cand_batches[batch_idx]))
            query_acq_vals = np.concatenate((query_acq_vals, all_acq_vals[batch_idx]))
            batch_idx += 1

        bb_task = hydra.utils.instantiate(
            self.task_config, tokenizer=self.tokenizer, candidate_pool=candidate_pool, batch_size=1
        )
        bb_out = bb_task.evaluate(query_points, return_as_dictionary=True)
        new_candidates = bb_out['X_cand'].reshape(-1)
        new_seqs = bb_out['X_seq'].reshape(-1)
        new_targets = bb_out["F"]
        bb_evals = query_points.shape[0]

        metrics = dict(
            acq_val=query_acq_vals.mean().item(),
            round_idx=round_idx,
            num_bb_evals=num_bb_evals,
            time_elapsed=time.time() - start_time,
        )
        metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
        wandb.log(metrics)

        return new_candidates, new_seqs, new_targets, bb_evals
