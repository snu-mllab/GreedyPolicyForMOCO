import time
import numpy as np
import torch
import wandb
import pandas as pd
from botorch.utils.multi_objective import infer_reference_point
from setbench.tools.misc_op import Normalizer
from setbench.tools.pareto_op import pareto_frontier
from setbench.tools.sample_op import weighted_resampling
from setbench.tools.data_op import safe_np_cat

class ActiveSetHelper:
    def __init__(self, bb_task, tokenizer, candidate_pool, pool_targets, all_seqs, all_targets, resampling_weight):
        self.bb_task = bb_task
        self.tokenizer = tokenizer
        self.batch_size = self.bb_task.batch_size
        self.resampling_weight = resampling_weight
        self.total_bb_evals = 0
        self.start_time = time.time()
        self.all_targets = all_targets
        self.all_seqs = all_seqs
        self.init_transform(all_targets)
        self.init_active_set(candidate_pool, pool_targets)
        self.init_ref_point()
    
    def init_transform(self, all_targets):
        target_min = all_targets.min(axis=0).copy()
        target_range = all_targets.max(axis=0).copy() - target_min
        self.hypercube_transform = Normalizer(
            loc=target_min + 0.5 * target_range,
            scale=target_range / 2.,
        )

    def init_active_set(self, pool_candidates, pool_targets):
        is_feasible = self.bb_task.is_feasible(pool_candidates)
        # Init Active Set
        self.active_candidates = pool_candidates[is_feasible]
        self.active_targets = pool_targets[is_feasible]
        self.active_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pool_candidates])[is_feasible]
        # Init History
        self.pareto_candidates, self.pareto_targets = pareto_frontier(self.active_candidates, self.active_targets)
        self.pareto_seqs = np.array([p_cand.mutant_residue_seq for p_cand in self.pareto_candidates])
        self.pareto_cand_history = self.pareto_candidates.copy()
        self.pareto_seq_history = self.pareto_seqs.copy()
        self.pareto_target_history = self.pareto_targets.copy()
    
    def init_ref_point(self):
        self.norm_pareto_targets = self.hypercube_transform(self.pareto_targets)
        self._ref_point = -infer_reference_point(-torch.tensor(self.norm_pareto_targets)).numpy()
        self.rescaled_ref_point = self.hypercube_transform.inv_transform(self._ref_point.copy())

    def update_active_set(self, pool_candidates, pool_targets, pool_seqs):
        # contract active pool to current Pareto frontier
        self.active_candidates, self.active_targets = pareto_frontier(
            self.active_candidates, self.active_targets
        )
        self.active_seqs = np.array([a_cand.mutant_residue_seq for a_cand in self.active_candidates])
        print(f'\nactive set contracted to {self.active_candidates.shape[0]} pareto points')
        # augment active set with old pareto points
        if self.active_candidates.shape[0] < self.batch_size:
            num_samples = min(self.batch_size, self.pareto_cand_history.shape[0])
            num_backtrack = min(num_samples, self.batch_size - self.active_candidates.shape[0])
            self.update_active_set_from_history(num_samples, num_backtrack)
        # augment active set with random points
        if self.active_candidates.shape[0] < self.batch_size:
            num_samples = min(self.batch_size, pool_candidates.shape[0])
            num_rand = min(num_samples, self.batch_size - self.active_candidates.shape[0])
            self.update_active_set_from_random(num_samples, num_rand, pool_candidates, pool_targets, pool_seqs)

        print(self.active_targets)
        for seq in self.active_seqs:
            if hasattr(self.tokenizer, 'to_smiles'):
                print(self.tokenizer.to_smiles(seq))
            else:
                print(seq)

    def update_active_set_from_history(self, num_samples, num_backtrack):
        _, weights, _ = weighted_resampling(self.pareto_target_history, k=self.resampling_weight)
        hist_idxs = np.random.choice(
            np.arange(self.pareto_cand_history.shape[0]), num_samples, p=weights, replace=False
        )
        is_active = np.in1d(self.pareto_seq_history[hist_idxs], self.active_seqs)
        hist_idxs = hist_idxs[~is_active]
        if hist_idxs.size > 0:
            hist_idxs = hist_idxs[:num_backtrack]
            backtrack_candidates = self.pareto_cand_history[hist_idxs]
            backtrack_targets = self.pareto_target_history[hist_idxs]
            backtrack_seqs = self.pareto_seq_history[hist_idxs]
            self.active_candidates = np.concatenate((self.active_candidates, backtrack_candidates))
            self.active_targets = np.concatenate((self.active_targets, backtrack_targets))
            self.active_seqs = np.concatenate((self.active_seqs, backtrack_seqs))
            print(f'active set augmented with {backtrack_candidates.shape[0]} backtrack points')
    
    def update_active_set_from_random(self, num_samples, num_rand, pool_candidates, pool_targets, pool_seqs):
        _, weights, _ = weighted_resampling(pool_targets, k=self.resampling_weight)
        rand_idxs = np.random.choice(
            np.arange(pool_candidates.shape[0]), num_samples, p=weights, replace=False
        )
        is_active = np.in1d(pool_seqs[rand_idxs], self.active_seqs)
        rand_idxs = rand_idxs[~is_active]
        if rand_idxs.size > 0:
            rand_idxs = rand_idxs[:num_rand]
            rand_candidates = pool_candidates[rand_idxs]
            rand_targets = pool_targets[rand_idxs]
            rand_seqs = pool_seqs[rand_idxs]
            self.active_candidates = np.concatenate((self.active_candidates, rand_candidates))
            self.active_targets = np.concatenate((self.active_targets, rand_targets))
            self.active_seqs = np.concatenate((self.active_seqs, rand_seqs))
            print(f'active set augmented with {rand_candidates.shape[0]} random points')

    # For LAMBO
    def get_new_candids(self, base_cand_batches, new_seq_batches, new_seq_scores, batch_entropy, round_idx, log_prefix):
        # score all decoded batches, observe the highest value batch
        new_seq_batches = np.stack(new_seq_batches)
        new_seq_scores = np.stack(new_seq_scores)
        best_batch_idx = new_seq_scores.argmin()

        base_candidates = base_cand_batches[best_batch_idx]
        base_seqs = np.array([b_cand.mutant_residue_seq for b_cand in base_candidates])
        new_seqs = new_seq_batches[best_batch_idx]

        # logging
        metrics = dict(
            acq_val=new_seq_scores[best_batch_idx].mean().item(),
            entropy=batch_entropy[best_batch_idx],
            round_idx=round_idx,
            num_bb_evals=self.total_bb_evals,
            time_elapsed=time.time() - self.start_time,
        )
        print(pd.DataFrame([metrics]).to_markdown(floatfmt='.4f'))
        metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
        wandb.log(metrics)

        print('\n---- querying objective function ----')
        new_candidates = self.bb_task.make_new_candidates(base_candidates, new_seqs)
        new_targets = self.bb_task.score(new_candidates)
        return self.filter_new_candids(new_candidates, new_seqs, new_targets, base_candidates, base_seqs)

    def filter_new_candids(self, new_candidates, new_seqs, new_targets, base_candidates=None, base_seqs=None):
        # filter infeasible candidates
        is_feasible = self.bb_task.is_feasible(new_candidates)
        is_finite = (new_targets < np.inf).prod(-1).astype(bool)
        is_feasible *= is_finite
        new_seqs = new_seqs[is_feasible]
        new_candidates = new_candidates[is_feasible]
        new_targets = new_targets[is_feasible]
        if base_candidates is not None:
            base_candidates = base_candidates[is_feasible]
        if base_seqs is not None:
            base_seqs = base_seqs[is_feasible]
        # new_tokens = new_tokens[is_feasible]
        if new_candidates.size == 0:
            print('no new candidates')
            return None

        # filter duplicate candidates
        new_seqs, unique_idxs = np.unique(new_seqs, return_index=True)
        new_candidates = new_candidates[unique_idxs]
        new_targets = new_targets[unique_idxs]
        if base_candidates is not None:
            base_candidates = base_candidates[unique_idxs]
        if base_seqs is not None:
            base_seqs = base_seqs[unique_idxs]

        # filter redundant candidates
        is_new = np.in1d(new_seqs, self.all_seqs, invert=True)
        new_seqs = new_seqs[is_new]
        new_candidates = new_candidates[is_new]
        new_targets = new_targets[is_new]
        if base_candidates is not None:
            base_candidates = base_candidates[is_new]
        if base_seqs is not None:
            base_seqs = base_seqs[is_new]
        if new_candidates.size == 0:
            print('no new candidates')
            return None

        return new_candidates, new_seqs, new_targets, base_candidates, base_seqs

    def update_new_candids(self, new_candidates, new_seqs, new_targets, base_candidates=None, base_seqs=None):
        self.all_targets = np.concatenate((self.all_targets, new_targets))
        self.all_seqs = np.concatenate((self.all_seqs, new_seqs))

        for seq in new_seqs:
            if hasattr(self.tokenizer, 'to_smiles'):
                print(self.tokenizer.to_smiles(seq))
            else:
                print(seq)

        assert new_seqs.shape[0] == new_targets.shape[0]
        if base_candidates is not None:
            assert base_seqs.shape[0] == new_seqs.shape[0] 
            for b_cand, n_cand, f_val in zip(base_candidates, new_candidates, new_targets):
                print(f'{len(b_cand)} --> {len(n_cand)}: {f_val}')

        # augment active pool with candidates that can be mutated again
        self.active_candidates = np.concatenate((self.active_candidates, new_candidates))
        self.active_targets = np.concatenate((self.active_targets, new_targets))
        self.active_seqs = np.concatenate((self.active_seqs, new_seqs))

        # overall Pareto frontier including terminal candidates
        self.pareto_candidates, self.pareto_targets = pareto_frontier(
            np.concatenate((self.pareto_candidates, new_candidates)),
            np.concatenate((self.pareto_targets, new_targets)),
        )
        self.pareto_seqs = np.array([p_cand.mutant_residue_seq for p_cand in self.pareto_candidates])

        if len(new_targets) > 0:
            print('\n new candidates')
            obj_vals = {f'obj_val_{i}': new_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
            print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

            print('\n best candidates')
            obj_vals = {f'obj_val_{i}': self.pareto_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
            print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

        # store good candidates for backtracking
        par_is_new = np.in1d(self.pareto_seqs, self.pareto_seq_history, invert=True)
        self.pareto_cand_history = safe_np_cat([self.pareto_cand_history, self.pareto_candidates[par_is_new]])
        self.pareto_seq_history = safe_np_cat([self.pareto_seq_history, self.pareto_seqs[par_is_new]])
        self.pareto_target_history = safe_np_cat([self.pareto_target_history, self.pareto_targets[par_is_new]])

        # logging
        self.norm_pareto_targets = self.hypercube_transform(self.pareto_targets)
