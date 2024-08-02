import hydra
import numpy as np
import torch
import random

from torch.nn import functional as F

from pymoo.termination import get_termination

from pymoo.optimize import minimize

from setbench.tasks.surrogate_task import UCBSurrogateTask, SurrogateTask
from pymoo.util.ref_dirs import get_reference_directions
from setbench.tools.sample_op import weighted_resampling
from setbench.tools.mutation_op import apply_mutation

class PyMooLoop(object):
    def __init__(self, bb_task, algorithm, tokenizer, encoder,
                 num_gens, seed, encoder_obj, 
                 residue_sampler='uniform', resampling_weight=1., **kwargs):
        self.bb_task = bb_task
        # tokenizer setup
        self.tokenizer = tokenizer
        self.tokenizer.set_sampling_vocab(None, bb_task.max_ngram_size)
        # algorithm setup
        self.algorithm = algorithm
        self.encoder = encoder

        self.batch_size = self.bb_task.batch_size
        self.num_gens = num_gens
        self.term_fn = get_termination("n_gen", num_gens)
        self.seed = seed
        self.encoder_obj = encoder_obj
        self.resampling_weight = resampling_weight

        self.residue_sampler = residue_sampler

    def optimize(self, surrogate_model, acq_fn, active_candidates, active_targets, log_prefix=''):
        self.surrogate_model = surrogate_model
        self.acq_fn = acq_fn
        self.active_candidates = active_candidates
        self.active_targets = active_targets
        
        if self.resampling_weight is None:
            active_weights = None #np.ones(self.active_targets.shape[0]) / self.active_targets.shape[0]
        else:
            _, active_weights, _ = weighted_resampling(self.active_targets, k=self.resampling_weight)

        if self.residue_sampler == 'uniform':
            mlm_obj = None
        else:
            raise ValueError

        if 'nsga3' in self.algorithm._target_ or 'nsga2' in self.algorithm._target_:
            problem = UCBSurrogateTask(self.tokenizer, self.active_candidates, self.acq_fn, batch_size=1, max_len=None, candidate_weights=active_weights,
                                    obj_dim=self.bb_task.obj_dim, allow_len_change=self.bb_task.allow_len_change, max_num_edits=self.bb_task.max_num_edits)
        elif self.algorithm._target_=='pymoo.algorithms.soo.nonconvex.ga.GA':
            problem = SurrogateTask(self.tokenizer, self.active_candidates, self.acq_fn, batch_size=self.batch_size, max_len=None, candidate_weights=active_weights,
                                    allow_len_change=self.bb_task.allow_len_change, max_num_edits=self.bb_task.max_num_edits)
        # Algorithm
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

        print('---- optimizing candidates ----')
        res = minimize(
            problem,
            algorithm,
            self.term_fn,
            save_history=False,
            verbose=True
        )

        # query outer task, append data
        new_seqs = self._evaluate_result(
            res, self.active_candidates, log_prefix
        )

        # Get best results
        best_seqs = new_seqs
        metrics = {}
        metrics['best_acq_val'] = self.acq_fn(best_seqs[None, :]).mean().item()
        metrics['best_seqs'] = new_seqs
        print("best_acq_val", metrics['best_acq_val'])
        return metrics, None

    def _evaluate_result(self, result, candidate_pool, log_prefix,
                         *args, **kwargs):
        all_x = result.pop.get('X')

        cand_batches = result.problem.x_to_query_batches(all_x)
        query_points = cand_batches[0]

        batch_idx = 1
        while query_points.shape[0] < self.bb_task.batch_size:
            query_points = np.concatenate((query_points, cand_batches[batch_idx]))
            batch_idx += 1

        # set bb_task 
        # bb_task_eval = hydra.utils.instantiate(self.task_config, tokenizer=self.tokenizer, candidate_pool=candidate_pool, batch_size=1)

        op_types = ['sub', 'ins', 'del'] if self.bb_task.allow_len_change else ['sub']
        x_seqs = []
        for query_pt in query_points:
            cand_idx = query_pt[0]
            base_candidate = candidate_pool[cand_idx]
            base_seq = base_candidate.mutant_residue_seq
            mut_seq = base_seq
            for i in range(self.bb_task.max_num_edits):
                mut_pos, mut_res_idx, op_idx = query_pt[1+3*i:1+3*(i+1)]
                op_type = op_types[op_idx]
                mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
                mut_seq = apply_mutation(mut_seq, mut_pos, mut_res, op_type, self.tokenizer)
            x_seqs.append(mut_seq)
        new_seqs = np.array(x_seqs).reshape(-1)
        return new_seqs