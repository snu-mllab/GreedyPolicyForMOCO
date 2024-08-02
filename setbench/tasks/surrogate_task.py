import numpy as np
import torch

from setbench.tasks.base_task import BaseTask
from setbench.tools.mutation_op import apply_mutation


class SurrogateTask(BaseTask):
    def __init__(self, tokenizer, candidate_pool, acq_fn, max_len=None, transform=lambda x: x, **kwargs):
        super().__init__(tokenizer, candidate_pool, obj_dim=acq_fn.out_dim, max_len=max_len,
                         transform=transform, **kwargs)
        self.acq_fn = acq_fn

    def _evaluate(self, x, out, *args, **kwargs):
        query_batches = self.x_to_query_batches(x)
        batch_shape, num_vars = query_batches.shape[:-1], query_batches.shape[-1]
        candidates = []
        for query_pt in query_batches.reshape(-1, num_vars):
            cand_idx = query_pt[0]
            base_seq = self.candidate_pool[cand_idx].mutant_residue_seq
            for i in range(self.max_num_edits):
                mut_pos, mut_res_idx, op_idx = query_pt[1+3*i:1+3*(i+1)]
                op_type = self.op_types[op_idx]
                mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
                base_seq = apply_mutation(base_seq, mut_pos, mut_res, op_type, self.tokenizer)
            candidates.append(base_seq)
        candidates = np.array(candidates).reshape(*batch_shape)
        with torch.inference_mode():
            acq_vals = self.acq_fn(candidates, batch_size=4).cpu().numpy()
        out["F"] = -acq_vals
    

class UCBSurrogateTask(BaseTask):
    def __init__(self, tokenizer, candidate_pool, acq_fn, max_len=None, transform=lambda x: x, obj_dim=None, **kwargs):
        super().__init__(tokenizer, candidate_pool, obj_dim=obj_dim, max_len=max_len,
                         transform=transform, **kwargs)
        self.acq_fn = acq_fn
        self.surrogate_model = acq_fn.acq_fn.model
        self.beta = acq_fn.acq_fn.beta if acq_fn.tag == 'ucbhvi' else 0.1

    def _evaluate(self, x, out, *args, **kwargs):
        assert x.ndim == 2
        candidates = []
        for query_pt in x:
            cand_idx = query_pt[0]
            base_seq = self.candidate_pool[cand_idx].mutant_residue_seq
            for i in range(self.max_num_edits):
                mut_pos, mut_res_idx, op_idx = query_pt[1+3*i:1+3*(i+1)]
                op_type = self.op_types[op_idx]
                mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
                base_seq = apply_mutation(base_seq, mut_pos, mut_res, op_type, self.tokenizer)
            candidates.append(base_seq)
        candidates = np.array(candidates).reshape(-1)
        with torch.inference_mode():
            posterior = self.surrogate_model.posterior(candidates)
            mean = posterior.mean
            std = torch.sqrt(posterior.variance)
            ucb_vector = (mean + self.beta * std).cpu().numpy()
        out["F"] = -ucb_vector