import numpy as np

import torch

from pymoo.core.mutation import Mutation
from setbench.tools.pymoo_op import get_mutation

from setbench.tools.data_op import str_to_tokens
from setbench.tasks.chem.logp import prop_func
from setbench.models.mlm import sample_tokens


def get_mlm_mutation(mlm_obj, problem, cand_idx, res_idx):
    seqs = [problem.candidate_pool[i].mutant_residue_seq for i in cand_idx]
    base_tok_idxs = str_to_tokens(seqs, mlm_obj.tokenizer)

    mask_idxs = res_idx.reshape(-1, 1)
    src_tok_idxs = base_tok_idxs.clone().to(mlm_obj.device)
    np.put_along_axis(src_tok_idxs, mask_idxs, mlm_obj.tokenizer.padding_idx, axis=1)

    with torch.no_grad():
        tgt_tok_logits, _ = mlm_obj.logits_from_tokens(src_tok_idxs)
    new_tok_idxs, _ = sample_tokens(
        base_tok_idxs, tgt_tok_logits, mlm_obj.tokenizer, replacement=False
    )
    new_tok_idxs = np.take_along_axis(new_tok_idxs, mask_idxs, axis=1).reshape(-1)
    new_toks = [mlm_obj.tokenizer.convert_id_to_token(t_idx) for t_idx in new_tok_idxs]
    sampling_vocab_idxs = np.array([
        mlm_obj.tokenizer.sampling_vocab.index(tok) for tok in new_toks
    ])
    return sampling_vocab_idxs


class LocalMutation(Mutation):
    def __init__(self, eta, prob, tokenizer=None, mlm_obj=None, max_num_edits=1):
        super().__init__()
        self.poly_mutation = get_mutation('int_pm', eta=eta, prob=prob)
        self.tokenizer = tokenizer
        self.mlm_obj = mlm_obj
        self.max_num_edits = max_num_edits

    def _do(self, problem, x, **kwargs):
        query_batches = problem.x_to_query_batches(x)
        batch_shape, num_vars = query_batches.shape[:-1], query_batches.shape[-1]
        flat_queries = query_batches.reshape(-1, num_vars)
        num_samples = flat_queries.shape[0]

        x0 = flat_queries[..., 0]
        samples = [x0]
        for i in range(self.max_num_edits):
            mut_x = self.poly_mutation._do(problem, x)
            mut_x = problem.x_to_query_batches(mut_x).reshape(-1, num_vars)
            x1 = mut_x[..., 3*i+1]

            for i, idx in enumerate(x0):
                num_tokens = len(self.tokenizer.encode(problem.candidate_pool[idx].mutant_residue_seq)) - 2
                x1[i] = min(num_tokens - 1, x1[i])

            if self.mlm_obj is None:
                x2 = np.random.randint(0, len(self.tokenizer.sampling_vocab), num_samples)
            else:
                x2 = get_mlm_mutation(self.mlm_obj, problem, x0, x1)

            x3 = np.random.randint(0, len(problem.op_types), num_samples)
            samples = samples + [x1, x2, x3]

        new_queries = np.stack(samples, axis=-1).reshape(*batch_shape, -1)
        new_x = problem.query_batches_to_x(new_queries)

        return new_x