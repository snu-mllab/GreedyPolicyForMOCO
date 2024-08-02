import numpy as np
from pymoo.core.sampling import Sampling


from setbench.optimizers.mutation import get_mlm_mutation


def _draw_samples(tokenizer, cand_pool, problem, num_samples, mlm_obj=None, max_num_edits=1):
    cand_weights = problem.candidate_weights
    if cand_weights is None:
        x0 = np.random.choice(
            np.arange(len(cand_pool)), num_samples, replace=True
        )
    else:
        x0 = np.random.choice(
            np.arange(len(cand_pool)), num_samples, p=cand_weights, replace=True
        )
    samples = [x0]
    for i in range(max_num_edits):
        # don't sample start or end token indexes
        x1 = []
        for idx in x0:
            num_tokens = len(tokenizer.encode(cand_pool[idx].mutant_residue_seq)) - 2
            x1.append(np.random.randint(0, num_tokens))
            # TODO always work with token indices?
            # num_tokens = len(tokenizer.encode(cand_pool[idx].mutant_residue_seq))
            # x1.append(np.random.randint(1, num_tokens - 1))
        x1 = np.array(x1)

        if mlm_obj is None:
            x2 = np.random.randint(0, len(tokenizer.sampling_vocab), num_samples)
        else:
            x2 = get_mlm_mutation(mlm_obj, problem, x0, x1)

        x3 = np.random.randint(0, len(problem.op_types), num_samples)
        samples = samples + [x1, x2, x3]

    return np.stack(samples, axis=-1)


class CandidateSampler(Sampling):
    def __init__(self, tokenizer=None, var_type=np.float64, mlm_obj=None, max_num_edits=1) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.var_type = var_type
        self.mlm_obj = mlm_obj
        self.max_num_edits = max_num_edits

    def _do(self, problem, n_samples, *args, **kwargs):
        cand_pool = problem.candidate_pool
        x = _draw_samples(self.tokenizer, cand_pool, problem, n_samples, self.mlm_obj, max_num_edits=self.max_num_edits)
        return x


class BatchSampler(CandidateSampler):
    def __init__(self, batch_size, tokenizer=None, var_type=np.float64, mlm_obj=None, max_num_edits=1) -> None:
        super().__init__(var_type)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.var_type = var_type
        self.mlm_obj = mlm_obj
        self.max_num_edits = max_num_edits

    def _do(self, problem, n_samples, *args, **kwargs):
        cand_pool = problem.candidate_pool
        batches = np.stack([
            _draw_samples(self.tokenizer, cand_pool, problem, self.batch_size, self.mlm_obj, max_num_edits=self.max_num_edits) for _ in range(n_samples)
        ])
        x = problem.query_batches_to_x(batches)
        return x