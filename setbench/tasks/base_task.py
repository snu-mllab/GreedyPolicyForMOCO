import numpy as np
from pymoo.core.problem import Problem

from setbench.tools.mutation_op import mutation_list


class BaseTask(Problem):
    def __init__(self, tokenizer, candidate_pool, obj_dim, max_len=None, transform=lambda x: x, batch_size=1,
                 candidate_weights=None, max_ngram_size=1, allow_len_change=True, **kwargs):
        self.op_types = ['sub', 'ins', 'del'] if allow_len_change else ['sub']
        self.max_num_edits = kwargs.get('max_num_edits', 1)
        if max_len is None:
            max_len = max([
                len(tokenizer.encode(cand.mutant_residue_seq)) - 2 for cand in candidate_pool
            ]) - 1
        if len(candidate_pool) == 0:
            xl = 0.
            xu = 1.
        else:
            xl = np.array([0] * (1+3*self.max_num_edits) * batch_size)
            xu = ([len(candidate_pool) - 1] + [2 * max_len, len(tokenizer.sampling_vocab) - 1, len(self.op_types) - 1] * self.max_num_edits) * batch_size
            xu = np.array(xu)

        n_var = (1+3*self.max_num_edits) * batch_size
        super().__init__(
            n_var=n_var, n_obj=obj_dim, n_constr=0, xl=xl, xu=xu, type_var=int
        )
        self.tokenizer = tokenizer
        self.candidate_pool = candidate_pool
        self.candidate_weights = candidate_weights
        self.obj_dim = obj_dim
        self.transform = transform
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_ngram_size = max_ngram_size
        self.allow_len_change = allow_len_change

    def make_new_candidates(self, base_candidates, new_seqs):
        assert base_candidates.shape[0] == new_seqs.shape[0]
        new_candidates = []
        for b_cand, n_seq in zip(base_candidates, new_seqs):
            mutation_ops = mutation_list(
                b_cand.mutant_residue_seq,
                n_seq,
                self.tokenizer
            )
            new_candidates.append(b_cand.new_candidate(mutation_ops, self.tokenizer))
        return np.stack(new_candidates)

    def task_setup(self, *args, **kwargs):
        raise NotImplementedError

    def x_to_query_batches(self, x):
        return x.reshape(-1, self.batch_size, 1+3*self.max_num_edits)

    def query_batches_to_x(self, query_batches):
        return query_batches.reshape(-1, self.n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        raise NotImplementedError

    def score(self, str_array):
        raise NotImplementedError

    def is_feasible(self, candidates):
        if self.max_len is None:
            is_feasible = np.ones(candidates.shape).astype(bool)
        else:
            is_feasible = np.array([len(cand) <= self.max_len for cand in candidates]).reshape(-1)
        return is_feasible