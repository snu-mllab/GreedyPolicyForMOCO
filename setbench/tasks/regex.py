import re

import numpy as np

from setbench.tasks.base_task import BaseTask
from setbench.tools.mutation_op import apply_mutation, mutation_list
from setbench.tools.string_op import StringCandidate, random_proteins


class RegexTask(BaseTask):
    def __init__(self, candidate_pool, regex_list, max_len, min_len, num_start_examples, tokenizer,
                 obj_dim, task_name, transform=lambda x: x, **kwargs):
        super().__init__(tokenizer, candidate_pool, obj_dim, max_len, transform, **kwargs)
        self.regex_list = regex_list
        self.max_len = max_len
        self.min_len = min_len
        self.num_start_examples = num_start_examples
        self.max_reward_per_dim = kwargs['max_score_per_dim']
        self.task_name = task_name

    def task_setup(self, *args, **kwargs):
        num_examples = 0
        selected_seqs = []
        selected_targets = []
        while num_examples < self.num_start_examples:
            # account for start and stop tokens
            all_seqs = random_proteins(self.num_start_examples, self.min_len, self.max_len - 2)
            base_candidates = np.array([
                StringCandidate(seq, mutation_list=[], tokenizer=self.tokenizer) for seq in all_seqs
            ]).reshape(-1)
            all_targets = self.score(base_candidates)
            positive_example_mask = (all_targets < 0).sum(-1).astype(bool)
            num_positive = positive_example_mask.astype(int).sum()
            num_negative = all_targets.shape[0] - num_positive
            num_selected = min(num_positive, num_negative)

            selected_seqs.append(all_seqs[positive_example_mask][:num_selected])
            selected_targets.append(all_targets[positive_example_mask][:num_selected])
            selected_seqs.append(all_seqs[~positive_example_mask][:num_selected])
            selected_targets.append(all_targets[~positive_example_mask][:num_selected])
            num_examples += num_selected

        all_seqs = np.concatenate(selected_seqs)[:self.num_start_examples]
        all_targets = np.concatenate(selected_targets)[:self.num_start_examples]

        base_candidates = np.array([
            StringCandidate(seq, mutation_list=[], tokenizer=self.tokenizer) for seq in all_seqs
        ]).reshape(-1)
        base_targets = all_targets.copy()

        return base_candidates, base_targets, all_seqs, all_targets

    def _evaluate(self, x, out, *args, **kwargs):
        assert x.ndim == 2
        x_cands, x_seqs, f_vals = [], [], []
        for query_pt in x:
            cand_idx = query_pt[0]
            base_candidate = self.candidate_pool[cand_idx]
            base_seq = base_candidate.mutant_residue_seq
            mut_seq = base_seq
            for i in range(self.max_num_edits):
                mut_pos, mut_res_idx, op_idx = query_pt[1+3*i:1+3*(i+1)]
                op_type = self.op_types[op_idx]
                mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
                mut_seq = apply_mutation(mut_seq, mut_pos, mut_res, op_type, self.tokenizer)
            mutation_ops = mutation_list(base_seq, mut_seq, self.tokenizer)
            candidate = base_candidate.new_candidate(mutation_ops, self.tokenizer)
            x_cands.append(candidate)
            x_seqs.append(candidate.mutant_residue_seq)
        x_seqs = np.array(x_seqs).reshape(-1)
        x_cands = np.array(x_cands).reshape(-1)

        out["X_cand"] = x_cands
        out["X_seq"] = x_seqs
        if kwargs.get('return_score', True):
            out["F"] = self.transform(self.score(x_cands))

    def score(self, candidates):
        if candidates[0].__class__.__name__ == 'StringCandidate':
            str_array = np.array([cand.mutant_residue_seq for cand in candidates])
        else:
            str_array = np.array(candidates)
        scores = []
        for regex in self.regex_list:
            scores.append(np.array([
                len(re.findall(regex, str(x))) / self.max_reward_per_dim for x in str_array
            ]).reshape(-1))
        scores = -np.stack(scores, axis=-1).astype(np.float64)
        return scores