import re

import numpy as np

from setbench.tasks.base_task import BaseTask
from nupack import Strand, Complex, ComplexSet, Model, SetSpec, complex_analysis
from omegaconf import ListConfig
from setbench.tools.mutation_op import apply_mutation, mutation_list


class NupackTask(BaseTask):
    def __init__(
        self,
        candidate_pool,
        regex_list,
        max_len,
        min_len,
        num_start_examples,
        tokenizer,
        objectives,
        task_name,
        transform=lambda x: x,
        **kwargs
    ):
        obj_dim = len(objectives)
        super().__init__(tokenizer, candidate_pool, obj_dim, max_len, transform, **kwargs)
        self.regex_list = None
        self.min_len = min_len
        self.max_len = max_len
        self.num_start_examples = num_start_examples
        self.max_reward_per_dim = kwargs["max_score_per_dim"]
        self.score_max = kwargs["score_max"]
        self.objectives = objectives
        self.task_name = task_name

    def task_setup(self, *args, **kwargs):
        return [], [], [], []
    
    def _evaluate(self, x, out, *args, **kwargs):
        assert x.ndim == 2
        x_cands, x_seqs, f_vals = [], [], []
        for query_pt in x:
            cand_idx, mut_pos, mut_res_idx, op_idx = query_pt
            op_type = self.op_types[op_idx]
            base_candidate = self.candidate_pool[cand_idx]
            base_seq = base_candidate.mutant_residue_seq
            mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
            # TODO add support for insertion and deletion here
            # mutation_ops = [base_candidate.new_mutation(mut_pos, mut_res, mutation_type='sub')]
            mut_seq = apply_mutation(base_seq, mut_pos, mut_res, op_type, self.tokenizer)
            mutation_ops = mutation_list(base_seq, mut_seq, self.tokenizer)
            candidate = base_candidate.new_candidate(mutation_ops, self.tokenizer)
            x_cands.append(candidate)
            x_seqs.append(candidate.mutant_residue_seq)
        x_seqs = np.array(x_seqs).reshape(-1)
        x_cands = np.array(x_cands).reshape(-1)

        out["X_cand"] = x_cands
        out["X_seq"] = x_seqs
        out["F"] = self.transform(self.score(x_cands))

    def score(self, candidates):
        """
        Computes multi-objective scores for each object in candidates.

        Args
        ----
        candidates : list of Candidate
            Aptamer sequences in letter format.

        Returns
        -------
        scores : np.array
            Multi-objective scores. Shape: [n_candidates, n_objectives]
        """
        if candidates[0].__class__.__name__ == 'StringCandidate':
            str_array = np.array([cand.mutant_residue_seq for cand in candidates])
        else:
            str_array = np.array(candidates)
        if len(str_array.shape) == 2 and str_array.shape[0] == 1:
            str_array = str_array[0]
        scores_dict = self.nupack_score(str_array, objectives=self.objectives)
        scores = [scores_dict[obj] for obj in self.objectives]
        scores = np.stack(scores, axis=-1).astype(np.float64)
        # Normalize and make positive
        scores = scores / self.score_max
        # scores = -1 * scores / self.score_max
        return scores

    def nupack_get_single_score(self, sequence, objectives="energy"):
        energy, ssString = self.nupack_get_single_result(sequence)

        n_pins = 0
        n_pairs = 0
        dict_return = {}
        if "pins" in objectives:
            indA = 0  # hairpin completion index
            for j in range(len(sequence)):
                if ssString[j] == "(":
                    indA += 1
                elif ssString[j] == ")":
                    indA -= 1
                    if indA == 0:  # if we come to the end of a distinct hairpin
                        n_pins += 1
            dict_return.update({"pins": -n_pins})
        if "pairs" in objectives:
            n_pairs = ssString.count("(") 
            dict_return.update({"pairs": -n_pairs})
        if "energy" in objectives:
            dict_return.update(
                {"energy": energy}
            )  # this is already negative by construction in nupack
        if "invlength" in objectives:
            invlength = self.min_len * (1.0 / len(sequence)) 
            dict_return.update({"invlength": -invlength})

        if "open loop" in objectives:
            biggest_loop = 0
            loops = [0]  # size of loops
            counting = 0
            indA = 0
            # loop completion index
            for j in range(len(sequence)):
                if ssString[j] == "(":
                    counting = 1
                    indA = 0
                if (ssString[j] == ".") and (counting == 1):
                    indA += 1
                if (ssString[j] == ")") and (counting == 1):
                    loops.append(indA)
                    counting = 0
            biggest_loop = max(loops)
            dict_return.update({"open loop": -biggest_loop})
        scores = [dict_return[obj] for obj in self.objectives]
        scores = np.stack(scores, axis=-1).astype(np.float64)
        # Normalize and make positive
        scores = -1 * scores / self.score_max
        return scores

    def nupack_get_single_result(self, sequence):
        temperature = 310.0  # Kelvin
        ionic_strength = 1.0  # molar

        strand = Strand(sequence, name="strand")
        comp = Complex([strand], name="comp")

        set = ComplexSet(strands=[strand], complexes=SetSpec(max_size=1, include=[comp]))
        model1 = Model(material="dna", celsius=temperature - 273, sodium=ionic_strength)
        results = complex_analysis(set, model=model1, compute=["mfe"])
        return results[comp].mfe[0].energy, str(results[comp].mfe[0].structure)

    def nupack_score(self, sequences, objectives="energy"):
        """
        Computes the score (energy, number of pins, number of pairs) of the (important)
        most probable structure according the nupack. Nupack requires Linux OS. Nupack is
        preferred over seqfold - more stable and higher quality predictions.

        Args
        ----
        candidates : list
            List of sequences.
            TODO: specify format when decided.

        objectives : string or list of strings
            Nupack objective(s) to return. Multiple objectives will be returned if a list
            of strings is used as argument.
        """
        temperature = 310.0  # Kelvin
        ionic_strength = 1.0  # molar

        energies = np.zeros(len(sequences))
        n_pins = np.zeros(len(sequences)).astype(int)
        n_pairs = 0
        ssStrings = np.zeros(len(sequences), dtype=object)

        # parallel evaluation - fast
        strandList = []
        comps = []
        i = -1
        for sequence in sequences:
            i += 1
            strandList.append(Strand(sequence, name="strand{}".format(i)))
            comps.append(Complex([strandList[-1]], name="comp{}".format(i)))

        set = ComplexSet(strands=strandList, complexes=SetSpec(max_size=1, include=comps))
        model1 = Model(material="dna", celsius=temperature - 273, sodium=ionic_strength)
        results = complex_analysis(set, model=model1, compute=["mfe"])
        for i in range(len(energies)):
            try:
                energies[i] = results[comps[i]].mfe[0].energy
                ssStrings[i] = str(results[comps[i]].mfe[0].structure)
            except Exception as e:
                energies[i], ssStrings[i] = self.nupack_get_single_result(sequences[i])

        dict_return = {}
        if "pins" in objectives:
            for i in range(len(ssStrings)):
                indA = 0  # hairpin completion index
                for j in range(len(sequences[i])):
                    if ssStrings[i][j] == "(":
                        indA += 1
                    elif ssStrings[i][j] == ")":
                        indA -= 1
                        if indA == 0:  # if we come to the end of a distinct hairpin
                            n_pins[i] += 1
            dict_return.update({"pins": -n_pins})
        if "pairs" in objectives:
            n_pairs = np.asarray([ssString.count("(") for ssString in ssStrings]).astype(int)
            dict_return.update({"pairs": -n_pairs})
        if "energy" in objectives:
            dict_return.update(
                {"energy": energies}
            )  # this is already negative by construction in nupack
        if "invlength" in objectives:
            invlength = np.asarray([self.min_len * (1.0 / len(seq)) for seq in sequences])
            dict_return.update({"invlength": -invlength})

        if "open loop" in objectives:
            biggest_loop = np.zeros(len(ssStrings))
            for i in range(
                len(ssStrings)
            ):  # measure all the open loops and return the largest
                loops = [0]  # size of loops
                counting = 0
                indA = 0
                # loop completion index
                for j in range(len(sequences[i])):
                    if ssStrings[i][j] == "(":
                        counting = 1
                        indA = 0
                    if (ssStrings[i][j] == ".") and (counting == 1):
                        indA += 1
                    if (ssStrings[i][j] == ")") and (counting == 1):
                        loops.append(indA)
                        counting = 0
                biggest_loop[i] = max(loops)
            dict_return.update({"open loop": -biggest_loop})

        if isinstance(objectives, (list, ListConfig)):
            if len(objectives) > 1:
                return dict_return
            else:
                return dict_return[objectives[0]]
        else:
            return dict_return[objectives]


def nupack_test():
    from setbench.tools.string_op import AptamerTokenizer
    sequence = ['CGCGCGCGCGCGCCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG',
 'GGCGCGGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCCG',
 'CGCCGCGCGCGCGCGCTGCGCCGCGCGCGCGCTCGCCGCCGCGCGGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCCGCGGCGCGCGACGCGCGCG',
 'CGCGCGCCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCCGCC',
 'CGCGCGCGCGCGCGCGCGCGCGCGGGCGGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCG',
 'GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGGGCGCGCGCGCGCGCGCGCGCGCGCGCGACGCGCGCGCGCGCGCGCGCGCGG',
 'CCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCCGCGC',
 'CCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCC',
 'GCGCCGCGGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCCGCGCACGCGCGCGCGCGCGC',
 'CCCGCGCCGCGCGCGCGCGCGCGGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGC',
 'CCGCGCGCGCACGCGCCGCGCGCGCGCGCGCGCGGCGCGCGCCACGCGCGCGCGCGCGGC',
 'CGCGCGCGCGCGCCGCGGCGCGCGCGCCGGCGCGCGCGCGCGCGCGCGCGCGCGCCGGCG',
 'CCGCGCGCGGCGCGCGCGCGCGCGGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCG',
 'GCGCGCGGCGCGCGGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCCGGCGCGCGCGCGGCGC',
 'GCGCGCGGCGCGCGCGGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGACGCGCGCGCGCGC',
 'GCGCGCCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCCGCGGCGCGCCGCG',
 'CGCGCCGCGCGCGCGCGGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCG',
 'CGCGGCGGCGCGCGCGCGCCGGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCCGCGCGCGCCGCGCGCGCGCCGCGCGCGACACGGCGCGCGCGCGC',
 'CGCGGCCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCACGCCGCGCGCGCGCGGCGCGCGCGCGCG',
 'GCGCGCGCGCGCGGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCCGCGCGGCGCGCGC',
 'GCGCGGCGCGCGCGCGCGGCGCCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCCGCGCC',
 'CGGCGGCGCGGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGGCGCGCGCGCCGCGTCGCGCGCG',
 'GCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCCGCGCGCGCCGCGCGCGCGCG',
 'CCGCCCGCGGCGCGCGCGCGCGCGGCGCGCGGCGCGCGCGCGCCGCGCGCGCGCGCGCCG',
 'CGCGCGCCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCCGCCGCGCGC',
 'CCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGGCGCGGC',
 'GGCGCGCGCGCGCGCGCGCCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCACGGCGCGC',
 'CGCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCCGCGCGCGCGCGGCGCG',
 'GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCACGCGC',
 'CGCGCGGCGCGGCGCGCGCGCGCGCGCGCGCGC',
 'ACGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGTGGCGC',
 'CGGCGCCGCGCGCGCGCGCGCGCGCCGCGCGCGGCCGCGCGGGGCGCGCGCGCGCGCGCG',
 'CCGCGGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG',
 'CGGCGCGCCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCCGCGGCTCGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGGCGGCGCGCGCGCGCCGCGCGCGCGCGAGCGCGCGCGCGC',
 'CCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGGCGCGCGCGCGCGCGCGCGCG',
 'GCGCGCGCGCCGGCGCGCGCGCCGCGCGCGGCGCCGCCGCGCGCCGCGCGCGCCGCCGCG',
 'CCGCGCGCGCGCGCGCGCGCGCCGCCGCGCGGCGCGCGCGCGCGGCGCGCGCCCGCCGCG',
 'CGCGCGCGGCGCGCGCCGCTGCGCGCGCGCCGCCGCGCGCGCGCGCGCGCGCGCGCCGCG',
 'CGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGGC',
 'GCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCCG',
 'GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGGCGCGCGCGCGCCGCCCGCGCGCG',
 'GCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGGCG',
 'CGCGGGCGCGCGCGCGCGCGGGCTGCGCGCGCGCGGCGCGCGCGCGCCGCGCCGCGGCGC',
 'GCGCGCGGCGGCGCGGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGGCGCGCG',
 'CGCGCGCCCGCCGCGGCGCCGCCGCGCGCGCGCGCGCG',
 'CGGCGCCGCGCGCCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGGCGCGCGCGCGCGCCGCGCGCGCGCGG',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGGCGCCGCGCCGCGCGCGCGCGCGCGCGCGCGACGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGCGGCGCGGCGCGCGCGCCCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGGCGCGCGCGCGCTGCGCGCGC',
 'GCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCG',
 'GGCGCACGCGCGGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCACGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGTCGCGCG',
 'GCCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCG',
 'GCGCGCGCGCGCCGCGCGCGCGCGCGCCGCCGCGCGGCGCCGCGCGGCGCGCGCGCCGCG',
 'CGCCGCCGCGGCGCGCGCGCCGCGGCGGCGCGCGCCGGCGCCCGCGCGCGGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC',
 'CGCGCGCGCCGCCGCGCGCGCCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGACGCGCGCGCGCGCGCGCGCGCGCG',
 'CCGCGCGCGGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCG',
 'CGCGCGCCGCGCGCGCGCCGCGCGACGCGCGCGCGCCGCGCGCGGCGCGCGCGGCGCGCG',
 'CGCGCGCGCGCGCCGCGCGCGCGGCCGCGGGGCGCGCGCGCGCGCGGCGCGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGCGACGCGGCGCGCGGCCGCGCCGCGCGCGGCGCGCGCGCG',
 'CGCGCCCGCGGCGCGGCGTCGCGGCGCGCGCGCGCGCGCGCGCGCCGCGCGGCGCGCGCG',
 'CGCGGCGCGCCGCGCGGCGCGCGCCGCGCGCGCGCGCGCCGCGCGCGCG',
 'GCGCCGCGCGCGGGGGCGCGCGCGCGCGCGCGCGTCGCGCGCGCGCGCGCGGCGCGCGCG',
 'GCGCGCGCGCGCGCGCACGCGGCGCGCGCGCGCGCGCGCGCGCGACCGCGCGCGCGCGGC',
 'CGCGCGCGGCGCGCGCGCGCGCGCGCGCGACGCGCGGCGCGACGCGCGCGGCGCGCGCGC',
 'CGCGCGCGCGCACGCGCGCGCGCGGGCGCGCCGCGCGCGCGCGCGCTGGCCCGCGCGGCG',
 'CGCGCGCGCGCGGTGCGCGCAGCGCGCCCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGC',
 'GCGCGGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCCGCGCGGCGCGCGCCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCG',
 'CGCGCGCGCGCCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCCCGCGCGCCG',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGACCGCCGCGCGGCGCGCGCCGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGCGCGGCCGCGCGCGCGCCGCGCGCCCGCGCGCGCGCGCGC',
 'CGGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGGCGCGCGCCGC',
 'CGCGCGGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCCGGCGCGCGGCGCGCGCGCGCGCGCGCGCGGCG',
 'CGCGCGCGGCGCGCGCGCGCAGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC',
 'CGCGGCGCGCGCGCGGGCGCGCGCGCGCGCGGCGCGCGCGGCGCGCCGCGCGCGCGGCGC',
 'CGCGCGCCGCGCCGCGCCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCTGGCGCGCGCGCCGCCGGGCGGCGCGCGCGC',
 'CGCCGCGCGCGCGCGCGCGCGCGGCGCGCGCCGCGCGCGGCGCGCGCGCGCCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCCGCGCGACGCGCGCGCGCGCGCGC',
 'CGCGCGGCGCGCGCGCGCGCGCGCCCGCGCGCGCGCGCGCCGGCGCGCGCGCGGCGCGCG',
 'GGCCGCGGCGGCGCGGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGGCCGCCGCGCGCGCGCGCGCGGCGCGCGCGGCGC',
 'CGCGCGCGGTCGCGCCGGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGG',
 'GCGCGCGCGCGCGCGCCGCGCGGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCG',
 'CCGCGCGCGCGCGCGCGCGCGCCGCGCCGCGCGCGCCGCGCCGCGGCGCGCGCGCGCGCG',
 'CGCGCCGGCGCGCGGCGCGCGCGCGCGCGCGCCGCGCGCGCGGCGCGCGCCGCCGCCGCG',
 'GCGCGCGCGCCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCAGCGCGCCGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGGCGCCGCGCGCGCGGCGCGCGCGCCGCGCGCGCC',
 'CCGCGCGCGCGGCAGCGCGCGCGCGCGCGCGCGCGCGCGCGCCCGCGCGCGGCGCGGCCG',
 'GCCGCGCGCCGCGCGCCGCGGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGGCGCGCGCGCGCGGCCGCGCGCGCCGCGGCGGCCGCGCGCGCGC',
 'CGGCGCGCGCGCGCGGCGCGCGCGCGGCGCGCGCGCGCCGCGCGGCGCGCGCGACGCGCG',
 'CGCGCGCGCGCGCGCCGCGCCGCGCGCGCGCGCGCGCCGCGCGCGCGCGCGCGCG',
 'CGGCGCGCGCCCACGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGC',
 'GCGCGGCGCGCGCGCGGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGGCGCGCGCGC',
 'CGCGCGCGCGCGCGCCGCGCGCCGGCGCGCGCGTCGCGC',
 'CGCGCGCCGCGCCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGC',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGCGCGGCGCGCGCGCTGCGCGC',
 'GGCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGCGCGCGCGCCCGCGCGCGCGCGCGCGCGCGC',
 'GCGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCCGCGCGCGTCGCGGCCGCGCGCGCG',
 'CGCGCGCGCGCGCGCGCGGGGCGCGCGCGCGCGCGCGCGCGCGCGCTCGCGCGCGCGCGC',
 'CGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGGCCCGCGCGGCGTCGCGCGCGCGCT',
 'CGCGCGCGCCGCCGCGGCGCGCGCGCCGCGCGCGGCGCGCGCGCGCGCGCGCGCG',
 'CCGCGCGCGCGCGCGCGGGCGCGCGCGCGCGCCGCGCGCCGCGCGCGGGGCGCGCGCGGC']
    
    tokenizer = AptamerTokenizer()
    nupack_task = NupackTask(
        regex_list=[],
        max_len=60,
        min_len=30,
        num_start_examples=0,
        tokenizer=tokenizer,
        objectives=['energy', 'pins', 'pairs'],
        task_name='nupack',
        max_score_per_dim=16,
        score_max=np.array([60,8,16])
    )
    scores = nupack_task.score(sequence)
    scores2 = [nupack_task.nupack_get_single_score(seq, nupack_task.objectives) for seq in sequence]
    for s1, s2 in zip(scores, scores2):
        assert np.allclose(s1, s2)
    print("Passed nupack test")
    
if __name__ == '__main__':
    nupack_test()