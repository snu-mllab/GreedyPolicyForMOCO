import itertools

import numpy as np
import torch

from cachetools import cached, LRUCache
import copy
import uuid

from setbench.tools.mutation_op import StringSubstitution, StringDeletion, StringInsertion, FoldxMutation

from pathlib import Path

AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
RESIDUE_ALPHABET = ["[PAD]", "[CLS]", "[UNK]", "[MASK]", "[SEP]"] + AMINO_ACIDS + ["0"]

APTAMER_BASES = [
    "A",
    "C",
    "T",
    "G",
]
APTAMER_ALPHABET = ["[PAD]", "[CLS]", "[UNK]", "[MASK]", "[SEP]"] + APTAMER_BASES + ["0"]

TOY_VOCAB = [
    "A",
    "R"
]
TOY_ALPHABET = ["[PAD]", "[CLS]", "[UNK]", "[MASK]", "[SEP]"] + TOY_VOCAB + ["0"]

class IntTokenizer:
    def __init__(self, non_special_vocab, full_vocab, padding_token="[PAD]",
                 masking_token="[MASK]", bos_token="[CLS]", eos_token="[SEP]"):
        self.non_special_vocab = non_special_vocab
        self.full_vocab = full_vocab
        self.special_vocab = set(full_vocab) - set(non_special_vocab)
        self.lookup = {a: i for (i, a) in enumerate(full_vocab)}
        self.inverse_lookup = {i: a for (i, a) in enumerate(full_vocab)}
        self.padding_idx = self.lookup[padding_token]
        self.masking_idx = self.lookup[masking_token]
        self.bos_idx = self.lookup[bos_token]
        self.eos_idx = self.lookup[eos_token]

        self.sampling_vocab = non_special_vocab
        self.non_special_idxs = [self.convert_token_to_id(t) for t in non_special_vocab]
        self.special_idxs = [self.convert_token_to_id(t) for t in self.special_vocab]
    
    @property
    def ntoks(self):
        return len(self.full_vocab)

    @cached(cache=LRUCache(maxsize=int(1e4)))
    def encode(self, seq, use_sep=True):
        if seq.endswith("%"): # hack for softql
            seq = seq[:-1]
            flag = True
            assert use_sep==False
        else:
            flag = False
        tokens = ["[CLS]"]
        buffer = []
        for char in seq:
            if char == '[':
                buffer.append(char)
            elif char == ']':
                buffer.append(char)
                tokens.append(''.join(buffer))
                buffer = []
            elif len(buffer) > 0:
                buffer.append(char)
            else:
                tokens.append(char)
        if use_sep:
            tokens.append("[SEP]")
        output = [self.convert_token_to_id(tok) for tok in tokens]
        if flag:
            return output + [self.full_vocab.index("[SEP]")]
        else:
            return output

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            return self.convert_id_to_token(token_ids)

        tokens = []
        for t_id in token_ids:
            token = self.convert_id_to_token(t_id)
            if token in self.special_vocab and token not in ["[MASK]", "[UNK]"]:
                continue
            tokens.append(token)
        return ' '.join(tokens)

    def convert_id_to_token(self, token_id):
        if torch.is_tensor(token_id):
            token_id = token_id.item()
        assert isinstance(token_id, int)
        return self.inverse_lookup.get(token_id, '[UNK]')

    def convert_token_to_id(self, token):
        unk_idx = self.lookup["[UNK]"]
        return self.lookup.get(token, unk_idx)

    def set_sampling_vocab(self, sampling_vocab=None, max_ngram_size=1):
        if sampling_vocab is None:
            sampling_vocab = []
            for i in range(1, max_ngram_size + 1):
                prod_space = [self.non_special_vocab] * i
                for comb in itertools.product(*prod_space):
                    sampling_vocab.append("".join(comb))
        else:
            new_tokens = set(sampling_vocab) - set(self.full_vocab)
            self.full_vocab.extend(list(new_tokens))
            self.lookup = {a: i for (i, a) in enumerate(self.full_vocab)}
            self.inverse_lookup = {i: a for (i, a) in enumerate(self.full_vocab)}

        self.sampling_vocab = sampling_vocab


class ResidueTokenizer(IntTokenizer):
    def __init__(self):
        super().__init__(AMINO_ACIDS, RESIDUE_ALPHABET)

class AptamerTokenizer(IntTokenizer):
    def __init__(self):
        super().__init__(APTAMER_BASES, APTAMER_ALPHABET)

class ToyTokenizer(IntTokenizer):
    def __init__(self):
        super().__init__(TOY_VOCAB, TOY_ALPHABET)


class StringCandidate:
    def __init__(self, wild_seq, mutation_list, tokenizer, wild_name=None, dist_from_wild=0.):
        self.wild_residue_seq = wild_seq
        self.uuid = uuid.uuid4().hex
        self.mutation_list = mutation_list
        self.wild_name = 'unnamed' if wild_name is None else wild_name
        self.mutant_residue_seq = self.apply_mutations(mutation_list, tokenizer)
        self.dist_from_wild = dist_from_wild
        self.tokenizer = tokenizer

    def __len__(self):
        tok_idxs = self.tokenizer.encode(self.mutant_residue_seq)
        return len(tok_idxs)

    def apply_mutations(self, mutation_list, tokenizer):
        if len(mutation_list) == 0:
            return self.wild_residue_seq

        mutant_seq = copy.deepcopy(self.wild_residue_seq)
        mutant_seq = tokenizer.encode(mutant_seq)[1:-1]
        for mutation_op in mutation_list:
            old_tok_idx = mutation_op.old_token_idx
            mut_pos = mutation_op.token_pos
            if mut_pos < len(mutant_seq):
                assert old_tok_idx == mutant_seq[mut_pos], str(mutation_op)
            if isinstance(mutation_op, StringSubstitution):
                new_tok_idx = mutation_op.new_token_idx
                mutant_seq = mutant_seq[:mut_pos] + [new_tok_idx] + mutant_seq[mut_pos + 1:]
            elif isinstance(mutation_op, StringDeletion):
                mutant_seq = mutant_seq[:mut_pos] + mutant_seq[mut_pos + 1:]
            elif isinstance(mutation_op, StringInsertion):
                new_tok_idx = mutation_op.new_token_idx
                mutant_seq = mutant_seq[:mut_pos] + [new_tok_idx] + mutant_seq[mut_pos:]
            else:
                raise RuntimeError('unrecognized mutation op')

        mutant_seq = tokenizer.decode(mutant_seq).replace(" ", "")
        return mutant_seq

    def new_candidate(self, mutation_list, tokenizer):
        cand_kwargs = dict(
            wild_seq=self.mutant_residue_seq,
            mutation_list=mutation_list,
            tokenizer=tokenizer,
            wild_name=self.wild_name,
            dist_from_wild=self.dist_from_wild + len(mutation_list),
        )
        return StringCandidate(**cand_kwargs)



def random_proteins(num, min_len=200, max_len=250):
    """
        output - np.array of str (num, min_len~max_len)
    """
    alphabet = AMINO_ACIDS

    proteins = []
    for _ in range(num):
        length = np.random.randint(min_len, max_len + 1)
        idx = np.random.choice(len(alphabet), size=length, replace=True)
        proteins.append("".join([alphabet[i] for i in idx]))
    proteins = np.array(proteins)

    return proteins



def random_aptamers(num, min_len=200, max_len=250):
    """
        output - np.array of str (num, min_len~max_len)
    """
    alphabet = APTAMER_BASES

    proteins = []
    for _ in range(num):
        length = np.random.randint(min_len, max_len + 1)
        idx = np.random.choice(len(alphabet), size=length, replace=True)
        proteins.append("".join([alphabet[i] for i in idx]))
    proteins = np.array(proteins)

    return proteins
