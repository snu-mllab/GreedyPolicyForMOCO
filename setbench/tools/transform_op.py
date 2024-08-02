import torch
import numpy as np
import random

from torch import LongTensor

def padding_collate_fn(batch, padding_value=0.0):
    """
        batch - list or tuple of torch.Tensor
    """
    with torch.no_grad():
        if isinstance(batch[0], tuple):
            k = len(batch[0])
            x = torch.nn.utils.rnn.pad_sequence(
                [b[0] for b in batch], batch_first=True, padding_value=padding_value
            )
            rest = [torch.stack([b[i] for b in batch]) for i in range(1, k)]
            return (x,) + tuple(rest)
        else:
            x = torch.nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=padding_value
            )
            return x

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class StringToLongTensor:
    def __init__(self, tokenizer, max_len=None):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, x: str):
        tok_idxs = self.tokenizer.encode(x)
        tok_idxs = torch.LongTensor(tok_idxs)
        num_tokens = tok_idxs.size(0)
        if self.max_len is not None and num_tokens < self.max_len:
            len_diff = self.max_len - num_tokens
            padding = LongTensor(
                [self.tokenizer.padding_idx] * len_diff
            )
            tok_idxs = torch.cat([tok_idxs, padding])
        elif self.max_len is not None and num_tokens > self.max_len:
            tok_idxs = tok_idxs[:self.max_len]

        return tok_idxs


class SequenceTranslation(object):
    """
    Performs a random cycle rotation of a tokenized sequence up to
    `max_shift` tokens either left or right.
    Assumes the sequence has start and stop tokens and NO padding tokens at the end.
    """

    def __init__(self, max_shift: int):
        self.max_shift = max_shift

    def __call__(self, x: LongTensor, shift=None):
        """
        Args:
            x: LongTensor with shape (num_tokens,)
            shift: (optional) magnitude and direction of shift, randomly sampled if None
        """
        if shift is None:
            shift = random.randint(-self.max_shift, self.max_shift)
        else:
            shift = min(shift, self.max_shift)
            shift = max(shift, -self.max_shift)

        num_valid_tokens = x.size(0) - 2
        if shift < 0:
            shift = -(-shift % num_valid_tokens)
        elif shift > 0:
            shift = shift % num_valid_tokens

        if shift == 0:
            return x

        # don't include start/stop tokens in rotation
        trimmed_x = x[1:-1]
        rot_x = x.clone()
        # left shift
        if shift < 0:
            rot_x[1: num_valid_tokens + shift + 1] = trimmed_x[-shift:]
            rot_x[num_valid_tokens + shift + 1: -1] = trimmed_x[:-shift]
        # right shift
        else:
            rot_x[1: shift + 1] = trimmed_x[-shift:]
            rot_x[shift + 1: -1] = trimmed_x[:-shift]

        return rot_x


class RandomMask(object):
    """
    Randomly replaces original tokens with masking tokens.
    Assumes the sequence has start and stop tokens and NO padding tokens.
    """

    def __init__(self, mask_size: int, masking_idx: int, contiguous: bool = True):
        """
        Args:
            mask_size: number of tokens to mask
            masking_idx: mask token index from tokenizer
            contiguous: if True, consecutive tokens will be masked, otherwise positions will
                        be drawn independently without replacement.
        """
        self.mask_size = mask_size
        self.masking_idx = masking_idx
        self.contiguous = contiguous

    def __call__(self, x: LongTensor):
        """
        Args:
            x: LongTensor with shape (num_tokens,)
        """
        num_tokens = x.size(0)
        # don't mask start or stop tokens
        start_min = 1
        stop_max = num_tokens - 1
        # don't mask all tokens
        if stop_max - start_min <= self.mask_size:
            return x
        # mask consecutive tokens
        if self.contiguous:
            offset = np.random.randint(start_min, start_min + self.mask_size)
            mask_start = np.random.randint(offset, stop_max - self.mask_size)
            mask_stop = mask_start + self.mask_size
            if mask_stop >= stop_max:
                mask_start = stop_max - self.mask_size
                mask_stop = stop_max
            x[mask_start:mask_stop] = self.masking_idx
        # mask random tokens
        else:
            mask_idxs = np.random.choice(
                np.arange(start_min, stop_max), self.mask_size, replace=False
            )
            x[mask_idxs] = self.masking_idx
        return x

def test_transforms():
    from setbench.tools.string_op import ResidueTokenizer
    tokenizer = ResidueTokenizer()
    transforms = [
        StringToLongTensor(tokenizer, max_len=12),
        SequenceTranslation(max_shift=3),
        RandomMask(mask_size=3, masking_idx=tokenizer.masking_idx, contiguous=True),
    ]
    composed_transform = Compose(transforms)
    orig = 'AAGCTT'
    print("orig", orig)
    cur = orig
    for t in transforms:
        cur = t(cur)
        print(t, cur)

    compose = composed_transform(orig[0])
    print("compose", compose)

def test_padding():
    batch1 = (
        torch.LongTensor([1,2,3]),
        torch.LongTensor([2,4,5,0]),
        torch.LongTensor([7,8])
    )
    batch2 = [
        torch.LongTensor([[1],[2],[3]]),
        torch.LongTensor([[2],[4],[5],[0]]),
        torch.LongTensor([[7],[8]])  
    ]

    padded1 = padding_collate_fn(batch1)
    padded2 = padding_collate_fn(batch2)

    print("batch1", batch1)
    print("padded1", padded1)

    print("batch2", batch2)
    print("padded2", padded2)


if __name__ == '__main__':
    test_transforms()
    test_padding()