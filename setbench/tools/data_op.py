from collections.__init__ import namedtuple
import numpy as np
import torch

from setbench.tools.transform_op import padding_collate_fn
from torch.utils.data import Dataset

fields = ("inputs", "targets")
defaults = (np.array([]), np.array([]))
DataSplit = namedtuple("DataSplit", fields, defaults=defaults)

class TransformTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, idx):
        x = self.tensors[0][idx]

        if self.transform:
            x = self.transform(x)

        if len(self.tensors) >= 2:
            rest = [self.tensors[i][idx] for i in range(1, len(self.tensors))]
            return (x,) + tuple(rest)
        else:
            return x

    def __len__(self):
        return len(self.tensors[0])

    def random_split(self, size_1, size_2):
        assert size_1 + size_2 == len(self)
        idxs = np.random.permutation(len(self))
        split_1 = TransformTensorDataset(
            [tensor[idxs[:size_1]] for tensor in self.tensors], self.transform
        )
        split_2 = TransformTensorDataset(
            [tensor[idxs[size_1:]] for tensor in self.tensors], self.transform
        )
        return split_1, split_2

def update_splits(
    train_split: DataSplit,
    val_split: DataSplit,
    test_split: DataSplit,
    new_split: DataSplit,
    holdout_ratio: float = 0.2,
):
    r"""
    This utility function updates train, validation and test data splits with
    new observations while preventing leakage from train back to val or test.
    New observations are allocated proportionally to prevent the
    distribution of the splits from drifting apart.

    New rows are added to the validation and test splits randomly according to
    a binomial distribution determined by the holdout ratio. This allows all splits
    to be updated with as few new points as desired. In the long run the split proportions
    will converge to the correct values.
    """
    train_inputs, train_targets = train_split
    val_inputs, val_targets = val_split
    test_inputs, test_targets = test_split

    # shuffle new data
    new_inputs, new_targets = new_split
    new_perm = np.random.permutation(
        np.arange(new_inputs.shape[0])
    )
    new_inputs = new_inputs[new_perm]
    new_targets = new_targets[new_perm]

    unseen_inputs = safe_np_cat([test_inputs, new_inputs])
    unseen_targets = safe_np_cat([test_targets, new_targets])

    num_rows = train_inputs.shape[0] + val_inputs.shape[0] + unseen_inputs.shape[0]
    num_test = min(
        np.random.binomial(num_rows, holdout_ratio / 2.),
        unseen_inputs.shape[0],
    )
    num_test = max(test_inputs.shape[0], num_test) if test_inputs.size else max(1, num_test)

    # first allocate to test split
    test_split = DataSplit(unseen_inputs[:num_test], unseen_targets[:num_test])

    resid_inputs = unseen_inputs[num_test:]
    resid_targets = unseen_targets[num_test:]
    resid_inputs = safe_np_cat([val_inputs, resid_inputs])
    resid_targets = safe_np_cat([val_targets, resid_targets])

    # then allocate to val split
    num_val = min(
        np.random.binomial(num_rows, holdout_ratio / 2.),
        resid_inputs.shape[0],
    )
    num_val = max(val_inputs.shape[0], num_val) if val_inputs.size else max(1, num_val)
    val_split = DataSplit(resid_inputs[:num_val], resid_targets[:num_val])

    # train split gets whatever is left
    last_inputs = resid_inputs[num_val:]
    last_targets = resid_targets[num_val:]
    train_inputs = safe_np_cat([train_inputs, last_inputs])
    train_targets = safe_np_cat([train_targets, last_targets])
    train_split = DataSplit(train_inputs, train_targets)

    return train_split, val_split, test_split

def to_tensor(*arrays, device=torch.device('cpu')):
    tensors = []
    for arr in arrays:
        if isinstance(arr, torch.Tensor):
            tensors.append(arr.to(device))
        else:
            tensors.append(torch.tensor(arr, device=device))

    if len(arrays) == 1:
        return tensors[0]

    return tensors

def to_cuda(batch):
    if torch.cuda.is_available():
        return tuple([x.to("cuda") for x in batch])
    else:
        return batch

def safe_np_cat(arrays, **kwargs):
    if all([arr.size == 0 for arr in arrays]):
        return np.array([])
    cat_arrays = [arr for arr in arrays if arr.size]
    return np.concatenate(cat_arrays, **kwargs)


def str_to_tokens(str_array, tokenizer, use_sep=True, return_len=False):
    tokens = [
        torch.tensor(tokenizer.encode(x, use_sep)) for x in str_array
    ]
    batch = padding_collate_fn(tokens, tokenizer.padding_idx)
    if return_len:
        lens = torch.tensor([len(x) for x in tokens])
        return batch, lens
    else:
        return batch


def tokens_to_str(tok_idx_array, tokenizer):
    str_array = np.array([
        tokenizer.decode(token_ids).replace(' ', '') for token_ids in tok_idx_array
    ])
    return str_array

def test_something():
    from setbench.tools.string_op import ResidueTokenizer
    tokenizer = ResidueTokenizer()
    str_array = [''] * 10
    x = str_to_tokens(str_array, tokenizer).to('cuda').t()[:1]
    print(x)
    print(x.shape)
    x = str_to_tokens(str_array, tokenizer, use_sep=False).to('cuda').t()[:1]
    print(x)
    print(x.shape)
if __name__ == '__main__':
    test_something()

