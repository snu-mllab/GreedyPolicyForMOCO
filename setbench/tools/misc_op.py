import numpy as np
import torch
from collections.abc import MutableMapping

def batched_call(fn, arg_array, batch_size, *args, **kwargs):
    batch_size = arg_array.shape[0] if batch_size is None else batch_size
    num_batches = max(1, arg_array.shape[0] // batch_size)

    if isinstance(arg_array, np.ndarray):
        arg_batches = np.array_split(arg_array, num_batches)
    elif isinstance(arg_array, torch.Tensor):
        arg_batches = torch.split(arg_array, num_batches)
    else:
        raise ValueError
    return [fn(batch, *args, **kwargs) for batch in arg_batches]

class Normalizer(object):
    def __init__(self, loc=0., scale=1.):
        self.loc = loc
        self.scale = np.where(scale != 0, scale, 1.)

    def __call__(self, arr):
        min_val = self.loc - 4 * self.scale
        max_val = self.loc + 4 * self.scale
        clipped_arr = np.clip(arr, a_min=min_val, a_max=max_val)
        norm_arr = (clipped_arr - self.loc) / self.scale

        return norm_arr

    def inv_transform(self, arr):
        return self.scale * arr + self.loc

def flatten_config(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class NewTQDM:
    def __init__(self, iterable, frequency=10):
        self.iterable = iterable
        self.frequency = frequency
        self.current_loop = 0
        self.description = ""

    def set_description(self, desc):
        self.description = desc

    def __iter__(self):
        for item in self.iterable:
            if self.current_loop % self.frequency == 0:
                print(self.description, f"({self.current_loop}/{len(self.iterable)})")
            yield item
            self.current_loop += 1
        print()

def test_misc():
    print("Test batched_call")
    norm = Normalizer(loc=2., scale=2.)
    arr = np.random.randn(1000)
    norm_arr = norm(arr)
    inv_arr = norm.inv_transform(norm_arr)

    print("arr", np.mean(arr), np.std(arr))
    print("norm", np.mean(norm_arr), np.std(norm_arr))
    print("inv", np.mean(inv_arr), np.std(inv_arr))

    print("Test batched_call")
    fn = lambda x: torch.mean(x, axis=0)
    arr = torch.FloatTensor([1,2,3,4,5,6,7,8,9,10])
    output = batched_call(fn, arr, 3)
    print("arr =",arr)
    print("batched_call(torch.mean, arr, 3)")
    print(output)

if __name__ == '__main__':
    test_misc()