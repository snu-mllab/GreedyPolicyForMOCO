import numpy as np
from scipy.stats import rankdata
from scipy.special import softmax
import random
import itertools
import math

def generate_simplex(dims, n_per_dim):
    spaces = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(dims)]
    return np.array([comb for comb in itertools.product(*spaces) 
                     if np.allclose(sum(comb), 1.0)])


def weighted_resampling(scores, k=1., num_samples=None):
    """
    Multi-objective ranked resampling weights.
    Assumes scores are being minimized.

    Args:
        scores: (num_rows, num_scores)
        k: softmax temperature
        num_samples: number of samples to draw (with replacement)
    """
    num_rows = scores.shape[0]
    scores = scores.reshape(num_rows, -1)

    ranks = rankdata(scores, method='dense', axis=0)  # starts from 1
    ranks = ranks.max(axis=-1)  # if A strictly dominates B it will have higher weight.

    weights = softmax(-np.log(ranks) / k)

    num_samples = num_rows if num_samples is None else num_samples
    resampled_idxs = np.random.choice(
        np.arange(num_rows), num_samples, replace=True, p=weights
    )
    return ranks, weights, resampled_idxs

def draw_bootstrap(*arrays, bootstrap_ratio=0.632, min_samples=1):
    """
    Returns bootstrapped arrays that (in expectation) have `bootstrap_ratio` proportion
    of the original rows. The size of the bootstrap is computed automatically.
    For large input arrays, the default value will produce a bootstrap
    the same size as the original arrays.

    :param arrays: indexable arrays (e.g. np.ndarray, torch.Tensor)
    :param bootstrap_ratio: float in the interval (0, 1)
    :param min_samples: (optional) instead specify the minimum size of the bootstrap
    :return: bootstrapped arrays
    """
    num_data = arrays[0].shape[0]
    assert all(arr.shape[0] == num_data for arr in arrays)

    if bootstrap_ratio is None:
        num_samples = min_samples
    else:
        assert bootstrap_ratio < 1
        num_samples = int(math.log(1 - bootstrap_ratio) / math.log(1 - 1 / num_data))
        num_samples = max(min_samples, num_samples)

    idxs = random.choices(range(num_data), k=num_samples)
    res = [arr[idxs] for arr in arrays]
    return res

def test_generate_simplex(dims, n_per_dim):
    print("Testing generate_simplex")
    print(f"dims {dims}, n_per_dim {n_per_dim}")
    simplexes = generate_simplex(dims, n_per_dim)
    print(simplexes[:5])
    print(simplexes.shape)

def test_weight_resampling():
    print("Testing weighted_resampling")
    scores = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.1], [0.15, 0.15, 0.2]])
    print(scores.shape)
    print(scores)
    print(weighted_resampling(scores, k=0.1))
    print(weighted_resampling(scores, k=10))

def test_draw_bootstrap():
    print("Testing draw_bootstrap")
    arrays = [np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])]
    print("arrays", arrays)
    print("bootstrap_ratio 0.632, min_samples 1")
    print(draw_bootstrap(*arrays, bootstrap_ratio=0.632, min_samples=1))

if __name__ == '__main__':
    test_generate_simplex(2,50)
    # test_generate_simplex(3,50)
    test_generate_simplex(4,10)
    # test_weight_resampling()
    # test_draw_bootstrap()