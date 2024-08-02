import pickle
import gzip
import numpy as np
from collections import defaultdict

def load_state(path):
    with gzip.open(path, 'rb') as f:
        state = pickle.load(f)
    return state

def get_best_hv(path, n):
    hv_dict = load_state(path)['hv']
    hv_ = [l[n] for l in hv_dict]
    
    hv = max(hv_)
    return hv

task = 'regex_4'
n = 16
print("TASK:", task)
results = defaultdict(dict)

result_dir = f'/home/deokjae/setbench/results_table1/{task}/'

# SetRL
pi_lr = 1e-5
rand = 0.05
train_max_size = 64
n_set_samples = 4

hvs = []
for seed in range(10):
    path = f'setrl/{pi_lr}_{rand}_{n}_{train_max_size}_{n_set_samples}_{seed}.pkl.gz'
    hv_dicts = load_state(result_dir + path)['hv']
    hv_ = [l[n] for l in hv_dicts]
    ind = np.argmax(np.array(hv_))
    hv = hv_[ind]
    hvs.append(hv)
best_hv = sum(hvs) / len(hvs)
best_std = np.std(hvs)
print("Ours", n, pi_lr, rand)
print(f"{best_hv:.3f} ({best_std:.3f})")
results[task][f'Ours'] = f"{best_hv:.3f} ({best_std:.3f})"
        
# MORL
for reward_type in ['tchebycheff', 'convex']:
    if reward_type == 'convex':
        pi_lr = 1e-5
        rand = 0
    else:
        pi_lr = 1e-4
        rand = 0

    hvs = []
    for seed in range(10):
        path = result_dir + f'realmorl_rl_norm/{pi_lr}_{rand}_{reward_type}_{n}_{seed}.pkl.gz'
        hvs.append(get_best_hv(path, n))

    best_hv = sum(hvs) / len(hvs)
    best_std = np.std(hvs)

    print("PC-RL", n, "(reward_type)", pi_lr, rand)
    print(f"{best_hv:.3f} ({best_std:.3f})")
    results[task]['PC-RL'+reward_type] = f"{best_hv:.3f} ({best_std:.3f})"

# Greedy + RL
n = 16
pi_lr = 1e-5
rand = 0.0

hvs = []
for seed in range(10):
    path = f'greedyrl_False/{pi_lr}_{rand}_{n}_{seed}.pkl.gz'
    hv = load_state(result_dir + path)['hv'][0]
    hvs.append(hv)
        
best_hv = sum(hvs) / len(hvs)
best_std = np.std(hvs)

print("Greedy+RL", n, pi_lr, rand)
print(f"{best_hv:.3f} ({best_std:.3f})")
results[task]['Greedy+RL '+f'{n}'] = f"{best_hv:.3f} ({best_std:.3f})"

# Greedy + HC
n = 16

hvs = []
for seed in range(10):
    path = f'randhill/{n}_0.0_20000_{seed}.pkl.gz'
    hv = load_state(result_dir + path)['hv'][0]
    hvs.append(hv)
best_hv = sum(hvs) / len(hvs)
best_std = np.std(hvs)
print(n, f"Greedy+HC {best_hv:.3f} ({best_std:.3f})")
results[task]['Greedy+HC '+f'{n}'] = f"{best_hv:.3f} ({best_std:.3f})"

# Greedy + RS
n = 16

hvs = []
for seed in range(10):
    path = f'randsearch/{n}_0.0_20000_{seed}.pkl.gz'
    hv = load_state(result_dir + path)['hv'][0]
    hvs.append(hv)
best_hv = sum(hvs) / len(hvs)
best_std = np.std(hvs)
print(n, f"Greedy+RS {best_hv:.3f} ({best_std:.3f})")
results[task]['Greedy+RS '+f'{n}'] = f"{best_hv:.3f} ({best_std:.3f})"


# print(results) in latex table style
print("Method & Bigram-2 & Unigram-2 & Bigram-3 & Unigram-3 & Bigram-4 & Unigram-4 & NUPACK \\\\")
for method in results[task]:
    print(method, end=' ')
    print('&', results[task][method], end=' ')
    print('\\\\')