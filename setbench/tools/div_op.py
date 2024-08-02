from polyleven import levenshtein
import itertools
import numpy as np


def is_similar(seq_a, seq_b, dist_type="edit", threshold=0):
    if dist_type == "edit":
        return edit_dist(seq_a, seq_b) < threshold
    return False

def edit_dist(seq1, seq2):
    return levenshtein(seq1, seq2) / 1.

def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)

def sum_pairwise_distances(seqs1, seqs2):
    sum_dist = 0
    for seq1 in seqs1:
        for seq2 in seqs2:
            sum_dist += edit_dist(seq1, seq2)
    return sum_dist