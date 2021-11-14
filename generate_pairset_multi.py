import numpy as np
import h5py
import os
import argparse

parser = argparse.ArgumentParser(description='Generating segment-level pairwise comparisons per each video w.r.t each preference')
parser.add_argument('--pairset', type=str, default='pairset/tvsum/pairs_2k.npy', help='global pairwise comparisons file')
parser.add_argument('--scores', type=str, default='dataset/clustering/scores_tvsum_4.npy', help='segment-level GT summaries per preference')
parser.add_argument('--output', type=str, default='pairset/tvsum/pairs_multi_2k_4.npy', help='local pairwise comparisons per each preference')
args = parser.parse_args()

pairs = np.load(args.pairset, encoding='latin1', allow_pickle=True)
scores = np.load(args.scores, encoding='latin1', allow_pickle=True)

pref = scores.shape[1]

pairs_multi = np.zeros((pairs.shape[0], pref*pairs.shape[1], 4))

for i in range(pairs_multi.shape[0]):
	for j in range(pairs_multi.shape[1]):

		d = pairs[i][j//pref]

		l = scores[i][j%pref][int(d[0])] == scores[i][j%pref][int(d[1])] 

		if not l:
			l = 1 if scores[i][j%pref][int(d[0])] > scores[i][j%pref][int(d[1])] else 0
		else:
			l = 0.5

		pairs_multi[i][j] = np.array([d[0], d[1], l, j%pref])

np.save(args.output, pairs_multi)
