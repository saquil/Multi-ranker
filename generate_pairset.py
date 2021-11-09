import numpy as np
import h5py
import os
import argparse

parser = argparse.ArgumentParser(description='Generating segment-level pairwise comparisons per each video')
parser.add_argument('--dataset', type=str, default='./dataset/iccv21_dataset_tvsum_google_pool5.h5', choices=['./dataset/iccv21_dataset_tvsum_google_pool5.h5', './dataset/iccv21_dataset_summe_google_pool5.h5'])
parser.add_argument('--npairs', type=int, default=2000, help='number of pairwise comparisons per each video')
parser.add_argument('--output', type=str, default='./pairset/tvsum/pairs_2k.npy', choices=['./pairset/tvsum/pairs_2k.npy', './pairset/summe/pairs_2k.npy'], help='destination file of pairwise comparisons')
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(os.path.dirname(args.output))

dataset = h5py.File(args.dataset, 'r')

nvids = len(dataset)
pairs = []

for i in range(nvids):
	list_pair = []
	gt_scores = dataset['video_' + str(i+1)]['gtscore']
	for j in range(args.npairs):
		pair = tuple(np.random.randint(len(gt_scores), size=2))
		label = gt_scores[pair[0]] == gt_scores[pair[1]]

		if not label:
			label = 1.0 if gt_scores[pair[0]] > gt_scores[pair[1]] else 0.0
		else:
			label = 0.5

		list_pair.append(np.array([pair[0], pair[1], label]))
	pairs.append(np.array(list_pair))

np.save(args.output, np.array(pairs))
		
	
