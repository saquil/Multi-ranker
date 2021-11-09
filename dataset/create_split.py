import json
import numpy as np
import argparse
import h5py

parser = argparse.ArgumentParser('Creating random splits for training, validation, and testing protocols')

parser.add_argument('--dataset', type=str, default='iccv21_dataset_tvsum_google_pool5.h5', choices=['iccv21_dataset_tvsum_google_pool5.h5', 'iccv21_dataset_summe_google_pool5.h5'])
parser.add_argument('--save', type=str, default='splits_tvsum.json', choices=['splits_tvsum.json', 'splits_summe.json'], help='output file of random splits')
parser.add_argument('--num_splits', type=int, default=5, help='number of dataset splits')
parser.add_argument('--num_vals', type=int, default=4, help='number of cross-validation folds')

args = parser.parse_args()

dataset = h5py.File(args.dataset, 'r')
keys = dataset.keys()
num_videos = len(keys)
num_train = int(num_videos*0.8) # 80% of videos are non-test set and 20% are test set
num_test = num_videos - num_train
num_val = num_test # the number of test videos is equal to the number of validation videos
splits = []

for idx in range(args.num_splits):
	rnd_idxs = np.random.choice(range(num_videos), size=num_train, replace=False)
	rem = rnd_idxs
	val_idxs = [None]*args.num_vals
	train_vals = []

	for i in range(args.num_vals):
		val_idxs[i] = np.random.choice(rem, size=num_val, replace=False)
		rem = [ r for r in rem if r not in val_idxs[i] ]
		train_keys, val_keys = [], []

		for idx, key in enumerate(keys):
			if idx in val_idxs[i]:
				val_keys.append(key)
			elif idx in rnd_idxs:
				train_keys.append(key)

		train_vals.append({'train': train_keys, 'val': val_keys})
	
	train_keys, val_keys, test_keys = [], [], []
	
	for idx, key in enumerate(keys):
		if idx in rnd_idxs:
			train_keys.append(key)
		else:
			test_keys.append(key)

	train_vals.append({'train': train_keys, 'val': val_keys})
	splits.append({'train_val': train_vals, 'test': test_keys})

	json.dump(splits, open(args.save, 'w'), indent=4, separators=(',', ': '))	
