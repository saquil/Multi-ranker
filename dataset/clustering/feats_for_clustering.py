import numpy as np
import h5py
import os
import argparse

parser = argparse.ArgumentParser(description='sample features for clustering')
parser.add_argument('--dataset', type=str, default='tvsum', choices=['tvsum', 'summe'], help='The name of dataset')
parser.add_argument('--size', type=int, default=6400, help='The sample size')
parser.add_argument('--output', type=str, default='feat_samples_tvsum.npy', choices=['feat_samples_tvsum.npy', 'feat_samples_summe.npy'], help='The output file')
parser.add_argument('--dataset_path', type=str, default='../iccv21_dataset_tvsum_google_pool5.h5', choices=['../iccv21_dataset_summe_google_pool5.h5', '../iccv21_dataset_tvsum_google_pool5.h5'], help='The path of dataset')

args = parser.parse_args()

if args.dataset == 'tvsum':
	args.dataset_path = '../iccv21_dataset_tvsum_google_pool5.h5'
elif args.dataset == 'summe':
	args.dataset_path = '../iccv21_dataset_summe_google_pool5.h5'

data = h5py.File(args.dataset_path, 'r')

videos = list(data.keys())

feats = []

for i in range(args.size):
	vid = np.random.randint(len(videos))
	video = videos[vid]
	idx = np.random.randint(data[video]['features'].shape[0])
	feats.append(data[video]['features'][idx,:])

feats = np.stack(feats, axis=0)

np.save(args.output, feats)
