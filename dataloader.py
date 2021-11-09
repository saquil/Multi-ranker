from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
import torch
import numpy as np
import os
import h5py
import json

# load the dataset of pairwise comparisons as global to avoid multiple allocations
data = None
data_pair = None

class Videos_pairs(Dataset):
	def __init__(self, args, setting):

		global data_pair
		global data
		
		if type(data_pair) == type(None):
			self.data_pair = np.load(args.pairset, allow_pickle=True)
			data_pair = self.data_pair
			self.data = h5py.File(args.dataset_path, 'r')
			data = self.data
		else:
			self.data_pair = data_pair
			self.data = data

		self.splits = json.load(open(args.split_path))
		self.setting = setting
		if self.setting == 'test':
			self.idxs = self.splits[args.split][self.setting]
		else:
			self.idxs = self.splits[args.split]['train_val'][args.validation][self.setting]

		self.n_vid = self.data_pair.shape[0]
		self.data_len = self.data_pair.shape[1]*len(self.idxs)
	
	def __getitem__(self, index):
		i = index//self.n_pairs()
		v_idx = self.idxs[i]
		idx = int(v_idx.split('_')[1]) - 1
		d1_tensor = self.data[ v_idx ]['features'][ int(self.data_pair[idx][index % self.n_pairs()][0]) ]
		d2_tensor = self.data[ v_idx ]['features'][ int(self.data_pair[idx][index % self.n_pairs()][1]) ]
		label = self.data_pair[ idx ][ index % self.n_pairs() ][2]

		return d1_tensor, d2_tensor, label

	def __len__(self):
		return self.data_len

	def n_vids(self):
		return self.n_vid

	def n_pairs(self):
		return self.data_pair.shape[1]

def dataloader(args):

	data_loader = DataLoader(Videos_pairs(args, 'train'), batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(Videos_pairs(args, 'val'), batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(Videos_pairs(args, 'test'), batch_size=args.batch_size, shuffle=False) 
	return data_loader, val_loader, test_loader

