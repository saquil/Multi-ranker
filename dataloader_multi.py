from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
import torch
import numpy as np
import os
import h5py
import json

# load the dataset of images as global to avoid multiple allocations
data = None
data_pair = None
data_pair_multi = None

class Videos_pairs(Dataset):
	def __init__(self, args, setting):

		global data_pair
		global data
		global data_pair_multi
		
		if type(data_pair) == type(None):
			self.data_pair = np.load(args.pairset, allow_pickle=True)
			data_pair = self.data_pair
			self.data_pair_multi = np.load(args.pairset_multi, allow_pickle=True)
			data_pair_multi = self.data_pair_multi
			self.data = h5py.File(args.dataset_path, 'r')
			data = self.data
		else:
			self.data_pair = data_pair
			self.data_pair_multi = data_pair_multi
			self.data = data

		self.n_vid = self.data_pair_multi.shape[0]
		self.splits = json.load(open(args.split_path))
		self.setting = setting
		self.preference = args.preference
		if self.setting == 'test':
			self.idxs = self.splits[args.split][self.setting]
		else:
			self.idxs = self.splits[args.split]['train_val'][args.validation][self.setting]
		self.data_len = len(self.idxs)*self.data_pair_multi.shape[1]
	
	def __getitem__(self, index):
		i = index//self.n_pairs()
		v_idx = self.idxs[i]
		idx = int(v_idx.split('_')[1]) - 1
		d1_tensor = self.data[ v_idx ]['features'][ int(self.data_pair_multi[idx][index % self.n_pairs()][0]) ]
		d2_tensor = self.data[ v_idx ]['features'][ int(self.data_pair_multi[idx][index % self.n_pairs()][1]) ]
		label_multi = self.data_pair_multi[ idx ][index % self.n_pairs()][2]
		label = self.data_pair[ idx ][(index % self.n_pairs())//self.preference][2]
		pref = self.data_pair_multi[ idx ][index % self.n_pairs()][3]		

		return d1_tensor, d2_tensor, label_multi, label, pref

	def __len__(self):
		return self.data_len

	def n_vids(self):
		return self.n_vid

	def n_pairs(self):
		return self.data_pair_multi.shape[1]

def dataloader(args):

	data_loader = DataLoader(Videos_pairs(args, 'train'), batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(Videos_pairs(args, 'val'), batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(Videos_pairs(args, 'test'), batch_size=args.batch_size, shuffle=False) 
	return data_loader, val_loader, test_loader

