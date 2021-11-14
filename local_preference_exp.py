import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
# aggregate the local evaluations of the multi-ranker models across the dataset splits for selected validation/test, number of pairs, batch size, lambda and preference options
parser.add_argument('--model_name', type=str, default='multi_ranker')
parser.add_argument('--save_dir', type=str, default='models/tvsum', choices=['models/tvsum', 'models/summe'])
parser.add_argument('--metric', type=str, default='kendall', choices=['kendall','spearman'])
parser.add_argument('--nsplits', type=int, default=5, help='The number of dataset existing splits')
parser.add_argument('--validation', type=int, default=4, choices=[0,1,2,3,4], help='Which fold of 4-fold cross validation to use. The option 4 denotes the test set')


args = parser.parse_args()

npairs = 2000
batch_size = 128

npref = [4]
lbda = [0.5]

pl = []

for pref in npref:
	for l in lbda:

		kendall_local = []
		spearman_local = []

		for split in range(args.nsplits):
			name = args.model_name + '_pr' + str(pref) + '_l' + str(l) + '_b' + str(batch_size) + '_p' + str(npairs//1000) + '_s' + str(split) + '_v' + str(args.validation)
			hist = np.load(os.path.join(args.save_dir, name, name + '_history.npy'), allow_pickle=True).item()

			kendall_local.append(hist['local'][0])
			spearman_local.append(hist['local'][1])

		kendall_local = np.array(kendall_local)
		spearman_local = np.array(spearman_local)
		kendall_mean_local = np.mean(kendall_local, axis=0)
		spearman_mean_local = np.mean(spearman_local, axis=0)
		kendall_std_local = np.std(kendall_local, axis=0)
		spearman_std_local = np.std(spearman_local, axis=0)

		
		if args.metric == 'kendall':
			for i in range(kendall_local.shape[1]):		
				print("Kendall tau local validation-test: [%.8f/%.8f]" % (kendall_mean_local[i], kendall_std_local[i]))	

		elif args.metric == 'spearman':
			for i in range(spearman_local.shape[1]):
				print("Spearman rho local validation-test: [%.8f/%.8f]" % (spearman_mean_local[i], spearman_std_local[i]))
			

