import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
# aggregate the evaluations of the models across the dataset splits for a selected validation/test option
parser.add_argument('--model_name', type=str, default='ranker')
parser.add_argument('--save_dir', type=str, default='models/tvsum', choices=['models/tvsum', 'models/summe'])
parser.add_argument('--metric', type=str, default='kendall', choices=['kendall','spearman'])
parser.add_argument('--nsplits', type=int, default=5, help='The number of dataset existing splits')
parser.add_argument('--validation', type=int, default=4, choices=[0,1,2,3,4], help='Which fold of 4-fold cross validation to use. The option 4 denotes the test set')

args = parser.parse_args()

npairs = [2000]
batch_size = [128]

pl = []

for p in npairs:
	for b in batch_size:

		kendall = []
		spearman = []
		kendall_human = []
		spearman_human = []

		for split in range(args.nsplits):
			name = args.model_name + '_b' + str(b) + '_p' + str(p//1000) + '_s' + str(split) + '_v' + str(args.validation)
			hist = np.load(os.path.join(args.save_dir, name, name + '_history.npy'), allow_pickle=True).item()
			kendall.append(hist['kendall'])
			spearman.append(hist['spearman'])
			kendall_human.append(hist['kendall_human'])
			spearman_human.append(hist['spearman_human'])
		
		kendall = np.array(kendall)
		spearman = np.array(spearman)
		kendall_mean = np.mean(kendall, axis=0)
		spearman_mean = np.mean(spearman, axis=0)
		kendall_std = np.std(kendall, axis=0)
		spearman_std = np.std(spearman, axis=0)
		kendall_human_mean = np.mean(kendall_human)
		kendall_human_std = np.std(kendall_human)
		spearman_human_mean = np.mean(spearman_human)
		spearman_human_std = np.std(spearman_human)		

		x = range(kendall.shape[1])
		
		pl = plt.plot(x, kendall_mean, lw=2, label='#pairs='+str(p)+' batch size='+str(b))
		if args.metric == 'kendall':
			print("Standard ranker kendall tau validation-test: [%.8f/%.8f]" % (kendall_mean[-1], kendall_std[-1]))
			plt.fill_between(x, kendall_mean-kendall_std, kendall_mean+kendall_std, facecolor=pl[0].get_color(), alpha=0.2)
		elif args.metric == 'spearman':
			print("Standard ranker spearman rho validation-test: [%.8f/%.8f]" % (spearman_mean[-1], spearman_std[-1]))
			plt.fill_between(x, spearman_mean-spearman_std, spearman_mean+spearman_std, facecolor=pl[0].get_color(), alpha=0.2)

if args.metric == 'kendall':
	mu = np.ones(len(x))*kendall_human_mean
	sig = np.ones(len(x))*kendall_human_std
elif args.metric == 'spearman':
	mu = np.ones(len(x))*spearman_human_mean
	sig = np.ones(len(x))*spearman_human_std

pl = plt.plot(x, mu, lw=2, label='human')
if args.metric == 'kendall':
	print("Human kendall tau validation-test: [%.8f/%.8f]" % (mu[-1], sig[-1]))
elif args.metric == 'spearman':
	print("Human spearman rho validation-test: [%.8f/%.8f]" % (mu[-1], sig[-1]))
plt.fill_between(x, mu-sig, mu+sig, facecolor=pl[0].get_color(), alpha=0.2)
if args.save_dir == 'save_dir/tvsum':
	plt.title('Standard ranker validation performance on TVSum dataset')
elif args.save_dir == 'save_dir/summe':
	plt.title('Standard ranker validation performance on SumMe dataset')				
plt.xlabel('Epoch')
if args.metric == 'kendall':
	plt.ylabel('Kendall Tau')
elif args.metric == 'spearman':
	plt.ylabel('Spearman Rho')

plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()		

path = os.path.join('results', args.model_name + '_' + args.metric + '.jpg')
plt.savefig(path)

