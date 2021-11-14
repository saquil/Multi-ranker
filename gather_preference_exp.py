import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
# aggregate the evaluations of the models across the dataset splits for selected validation/test, number of pairs, batch size, lambda and preference options
parser.add_argument('--model_name', type=str, default='multi_ranker')
parser.add_argument('--save_dir', type=str, default='models/tvsum', choices=['models/tvsum', 'models/summe'])
parser.add_argument('--metric', type=str, default='kendall', choices=['kendall','spearman'])
parser.add_argument('--nsplits', type=int, default=5, help='The number of dataset existing splits')
parser.add_argument('--validation', type=int, default=4, choices=[0,1,2,3,4], help='Which fold of 4-fold cross validation to use. The option 4 denotes the test set')
parser.add_argument('--mode', type=str, default='lambda', choices=['lambda', 'preference'], help='"lambda" show the global and local evaluations with different lambda values. "preference" show the global evaluations with different number of preferences')

args = parser.parse_args()

npairs = 2000
batch_size = 128

if args.mode == 'preference':
	#npref = [2,4,8]
	npref = [4]
	lbda = [0.5]
elif args.mode == 'lambda':
	npref = [4]
	#lbda = [0.25,0.5,0.75]
	lbda = [0.5]

pl = []

for pref in npref:
	for l in lbda:

		if args.mode == 'lambda':
			kendall_local = []
			spearman_local = []
			kendall_human_local = []
			spearman_human_local = []

		kendall_global = []
		spearman_global = []
		kendall_human_global = []
		spearman_human_global = []

		for split in range(args.nsplits):
			name = args.model_name + '_pr' + str(pref) + '_l' + str(l) + '_b' + str(batch_size) + '_p' + str(npairs//1000) + '_s' + str(split) + '_v' + str(args.validation)
			hist = np.load(os.path.join(args.save_dir, name, name + '_history.npy'), allow_pickle=True).item()

			if args.mode == 'lambda':
				kendall_local.append(hist['kendall_local'][1:])
				spearman_local.append(hist['spearman_local'][1:])
				kendall_human_local.append(hist['kendall_human_local'])
				spearman_human_local.append(hist['spearman_human_local'])
			
			kendall_global.append(hist['kendall_global'][1:])
			spearman_global.append(hist['spearman_global'][1:])
			kendall_human_global.append(hist['kendall_human_global'])
			spearman_human_global.append(hist['spearman_human_global'])

		if args.mode == 'lambda':	
			kendall_local = np.array(kendall_local)
			spearman_local = np.array(spearman_local)
			kendall_mean_local = np.mean(kendall_local, axis=0)
			spearman_mean_local = np.mean(spearman_local, axis=0)
			kendall_std_local = np.std(kendall_local, axis=0)
			spearman_std_local = np.std(spearman_local, axis=0)
			kendall_human_mean_local = np.mean(kendall_human_local)
			kendall_human_std_local = np.std(kendall_human_local)
			spearman_human_mean_local = np.mean(spearman_human_local)
			spearman_human_std_local = np.std(spearman_human_local)
		
		kendall_global = np.array(kendall_global)
		spearman_global = np.array(spearman_global)
		kendall_mean_global = np.mean(kendall_global, axis=0)
		spearman_mean_global = np.mean(spearman_global, axis=0)
		kendall_std_global = np.std(kendall_global, axis=0)
		spearman_std_global = np.std(spearman_global, axis=0)
		kendall_human_mean_global = np.mean(kendall_human_global)
		kendall_human_std_global = np.std(kendall_human_global)	
		spearman_human_mean_global = np.mean(spearman_human_global)
		spearman_human_std_global = np.std(spearman_human_global)

		x = range(1, kendall_global.shape[1]+1)
		
		if args.metric == 'kendall':
			if args.mode == 'lambda':
				pl = plt.plot(x, kendall_mean_global, lw=2, label='global lambda='+str(l))
			elif args.mode == 'preference':
				pl = plt.plot(x, kendall_mean_global, lw=2, label='#preferences='+str(pref))
			print("Kendall tau global validation-test: [%.8f/%.8f]" % (kendall_mean_global[-1], kendall_std_global[-1]))
			plt.fill_between(x, kendall_mean_global-kendall_std_global, kendall_mean_global+kendall_std_global, facecolor=pl[0].get_color(), alpha=0.2)
			
			if args.mode == 'lambda':
				print("Kendall tau local validation-test: [%.8f/%.8f]" % (kendall_mean_local[-1], kendall_std_local[-1]))
				plt.plot(x, kendall_mean_local, lw=2, label='local lambda='+str(l), color=pl[0].get_color(), alpha=0.5)
				plt.fill_between(x, kendall_mean_local-kendall_std_local, kendall_mean_local+kendall_std_local, facecolor=pl[0].get_color(), alpha=0.2)

		elif args.metric == 'spearman':
			if args.mode == 'lambda':
				pl = plt.plot(x, spearman_mean_global, lw=2, label='global lambda='+str(l))
			elif args.mode == 'preference':
				pl = plt.plot(x, spearman_mean_global, lw=2, label='#preferences='+str(sub))
			print("Spearman rho global validation-test: [%.8f/%.8f]" % (spearman_mean_global[-1], spearman_std_global[-1]))
			plt.fill_between(x, spearman_mean_global-spearman_std_global, spearman_mean_global+spearman_std_global, facecolor=pl[0].get_color(), alpha=0.2)
			
			if args.mode == 'lambda':
				print("Spearman rho local validation-test: [%.8f/%.8f]" % (spearman_mean_local[-1], spearman_std_local[-1]))
				plt.plot(x, spearman_mean_local, lw=2, label='local lambda='+str(l), color=pl[0].get_color(), alpha=0.5) 
				plt.fill_between(x, spearman_mean_local-spearman_std_local, spearman_mean_local+spearman_std_local, facecolor=pl[0].get_color(), alpha=0.2)

if args.metric == 'kendall':
	if args.mode == 'lambda':
		mu_local = np.ones(len(x))*kendall_human_mean_local
		sig_local = np.ones(len(x))*kendall_human_std_local

	mu_global = np.ones(len(x))*kendall_human_mean_global
	sig_global = np.ones(len(x))*kendall_human_std_global
elif args.metric == 'spearman':
	if args.mode == 'lambda':
		mu_local = np.ones(len(x))*spearman_human_mean_local
		sig_local = np.ones(len(x))*spearman_human_std_local
	
	mu_global = np.ones(len(x))*spearman_human_mean_global
	sig_global = np.ones(len(x))*spearman_human_std_global

if args.metric == 'kendall':
	print("Global human kendall tau validation-test: [%.8f/%.8f]" % (mu_global[-1], sig_global[-1]))
elif args.metric == 'spearman':
	print("Global human spearman rho validation-test: [%.8f/%.8f]" % (mu_global[-1], sig_global[-1]))
pl = plt.plot(x, mu_global, lw=2, label='global human')
plt.fill_between(x, mu_global-sig_global, mu_global+sig_global, facecolor=pl[0].get_color(), alpha=0.2)

if args.mode == 'lambda':
	if args.metric == 'kendall':
		print("Local human kendall tau validation-test: [%.8f/%.8f]" % (mu_local[-1], sig_local[-1]))
	elif args.metric == 'spearman':
		print("Local human spearman rho validation-test: [%.8f/%.8f]" % (mu_local[-1], sig_local[-1]))
	pl = plt.plot(x, mu_local, lw=2, label='local human', color=pl[0].get_color(), alpha=0.5)
	plt.fill_between(x, mu_local-sig_local, mu_local+sig_local, facecolor=pl[0].get_color(), alpha=0.2)

if args.save_dir == 'models/tvsum':
	plt.title('Multi-ranker validation/test performance on TVSum dataset')
elif args.save_dir == 'models/summe':
	plt.title('Multi-ranker validation/test performance on SumMe dataset')				
plt.xlabel('Epoch')

if args.metric == 'kendall':
	plt.ylabel('Kendall Tau')
elif args.metric == 'spearman':
	plt.ylabel('Spearman Rho')

plt.legend(loc=1)
plt.grid(True)
#plt.tight_layout()		

path = os.path.join('results', args.model_name + '_' + args.save_dir.split('/')[1]  + '_' + args.mode + '_' + args.metric + '.jpg')
plt.savefig(path)
