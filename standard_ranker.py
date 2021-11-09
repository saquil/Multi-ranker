import utils, torch, time, os, pickle, itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import json

class Ranker(nn.Module): 
	def __init__(self, input_dim=1024, output_dim=1):
		super(Ranker, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		layers = []

		layers.append(nn.Sequential(
		    nn.Conv2d(input_dim, 1, 1),
		))
		self.main = nn.Sequential(*layers)

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.main(input.view([input.shape[0], input.shape[1], 1, 1]))
		x = x.view([-1])
		return x

class ranker(object):
	def __init__(self, args, SUPERVISED=True):
		# parameters
		self.epoch = args.epoch
		self.batch_size = args.batch_size
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = args.dataset
		self.dataset_path = args.dataset_path
		self.split_path = args.split_path
		self.log_dir = args.log_dir
		self.gpu_mode = args.gpu_mode
		self.input_dim = args.input_dim
		self.model_name = args.model_name
		self.pairset = args.pairset
		self.split = args.split
		self.validation = args.validation
		
		# load dataset
		from dataloader import dataloader
		(self.data_loader, self.val_loader, self.test_loader) = dataloader(args)

		# networks init
		self.R = Ranker(input_dim=self.input_dim, output_dim=1)
		self.R_optimizer = optim.Adam(self.R.parameters(), lr=args.lrR, betas=(args.beta1, args.beta2))

		if self.gpu_mode:
			self.R.cuda()
			self.BCE_loss = nn.BCELoss().cuda()
		else:
			self.BCE_loss = nn.BCELoss()

		print('---------- Networks architecture -------------')
		utils.print_network(self.R)
		print('-----------------------------------------------')

	def prediction(self, load=True):
	# predict the ranking score for each segment in the whole dataset and transform it to frame-level scores (r), then return it with the frame-level reference summaries (scores) 
		with torch.no_grad():
			# whether we load a saved ranking model or use the current one
			if load:
				self.load()
			self.R.eval()

			# get the full dataset
			data = self.data_loader.dataset.data

			# initialize and fill the predicted and reference summaries
			scores = []
			r = []
 
			for i in range(len(data)):
				tmp = np.zeros((data['video_' + str(i+1)]['features'].shape[0]))
				for j in range(0, data['video_' + str(i+1)]['features'].shape[0], self.batch_size*2):
					x = data['video_' + str(i+1)]['features'][j:min(j+self.batch_size*2, data['video_' + str(i+1)]['features'].shape[0]),:]
					x_ = self.R(torch.Tensor(x).cuda())
					tmp[j:min(j+self.batch_size*2, data['video_' + str(i+1)]['features'].shape[0])] = x_.cpu().numpy()
				r.append(tmp)
				scores.append(data['video_' + str(i+1)]['user_summary'][:])

			self.R.train()
			frames = 15 if self.input_dim == 1024 else 16

			for i in range(len(scores)):
				tmp = [0]*len(scores[i][0])
				for j in range(len(tmp)):
					tmp[j] = r[i][j//frames] if j//frames<len(r[i]) else r[i][-1]
				r[i] = np.array(tmp)

		return scores, r


	def kendall(self, data=None):
	# calculate the kendall's tau metric between the frame-level predicted and reference summaries w.r.t each annotator/user and video 
		if not data:
			scores, r = self.prediction()
		else:
			scores, r = data

		kendall = []
		for i in range(len(r)):
			tmp = []
			for j in range(len(scores[i])):	
				if np.sum(scores[i][j])>0:	
					tmp.append(stats.kendalltau(scores[i][j], r[i])[0])
				else:
					tmp.append(float('nan'))
			kendall.append(tmp)

		return np.array(kendall)
	
	def spearman(self, data=None):
	# calculate the spearman's rho metric between the frame-level predicted and reference summaries w.r.t each annotator/user and video
		if not data:
			scores, r = self.prediction()
		else:
			scores, r = data

		spearman = []
		for i in range(len(r)):
			tmp = []
			for j in range(len(scores[i])):
				if np.sum(scores[i][j])>0:
					tmp.append(stats.spearmanr(scores[i][j],r[i])[0])
				else:
					tmp.append(float('nan'))
			spearman.append(tmp)
		
		return np.array(spearman)

	def human(self, load=True):
	# calculate the human baseline between the reference summaries using leave-one-out approach w.r.t each annotator/user and video
		scores, r = self.prediction(load)
		kendall = []
		spearman = []
		for i in range(len(r)):
			ken = []
			sp = []
			for j in range(len(scores[i])):
				ken_avg = []
				sp_avg = []
				for k in range(len(scores[i])):
					if j != k:
						ken_avg.append(stats.kendalltau(scores[i][j], scores[i][k])[0])
						sp_avg.append(stats.spearmanr(scores[i][j], scores[i][k])[0])
				ken.append(np.mean(ken_avg))
				sp.append(np.mean(sp_avg))
			kendall.append(ken)
			spearman.append(sp)

		return (np.array(kendall), np.array(spearman))

	def both(self, load=True):
	# calculate the kendall's tau and spearman's rho metrics between the frame-level predicted and reference summaries w.r.t each annotator/user and video
		data = self.prediction(load)
		return (self.kendall(data), self.spearman(data))

	def val(self, load=True, human=False, stats=None):
	# calculate the human or predicted metrics on the selected split and validation/test set
		kendall, spearman = [], []
		if human:
			if not os.path.exists('./dataset/human_stats_'+self.dataset+'.npy'): 
				np.save('./dataset/human_stats_'+self.dataset+'.npy', self.human(load)) 
			ken, sp = np.load('./dataset/human_stats_'+self.dataset+'.npy', allow_pickle=True) 
		else:	
			ken, sp = stats

		splits = json.load(open(self.split_path))

		vids = splits[self.split]['test'] if self.validation == 4 else splits[self.split]['train_val'][self.validation]['val']

		for i in range(ken.shape[0]):	
			if 'video_'+str(i+1) in vids:
				kendall.append(np.mean(ken[i]))
				spearman.append(np.mean(sp[i]))

		return np.mean(kendall), np.mean(spearman)

	def train(self):
		self.train_hist = {}
		self.train_hist['R_loss'] = []
		self.train_hist['per_epoch_time'] = []
		self.train_hist['total_time'] = []
		self.train_hist['kendall'] = []
		self.train_hist['spearman'] = []
		self.train_hist['kendall_human'] = []
		self.train_hist['spearman_human'] = []

		self.y_real_, self.y_fake_ = torch.ones(self.batch_size), torch.zeros(self.batch_size)
		if self.gpu_mode:
			self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

		self.R.train()
		print('training start!!')
		start_time = time.time()
		
		kendall, spearman = self.val(load=False, human=False, stats=self.both(False))
		print("Epoch : [%2d] [%.8f/%.8f] kendall/spearman" % (0, kendall, spearman))
		self.train_hist['per_epoch_time'].append(time.time() - start_time)
		self.train_hist['kendall'].append(kendall)
		self.train_hist['spearman'].append(spearman)

		for epoch in range(self.epoch):
			epoch_start_time = time.time()

			for iter, x_ in enumerate(self.data_loader):

				(x1_, x2_, r_) = x_

				if r_.shape[0] < self.batch_size:
					break	

				if self.gpu_mode:
					x1_, x2_, r_ = x1_.float().cuda(), x2_.float().cuda(), r_.float().cuda()

				x_p = torch.cat([x1_, x2_], 0)

				#update R network
				self.R_optimizer.zero_grad()

				R_ = self.R(x_p)
				diff = R_[:self.batch_size] - R_[self.batch_size:]
				sig = nn.Sigmoid().cuda()
				prob = sig(diff)
				R_loss = self.BCE_loss(prob, r_)

				self.train_hist['R_loss'].append(R_loss.item())		
				R_loss.backward()
				self.R_optimizer.step()		
					
				if ((iter + 1) % 100) == 0:
					print("Epoch: [%2d] [%4d/%4d] rank_loss: %.8f" %
					  ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, R_loss.item()))

			kendall, spearman = self.val(load=False, human=False, stats=self.both(False))
			print("Epoch : [%2d] [%.8f/%.8f] kendall/spearman" % ((epoch + 1), kendall, spearman))
			self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
			self.train_hist['kendall'].append(kendall)
			self.train_hist['spearman'].append(spearman)

		self.train_hist['total_time'].append(time.time() - start_time)
		kendall_human, spearman_human = self.val(load=False, human=True)
		self.train_hist['kendall_human'].append(kendall_human)
		self.train_hist['spearman_human'].append(spearman_human)
		print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
									self.epoch, self.train_hist['total_time'][0]))
		print("Training finish!... save training results")

		self.save()
		self.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name, 'kendall')
		self.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name, 'spearman')

	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.R.state_dict(), os.path.join(save_dir, self.model_name + '_R.pkl'))

		np.save(os.path.join(save_dir, self.model_name + '_history.npy'), self.train_hist)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		self.R.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_R.pkl')))

	def loss_plot(self, hist, path='Train_hist.png', model_name='', metric='kendall'):
		x = range(len(hist[metric]))
		z = np.ones(len(hist[metric]))*hist[metric + '_human']
		y = hist[metric]

		plt.plot(x, y, label='model '+ metric)
		plt.plot(x, z, label='human '+ metric)

		plt.xlabel('Iter')
		plt.ylabel('Kendall\'s Tau' if metric == 'kendall' else 'Spearman\'s Rho')

		plt.legend(loc=4)
		plt.grid(True)
		plt.tight_layout()

		path = os.path.join(path, model_name + '_' + metric + '_loss.png')

		plt.savefig(path)
