import numpy as np
import argparse
from sklearn.cluster import KMeans
import h5py  

parser = argparse.ArgumentParser(description='Generate segment-level GT summaries (scores_*) and frame-level reference summaries (preferences_*) per each cluster/preference')
parser.add_argument('--n_clusters', type=int, default=4, choices=[2,4,8], help='The number of clusters or preferences')
parser.add_argument('--dataset', type=str, default='tvsum', choices=['tvsum', 'summe'], help='The name of dataset')
parser.add_argument('--dataset_path', type=str, default='../iccv21_dataset_tvsum_google_pool5.h5', choices=['../iccv21_dataset_tvsum_google_pool5.h5', '../iccv21_dataset_summe_google_pool5.h5'])
parser.add_argument('--input_dim', type=int, default=1024, choices=[1024,2048], help='The segment feature dimension. The baseline features have 1024 and the 3D ResNet features have 2048')
parser.add_argument('--samples_file', type=str, default='feat_samples_tvsum.npy', choices=['feat_samples_tvsum.npy', 'feat_samples_summe.npy'], help='The output file')

args = parser.parse_args()

if args.dataset == 'tvsum':
	args.dataset_path = '../iccv21_dataset_tvsum_google_pool5.h5'
elif args.dataset == 'summe':
	args.dataset_path = '../iccv21_dataset_summe_google_pool5.h5'

frames = 15 if args.input_dim == 1024 else 16

dataset = h5py.File(args.dataset_path, 'r')

clusters = [None]*len(dataset)
preferences = [None]*args.n_clusters

samples = np.load(args.samples_file, allow_pickle=True)

kmeans = KMeans(args.n_clusters, n_jobs=8)
kmeans.fit(samples)

for i in range(len(dataset)):
	clusters[i] = [None]*args.n_clusters

for i in range(args.n_clusters):
	preferences[i] = [[] for j in range(len(dataset))]

for i in range(len(dataset)):
	pred = kmeans.predict(dataset['video_' + str(i+1)]['features'][:])

	for j in range(args.n_clusters):
		clusters[i][j] = np.zeros_like(dataset['video_' + str(i+1)]['gtscore'])
	for j in range(len(pred)):
		clusters[i][pred[j]][j] = dataset['video_' + str(i+1)]['gtscore'][j]
	for j in range(args.n_clusters):
		for k in range(len(dataset['video_' + str(i+1)]['user_summary'])):
			preferences[j][i].append(np.zeros_like(dataset['video_' + str(i+1)]['user_summary'][k]))

	pred = list(pred)
	pred.append(pred[-1])
	for j in range(len(pred)):
		for k in range(len(dataset['video_' + str(i+1)]['user_summary'])):
			preferences[pred[j]][i][k][frames*j:frames*(j+1)] = dataset['video_' + str(i+1)]['user_summary'][k][j*frames:frames*(j+1)]

np.save('scores_'+str(args.dataset)+'_'+str(args.n_clusters)+'.npy', np.array(clusters))
np.save('preferences_'+str(args.dataset)+'_'+str(args.n_clusters)+'.npy', np.array(preferences))

