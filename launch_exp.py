import numpy as np
import subprocess as sp
import argparse
import time
import sys

parser = argparse.ArgumentParser()
# launch training models per each split for a selected validation/test option
parser.add_argument('--type', type=str, default='ranker', choices=['ranker', 'multi_ranker'], help='The initial of the name of the model')
parser.add_argument('--batch_size', type=int, default=128, choices=[32,64,128], help='The size of the mini-batch')
parser.add_argument('--epoch', type=int, default=1, help='The number of epochs to run')
parser.add_argument('--dataset', type=str, default='tvsum', choices= ['tvsum', 'summe'], help='The name of dataset')
parser.add_argument('--ngpus', type=int, default=6, help='The number of the available gpus to use')
parser.add_argument('--gpu_mem', type=int, default=7952, help='The maximum memory available per gpu')
parser.add_argument('--nsplits', type=int, default=5, help='The number of dataset existing splits')
parser.add_argument('--npairs', type=int, default=2000, choices=[2000,5000,10000], help='The number of pairwise comparisons per each video')
parser.add_argument('--preference', type=int, default=4, choices=[2,4,8], help='The number of preferences in case of Multi-ranker')
parser.add_argument('--lbda', type=float, default=0.5, help='The value of the hyperparameter lambda')
parser.add_argument('--validation', type=int, default=4, choices=[0,1,2,3,4], help='Which fold of 4-fold cross validation to use. The option 4 denotes the test set')


args = parser.parse_args()

max_memory = args.gpu_mem
if args.type == 'ranker':
	memory = 865
elif args.type == 'multi_ranker':
	memory = 902

_output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

def get_gpu_memory():
	COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
	memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
	memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
	print(memory_free_values)
	return memory_free_values

finished = False
ps = []
for split in range(args.nsplits):
	free_memory = get_gpu_memory()
	if args.type == 'ranker':
		model_name = "ranker_b"+str(args.batch_size)+"_p"+str(args.npairs//1000)+"_s"+str(split)+"_v"+str(args.validation)
	elif args.type == 'multi_ranker':
		model_name = "multi_ranker_pr"+str(args.preference)+"_l"+str(args.lbda)+"_b"+str(args.batch_size)+"_p"+str(args.npairs//1000)+"_s"+str(split)+"_v"+str(args.validation)

	i = 0
	while(True):
		if free_memory[i] - memory > 1000:
			cmd = "CUDA_VISIBLE_DEVICES="+str(i)
			break
		if i == args.ngpus-1:
			time.sleep(10*60)
		i = (i+1)%args.ngpus

	if args.type == 'ranker':
		cmd = cmd + " python3 main.py --epoch="+str(args.epoch)+" --batch_size="+ str(args.batch_size) +" --dataset="+ args.dataset +" --mode=training --model_name="+ model_name +" --pairset=./pairset/"+ args.dataset +"/pairs_"+str(args.npairs//1000)+"k.npy --split="+str(split)+" --validation="+str(args.validation)
	elif args.type == 'multi_ranker':
		cmd = cmd + " python3 main.py --epoch="+str(args.epoch)+" --batch_size="+ str(args.batch_size) +" --dataset="+ args.dataset +" --mode=training --model_name="+ model_name +" --pairset_multi=./pairset/"+ args.dataset +"/pairs_multi_"+str(args.npairs//1000)+"k_"+str(args.preference)+".npy --pairset=./pairset/"+ args.dataset  +"/pairs_"+str(args.npairs//1000)+"k.npy --users=dataset/clustering/preferences_"+args.dataset+"_"+str(args.preference)+".npy  --multi=True --split="+str(split)+" --validation="+str(args.validation)+" --preference="+str(args.preference)+" --lbda="+str(args.lbda)

	ps.append(sp.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr))
	time.sleep(60)
	
while(not finished):
	finished = True
	for i, p in enumerate(ps):
		if p.poll() is not None:
			pass	
		else:
			finished = False
