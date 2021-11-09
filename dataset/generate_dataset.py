import numpy as np
import h5py
import os
import argparse

# generate frame-level reference summaries () and segment-level GT summary () per each video

parser = argparse.ArgumentParser(description='generate GT summaries')
parser.add_argument('--input', type=str, default='eccv16_dataset_tvsum_google_pool5.h5', choices=['eccv16_dataset_tvsum_google_pool5.h5', 'eccv16_dataset_summe_google_pool5.h5'])
parser.add_argument('--reference', type=str, )
parser.add_argument('--gt', type=str, )
args = parser.parse_args()

source = h5py.File(args.input, 'r')

reference = [None]*len(source)
gt_segment = [None]*len(source)

for i in range(len(source)):
    reference[i] = source['video_' + str(i+1)]['user_summary'].copy()
    gt[i] = source['video_' + str(i+1)]['gtscore'].copy()

reference = np.array(reference)
gt = np.array(reference)

np.save('')
