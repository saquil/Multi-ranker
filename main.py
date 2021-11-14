import argparse, os, torch
#from ranker import ranker
import numpy as np

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of the parameters of Standard ranker and Multi-ranker"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of the mini-batch')
    parser.add_argument('--input_dim', type=int, default=1024, choices=[1024,2048], help='The segment feature dimension. The baseline features have 1024 and the 3D ResNet features have 2048')
    parser.add_argument('--dataset', type=str, default='tvsum', choices= ['tvsum', 'summe'], help='The name of dataset')
    parser.add_argument('--dataset_path', type=str, default='./dataset/iccv21_dataset_tvsum_google_pool5.h5', choices=['./dataset/iccv21_dataset_summe_google_pool5.h5', './dataset/iccv21_dataset_tvsum_google_pool5.h5'], help='The path of dataset')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated plots')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--lrR', type=float, default=0.0002)
    parser.add_argument('--lrEz', type=float, default=0.0002)
    parser.add_argument('--lrEr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--mode', type=str, default='training', choices=['training', 'local_preference', 'comb_preference'], help='The task to perform by the script')
    parser.add_argument('--model_name', type=str, default='ranker', help='The initial of the name of the model')
    parser.add_argument('--pairset', type=str, default='./pairset/tvsum/pairs_2k.npy', help='The global pairwise comparisons of the segments')
    parser.add_argument('--pairset_multi', type=str, default='./pairset/tvsum/pairs_multi_2k_4.npy', help='The local pairwise comparisons of the segments w.r.t. each preference for Multi-ranker')
    parser.add_argument('--users', type=str, default='dataset/clustering/preferences_tvsum_4.npy', help='frame-level reference summaries per each preference')
    parser.add_argument('--split_path', type=str, default='./dataset/splits_tvsum.json', choices=['./dataset/splits_tvsum.json', './dataset/splits_summe.json'], help='The path to the json file containing the dataset splits and sets')
    parser.add_argument('--multi', type=bool, default=False, help='Using Standard ranker or Multi-ranker')
    parser.add_argument('--split', type=int, default=0, choices=[0,1,2,3,4], help='Which dataset split to use')
    parser.add_argument('--validation', type=int, default=0, choices=[0,1,2,3,4], help='Which fold of 4-fold cross validation to use. The option 4 denotes an empty validation set and a full training set to produce the final model test results.' )
    parser.add_argument('--preference', type=int, default=4, choices=[2,4,8], help='The number of preferences in case of Multi-ranker')
    parser.add_argument('--lbda', type=float, default=0.5, help='The value of the hyperparameter lambda')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)	
		
    # --log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.dataset == 'tvsum':
        args.dataset_path = './dataset/iccv21_dataset_tvsum_google_pool5.h5'
        args.split_path = './dataset/splits_tvsum.json'
    elif args.dataset == 'summe':
        args.dataset_path = './dataset/iccv21_dataset_summe_google_pool5.h5'
        args.split_path = './dataset/splits_summe.json'

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    if args.multi:
       from multi_ranker import ranker 
    else:
       from standard_ranker import ranker

    model = ranker(args)	

    # launch the graph in a session
    if args.mode == 'training':
       model.train()
       print(" [*] Training finished!")
    elif args.mode == 'local_preference':
       save_dir = os.path.join(args.save_dir, args.dataset, args.model_name)
       train_hist = np.load(os.path.join(save_dir, args.model_name + '_history.npy'), allow_pickle=True).item()
       train_hist['local'] = model.local_preference(load=True)
       np.save(os.path.join(save_dir, args.model_name + '_history.npy'), train_hist)
       print(" [*] Evaluation finished!")
    elif args.mode == 'comb_preference':
       save_dir = os.path.join(args.save_dir, args.dataset, args.model_name)
       train_hist = np.load(os.path.join(save_dir, args.model_name + '_history.npy'), allow_pickle=True).item()
       train_hist['combine'] = model.comb_preference(load=True)
       np.save(os.path.join(save_dir, args.model_name + '_history.npy'), train_hist)
       print(" [*] Evaluation finished!")

if __name__ == '__main__':
    main()
