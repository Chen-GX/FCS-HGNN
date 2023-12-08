import os
import os.path as osp
import time
import argparse
from log_utils import log_params, stream_logging

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


def get_args():
    parser = argparse.ArgumentParser()  # 参数解释器

    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--edge_feats', type=int, default=8)  # 每个头的维度
    parser.add_argument('--hidden_dim', type=int, default=64)  # 总共的维度
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--slope', type=float, default=0.05)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--residual', type=str2bool, default=False)
    parser.add_argument('--use_batchnorm', type=str2bool, default=True)

    ## search
    parser.add_argument('--search_homo', type=str2bool, default=True)  # 是否用同构图进行搜索
    parser.add_argument('--before_num_node', type=int, default=0)
    parser.add_argument('--max_depth', type=int, default=10)

    ## dataset 
    parser.add_argument('--data_dir', type=str, default='../../datasets')
    parser.add_argument('--data_name', type=str, default='Freebase')
    parser.add_argument('--query_name', type=str, default=None)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="../../../output_dir/V2/test")

    # parser.add_argument('--log', action='store_true', help='Whether to use wandb')
    args = parser.parse_args()  # 解析参数

    if args.data_name == 'DBLP_A':
        args.query_name = 'query_data_A.json'
        args.community_types = [0]
        args.before_num_node = 0
    elif args.data_name == 'DBLP_P':
        args.query_name = 'query_data_P.json'
        args.community_types = [1]
        args.before_num_node = 4057

    elif args.data_name == 'ACM':
        args.query_name = 'query_data_P.json'
        args.community_types = [0]
        args.before_num_node = 0
    
    elif args.data_name == 'IMDB':
        args.query_name = 'query_data_M.json'
        args.community_types = [0]
        args.before_num_node = 0
    
    elif args.data_name == 'Freebase':
        args.query_name = 'query_data_S.json'
        args.community_types = [0]
        args.before_num_node = 0

        
    # 时间戳后缀，
    timestamp = time.strftime("%m-%d_%H-%M-%S",time.localtime())
    args.timestamp = timestamp
    file_name = f"ep_{args.epoch}_lr_{args.lr}_bs_{args.batch_size}_md_{args.max_depth}_{args.timestamp}"
    if args.search_homo:
        args.output_dir = osp.join(args.output_dir, args.data_name + "_search_homo", file_name)
    else:
        args.output_dir = osp.join(args.output_dir, args.data_name, file_name)
    os.makedirs(args.output_dir, exist_ok=True)

    log_params(args)

    return args



