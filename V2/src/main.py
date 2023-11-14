import os, sys
os.chdir(sys.path[0])
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import copy
import json
from graph_dataset import CS_Graphdataset
from query_dataset import load_query
from arguments import get_args
from utils import set_seed, get_metric
from model import load_model, EarlyStopping
from community_search import Community_Search, search_threshold_parallel, search_threshold

from tqdm import tqdm

import logging
logger = logging.getLogger()

def main(args):
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载图

    graph_data = CS_Graphdataset(args)
    num_nodes = graph_data.g.number_of_nodes()
    
    lc = Community_Search(args, graph_data, args.community_types, max_depth=args.max_depth)  # 搜索局部子图
    graph_data.g = graph_data.g.to(args.device)
    # 加载query community
    primary, train_data, valid_data, test_data = load_query(args, num_nodes, check=False)

    model = load_model(args, graph_data).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    early_stopping = EarlyStopping(patience=args.patience, save_path=args.output_dir, verbose=True)  # 初始化早停类

    for epoch in tqdm(range(args.epoch)):
        # Train Phrase
        total_loss = []
        model.train()
        for j, data in enumerate(train_data):
            query, train_nodes, labels = data['query'].to(args.device), data['train_nodes'].to(args.device), data['label'].float().reshape(-1, 1).to(args.device)
            # 形成q向量
            q = torch.zeros((num_nodes, 1), device=args.device)
            q[query] = 1
            outputs = model(graph_data.features_list, graph_data.e_feat, q)
            loss = criterion(outputs[train_nodes], labels[train_nodes])
            total_loss.append(loss.item())
            loss = loss / args.batch_size   # 规范化损失
            loss.backward()                    # 反向传播，计算当前梯度
            if (j+1) % args.batch_size == 0 or (j+1) == len(train_data):  # 每accumulation_steps次迭代，或者是最后一个batch，执行一次梯度更新
                optimizer.step()                 # 更新参数
                optimizer.zero_grad()            # 清空梯度缓存
        
        logger.info(f'epoch {epoch} | loss {np.mean(total_loss)}')

        # 在验证集上搜索最优阈值
        with torch.no_grad():
            model.eval()
            scorelists = []
            true_len = []
            for j, data in enumerate(valid_data):
                query, comm = data['query'].to(args.device), data['comm']
                # 形成q向量
                q = torch.zeros((num_nodes, 1), device=args.device)
                q[query] = 1
                outputs = model(graph_data.features_list, graph_data.e_feat, q)
                probs = torch.sigmoid(outputs).cpu().numpy().reshape(-1).tolist()
                scorelists.append([query.item(), comm, probs])
                true_len.append(len(comm))

            best_valid_threshold, best_valid_f1, avg_len = search_threshold_parallel(lc, scorelists, len(valid_data), max_processes=20)
            # best_valid_threshold, best_valid_f1 = search_threshold(lc, scorelists, len(valid_data))
            # logger.info(f"Predict avg len: {avg_len}, True avg len: {np.mean(true_len)}")

        early_stopping(epoch, best_valid_f1, best_valid_threshold, model)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    
    # 加载最优模型
    model = load_model(args, graph_data)
    # 加载预训练权重
    model.load_state_dict(torch.load(osp.join(args.output_dir, 'checkpoint.pt')))
    model = model.to(args.device)
    with open(osp.join(args.output_dir, 'checkpoint_params.json'), 'r') as f:
        best_threshold = json.load(f)['threshold']
    with torch.no_grad():
        model.eval()
        # 找到最佳阈值后，在测试集上进行测试
        test_f1, test_js, test_nmi = [], [], []
        scorelists = []
        com_len, find_com_len = [], []
        for j, data in enumerate(test_data):
            query, comm = data['query'].to(args.device), data['comm']
            # 形成q向量
            q = torch.zeros((num_nodes, 1), device=args.device)
            q[query] = 1
            outputs = model(graph_data.features_list, graph_data.e_feat, q)
            probs = torch.sigmoid(outputs).cpu().numpy().reshape(-1).tolist()
            
            comm_find = lc.bfs(query.item(), best_threshold, probs)

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            com_len.append(len(comm))
            find_com_len.append(len(comm_find))
            metric = get_metric(comm_find, comm, n_nodes=num_nodes)
            test_f1.append(metric['f1'])
            test_js.append(metric['js'])
            test_nmi.append(metric['nmi'])

        logger.info(f'comm len: {np.mean(com_len)}')
        logger.info(f'find_comm len: {np.mean(find_com_len)}')
        logger.info(f'f1: {np.mean(test_f1)}')
        logger.info(f'js: {np.mean(test_js)}')
        logger.info(f'nmi: {np.mean(test_nmi)}')

        result = {
            'f1': np.mean(test_f1),
            'js': np.mean(test_js),
            'nmi': np.mean(test_nmi)
        }
        with open(osp.join(args.output_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=4)

if __name__ == '__main__':
    args = get_args()
    main(args)