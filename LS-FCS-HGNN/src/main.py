import os, sys
os.chdir(sys.path[0])
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import copy
import json
import dgl
import time
from graph_dataset import CS_Graphdataset
from query_dataset import load_query
from arguments import get_args
from utils import set_seed, get_metric
from model import load_model, EarlyStopping
from community_search import Community_Search, search_threshold_parallel, search_threshold
from large_scale import MultiLayerNeighborSampler, DFSGraphExplorer, BFSGraphExplorer
from tqdm import tqdm

import logging
logger = logging.getLogger()

def main(args):
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 验证集节点
    inference_node_ids = []
    start_id = 0

    for node_type in range(max(args.num_nodes_type.keys()) + 1):
        end_id = start_id + args.num_nodes_type[node_type]

        if node_type in args.community_types:
            inference_node_ids.extend(range(start_id, end_id))

    valid_nodes = torch.tensor(list(inference_node_ids), dtype=torch.int64, device=args.device)

    # 加载图
    begin_time = time.time()
    graph_data = CS_Graphdataset(args)  # 0.88538813   0.1255825  变快很多
    print(f'gd time: {(time.time()-begin_time) / 60}')
    num_nodes = graph_data.g.number_of_nodes()

    begin_time = time.time()
    lc = Community_Search(args, graph_data, args.community_types, num_nodes, max_depth=args.max_depth)  # 搜索局部子图  0.40597934
    print(f'cs time: {(time.time()-begin_time) / 60}')
    # begin_time = time.time()
    sampler = MultiLayerNeighborSampler(fanouts=[args.fanouts_0, args.fanouts_1])  # list间接表示层数
    # node_explorer = BFSGraphExplorer(args, graph_data.id2type, args.community_types, max_depth=max_depth, prob=1.0, device=args.device, num_nodes=num_nodes)  # 0.39473409
    # print(f'node_explorer: {(time.time()-begin_time) / 60}')
    begin_time = time.time()
    graph_data.g = graph_data.g.to(args.device)
    print(f'g to device: {(time.time()-begin_time) / 60}')  # 0.011824301
    # 加载query community
    begin_time = time.time()
    primary, train_data, valid_data, test_data = load_query(args, num_nodes, check=False)  # 0.152203905
    print(f'lq time: {(time.time()-begin_time) / 60}')

    model = load_model(args, graph_data).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    early_stopping = EarlyStopping(patience=args.patience, save_path=args.output_dir, verbose=True)  # 初始化早停类

    for epoch in tqdm(range(args.epoch), desc=f'training epoch'):
        # Train Phrase
        total_loss = []
        model.train()
        for j, data in enumerate(tqdm(train_data)):
            # if j > 2:
            #     break
            query, train_nodes, labels = data['query'].to(args.device), data['train_nodes'].to(args.device), data['label'].float().reshape(-1, 1).to(args.device)
            # 形成q向量
            # begin_time = time.time()
            q = torch.zeros((num_nodes, 1), device=args.device)
            q[query] = 1
            
            graph_data.g.ndata['q'] = q
            # print(f'q: {(time.time()-begin_time) / 60}')  # 3.625154495
            # begin_time = time.time()
            dataloader = dgl.dataloading.DataLoader(
                graph_data.g, train_nodes, sampler,
                batch_size=1024,
                shuffle=True,
                drop_last=False,
                num_workers=0)
            # print(f'dataloader time: {(time.time()-begin_time) / 60}')  # 2.0702679951985676e-05

            # begin_time = time.time()
            for input_nodes, output_nodes, blocks in dataloader:
                # input_nodes 当前批次下用到的所有的节点，output_nodes 目标节点，信息汇聚到这里
                # blocks = [b.to(args.device) for b in blocks]
                outputs = model(blocks, graph_data.features_list, input_nodes)
                loss = criterion(outputs, labels[train_nodes])
                total_loss.append(loss.item())
                loss = loss / args.batch_size   # 规范化损失
                loss.backward()                    # 反向传播，计算当前梯度
                if (j+1) % args.batch_size == 0 or (j+1) == len(train_data):  # 每accumulation_steps次迭代，或者是最后一个batch，执行一次梯度更新
                    optimizer.step()                 # 更新参数
                    optimizer.zero_grad()            # 清空梯度缓存
            # print(f'1 example time: {(time.time()-begin_time) / 60}')
        
        logger.info(f'epoch {epoch} | loss {np.mean(total_loss)}')

        # 在验证集上搜索最优阈值
        model.eval()
        scorelists = []
        true_len = []
        with torch.no_grad():
            for j, data in enumerate(tqdm(valid_data, desc='valid data')):
                # if j > 2:
                #     break
                query, comm = data['query'].to(args.device), data['comm']
                # begin_time = time.time()
                # valid_nodes = node_explorer.explore(query.item())
                # print(f'node_explorer: {(time.time()-begin_time) / 60}')
                

                # begin_time = time.time()
                all_probs = np.zeros((num_nodes, 1))
                # 形成q向量
                q = torch.zeros((num_nodes, 1), device=args.device)
                q[query] = 1
                graph_data.g.ndata['q'] = q
                dataloader = dgl.dataloading.DataLoader(
                    graph_data.g, valid_nodes, sampler,
                    batch_size=args.infer_batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0)
                # print(f'dataloader: {(time.time()-begin_time) / 60}')
                
                # begin_time = time.time()
                for input_nodes, output_nodes, blocks in dataloader:
                    outputs = model(blocks, graph_data.features_list, input_nodes)
                    all_probs[output_nodes.cpu().numpy()] = torch.sigmoid(outputs).cpu().numpy()
                # print(f'inference: {(time.time()-begin_time) / 60}')

                scorelists.append([query.item(), comm, all_probs.reshape(-1).tolist()])
                true_len.append(len(comm))

            # best_valid_threshold, best_valid_f1, avg_len = search_threshold_parallel(lc, scorelists, len(valid_data), max_processes=20)
            best_valid_threshold, best_valid_f1, avg_len = search_threshold(lc, scorelists, len(valid_data))
            logger.info(f"Predict avg len: {avg_len}, True avg len: {np.mean(true_len)}")


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
        for j, data in enumerate(tqdm(test_data, desc='test data')):
            # if j > 3:
            #     break
            query, comm = data['query'].to(args.device), data['comm']
            # test_nodes = node_explorer.explore(query.item())
            all_probs = np.zeros((num_nodes, 1))
            # 形成q向量
            q = torch.zeros((num_nodes, 1), device=args.device)
            q[query] = 1
            graph_data.g.ndata['q'] = q

            dataloader = dgl.dataloading.DataLoader(
                    graph_data.g, valid_nodes, sampler,
                    batch_size=args.infer_batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0)
            
            for input_nodes, output_nodes, blocks in dataloader:
                outputs = model(blocks, graph_data.features_list, input_nodes)
                all_probs[output_nodes.cpu().numpy()] = torch.sigmoid(outputs).cpu().numpy()
            
            comm_find = lc.bfs(query.item(), best_threshold, all_probs)

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