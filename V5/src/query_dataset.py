import os.path as osp
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)

def load_query(args, num_nodes, check=False):
    dataset_path = osp.join(args.data_dir, args.data_name, f'querydataset_train.pt')
    if osp.exists(dataset_path):
        train_data = QueryDataset(args, 'paper', None, num_nodes, 'train')
        valid_data = QueryDataset(args, 'paper', None, num_nodes, 'valid')
        test_data = QueryDataset(args, 'paper', None, num_nodes, 'test')

    else:
        data_dir = osp.join(args.data_dir, args.data_name, args.query_name)

        with open(data_dir, 'r') as f:
            data = json.load(f)
        
        if check:
            flag = False
            for i in ['train', 'valid', 'test']:
                for q, _, _, com in tqdm(data[i]):
                    if q >= num_nodes or q < 0:
                        flag = True
                        break
                    for j in com:
                        if j >= num_nodes or j < 0:
                            flag = True
                            break

            if flag:
                assert False, 'community node error'


        # logger.info()
        logger.info(f"Primary node: {data['primary']}")
        logger.info(f"Train data size: {len(data['train'])}")
        logger.info(f"Validation data size: {len(data['valid'])}")
        logger.info(f"Test data size: {len(data['test'])}")

        # 这里所有的数据都是从0开始编号的，异构图上要把0加回去？
        train_data = QueryDataset(args, data['primary'], data['train'], num_nodes, 'train')
        valid_data = QueryDataset(args, data['primary'], data['valid'], num_nodes, 'valid')
        test_data = QueryDataset(args, data['primary'], data['test'], num_nodes, 'test')

    return 'paper', train_data, valid_data, test_data




class QueryDataset(Dataset):
    def __init__(self, args, primary, data, num_nodes, d_split):
        """data 是 [[q, [com]], ...]"""
        super(QueryDataset, self).__init__()
        self.primary = primary
        self.num_nodes = num_nodes
        dataset_path = osp.join(args.data_dir, args.data_name, f'querydataset_{d_split}.pt')
        if osp.exists(dataset_path):
            all_data = torch.load(dataset_path)
            self.queries = all_data['queries']
            self.train_nodes = all_data['train_nodes']
            self.test_nodes = all_data['test_nodes']
            self.labels = all_data['labels']
            self.coms = all_data['coms']
        
        else:
            self.process(args, data, dataset_path)
        

    def process(self, args, data, dataset_path):
        all_nodes = np.zeros((len(data), self.num_nodes))  # 样本数，节点个数
        queries, train_nodes, test_nodes, coms = [], [], [], []
        all_node_id = list(range(self.num_nodes))
        for idx, (q, pos, neg, com) in enumerate(data):
            # 修正节点id
            q = q + args.before_num_node
            pos = [pos_item + args.before_num_node for pos_item in pos]
            neg = [neg_item + args.before_num_node for neg_item in neg]
            com = [com_item + args.before_num_node for com_item in com]

            queries.append(q)
            train_nodes.append([q] + pos + neg)
            test_nodes.append(list(set(all_node_id) - set([q] + pos + neg)))
            all_nodes[idx][com] = 1  # 构造label  节点是否属于community
            coms.append(com)
        
        assert len(queries) == len(train_nodes) == len(test_nodes) == all_nodes.shape[0], \
            "queries, trains and labels must have the same length"
        
        self.queries = torch.tensor(queries, dtype=torch.long).reshape(-1, 1)
        self.train_nodes = torch.tensor(train_nodes, dtype=torch.long)
        self.test_nodes = torch.tensor(test_nodes, dtype=torch.long)  # 除了标注节点之外的节点id
        self.labels = torch.tensor(all_nodes, dtype=torch.long)
        self.coms = coms

        all_data = {
            'queries': self.queries,
            'train_nodes': self.train_nodes,
            'test_nodes': self.test_nodes,
            'labels': self.labels,
            'coms': self.coms,
        }
        torch.save(all_data, dataset_path)




    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        train_nodes = self.train_nodes[idx]
        test_nodes = self.test_nodes[idx]
        label = self.labels[idx]
        comm = self.coms[idx]
        return {"query": query, "train_nodes": train_nodes, "test_nodes": test_nodes, "label": label, "comm": comm}
