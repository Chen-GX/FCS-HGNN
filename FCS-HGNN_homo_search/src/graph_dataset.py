import os.path as osp
import os, sys
os.chdir(sys.path[0])
import numpy as np
import torch
import scipy.sparse as sp
import pickle
import torch
import dgl
import json

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

class CS_Graphdataset(object):
    def __init__(self, args):
        super(CS_Graphdataset, self).__init__()
        self.name = args.data_name
        self.raw_dir = osp.join(args.data_dir, args.data_name)
        self.device = args.device
        self.process()

    def process(self):
        # 加载homo adj for search
        with open(osp.join(self.raw_dir, 'homo_adj.json'), 'r') as f:
            homo_dict = json.load(f)  # key是str

        self.homo_dict = {int(k): v for k, v in homo_dict.items()}

        adj = sp.load_npz(osp.join(self.raw_dir, 'adj.npz'))
        # 构建图
        g = dgl.DGLGraph(adj+(adj.T))  # 保证无向图  adj csr
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        self.g = g

        with open(osp.join(self.raw_dir, 'features.pickle'), 'rb') as f:
            features_list = pickle.load(f)
        
        self.features_list = [mat2tensor(features).to(self.device) for features in features_list]
        self.in_dims = [features.shape[1] for features in features_list]

        with open(osp.join(self.raw_dir, 'edge2type.pickle'), 'rb') as f:
            edge2type = pickle.load(f)
        
        assert len(edge2type) == g.number_of_edges()
        e_feat = []
        for u, v in zip(*g.edges()):
            u = u.item()
            v = v.item()
            e_feat.append(edge2type[(u,v)])
        self.e_feat = torch.tensor(e_feat, dtype=torch.long).to(self.device)

        node_types = np.load(osp.join(self.raw_dir, 'node_types.npy'))
        # 节点id到节点type的映射
        self.id2type = {i: val for i, val in enumerate(node_types)}


        


