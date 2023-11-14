import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import dgl
from itertools import accumulate
from conv import myGATConv

import logging
logger = logging.getLogger()


def load_model(args, graph_data):
    heads = [args.num_heads] * args.num_layers
    return V1(args, graph_data.g, torch.max(graph_data.g.edata['w']).item() + 1,
              graph_data.in_dims, heads, residual=args.residual, use_batchnorm=args.use_batchnorm)


class V1(nn.Module):
    def __init__(self,
                 args,
                 g,
                 num_etypes,
                 in_dims,
                 heads,
                 activation=F.elu,
                 feat_drop=0.5,
                 attn_drop=0.5,
                 negative_slope=0.05,
                 residual=True,
                 alpha=0.05,
                 use_batchnorm=False,
                 fanout=10,  # 新增：每个节点需要采样的邻居数量
                 ):
        super(V1, self).__init__()
        num_hidden, edge_dim, feat_drop, attn_drop, negative_slope = args.hidden_dim, args.edge_feats, args.dropout, args.dropout, args.slope
        self.g = g
        self.fanout = fanout
        self.num_hidden = num_hidden
        self.num_layers = args.num_layers
        self.residual = residual
        self.use_batchnorm = use_batchnorm
        self.graph_gat_layers = nn.ModuleList()
        self.query_gat_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if use_batchnorm:
            # self.bn_layers = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(args.num_layers - 1)])
            self.bn_fc = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(len(in_dims))])
        # self.epsilon = torch.FloatTensor([1e-12]).cuda()
        
        ####### Graph Encoder
        # input projection (no residual)
        self.graph_gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden // heads[0], heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha, use_batchnorm=use_batchnorm))
        # hidden layers
        for l in range(1, self.num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.graph_gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden, num_hidden // heads[l], heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha, use_batchnorm=use_batchnorm))
        
        ####### Query Encoder
        self.query_gat_layers.append(myGATConv(edge_dim, num_etypes,
            1, num_hidden // heads[0], heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha, use_batchnorm=use_batchnorm))
        # hidden layers
        for l in range(1, self.num_layers):
            self.fusion_layers.append(nn.Linear(2 * num_hidden, num_hidden))
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.query_gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden, num_hidden // heads[l], heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha, use_batchnorm=use_batchnorm))
        
        self.fusion_layers.append(nn.Linear(2 * num_hidden, args.num_classes))


    def forward(self, blocks, features_list, input_nodes):
        # 先把所有节点的特征进行转换
        x = []
        for i, (fc, feature) in enumerate(zip(self.fc_list, features_list)):
            out = fc(feature)
            
            # 使用BatchNorm层（如果启用）
            if self.use_batchnorm:
                out = self.bn_fc[i](out)
                
            x.append(F.dropout(F.relu(out), training=self.training))

        # 然后取需要的部分
        x = torch.cat(x, 0)
        x = x[input_nodes]
        q = blocks[0].ndata['q']['_N']  # 初始的q，后续的q不再根据这个
        for l in range(self.num_layers):  # e_feat[g_l.edata['_ID']]
            x, _ = self.graph_gat_layers[l](blocks[l], x, blocks[l].edata['w'], res_attn=None)
            x = F.dropout(x.flatten(1), training=self.training)
            q, _ = self.query_gat_layers[l](blocks[l], q, blocks[l].edata['w'], res_attn=None)
            q = F.dropout(q.flatten(1), training=self.training)
            q = self.fusion_layers[l](torch.cat((x, q), dim=-1))
            if l < self.num_layers - 1:
                q = F.dropout(F.relu(q), training=self.training)

        return q
        # q = q[input_nodes]
        # new_feature_list = [[] for i in features_list]
        # feature_node = [feature.size(0) for feature in features_list]
        
        # ls = list(accumulate(feature_node))
        # for i_n in input_nodes:
        #     for idx, (feat, num_nodes) in enumerate(zip(features_list, ls)):
        #         if i_n < num_nodes:
        #             if idx == 0:
        #                 new_feature_list[idx].append(feat[i_n])
        #             else:
        #                 new_feature_list[idx].append(feat[i_n - ls[idx - 1]])
        #             break
        # new_feature_list = [f_l if len(f_l) == 0 else torch.stack(f_l, dim=0) for f_l in new_feature_list]
        # x = []
        # for i, (fc, feature) in enumerate(zip(self.fc_list, new_feature_list)):
        #     if type(feature) == list:
        #         pass
        #     else:
        #         out = fc(feature)
                
        #         # 使用BatchNorm层（如果启用）
        #         if self.use_batchnorm and out.size(0) > 1:
        #             out = self.bn_fc[i](out)
                    
        #         x.append(F.dropout(F.relu(out), training=self.training))
            
        # x = torch.cat(x, 0)
        # # graph_res_attn, query_res_attn = None, None
        # for l in range(self.num_layers):  # e_feat[g_l.edata['_ID']]
        #     x, graph_res_attn = self.graph_gat_layers[l](blocks[l], x, e_feat[blocks[l].edata['_ID']], res_attn=None)
        #     x = F.dropout(x.flatten(1), training=self.training)
        #     q, query_res_attn = self.query_gat_layers[l](blocks[l], q, e_feat[blocks[l].edata['_ID']], res_attn=None)
        #     q = F.dropout(q.flatten(1), training=self.training)
        #     q = self.fusion_layers[l](torch.cat((x, q), dim=-1))
        #     if l < self.num_layers - 1:
        #         q = F.dropout(F.relu(q), training=self.training)

        # return q
    
    def inference(self, g, features_list, e_feat, all_q, batch_size, device):
        with torch.no_grad():
            for l in range(self.num_layers):
                total_x = torch.zeros(g.num_nodes(),
                            self.num_hidden)
                total_q = torch.zeros(g.num_nodes(),
                            self.num_hidden
                            if l != self.num_layers - 1
                            else 1)
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.DataLoader(g, torch.arange(736389).to(device), sampler, batch_size=batch_size)
                # Within a layer, iterate over nodes in batches
                for input_nodes, output_nodes, blocks in dataloader:
                    block = blocks[0]
                    if l == 0:
                        # 特征变换
                        q = all_q[input_nodes]
                        new_feature_list = [[] for i in features_list]
                        feature_node = [feature.size(0) for feature in features_list]
                
                        ls = list(accumulate(feature_node))
                        for i_n in input_nodes:
                            for idx, (feat, num_nodes) in enumerate(zip(features_list, ls)):
                                if i_n < num_nodes:
                                    if idx == 0:
                                        new_feature_list[idx].append(feat[i_n])
                                    else:
                                        new_feature_list[idx].append(feat[i_n - ls[idx - 1]])
                                    break
                        new_feature_list = [f_l if len(f_l) == 0 else torch.stack(f_l, dim=0) for f_l in new_feature_list]
                        x = []
                        for i, (fc, feature) in enumerate(zip(self.fc_list, new_feature_list)):
                            if type(feature) == list:
                                pass
                            else:
                                out = fc(feature.to(device))
                                
                                # 使用BatchNorm层（如果启用）
                                if self.use_batchnorm and out.size(0) > 1:
                                    out = self.bn_fc[i](out)
                                    
                                x.append(F.dropout(F.relu(out), training=self.training))
                            
                        x = torch.cat(x, 0)
                    else:
                        x = all_x[input_nodes.cpu()].to(device)
                        q = all_q[input_nodes.cpu()].to(device)

                    x, _ = self.graph_gat_layers[l](block, x, e_feat[block.edata['_ID']], res_attn=None)
                    x = F.dropout(x.flatten(1), training=self.training)
                    q, _ = self.query_gat_layers[l](block, q, e_feat[block.edata['_ID']], res_attn=None)
                    q = F.dropout(q.flatten(1), training=self.training)
                    q = self.fusion_layers[l](torch.cat((x, q), dim=-1))
                    if l < self.num_layers - 1:
                        q = F.dropout(F.relu(q), training=self.training)
                    
                    total_x[output_nodes] = x.cpu()
                    total_q[output_nodes] = q.cpu()
                
                all_x = total_x
                all_q = total_q

        return all_q



# 定义早停类
class EarlyStopping:
    def __init__(self, patience=7, save_path=None, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -1
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

        self.best_max_threshold = -1
        self.best_max_score = -1

    def __call__(self, epoch, score, threshold, model):
        # score 是验证集上的F1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, score, threshold, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"epoch {epoch} F1 {score} with threshold {threshold}")
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, score, threshold, model)
            self.counter = 0

    def save_checkpoint(self, epoch, score, threshold, model):
        if self.verbose:
            logger.info(f"epoch {epoch} best F1 {score} with threshold {threshold}")
            logger.info(f"Valid score improve ({self.best_max_score:.5f} | {self.best_max_threshold:.3f} --> {score:.5f} | {threshold:.3f}). Saving model...")
        torch.save(model.state_dict(), osp.join(self.save_path, 'checkpoint.pt'))
        self.best_max_score = score
        self.best_max_threshold = threshold

        with open(osp.join(self.save_path, 'checkpoint_params.json'), 'w') as f:
            json.dump({'threshold': self.best_max_threshold}, f)