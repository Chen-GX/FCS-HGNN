import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from conv import myGATConv

import logging
logger = logging.getLogger()


def load_model(args, graph_data):
    heads = [args.num_heads] * args.num_layers
    return V1(args, graph_data.g, torch.max(graph_data.e_feat).item() + 1,
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
                 ):
        super(V1, self).__init__()
        num_hidden, edge_dim, feat_drop, attn_drop, negative_slope = args.hidden_dim, args.edge_feats, args.dropout, args.dropout, args.slope
        self.g = g
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


    def forward(self, features_list, e_feat, q):
        x = []
        for i, (fc, feature) in enumerate(zip(self.fc_list, features_list)):
            out = fc(feature)
            
            # 使用BatchNorm层（如果启用）
            if self.use_batchnorm:
                out = self.bn_fc[i](out)
                
            x.append(F.dropout(F.relu(out), training=self.training))
            
        x = torch.cat(x, 0)
        graph_res_attn, query_res_attn = None, None
        for l in range(self.num_layers):
            x, graph_res_attn = self.graph_gat_layers[l](self.g, x, e_feat, res_attn=graph_res_attn)
            x = F.dropout(x.flatten(1), training=self.training)
            q, query_res_attn = self.query_gat_layers[l](self.g, q, e_feat, res_attn=query_res_attn)
            q = F.dropout(q.flatten(1), training=self.training)
            q = self.fusion_layers[l](torch.cat((x, q), dim=-1))
            if l < self.num_layers - 1:
                q = F.dropout(F.relu(q), training=self.training)

        return q



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