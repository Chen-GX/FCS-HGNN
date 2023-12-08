import dgl
import random
import torch
import pickle
from collections import deque
import numpy as np
import os.path as osp
from dgl.dataloading import NeighborSampler

class MultiLayerNeighborSampler(NeighborSampler):
    def __init__(self, fanouts):
        super().__init__(len(fanouts))

        self.fanouts = fanouts

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        if fanout is None:
            frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
        return frontier
    


# 以深度为超参，探索query节点可能为后续节点的列表

class DFSGraphExplorer:
    def __init__(self, args, node_types, community_types, max_depth, prob, device):
        """
        graph: 待搜索的图
        node_types: 图中每一个节点的type  list
        community_types: 社区节点
        max_depth: 探索深度，-1表示探索到底
        prob: 表示满足约束的节点被加入结果集合的概率
        """
        graph_path = osp.join(args.data_dir, args.data_name, 'search_graph.pickle')
        if osp.exists(graph_path):
            with open(graph_path, 'rb') as handle:
                self.graph = pickle.load(handle)
        else:
            raise FileNotFoundError
        # self.graph = graph
        self.node_types = node_types
        self.community_types = community_types
        self.max_depth = max_depth
        self.prob = prob
        self.device = device
        self.visited = set()
        self.result = set()

    def _dfs(self, node, depth):
        if node in self.visited or (self.max_depth != -1 and depth > self.max_depth):
            return
        self.visited.add(node)

        # 如果节点类型满足要求，以概率 p 将其加入结果集中
        if self.node_types[node] in self.community_types and random.random() < self.prob:
            self.result.add(node)

        # 探索邻居节点
        for neighbor in self.graph[node]:
            self._dfs(neighbor, depth + 1)

    def explore(self, query):
        self.visited.clear()
        self.result.clear()
        self._dfs(query, 0)
        return torch.tensor(list(self.result), dtype=torch.int64, device=self.device)


class BFSGraphExplorer:
    def __init__(self, args, node_types, community_types, max_depth, prob, device, num_nodes):
        graph_path = osp.join(args.data_dir, args.data_name, 'search_graph.pickle')
        if osp.exists(graph_path):
            with open(graph_path, 'rb') as handle:
                self.graph = pickle.load(handle)
        else:
            raise FileNotFoundError

        self.node_types = node_types
        self.community_types = community_types
        self.max_depth = max_depth
        self.prob = prob
        self.device = device
        
        self.num_nodes = num_nodes
        self.visited = np.zeros(self.num_nodes, dtype=bool)
        self.result = set()

    def _bfs(self, node):
        queue = deque([(node, 0)])

        while queue:
            curr_node, depth = queue.popleft()
            
            if not self.visited[curr_node] and (self.max_depth == -1 or depth <= self.max_depth):
                self.visited[curr_node] = True

                # 如果节点类型满足要求，以概率 p 将其加入结果集中
                if self.node_types[curr_node] in self.community_types:  #  and random.random() < self.prob
                    self.result.add(curr_node)
                
                # 把邻居节点加入队列，深度加一
                if self.max_depth == -1 or depth < self.max_depth:
                    for neighbor in self.graph[curr_node]:
                        queue.append((neighbor, depth + 1))

    def explore(self, query):
        self.visited.fill(False)
        self.result.clear()
        self._bfs(query)
        return torch.tensor(list(self.result), dtype=torch.int64, device=self.device)

    

# from multiprocessing import Pool

# def dfs_wrapper(args):
#     return GraphExplorer(*args[:-3]).explore(*args[-3:])

# def parallel_explore(queries, graph, node_types, max_depth, prob):
#     with Pool() as pool:
#         # 创建参数列表
#         args_list = [(graph, node_types, query, max_depth, prob) for query in queries]
        
#         # 使用 multiprocessing Pool 进行并行处理
#         results = pool.map(dfs_wrapper, args_list)
    
#     return results
