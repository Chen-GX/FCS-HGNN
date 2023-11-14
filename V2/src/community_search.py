import numpy as np
import multiprocessing as mp
from collections import deque
from utils import f1_score_
import logging
logger = logging.getLogger()

def dgl_to_adj(g): # g是dgl.graph
    adj = g.adj()  # get adjacency matrix in coo format
    num_nodes = g.number_of_nodes()

    edge_index = adj.row.numpy(), adj.col.numpy() # get edges from adjacency matrix

    adjacency_dict = {i: set() for i in range(num_nodes)}
    for (src, dst) in zip(*edge_index):
        if src != dst:  # 不加入自环，减少搜索空间
            adjacency_dict[src].add(dst)
    
    return adjacency_dict

class Community_Search(object):
    def __init__(self, args, graph_data, community_types, max_depth=-1):
        self.args = args
        if args.search_homo:
            self.graph = graph_data.homo_dict
            self.before_num_node = args.before_num_node  # 通过图上搜，要减去
        else:
            self.graph = dgl_to_adj(graph_data.g)
            self.before_num_node = 0  # 如果是全局搜，不需要减去
        self.id2type = graph_data.id2type
        self.community_types = community_types  # 社区所需要的节点的类型
        self.max_depth = max_depth  # =-1代表搜到底

    def bfs(self, query, threshold, probs):
        # 如果是同构图，那图中所有的节点都缩小了（减去了before num node）
        # 所以这里的query也应该减小，使得加入队列的id是减去后的id（因为同构图邻接表存储的就是减去后的）

        # 但是注意probs是所有的节点的概率，所以在probs中又得加回去

        # 如果是异构图，减去0也没影响，所以query这一步就应该减去before num node

        # 但是真实的社区的id是修正过的（从原来都是0开始，变成了在大图上的id）
        # 所以这里的结果集合community，返回之前也应该修正
        query = query - self.before_num_node
        visited = set()
        queue = deque([(query, 0)])  # 队列中的元素是一个元组，包括当前节点和到达该节点的深度
        community = set()

        while queue:
            current_node, depth = queue.popleft()

            if current_node not in visited and (depth <= self.max_depth or self.max_depth == -1):
                visited.add(current_node)

                # 只有当节点的类型和概率都满足条件时，才将其添加到结果中
                if self.id2type[current_node + self.before_num_node] in self.community_types and probs[current_node + self.before_num_node] >= threshold:
                    community.add(current_node)

                # 不论节点类型是否符合，只要深度允许，就将其所有邻居加入队列以延续搜索
                if depth < self.max_depth or self.max_depth == -1:
                    for next_node in self.graph[current_node]:  # 如果是同质图，这里存储的就是减去后的id
                        queue.append((next_node, depth + 1))
        
        # 修正返回的社区
        community_find = [com + self.before_num_node for com in community]
        return set(community_find)

    

def search_threshold(lc, scorelists, num_data):
    s_ = 0.1  # 阈值
    f1_m = 0.0  # 最佳f1
    s_m = s_  # 最佳阈值
    while(s_<=0.96):
        f1_x = 0.0
        logger.info(f"------------------------------ {s_}")
        for q, comm, probs in scorelists:
            comm_find = lc.bfs(q, s_, probs)

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            f1_x = f1_x + f1#pre
        f1_x = f1_x / num_data
        if f1_m < f1_x:
            f1_m = f1_x
            s_m = s_
        s_ = s_+0.05
    logger.info(f"------------------------ {s_m} {f1_m}")
    return s_m, f1_m


def worker(args):
    s_, lc, scorelists, num_data = args
    total_f1_score = 0.0
    total_length = 0

    for q, comm, probs in scorelists:
        predicted_communities = list(set(lc.bfs(q, s_, probs)))
        total_f1_score += f1_score_(predicted_communities, list(set(comm)))[0]
        total_length += len(predicted_communities)

    f1_x = total_f1_score / num_data
    average_length = total_length / num_data
    return s_, f1_x, average_length

def search_threshold_parallel(lc, scorelists, num_data, max_processes=20):
    threshold_range = np.arange(0.05, 0.96, 0.05)
    max_processes = min(max_processes, mp.cpu_count())  
    
    with mp.Pool(processes=max_processes) as pool:
        results = pool.map(worker, [(s_, lc, scorelists, num_data) for s_ in threshold_range])

    result_max_f1 = max(results, key=lambda x: x[1])
    s_m, f1_m, avg_len = result_max_f1  

    return s_m, f1_m, avg_len 



# def worker(args):
#     s_, lc, scorelists, num_data = args
#     f1_x = sum(f1_score_(list(set(lc.bfs(q, s_, probs))), list(set(comm)))[0] for q, comm, probs in scorelists) / num_data
#     return s_, f1_x

# def search_threshold_parallel(lc, scorelists, num_data, max_processes=20):
#     threshold_range = np.arange(0.05, 0.96, 0.005)
#     # logger.info(mp.cpu_count())  96
#     max_processes = min(max_processes, mp.cpu_count())  # 获取CPU的核心数
#     pool = mp.Pool(processes=max_processes)

#     results = pool.map(worker, [(s_, lc, scorelists, num_data) for s_ in threshold_range])

#     s_m, f1_m = max(results, key=lambda x: x[1])

#     return s_m, f1_m   # 将结果返回

# search_threshold_parallel(lc, scorelists, valid_data)