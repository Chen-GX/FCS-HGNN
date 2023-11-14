import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle
import os.path as osp
import torch
# from torch_geometric.data import HeteroData

def create_homograph(graph, meta_paths):
    """
    Create a homograph based on the given metapaths using sparse matrix multiplication.

    Args:
        graph (HeteroData): The input heterogeneous graph.
        meta_paths (list): A list of metapaths, each as a list of node types.

    Returns:
        dict: A dictionary contains adjacency matrix and nodes mapping between old graph and new graph.
    """

    # Get the number of nodes for each type
    # num_nodes_dict = {ntype: data.num_nodes for ntype, data in graph}
    num_nodes_dict = graph.num_nodes_dict  # 每个节点type的节点数

    # Convert edge_index to csr_matrix for each relation
    # adj_dict = {(src_type, tar_type): sp.coo_matrix(
    #                 (np.ones(edge_index.shape[1]), 
    #                  (edge_index[0], edge_index[1])),
    #                  shape=(num_nodes_dict[src_type], num_nodes_dict[tar_type])).tocsr()
    #             for (src_type, _, tar_type), edge_index in graph.edges.items()}
    adj_dict = {}
    for (src_type, _, tar_type) in graph.edge_types:
        edge_index = graph[(src_type, _, tar_type)].edge_index.numpy()
        adj_dict[(src_type, tar_type)] = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(num_nodes_dict[src_type], num_nodes_dict[tar_type])).tocsr()


    results = {}
    
    # Iterate over each metapath.
    for path in meta_paths:
        # Initialize the adjacency matrix with the first relation in the metapath
        src_type, dst_type = path[0], path[1]
        adj_matrix = adj_dict.get((src_type, dst_type), None)
        if adj_matrix is None:
            raise ValueError(f"No edges from {src_type} to {dst_type}")

        # Multiply the adjacency matrix with each subsequent relation in the metapath
        for i in range(1, len(path) - 1):
            src_type, dst_type = path[i], path[i + 1]
            next_adj_matrix = adj_dict.get((src_type, dst_type), None)
            if next_adj_matrix is None:
                raise ValueError(f"No edges from {src_type} to {dst_type}")
            adj_matrix = adj_matrix.dot(next_adj_matrix)

        # Add the adjacency matrix to the results (as a coo_matrix for easier interpretation)
        results[str(path)] = adj_matrix.tocoo()

    return results

# # Define your graph here.
# graph = ...

# # Define your metapaths.
# meta_paths = [['author', 'paper', 'author'], ['author', 'paper', 'conference', 'paper', 'author'], ['author', 'paper', 'term', 'paper', 'author']]

# # Create the homograph.
# homograph = create_homograph(graph, meta_paths)


def homograph_base_meta_path(args, graph: HeteroData, meta_paths, recreate=False):
    mp = []
    for meta_path in meta_paths:
        mp_str = ""
        for node_type in meta_path:
            mp_str += node_type[0]
        mp.append(mp_str)
    graph_file = "_".join(mp)
    if not recreate and osp.exists(osp.join(args.data_dir, args.data_name, f'{graph_file}.pkl')):
        # 只有不需要重新构造，并且文件存在，才可以直接读取
        with open(osp.join(args.data_dir, args.data_name, f'{graph_file}.pkl'), 'rb') as file:
            G = pickle.load(file)
    else:
        meta_path_based_adj = create_homograph(graph, meta_paths)

        # 合并多条元路径得到的邻接矩阵
        adj_matrix = None
        for path, matrix in meta_path_based_adj.items():
            if adj_matrix is None:
                adj_matrix = matrix
            else:
                adj_matrix += matrix
        adj_matrix.data = adj_matrix.sign().data  # 权重全部设为1
        G = nx.from_scipy_sparse_matrix(adj_matrix)
        with open(osp.join(args.data_dir, args.data_name, f'{graph_file}.pkl'), 'wb') as file:
            pickle.dump(G, file)
    return G

# 如果读取太慢
# import scipy.sparse as sp

# # 假设你有一个稀疏邻接矩阵 adj_matrix
# adj_matrix = sp.csr_matrix(...)  # 用你的数据填充这个矩阵

# def get_out_neighbors(matrix, node):
#     return set(matrix[node].nonzero()[1])  # 出度邻居

# def get_in_neighbors(matrix, node):
#     return set(matrix[:,node].nonzero()[0])  # 入度邻居

# cnode = ...  # 这是你要查询的节点

# out_neighbors = get_out_neighbors(adj_matrix, cnode)  # networkx是出度邻居，矩阵最好是csr
# in_neighbors = get_in_neighbors(adj_matrix, cnode)


class homo_graph(object):
    def __init__(self, hetero_graph, meta_path, max_depth=2) -> None:
        self.hetero_graph = hetero_graph
        self.meta_path = meta_path
        self.max_depth = max_depth

    def get_connected_nodes(self, start_node, edge_type):
        row, col = self.hetero_graph[edge_type].edge_index
        mask = row == start_node.item()
        connected_nodes = col[mask]
        return connected_nodes
    
    def dfs_through_metapath(self, query_node, path, depth=0):
        """基于dfs 查找query节点的元路径"""
        if depth == len(path) - 1:
            return [query_node]
        
        found_nodes = []
        nodes_to_search = self.get_connected_nodes(torch.tensor([query_node]), (path[depth], path[depth + 1])).tolist()

        for node in nodes_to_search:
            found_nodes.extend(self.dfs_through_metapath(node, path, depth + 1))

        return found_nodes

    def construct_homogeneous_graph(self, query_node):
        homo_G = nx.Graph()
        visited = set()  # 维护作为哪些节点已经查询过了

        stack = [(query_node, 0)]
        while stack:  # 外面这一层是bfs
            current_node, depth = stack.pop(0)  # 返回并删除第一个元素

            if depth == self.max_depth:  # 因为是bfs，有一个深度超过max，后面一定就是超过
                break

            if current_node in visited:
                continue     

            visited.add(current_node)
            current_found_nodes = []
            # 寻找query节点，所有元路径的邻居
            for path in self.meta_path:
                current_found_nodes.extend(self.dfs_through_metapath(current_node, path))
            
            # 去重
            current_found_nodes = list(set(current_found_nodes))
            for target_node in current_found_nodes:
                homo_G.add_edge(current_node, target_node)
                stack.append((target_node, depth + 1))

        print("Number of edges:", homo_G.number_of_edges())
        print("Number of nodes:", homo_G.number_of_nodes()) 
        return homo_G
    
