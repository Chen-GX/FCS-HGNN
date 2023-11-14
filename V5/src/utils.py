import os
import os.path as osp
import random
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

import logging
logger = logging.getLogger(__name__)

def set_seed(seed: int = 1024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return float(intersection) / union

# def f1_score_(comm_find, comm):

#     lists = [x for x in comm_find if x in comm]
#     if len(lists) == 0:
#         #print("f1, pre, rec", 0.0, 0.0, 0.0)
#         return 0.0, 0.0, 0.0
#     #pre = (len(lists)-1) * 1.0 / len(comm_find)
#     pre = len(lists) * 1.0 / len(comm_find)
#     rec = len(lists) * 1.0 / len(comm)
#     f1 = 2 * pre * rec / (pre + rec)
#     #print("f1, pre, rec", f1, pre, rec)
#     return f1, pre, rec

def f1_score_(comm_find, comm):
    common = set(comm_find) & set(comm)
    if len(common) == 0:
        return 0.0, 0.0, 0.0

    pre = len(common) / len(comm_find)
    rec = len(common) / len(comm)
    f1 = 2 * pre * rec / (pre + rec)

    return f1, pre, rec


def NMI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = normalized_mutual_info_score(truthlabel, prelabel)
    #print("q, nmi:", score)
    return score

def get_metric(find_comms, comms, n_nodes):
    """
    find_comms是寻找到的社区中节点的 id
    comms 真实的社区的节点id
    n_nodes: 节点的数量
    """
    # js, 要求都是集合，把label转换为
    metric = {}
    metric['js'] = jaccard_similarity(set(find_comms), set(comms))
    metric['f1'], metric['pre'], metric['rec'] = f1_score_(find_comms, comms)
    metric['nmi'] = NMI_score(find_comms, comms, n_nodes)

    return metric

