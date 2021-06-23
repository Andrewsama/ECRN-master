from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import scipy.sparse as sp
import torch
import pickle as pkl
from sklearn.metrics import f1_score
from collections import defaultdict

import networkx as nx
from networkx.readwrite import json_graph

version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
# assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN = 5
N_WALKS = 50


def load_ppi_data(prefix="ppi", normalize=True):
    G_data = json.load(open("data/ppi/"+prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    if os.path.exists("data/ppi/"+prefix + "-feats.npy"):
        feats = np.load("data/ppi/"+prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open("data/ppi/"+prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    walks = []
    class_map = json.load(open("data/ppi/"+prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")

    # "train_removed"表示该边连接的两节点是否有测试集/验证集节点
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        # train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        # train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(feats)
        feats = scaler.transform(feats)


    adj = nx.adjacency_matrix(G)

    idx_train,idx_val,idx_test = [],[],[]

    idx_val = [n for n in G.nodes() if G.node[n]['val']]  # 验证集
    idx_test = [n for n in G.nodes() if G.node[n]['test']]  # 测试集

    no_train_nodes_set = set(idx_val + idx_test)  # 非训练集
    idx_train = set(G.nodes()).difference(no_train_nodes_set)  # 训练集
    return adj, feats, id_map,class_map,idx_train,idx_val,idx_test



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return r_mat_inv_sqrt.dot(mx).dot(r_mat_inv_sqrt)
    # return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def score_f1(output,labels,f1_s = 'micro',nclass = 7):
    num = [i for i in range(nclass)]
    preds = output.max(1)[1].type_as(labels)
    f1_val = f1_score(preds, labels, labels=num, average=f1_s)
    return f1_val

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index