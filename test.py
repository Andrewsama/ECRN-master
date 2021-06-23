import numpy as np
import torch
from copy import deepcopy
import scipy.sparse as sp

def load_npz(f):
    features = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape'])
    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                        f['adj_shape']).tocoo()
    nodes_num = adj.shape[0]
    labels_tmp = f['labels']
    labels_num = len(set(labels_tmp))
    idx_tmp = []
    idx_train = []
    idx_val = []
    idx_test = []
    idx_train_val = [np.random.choice(np.squeeze(np.argwhere(labels_tmp == i)),
                                      size=60,replace=False)
                     for i in range(labels_num)]
    for i in range(len(idx_train_val)):
        idx_tmp.extend(idx_train_val[i])
        choose = np.random.choice(idx_train_val[i],size=30,replace=False)
        idx_train.extend(choose)
        idx_val.extend([j for j in idx_train_val[i] if j not in choose])
    idx_test = np.delete(range(nodes_num), idx_tmp)

    labels = np.zeros([nodes_num, labels_num])
    labels[np.arange(nodes_num), labels_tmp] = 1
    return adj, features, labels, idx_train, idx_val, idx_test


if __name__ == '__main__':
    path = 'data/amazon_electronics_computers.npz'
    data = np.load(path)
    read_npz(data)