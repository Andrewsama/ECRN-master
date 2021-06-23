import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))
       
    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_blogCatalog():
    graph_file = 'data/blogCatalog/bc_edgelist.txt'
    g = nx.read_edgelist(graph_file, create_using=nx.Graph(), nodetype=None, data=[('weight', int)])
    num_of_nodes = g.number_of_nodes()  # 节点数
    num_of_edges = g.number_of_edges()  # 边数
    edges_raw = g.edges()  # 边数组
    nodes_raw = g.nodes()  # 节点数组

    embedding = []  # 嵌入矩阵
    neg_nodeset = []  # 所有不在路径中的节点的下标序列
    node_index = {}  # 节点映射下标
    node_index_reversed = {}  # 下标映射节点
    for index, node in enumerate(nodes_raw):
        node_index[node] = index          #节点 -> 下标
        node_index_reversed[index] = node #下标 -> 节点
    edges = np.array([(node_index[u], node_index[v]) for u, v in edges_raw]).reshape([-1,2])  # 边的下标数组：(节点下标，节点下标)
    adj = sp.coo_matrix((np.ones(num_of_edges), (edges[:, 0], edges[:, 1])),
                        shape=(num_of_nodes, num_of_nodes),
                        dtype=np.float32)
    feat_data = sp.eye(num_of_nodes)

    label_node = []
    label_multi = []
    labels_num = []
    with open('data/blogCatalog/bc_labels.txt','r') as fin:
        l = fin.readline()
        while l:
            vec = l.strip().split(' ')
            label_node.append(node_index[vec[0]])
            label_multi.append(vec[1:])
            for ii in vec[1:]:
                labels_num.append(ii)
            l = fin.readline()
    labels_num = set(labels_num)
    #print('labels_num:',labels_num)
    labels = np.zeros([num_of_nodes,len(labels_num)])
    for i in range(len(label_node)):
        for j in label_multi[i]:
            labels[label_node[i]][int(j)] = 1
    rand_indices = np.random.permutation(num_of_nodes)
    idx_train = rand_indices[:200]
    idx_val = rand_indices[500:1000]
    idx_test = list(rand_indices[1000:2000])
    return adj,feat_data,labels,idx_train, idx_val, idx_test
    # return feat_data, labels, adj, idx_train, idx_val, idx_test

def load_npz(f):
    features = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape'])
    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                        f['adj_shape']).tocoo()
    nodes_num = adj.shape[0]
    adj_dense = adj.todense()
    x = np.sum(adj_dense, 1)
    labels_tmp = f['labels']
    assert np.sum(np.where(adj_dense < 0)) == 0  # 邻接矩阵值为正数
    assert np.sum(np.where(np.diag(adj_dense) != 0)) == 0  # 无自环
    #转成对称矩阵
    adj = adj + adj.transpose()
    adj[adj > 0] = 1
    adj_dense = adj.todense()
    assert np.sum(np.where(adj_dense != adj_dense.transpose())) == 0  # 对称矩阵
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
    idx_test = list(np.delete(range(nodes_num), idx_tmp))

    labels = np.zeros([nodes_num, labels_num])
    labels[np.arange(nodes_num), labels_tmp] = 1
    return adj, features, labels, idx_train, idx_val, idx_test
from sklearn.model_selection import train_test_split

def load_dblp(alpha=0.1):
    adj = np.load('data/dblp_adj.npy')   #2维，ndarray
    features = np.load('data/dblp_features.npy')#2维,ndarray
    labels = np.load('data/dblp_label.npy')#2维onehot,ndarray

    idx = np.arange(labels.shape[0])
    idx = idx.reshape(-1, 1)
    idx_train, idx_test, _, _ = train_test_split(idx, labels,
                                                 test_size=1 - alpha)
    adj = sp.csr_matrix(adj)
    features = sp.lil_matrix(features)
    idx_train = list(np.squeeze(idx_train))
    idx_test = list(np.squeeze(idx_test))
    return adj, features, labels, idx_train, idx_test
