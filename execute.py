import numpy as np
import scipy.io as scio
from copy import deepcopy
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import metric
import matplotlib
import matplotlib.pyplot as plt
from sklearn import manifold
from models import ECRN, LogReg
from utils import process
import random
def message_print(adj, features, labels, idx_train, idx_val, idx_test):
    print('adj',type(adj),adj.shape)
    print('features',type(features),features.shape)
    print('labels',type(labels),labels.shape)
    print('idx_train',type(idx_train),len(idx_train))
    print('idx_val',type(idx_val),len(idx_val))
    print('idx_test',type(idx_test),len(idx_test))

dataset = ' '
path = 'data/ms_academic_phy.npz'
# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0

hid_units = 512
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

def train(dataset,ii):
    # adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    if dataset == 'dblp':
        adj, features, labels, idx_train, idx_test = process.load_dblp()
    else:
        adj, features, labels, idx_train, idx_val, idx_test = process.load_npz(np.load(path))

    # message_print(adj, features, labels, idx_train, idx_val, idx_test)
    lab = deepcopy(labels)

    features, _ = process.preprocess_features(features)

    if dataset == 'citeseer':
        labels_index_list = np.where(labels == 1)[0]
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]
    ''''''
    adj_label_u = deepcopy(adj).todense()
    for i in range(nb_nodes):
        if np.sum(adj_label_u[i])==0:
            adj_label_u[i,i]=1
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = adj_label.todense()

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if sparse:
        # sp_adj_label = process.sparse_mx_to_torch_sparse_tensor(adj_label)
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
        # adj_label = torch.FloatTensor(adj_label[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    if dataset !='dblp':
        idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    model = ECRN(ft_size, hid_units, nonlinearity,nb_nodes)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        # print('Using CUDA')
        model.cuda()
        features = features.cuda()
        if sparse:
            sp_adj = sp_adj.cuda()
            # adj_label = adj_label.cuda()
        else:
            adj = adj.cuda()
            # adj_label = adj_label.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    b_xent = nn.BCEWithLogitsLoss()
    bce_xent = nn.BCELoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]
        ''''''
        row = np.arange(nb_nodes)
        col_true = np.array([np.random.choice(np.where(adj_label_u[i] != 0)[1]) for i in np.arange(nb_nodes)])
        # print(col_true)
        col_false = np.array([np.random.choice(np.where(adj_label[i] == 0)[1]) for i in np.arange(nb_nodes)])
        col_label_true = torch.ones(nb_nodes).cuda()
        col_label_false = torch.zeros(nb_nodes).cuda()

        lbl_1 = torch.ones(batch_size, 2*nb_nodes)
        lbl_2 = torch.zeros(batch_size, 2*nb_nodes)

        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()



        logits, emb_1, emb_2 = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
        # logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

        loss1 = b_xent(logits, lbl)

        ''''''
        nei_1_true = torch.sum(torch.mul(emb_1[row], emb_1[col_true]), 1)
        nei_1_false = torch.sum(torch.mul(emb_1[row], emb_1[col_false]), 1)

        nei_2_true = torch.sum(torch.mul(emb_2[row], emb_2[col_true]), 1)
        nei_2_false = torch.sum(torch.mul(emb_2[row], emb_2[col_false]), 1)
        loss2 = b_xent(nei_1_true, col_label_true)
        loss3 = b_xent(nei_1_false, col_label_false)
        loss4 = b_xent(nei_2_true, col_label_true)
        loss5 = b_xent(nei_2_false, col_label_false)
        loss = loss1+loss2+loss3+loss4+loss5
        # print('Loss:', loss)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_ecrn.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            # print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    # print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_ecrn.pkl'))

    embeds= model.embed(features, sp_adj if sparse else adj, sparse, None)

    # '''mutili-labels'''
    # from sklearn.metrics import f1_score,accuracy_score
    # train_embs = embeds[0, idx_train]
    # val_embs = embeds[0, idx_val]
    # test_embs = embeds[0, idx_test]
    #
    # train_lbls = labels[0, idx_train]
    # val_lbls = labels[0, idx_val]
    # test_lbls = labels[0, idx_test].cpu()
    #
    # tot_acc = torch.zeros(1)
    # tot_acc = tot_acc.cuda()
    # tot_micro = torch.zeros(1)
    # tot_micro = tot_micro.cuda()
    # tot_macro = torch.zeros(1)
    # tot_macro = tot_macro.cuda()
    #
    # accs = []
    # micros = 0
    # macros = 0
    #
    # for _ in range(50):
    #     log = LogReg(hid_units, nb_classes,'multi')
    #     opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    #     log.cuda()
    #
    #     pat_steps = 0
    #     best_acc = torch.zeros(1)
    #     best_acc = best_acc.cuda()
    #     for _ in range(100):
    #         log.train()
    #         opt.zero_grad()
    #
    #         logits = log(train_embs)
    #         loss = bce_xent(logits, train_lbls)
    #
    #         loss.backward()
    #         opt.step()
    #
    #     logits = log(test_embs)
    #
    #     preds = logits.detach().cpu()
    #     preds[preds>0.5] = 1
    #     preds[preds<=0.5] = 0
    #     micro = f1_score(test_lbls,preds,average='micro')
    #     macro = f1_score(test_lbls,preds,average='macro')
    #     acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]/nb_classes
    #
    #     accs.append(acc * 100)
    #     micros += micro * 100
    #     macros += macro * 100
    #     # print(acc)
    #     # tot_acc += acc
    #     # tot_micro +=micro
    #     # tot_macro +=macro
    #
    # # print('Average accuracy:', tot / 50)
    #
    # accs = torch.stack(accs)
    # # tot_micro = torch.stack(tot_micro)
    # # tot_macro = torch.stack(tot_macro)
    # print(str(ii), 'f1-micro:', micros/50,'f1-macro:', macros/5,
    #       'acc:', accs.mean().item(), accs.std().item(),)

    train_embs = embeds[0, idx_train]
    if dataset != 'dblp':
        val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    if dataset != 'dblp':
        val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    tot = torch.zeros(1)
    tot = tot.cuda()

    accs = []
    print(111)
    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        print(222)
        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print(acc)
        tot += acc

    # print('Average accuracy:', tot / 50)

    accs = torch.stack(accs)
    print(str(ii),':',accs.mean().item(), accs.std().item())

    # emb = embeds[0, :]
    # emb = emb.cpu()
    # em = deepcopy(emb)
    # if dataset == 'citeseer':
    #     em = em[labels_index_list]
    # kmeans = KMeans(n_clusters=nb_classes, random_state=0).fit(em)
    # predict_labels = kmeans.predict(em)
    # cm = metric.clustering_metrics(labels_list, predict_labels)
    # cm.evaluationClusterModelFromLabel()


    # if ii == 9:
    #     labels_num = lab.shape[1]
    #     color_list = []
    #     if dataset == 'citeseer':
    #         for i in range(lab.shape[0]):
    #             if np.sum(lab[i]) == 0:
    #                 color_list.append(lab.shape[1])
    #             else:
    #                 color_list.append(np.where(lab[i] == 1)[0][0])
    #     else:
    #         color_list = np.where(lab == 1)[1]
    #         color_list = [int(i) for i in color_list]
    #
    #     emb = embeds[0, :]
    #     emb = emb.cpu()
    #     t_sne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    #     X_tsne = t_sne.fit_transform(emb)
    #     # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    #     # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    #     if dataset != 'pubmed':
    #         colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
    #     else:
    #         colors = ['red', 'green', 'blue']
    #     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_list,s=5,
    #                 cmap=matplotlib.colors.ListedColormap(colors), linewidths=0.05)
    #     plt.show()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    torch.cuda.set_device(0)
    # dataset = 'cora'
    file_name = 'test1.txt'
    with open(dataset+'_'+file_name,'a') as ff:
        ff.writelines('')
    # 'cora', 'citeseer', 'pubmed'

    for i in range(50):
        train(dataset, i)