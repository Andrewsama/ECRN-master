import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCN, AvgReadout, Discriminator

class ECRN(nn.Module):
    def __init__(self, n_in, n_h, activation, nb_nodes):
        super(ECRN, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2, adj)
        #
        emb_1 = torch.squeeze(h_1,0)
        emb_2 = torch.squeeze(h_2,0)

        emb_1 = F.normalize(emb_1,p=2,dim=1)
        emb_2 = F.normalize(emb_2,p=2,dim=1)
        return ret, emb_1, emb_2


    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        # c = self.read(h_1, msk)

        return h_1.detach()#, c.detach()

