import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None,adj=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        h_pl_attr = torch.spmm(adj, torch.squeeze(h_pl, 0)) # (node,n_h)
        h_mi_attr = torch.spmm(adj, torch.squeeze(h_mi, 0)) # (node,n_h)
        sc_1_attr = torch.unsqueeze(torch.squeeze(torch.mm(h_pl_attr,c.t()),1),0)# (batch, node)
        sc_2_attr = torch.unsqueeze(torch.squeeze(torch.mm(h_mi_attr,c.t()),1),0)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2



        logits = torch.cat((sc_1,sc_1_attr, sc_2,sc_2_attr), 1)

        return logits

