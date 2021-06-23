import torch
import torch.nn as nn
import torch.nn.functional as F

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes,flag=None):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.flag = flag

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        if self.flag == 'multi':
            ret = torch.sigmoid(ret)
        return ret

