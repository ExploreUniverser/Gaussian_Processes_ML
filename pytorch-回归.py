import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
class MLP(nn.Module):
    def __init__(self, ninp, nhid):
        super(MLP, self).__init__()
        self.inlayer = nn.Linear(ninp,nhid)
        self.hidlayer = nn.Linear(nhid, nhid)
        # self.bn = nn.LayerNorm(nhid)
        self.outlayer = nn.Linear(nhid,nhid)
    def forward(self,x):

        out = self.inlayer(x)
        out = F.relu(out)
        out = self.hidlayer(out)
        out = F.relu(out)
        out = self.outlayer(out)
        # out = F.normalize(out, dim=-1)
        return out