import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

IN_DIM = 3
OUT_DIM = 1
H_LAYER1 = 50
H_LAYER2 = 10
H_LAYER3 = 10
H_LAYER4 = 10


class model(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(model, self).__init__()
        IN_DIM = dim_input 
        OUT_DIM = dim_output 
        self.linear1 = nn.Linear(IN_DIM, H_LAYER1)
        self.linear2 = nn.Linear(H_LAYER1, H_LAYER2)
        self.linear3 = nn.Linear(H_LAYER2, H_LAYER3)
        self.linear4 = nn.Linear(H_LAYER3, H_LAYER4)
        self.linear5 = nn.Linear(H_LAYER4, OUT_DIM)

        self.batch_norm1 = nn.BatchNorm1d(H_LAYER1)
        self.batch_norm2 = nn.BatchNorm1d(H_LAYER2)
        self.batch_norm3 = nn.BatchNorm1d(H_LAYER3)
        self.batch_norm4 = nn.BatchNorm1d(H_LAYER4)

    
    def forward(self,s,a):
        np_x = np.concatenate((s,a),axis=1)
        x = Variable(torch.FloatTensor(np_x))

        a1 = F.relu(self.linear1(x))
        b1 = self.batch_norm1(a1)
        a2 = F.relu(self.linear2(b1))
        b2 = self.batch_norm2(a2)
        a3 = F.relu(self.linear3(b2))
        b3 = self.batch_norm3(a3)
        a4 = F.relu(self.linear4(b3))
        b4 = self.batch_norm4(a4)
        y = self.linear5(b4)
        return y

	
