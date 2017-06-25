import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import dill

H_LAYER1 = 50
H_LAYER2 = 10
H_LAYER3 = 10
H_LAYER4 = 10

class Critic(nn.Module):
    def __init__(self,dim_input, dim_output):
        super(Critic, self).__init__()
        self._dim_input = dim_input
        self._dim_output = dim_output
        self.linear1 = nn.Linear(self._dim_input, H_LAYER1)
        self.linear2 = nn.Linear(H_LAYER1, H_LAYER2)
        self.linear3 = nn.Linear(H_LAYER2, H_LAYER3)
        self.linear4 = nn.Linear(H_LAYER3, H_LAYER4)
        self.linear5 = nn.Linear(H_LAYER4, self._dim_output)

    def forward(self,s,a):
        '''
        s = Variable(torch.FloatTensor(np.array(s,dtype=np.float32)))
        if(type(a)!=type(s)):
            a = Variable(torch.FloatTensor(np.array(a,dtype=np.float32)))
        '''
        x = torch.cat([s,a],1)

        a1 = F.relu(self.linear1(x))
        a2 = F.relu(self.linear2(a1))
        a3 = F.relu(self.linear3(a2))
        a4 = F.relu(self.linear4(a3))
        y = self.linear5(a4)
        return y

#TODO: Change to use config file instead, add main function and all that
def generate(path, idim, odim):
    M = critic(idim, odim)
    with open(path, "wb") as f:
        dill.dump(M,f)
