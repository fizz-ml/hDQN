import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dill


#don't ask how I came up with these numbers

SIZE_H1 = 50
SIZE_H2 = 100
SIZE_H3 = 60

class Actor(torch.nn.Module):
    """Defines custom model
    Inherits from torch.nn.Module
    """
    def __init__(self, dim_input, dim_output):

        super(Actor, self).__init__()
        self._dim_input = dim_input
        self._dim_output = dim_output

        '''Initialize nnet layers'''
        self._l1 = torch.nn.Linear(self._dim_input, SIZE_H1)
        self._l2 = torch.nn.Linear(SIZE_H1, SIZE_H2)
        self._l3 = torch.nn.Linear(SIZE_H2, SIZE_H3)
        self._l4 = torch.nn.Linear( SIZE_H3, self._dim_output)

    def forward(self,s_t, aux_array): #TODO: Add aux task support, experiment with inputting previous action as well
        #inp = np.concatenate((s_t,r_t), axis = 0)
        #inp = np.expand_dims(inp, axis = 0)
        x = s_t # hVariable(torch.FloatTensor(s_t.astype(np.float32)))
        #print(s_t)
        self._l1_out = F.relu(self._l1(x))
        self._l2_out = F.relu(self._l2(self._l1_out))
        self._l3_out = F.relu(self._l3(self._l2_out))
        self._out = F.tanh(self._l4(self._l3_out))

        #print('_out',self._out)
        return self._out, {}

#TODO: Change to use config file instead, add main function and all that
def generate(path, idim, odim):
    M = actor(idim, odim)
    with open(path, "wb") as f:
        dill.dump(M, f)
