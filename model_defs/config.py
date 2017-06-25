import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#don't ask how I came up with these numbers

SIZE_H1 = 50
SIZE_H2 = 100
SIZE_H3 = 60



class model(torch.nn.Module):
    """Defines custom model
    Inherits from torch.nn.Module 
    """
    def __init__(self, dim_input, dim_output):

        super(model, self).__init__()
        self._dim_input = dim_input
        self._dim_output = dim_output

        '''Initialize nnet layers'''
        self._l1 = torch.nn.Linear(self._dim_input, SIZE_H1)
        self._l2 = torch.nn.Linear(SIZE_H1, SIZE_H2)
        self._l3 = torch.nn.Linear(SIZE_H2, SIZE_H3)
        self._l4 = torch.nn.Linear( SIZE_H3, self._dim_output)

    def forward(self,s_t, r_t, aux_array): #TODO: Add aux task support, experiment with inputting previous action as well
        print(r_t.shape)
        print(s_t.shape)
        x = Variable(torch.FloatTensor(
            np.concatenate((s_t,np.expand_dims(r_t, axis=1)), axis = 1))
        )
        self._l1_out = F.relu(self._l1(x))
        self._l2_out = F.relu(self._l2(self._l1_out))
        self._l3_out = nn.BatchNorm1d(SIZE_H3)(self._l3(self._l2_out))
        self._out = F.tanh(self._l4(self._l3_out))


        return self._out
