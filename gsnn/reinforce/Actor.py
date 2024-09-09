'''
'''

import torch 
from gsnn.models import NN

class Actor(torch.nn.Module): 
    def __init__(self, in_channels, bias=None, model='linear', hidden_channels=25): 
        '''
        N       num entities to select for 
        '''
        super().__init__()

        if model == 'linear':
            self.f = torch.nn.Linear(in_channels, 1, bias=False)
        elif model == 'nn': 
            self.f = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.GELU(), torch.nn.Linear(hidden_channels,1))
        else: 
            raise Exception()

        if bias is not None: 
            self.bias = torch.nn.Parameter(torch.tensor([bias], dtype=torch.float32))
        else: 
            self.bias = 0

    def forward(self, x): 

        return self.f(x) + self.bias
        
