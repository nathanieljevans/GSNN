'''
Distributional Neural Network. Predicts probability distribution parameters. 

'''

import torch
import numpy as np

class DistNN(torch.nn.Module): 

    def __init__(self, in_channels, hidden_channels, layers, dropout, dist='gaussian', nonlin=torch.nn.ELU, norm=torch.nn.BatchNorm1d, eps=1e-6, transform=None, input_idxs=None): 
        '''
        predicts 2-parameter distribution. Assumes a single output/target. 

        Args: 
            in_channels             int                 number of input channels 
            hidden_channels         int                 number of hidden channels per layer 
            layers                  int                 number of hidden layers 
            dropout                 float               dropout regularization probability 
            nonlin                  pytorch.module      non-linear activation function 
            norm                    pytorch.module      normalization method to use 
            eps                     float               positive small value to stabilize softplus and ensure non-zer variance. 
        '''
        super().__init__()
        self.dist = dist
        self.eps = eps

        self.transform = transform
        self.register_buffer('input_idxs', input_idxs)

        if transform is not None: 
            mu, sigma = transform
            self.register_buffer('mu', mu)
            self.register_buffer('sigma', sigma)

        seq = [torch.nn.Linear(in_channels, hidden_channels)]
        if norm is not None: seq.append(norm(hidden_channels))
        seq += [nonlin(), torch.nn.Dropout(dropout)] 
        for _ in range(layers - 1): 
            seq += [torch.nn.Linear(hidden_channels, hidden_channels)]
            if norm is not None: seq.append(norm(hidden_channels))
            seq += [nonlin(), torch.nn.Dropout(dropout)]
        seq += [torch.nn.Linear(hidden_channels, 2)]
        self.nn = torch.nn.Sequential(*seq)

    def forward(self, x): 
        '''
        input size: (B, N)
        output: mean, var 
        '''
        # zscore inputs 
        if self.transform is not None: 
            x = (x - self.mu)/(self.sigma + 1e-6)

        # input variance filter 
        if self.input_idxs is not None: 
            x = x[:, self.input_idxs]

        out = self.nn(x).squeeze()

        # return predictions as a torch.distributions.Distribution object 
        if self.dist == 'gaussian': 
            mu, logvar = out.T
            std = torch.exp(0.5 * logvar)
            pred_dist = torch.distributions.Normal(mu, std)
        elif self.dist == 'beta': 
            a,b = torch.nn.functional.softplus(out).T + 1e-8 # ensure positive value parameters
            pred_dist = torch.distributions.Beta(a,b)
        else: 
            raise NotImplementedError('unrecognized `dist` parameter.')

        return pred_dist

        