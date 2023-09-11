'''
Batched sparse matrix multiplication that scales with GPU's better. 
'''

import torch
import torch_geometric as pyg 
import numpy as np

class Conv(pyg.nn.MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight, bias, size):
        
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        return out.view(-1, 1) + bias.view(-1, 1)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j.view(-1, 1)



class SparseLinear2(torch.nn.Module): 
    def __init__(self, indices, size, dtype=torch.float32, bias=True, init='kaiming'):
        '''
        Sparse Linear layer, equivalent to sparse matrix multiplication as provided by indices. 

        Args: 
            indices         COO coordinates for the sparse matrix multiplication 
            size            size of weight matrix 
            dtype           weight matrix type 
            bias            whether to include a bias term; Wx + B
        '''
        super().__init__() 

        self.N, self.M = size
        self.size = size

        self.conv = Conv()

        src, dst = indices.type(torch.long)
        values = torch.randn(indices.size(1), dtype=dtype)

        # weight initialization 
        fan_in = pyg.utils.degree(dst, num_nodes=self.M)
        fan_out = pyg.utils.degree(src, num_nodes=self.N)
        n_in = fan_in[dst]      # number of input units 
        n_out = fan_out[src]    # number of output units 
        if init == 'xavier':  # glorot
            std = (2/(n_in + n_out))**0.5
        elif init == 'kaiming': # he
            std = (2/n_in)**(0.5)
        elif init == 'normal': 
            std = torch.ones_like(values)
        else:
            raise ValueError('unrecognized weight initialization method, options: xavier, kaiming, normal')
        values *= std

        self.values = torch.nn.Parameter(values) # torch optimizer require dense parameters 
        self.register_buffer('indices', indices.type(torch.long))
        if bias: self.bias = torch.nn.Parameter(torch.zeros((self.M, 1), dtype=dtype))

    def forward(self, x): 
        '''
        Assumes x is in shape: (B, N, 1), where B is batch dimension
        weight shape is (N, M)
        Returns shape (B, M, 1)

        batch dimension is handled in `torch_geometric` fashion, e.g., concatenated batch graphs via incremented node idx 
        '''
        B = x.size(0)
        E = self.indices.size(1)

        edge_index = self.indices.repeat(1, B) + torch.stack((torch.arange(B, device=x.device).repeat_interleave(E)*self.N,
                                                                  torch.arange(B, device=x.device).repeat_interleave(E)*self.M), dim=0)
        edge_id = torch.arange(self.indices.size(1)).repeat(B)
        bias_idx = torch.arange(self.M).repeat(B)
        batched_size = (self.N*B, self.M*B)

        edge_id = edge_id.to(x.device)
        bias_idx = bias_idx.to(x.device)
        edge_index = edge_index.to(x.device)

        edge_weight = self.values[edge_id]

        if hasattr(self, 'bias'):
            bias = self.bias[bias_idx]
        else: 
            bias = torch.zeros((self.M, 1)[bias_idx], device=x.device)

        x = x.view(-1,1)

        x = self.conv(x, edge_index, edge_weight, bias, size=batched_size)

        x = x.view(B, -1, 1)

        return x

