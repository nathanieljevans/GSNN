'''
Batched sparse matrix multiplication that scales with GPU's better. 
'''

import torch
import torch_geometric as pyg 
import numpy as np
import time

class Conv(pyg.nn.MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight, bias, size):
        
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size).view(-1, 1)
        
        if bias is not None: out = out + bias.view(-1, 1)
        
        return out 

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j.view(-1, 1)


def batch_graphs(N, M, edge_index, B, device, edge_mask=None):
    '''
    Create batched edge_index/edge_weight tensors for bipartite graphs.

    Args:
        N (int): Size of the first set of nodes in each bipartite graph.
        M (int): Size of the second set of nodes in each bipartite graph.
        edge_index (tensor): edge index tensor to batch.
        B (int) batch size
        device (str)

    Returns:
        torch.Tensor: Batched edge index.
    '''
    E = edge_index.size(1)
    batched_edge_indices = edge_index.repeat(1, B).contiguous()
    batch_idx = torch.repeat_interleave(torch.arange(B, dtype=torch.long, device=device), E).contiguous()

    if edge_mask is not None: 
        ## this batching process takes much longer
        edge_subset = torch.cat([mask + E*i for i,mask in enumerate(edge_mask)], dim=-1) # this takes most of the time
        batched_edge_indices = batched_edge_indices[:, edge_subset]
        batch_idx = batch_idx[edge_subset]
    else: 
        edge_subset=None

    src_incr = batch_idx*N
    dst_incr = batch_idx*M
    incr = torch.stack((src_incr, dst_incr), dim=0)
    batched_edge_indices += incr

    return batched_edge_indices, edge_subset

class SparseLinear(torch.nn.Module): 
    def __init__(self, indices, size, dtype=torch.float32, bias=True, init='kaiming'):
        '''
        Sparse Linear layer, equivalent to sparse matrix multiplication as provided by indices. 

        Args: 
            indices         COO coordinates for the sparse matrix multiplication 
            size            size of weight matrix 
            dtype           weight matrix type 
            bias            whether to include a bias term; Wx + B
            init            weight initialization strategy 
        '''
        super().__init__() 

        self.N, self.M = size
        self.size = size

        self.conv = Conv()

        src, dst = indices.type(torch.long)

        # weight initialization 
        fan_in = pyg.utils.degree(dst, num_nodes=self.M)
        fan_out = pyg.utils.degree(src, num_nodes=self.N)
        n_in = fan_in[dst]      # number of input channels 
        n_out = fan_out[src]    # number of output channels 
        if init in ['xavier', 'glorot']:
            std = (2/(n_in + n_out))**0.5
        elif init in ['kaiming', 'he']:
            std = (2/n_in)**(0.5)
        elif init == 'lecun': 
            std = (1/n_in)**(0.5)
        elif init == 'normal': 
            std = torch.ones_like(values)
        else:
            raise ValueError('unrecognized weight initialization method, options: xavier, kaiming, lecun, normal')
        
        # scale normal distribution
        values = torch.randn(indices.size(1), dtype=dtype)
        values *= std**2 # N(mean, variance)

        self.values = torch.nn.Parameter(values) # torch optimizer require dense parameters 
        self.register_buffer('indices', indices.type(torch.long))
        if bias: self.bias = torch.nn.Parameter(torch.zeros((self.M, 1), dtype=dtype))

        # caching 
        self._B = 0 
        self._edge_index = None

    def forward(self, x, batched_indices=None, edge_subset=None): 
        '''
        batch dimension is handled in `torch_geometric` fashion, e.g., concatenated batch graphs via incremented node idx 

        Args: 
            x               input (B, N, 1)
            edge_mask       boolean mask size (B, E); edges to include in the forward pass 
        
        Returns:
          Tensor (B, M, 1)
        '''
        device = x.device
        B = x.size(0)

        edge_weight = torch.cat([self.values for _ in range(B)], dim=0).contiguous()

        if batched_indices is None: 
            batched_indices, edge_subset = batch_graphs(N=self.N, M=self.M, edge_index=self.indices, B=B, device=device, edge_mask=edge_subset)

        if edge_subset is not None: 
            edge_weight = edge_weight[edge_subset]

        if hasattr(self, 'bias'):
            bias_idx = torch.arange(self.M, device=device).repeat(B)
            bias = self.bias[bias_idx]
        else: 
            bias = None

        x = x.view(-1,1)

        x = self.conv(x, batched_indices, edge_weight, bias, size=(self.N*B, self.M*B))

        x = x.view(B, -1, 1)

        return x