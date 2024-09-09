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

        # weight initialization 
        fan_in = pyg.utils.degree(dst, num_nodes=self.M)
        fan_out = pyg.utils.degree(src, num_nodes=self.N)
        n_in = fan_in[dst]      # number of input units 
        n_out = fan_out[src]    # number of output units 
        if init in ['xavier', 'glorot']:  # glorot
            std = (2/(n_in + n_out))**0.5
        elif init in ['kaiming', 'he']: # he
            std = (2/n_in)**(0.5)
        elif init == 'lecun': 
            std = (1/n_in)**(0.5)
        elif init == 'normal': 
            std = torch.ones_like(values)
        else:
            raise ValueError('unrecognized weight initialization method, options: xavier, kaiming, normal')
        
        # scale normal distribution
        values = torch.randn(indices.size(1), dtype=dtype)
        values *= std**2 # N(mean, variance)

        self.values = torch.nn.Parameter(values) # torch optimizer require dense parameters 
        self.register_buffer('indices', indices.type(torch.long))
        if bias: self.bias = torch.nn.Parameter(torch.zeros((self.M, 1), dtype=dtype))

        # caching 
        self._B = 0 
        self._edge_index = None



    def forward(self, x): 
        '''
        Assumes x is in shape: (B, N, 1), where B is batch dimension
        weight shape is (N, M)
        Returns shape (B, M, 1)

        batch dimension is handled in `torch_geometric` fashion, e.g., concatenated batch graphs via incremented node idx 
        '''
        device = x.device
        B = x.size(0)

        # TODO: implement drug specific forward pass 
        #if subedges_list is not None: 
        #
        #   edge_index_list = [self.indices for subedges in subedges_list]
        #    edge_weight_list = [self.values for _ in range(B)]
        #else: 
        
        edge_weight = torch.cat([self.values for _ in range(B)], dim=0)

        if (self._B == B) and (self._edge_index is not None): 
            edge_index = self._edge_index 
        else: 
            edge_index = batch_graphs(N=self.N, M=self.M, edge_index=self.indices, B=B, device=device)
            self._edge_index = edge_index

        if hasattr(self, 'bias'):
            bias_idx = torch.arange(self.M, device=device).repeat(B)
            bias = self.bias[bias_idx]
        else: 
            bias = None

        x = x.view(-1,1)

        x = self.conv(x, edge_index, edge_weight, bias, size=(self.N*B, self.M*B))

        x = x.view(B, -1, 1)

        return x
    


def batch_graphs(N, M, edge_index, B, device):
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
    batched_edge_indices = edge_index.repeat(1, B)

    batch_idx = torch.repeat_interleave(torch.arange(B, dtype=torch.long, device=device), E)
    src_incr = batch_idx*N
    dst_incr = batch_idx*M
    incr = torch.stack((src_incr, dst_incr), dim=0)
    batched_edge_indices += incr

    return batched_edge_indices

    '''

    N_offset = 0
    M_offset = 0
    for edge_index in edge_index_list:

        incr = torch.tensor([[N_offset], 
                             [M_offset]], device=edge_index.device)
        # Increment the node indices for the two sets
        edge_index = edge_index + incr 

        batched_edge_indices.append(edge_index)

        # Update the offsets
        N_offset += N
        M_offset += M

    # Concatenate all edge indices and weights to form the batched graph
    batched_edge_index = torch.cat(batched_edge_indices, dim=1)
    
    return batched_edge_index
    '''




'''   
# old batching method 
def forward(self, x): 
        \'''
        Assumes x is in shape: (B, N, 1), where B is batch dimension
        weight shape is (N, M)
        Returns shape (B, M, 1)

        batch dimension is handled in `torch_geometric` fashion, e.g., concatenated batch graphs via incremented node idx 
        \'''
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
'''