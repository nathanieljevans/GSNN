'''
Batched sparse matrix multiplication that scales with GPU's better. 
'''

import torch
import torch_geometric as pyg 


class Conv(pyg.nn.MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight, bias, size):
        
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size).view(-1, 1)
        
        if bias is not None: out = out + bias.view(-1, 1)
        
        return out 

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j.view(-1, 1)


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
    batched_edge_indices = edge_index.repeat(1, B).contiguous()
    batch_idx = torch.repeat_interleave(torch.arange(B, dtype=torch.long, device=device), E).contiguous()

    src_incr = batch_idx*N
    dst_incr = batch_idx*M
    incr = torch.stack((src_incr, dst_incr), dim=0)
    batched_edge_indices += incr

    return batched_edge_indices

def xavier_uniform(size, fan_in, fan_out, gain=1, dtype=torch.float32): 
    a = gain * torch.sqrt((6/(fan_in + fan_out)))
    out = torch.empty(size, dtype=dtype)
    out = torch.nn.init.uniform_(out, a=-1, b=1) 
    out = out * a 
    return out 

def xavier_normal(size, fan_in, fan_out, gain=1, dtype=torch.float32): 
    a = gain * torch.sqrt((2/(fan_in + fan_out)))
    out = torch.empty(size, dtype=dtype)
    out = torch.nn.init.normal_(out, mean=0, std=1) 
    out = out * a 
    return out 

def uniform(size, gain=1., dtype=torch.float32): 
    a = gain 
    out = torch.empty(size, dtype=dtype)
    out = torch.nn.init.uniform_(out, a=-1, b=1) 
    out = out * a 
    return out 

def normal(size, gain=1, dtype=torch.float32): 
    a = gain
    out = torch.empty(size, dtype=dtype)
    out = torch.nn.init.normal_(out, mean=0, std=1) 
    out = out * a 
    return out 

def kaiming_uniform(size, fan_in, fan_out, fan_mode='fan_in', gain=1, dtype=torch.float32): 
    fan_val = fan_in if fan_mode == 'fan_in' else fan_out
    a = gain * torch.sqrt( 3 / fan_val )
    out = torch.empty(size, dtype=dtype)
    out = torch.nn.init.uniform_(out, a=-1, b=1) 
    out = out * a 
    return out  

def kaiming_normal(size, fan_in, fan_out, fan_mode='fan_in', gain=1, dtype=torch.float32): 
    fan_val = fan_in if fan_mode == 'fan_in' else fan_out
    a = gain / torch.sqrt( fan_val )
    out = torch.empty(size, dtype=dtype)
    out = torch.nn.init.normal_(out, mean=0, std=1) 
    out = out * a 
    return out  


class SparseLinear(torch.nn.Module): 
    def __init__(self, indices, size, dtype=torch.float32, bias=True, init='kaiming', init_gain=1.):
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

        if init == 'xavier_uniform': 
            values = xavier_uniform(indices.size(1), n_in, n_out, gain=init_gain, dtype=dtype)
        elif init == 'xavier_normal': 
            values = xavier_normal(indices.size(1), n_in, n_out, gain=init_gain, dtype=dtype)
        elif init == 'kaiming_uniform':
            values = kaiming_uniform(indices.size(1), n_in, n_out, gain=init_gain, dtype=dtype)
        elif init == 'kaiming_normal': 
            values = kaiming_normal(indices.size(1), n_in, n_out, gain=init_gain, dtype=dtype)
        elif init == 'uniform': 
            values = uniform(indices.size(1), gain=init_gain, dtype=dtype)
        elif init == 'normal': 
            values = normal(indices.size(1), gain=init_gain, dtype=dtype)
        else:
            raise ValueError('unrecognized weight initialization method, options: ')
        
        self.values = torch.nn.Parameter(values) # torch optimizer require dense parameters 
        self.register_buffer('indices', indices.type(torch.long))
        
        if bias: self.bias = torch.nn.Parameter(torch.zeros((self.M, 1), dtype=dtype))

        # caching 
        self._B = 0 
        self._edge_index = None

    def forward(self, x, batched_indices=None): 
        '''
        batch dimension is handled in `torch_geometric` fashion, e.g., concatenated batch graphs via incremented node idx 

        Args: 
            x               input (B, N, 1)
            batched_indices 
        
        Returns:
          Tensor (B, M, 1)
        '''
        device = x.device
        B = x.size(0)

        edge_weight = self.values.expand(B, *self.values.shape).reshape(-1)

        if batched_indices is None: 
            batched_indices = batch_graphs(N=self.N, M=self.M, edge_index=self.indices, B=B, device=device)

        if hasattr(self, 'bias'):
            bias_idx = torch.arange(self.M, device=device).repeat(B)
            bias = self.bias[bias_idx]
        else: 
            bias = None

        x = x.view(-1,1)

        x = self.conv(x, batched_indices, edge_weight, bias, size=(self.N*B, self.M*B))

        x = x.view(B, -1, 1)

        return x

    def prune(self, idxs):
        """
        """
        self.values = torch.nn.Parameter(self.values[idxs])
        self.register_buffer('indices', self.indices[:, idxs])

        
            
