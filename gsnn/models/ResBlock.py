
from gsnn.models.SparseLinear import SparseLinear
from gsnn.models import utils
import numpy as np
import torch
from gsnn.models.GroupLayerNorm import GroupLayerNorm
import networkx as nx

class ResBlock(torch.nn.Module): 

    def __init__(self, edge_index, channels, function_nodes, fix_hidden_channels, bias, nonlin, 
                 residual, two_layers, indices_params, fix_inputs=True, dropout=0., norm='layer', init='xavier', dropout_type='node', 
                 subgraph_dict=None): 
        super().__init__()
        assert norm in ['layer', 'none'], 'unrecognized `norm` type'
        
        w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups = indices_params
        
        self.two_layers = two_layers
        self.dropout = dropout
        self.residual = residual
        self.fix_inputs = fix_inputs
        self.dropout_type = dropout_type
        self.channel_groups = torch.tensor(channel_groups, dtype=torch.long)

        self.lin1 = SparseLinear(indices=w1_indices, size=w1_size, bias=bias, init=init)
        if norm == 'layer': self.norm1 = GroupLayerNorm(channel_groups)
        if two_layers: 
            self.lin2 = SparseLinear(indices=w2_indices, size=w2_size, bias=bias)
            if norm == 'layer':self.norm2 = GroupLayerNorm(channel_groups)
        self.lin3 = SparseLinear(indices=w3_indices, size=w3_size, bias=bias, init=init)

        self.nonlin = nonlin()
        self.mask = None 
        self.x0 = None 

    def set_x0(self, x0):
        self.x0 = x0

    def set_mask(self, mask): 
        self.mask = mask

    def dropout_(self, x): 
        '''assumes x is nn latent channels not edge channels'''

        if (not self.training) or (self.dropout == 0): return x

        if self.dropout_type=='channel': 
            # dropout random channels independant of node
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        elif self.dropout_type=='node': 
            # dropout all channels from a given node
            N = max(self.channel_groups) + 1
            B = x.size(0)
            do = torch.tensor(np.random.choice([0,1], p=[self.dropout, 1-self.dropout], size=(B,N)), dtype=torch.float32, device=x.device)
            mask = do[:, self.channel_groups]
            x = mask.unsqueeze(-1)*x
        else: 
            raise Exception('unrecognized dropout type [channel, node]')
        
        return x 

    def forward(self, x, batch_params): 
        '''
        Args: 
            x       torch.tensor        (B, C)
            subset  list                (B) or None         a list of graph subsets that should be applied to each obs in forward pass. 
        '''
        batched_edge_indices1, batched_edge_indices2, batched_edge_indices3, edge_subset1, edge_subset2, edge_subset3 = batch_params

        out = self.lin1(x, batched_indices=batched_edge_indices1, edge_subset=edge_subset1)      
        if hasattr(self, 'norm1'): out = self.norm1(out)
        out = self.nonlin(out)  

        out = self.dropout_(out)

        if self.two_layers: 
            out = self.lin2(out, batched_indices=batched_edge_indices2, edge_subset=edge_subset2) 
            if hasattr(self, 'norm2'): out = self.norm2(out)
            out = self.nonlin(out)  
            out = self.dropout_(out)

        out = self.lin3(out, batched_indices=batched_edge_indices3, edge_subset=edge_subset3) 

        if self.residual: 
            out = out.squeeze(-1) + x
        else: 
            if self.fix_inputs: 
                out = out.squeeze(-1) + self.x0
            else: 
                out = out.squeeze(-1)
        
        if self.mask is not None: x = self.mask * x

        return out