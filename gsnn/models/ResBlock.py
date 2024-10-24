
from gsnn.models.SparseLinear import SparseLinear
from gsnn.models import utils
import numpy as np
import torch
from gsnn.models.GroupLayerNorm import GroupLayerNorm
import networkx as nx

class ResBlock(torch.nn.Module): 

    def __init__(self, bias, nonlin, residual, two_layers, indices_params, fix_inputs=True, dropout=0., norm='layer', init='xavier', dropout_type='node'): 
        super().__init__()
        assert norm in ['layer', 'batch', 'none'], 'unrecognized `norm` type'
        
        w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups = indices_params
        
        self.norm = norm
        self.two_layers = two_layers
        self.dropout = dropout
        self.residual = residual
        self.fix_inputs = fix_inputs
        self.dropout_type = dropout_type
        self.register_buffer('channel_groups', torch.tensor(channel_groups, dtype=torch.long))

        if norm == 'layer': 
            _norm = lambda: GroupLayerNorm(channel_groups)
        elif norm == 'batch': 
            _norm = lambda: torch.nn.BatchNorm1d(len(channel_groups), eps=1e-2)
        else: 
            _norm = lambda: torch.nn.Identity()

        self.lin1 = SparseLinear(indices=w1_indices, size=w1_size, bias=bias, init=init)
        self.norm1 = _norm()
        if two_layers: 
            self.lin2 = SparseLinear(indices=w2_indices, size=w2_size, bias=bias)
            self.norm2 = _norm()
        self.lin3 = SparseLinear(indices=w3_indices, size=w3_size, bias=bias, init=init)

        self.nonlin = nonlin()
        self.mask = None 
        self.x0 = None 

    def set_x0(self, x0):
        self.x0 = x0

    def set_node_mask(self, mask): 
        self.node_mask = mask

    def set_dropout(self, mask): 
        self.dropout_mask_ = mask

    def forward(self, x, batch_params): 
        '''
        Args: 
            x       torch.tensor        (B, C)
            subset  list                (B) or None         a list of graph subsets that should be applied to each obs in forward pass. 
        '''
        batched_edge_indices1, batched_edge_indices2, batched_edge_indices3 = batch_params

        out = self.lin1(x, batched_indices=batched_edge_indices1)      
        out = self.norm1(out)
        out = self.nonlin(out)  

        # node, channel dropout 
        if (self.dropout_type == 'node') and (self.training) and (self.dropout > 0): out = torch.nn.functional.dropout(out, p=self.dropout)
        if (self.dropout_type == 'node') and (self.training) and (self.dropout > 0): self.dropout_mask_*out

        if self.two_layers: 
            out = self.lin2(out, batched_indices=batched_edge_indices2) 
            out = self.norm2(out)
            out = self.nonlin(out)  
            if (self.dropout_type == 'node') and (self.training) and (self.dropout > 0): out = torch.nn.functional.dropout(out, p=self.dropout)
            if (self.dropout_type == 'node') and (self.training) and (self.dropout > 0): self.dropout_mask_*out

        if self.node_mask is not None:
            out = out.squeeze(-1) * self.node_mask.squeeze(-1)

        out = self.lin3(out, batched_indices=batched_edge_indices3) 
        if (self.dropout_type == 'edge') and (self.training) and (self.dropout > 0): out = self.dropout_mask_*out

        if self.residual: 
            out = out.squeeze(-1) + x 
        else: 
            if self.fix_inputs: 
                out = out.squeeze(-1) + self.x0
            else: 
                out = out.squeeze(-1)
        
        return out