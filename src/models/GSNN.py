import torch 
import numpy as np 
#from src.models.SparseLinear import SparseLinear
from src.models.SparseLinear2 import SparseLinear2 as SparseLinear
from src.models import utils
from src.models.GroupLayerNorm import GroupLayerNorm

def get_conv_indices(edge_index, channels, function_nodes, fix_hidden_channels): 
    E = edge_index.size(1)  
    w1_indices, node_hidden_channels = utils.get_W1_indices(edge_index, channels, function_nodes, scale_by_degree=not fix_hidden_channels)
    w2_indices = utils.get_W2_indices(function_nodes, node_hidden_channels)
    w3_indices = utils.get_W3_indices(edge_index, function_nodes, node_hidden_channels)
    w1_size = (E, np.sum(node_hidden_channels))
    w2_size = (np.sum(node_hidden_channels), np.sum(node_hidden_channels))
    w3_size = (np.sum(node_hidden_channels), E)

    channel_groups = [] 
    for node_id, c in enumerate(node_hidden_channels): 
        for i in range(c): 
            channel_groups.append(node_id)

    return w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups


class ResBlock(torch.nn.Module): 

    def __init__(self, edge_index, channels, function_nodes, fix_hidden_channels, bias, nonlin, residual, two_layers, dropout=0., norm='layer', init='xavier'): 
        super().__init__()
        assert norm in ['layer', 'none'], 'unrecognized `norm` type'
        
        w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups = get_conv_indices(edge_index, channels, function_nodes, fix_hidden_channels)
        
        self.two_layers = two_layers
        self.dropout = dropout
        self.residual = residual

        self.lin1 = SparseLinear(indices=w1_indices, size=w1_size, bias=bias, init=init)
        if norm == 'layer': self.norm1 = GroupLayerNorm(channel_groups)
        if two_layers: 
            self.lin2 = SparseLinear(indices=w2_indices, size=w2_size, bias=bias)
            if norm == 'layer':self.norm2 = GroupLayerNorm(channel_groups)
        self.lin3 = SparseLinear(indices=w3_indices, size=w3_size, bias=bias, init=init)

        self.nonlin = nonlin()

    def forward(self, x): 

        out = self.lin1(x)      
        if hasattr(self, 'norm1'): out = self.norm1(out)
        out = self.nonlin(out)  
        out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)

        if self.two_layers: 
            out = self.lin2(out) 
            if hasattr(self, 'norm2'): out = self.norm2(out)
            out = self.nonlin(out)  
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)

        out = self.lin3(out) 
        if self.residual: out = out + x 

        return out
    
        

class GSNN(torch.nn.Module): 

    def __init__(self, edge_index, channels, input_node_mask, output_node_mask, layers, residual=True, dropout=0., 
                            nonlin=torch.nn.ELU, bias=True, share_layers=False, fix_hidden_channels=False, two_layer_conv=False, 
                                add_function_self_edges=False, norm='layer', init='xavier'):
        super().__init__()

        self.share_layers = share_layers            # whether to share function node parameters across layers
        self.register_buffer('output_node_mask', output_node_mask)
        self.input_node_mask = input_node_mask
        self.layers = layers 
        self.residual = residual
        self.channels = channels
        self.add_function_self_edges = add_function_self_edges
        self.fix_hidden_channels = fix_hidden_channels 
        self.two_layer_conv = two_layer_conv

        function_nodes = (~(input_node_mask | output_node_mask)).nonzero(as_tuple=True)[0]
        if add_function_self_edges: 
            print('Augmenting `edge index` with function node self-edges.')
            edge_index = torch.cat((edge_index, torch.stack((function_nodes, function_nodes), dim=0)), dim=1)
        self.register_buffer('edge_index', edge_index)
        self.E = self.edge_index.size(1)                             # number of edges 
        self.N = torch.unique(self.edge_index.view(-1)).size(0)      # number of nodes

        self.register_buffer('function_edge_mask', torch.isin(edge_index[0], function_nodes)) # edges from a function node / e.g., not an input or output edge 
        self.register_buffer('input_edge_mask', self.input_node_mask[self.edge_index[0]].type(torch.float32))

        self.dropout = dropout

        _n = 1 if self.share_layers else self.layers
        self.ResBlocks = torch.nn.ModuleList([ResBlock(self.edge_index, channels, function_nodes, fix_hidden_channels, 
                                                       bias, nonlin, residual=residual, two_layers=two_layer_conv, 
                                                       dropout=dropout, norm=norm, init=init) for i in range(_n)])
        


    def forward(self, x, mask=None):
        '''
        Assumes x is `node` indexed 
        ''' 
        x = utils.node2edge(x, self.edge_index)  # convert x to edge-indexed
        x0 = x
        for l in range(self.layers): 
            x = self.ResBlocks[0 if self.share_layers else l](x)
            if not self.residual: x += x0
            if mask is not None: x = mask * x

        if self.residual: x /= self.layers

        return utils.edge2node(x, self.edge_index, self.output_node_mask)  # convert x from edge-indexed to node-indexed

