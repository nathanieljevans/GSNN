import torch 
import numpy as np 
#from src.models.SparseLinear import SparseLinear
from gsnn.models.SparseLinear import SparseLinear2 as SparseLinear
from gsnn.models import utils
from gsnn.models.GroupLayerNorm import GroupLayerNorm
import time 
from torch.utils.checkpoint import checkpoint_sequential
from torch.autograd import Variable

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

    def __init__(self, edge_index, channels, function_nodes, fix_hidden_channels, bias, nonlin, residual, two_layers, fix_inputs=True, dropout=0., norm='layer', init='xavier'): 
        super().__init__()
        assert norm in ['layer', 'none'], 'unrecognized `norm` type'
        
        w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups = get_conv_indices(edge_index, channels, function_nodes, fix_hidden_channels)
        
        self.two_layers = two_layers
        self.dropout = dropout
        self.residual = residual
        self.fix_inputs = fix_inputs

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

        if self.residual: 
            out = out.squeeze(-1) + x
        else: 
            if self.fix_inputs: 
                out = out.squeeze(-1) + self.x0
            else: 
                out = out.squeeze(-1)
        
        if self.mask is not None: x = self.mask * x

        return out
    
def hetero2homo(edge_index_dict, node_names_dict): 

    # convert edge_index_dict to edge_index (homogenous)
    input_edge_index = edge_index_dict['input', 'to', 'function'].clone()
    function_edge_index = edge_index_dict['function', 'to', 'function'].clone()
    output_edge_index = edge_index_dict['function', 'to', 'output'].clone()

    N_input = len(node_names_dict['input'])
    N_function = len(node_names_dict['function'])
    N_output = len(node_names_dict['output'])

    # add offsets to treat as unique nodes 
    input_edge_index[0, :] = input_edge_index[0,:] + N_function  # increment input nodes only 

    output_edge_index[1, :] = output_edge_index[1, :] + N_function + N_input # increment output nodes only 

    edge_index = torch.cat((function_edge_index, input_edge_index, output_edge_index), dim=1)
    
    input_node_mask = torch.zeros((N_input + N_function + N_output), dtype=torch.bool)
    input_nodes = torch.arange(N_input) + N_function
    input_node_mask[input_nodes] = True

    output_node_mask = torch.zeros((N_input + N_function + N_output), dtype=torch.bool)
    output_nodes = torch.arange(N_output) + N_function + N_input
    output_node_mask[output_nodes] = True

    num_nodes = N_input + N_function + N_output

    return edge_index, input_node_mask, output_node_mask, num_nodes 

class GSNN(torch.nn.Module): 

    def __init__(self, edge_index_dict, node_names_dict, channels, layers, residual=True, dropout=0., 
                            nonlin=torch.nn.GELU, bias=True, share_layers=True, fix_hidden_channels=True, two_layer_conv=False, 
                                add_function_self_edges=False, norm='layer', init='kaiming', verbose=False, edge_channels=1, checkpoint=False, fix_inputs=True):
        super().__init__()

        # add multiple latent edge features per edge
        if edge_channels > 1:
            edge_index_dict['function', 'to', 'function'] = edge_index_dict['function', 'to', 'function'].repeat(1, edge_channels)

        # convert edge_index_dict to edge_index (homogenous)
        edge_index, input_node_mask, output_node_mask, self.num_nodes = hetero2homo(edge_index_dict, node_names_dict)
        self.share_layers = share_layers            # whether to share function node parameters across layers
        self.register_buffer('output_node_mask', output_node_mask)
        self.register_buffer('input_node_mask', input_node_mask)
        self.layers = layers 
        self.residual = residual
        self.channels = channels
        self.add_function_self_edges = add_function_self_edges
        self.fix_hidden_channels = fix_hidden_channels 
        self.two_layer_conv = two_layer_conv
        self.verbose = verbose
        self.edge_channels = edge_channels
        self.checkpoint = checkpoint
        self.fix_inputs = fix_inputs

        function_nodes = (~(input_node_mask | output_node_mask)).nonzero(as_tuple=True)[0]
        if add_function_self_edges: 
            if verbose: print('Augmenting `edge index` with function node self-edges.')
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
                                                       dropout=dropout, norm=norm, init=init, fix_inputs=fix_inputs) for i in range(_n)])

    def forward(self, x, mask=None):
        '''
        Assumes x is the values of the "input" nodes ONLY
        ''' 
        B = x.size(0)
        x_node = torch.zeros((B, self.num_nodes), device=x.device, dtype=torch.float32)
        x_node[:, self.input_node_mask] = x

        x = utils.node2edge(x_node, self.edge_index)  # convert x to edge-indexed
        
        x0 = x.clone()
        if self.share_layers: 
            modules = [self.ResBlocks[0] for i in range(self.layers)]
        else: 
            modules = [self.ResBlocks[i] for i in range(self.layers)]

        if mask is not None: 
            for mod in modules: mod.set_mask(mask)
        if not self.residual: 
            for mod in modules: mod.set_x0(x0)
        
        if self.checkpoint and self.training: 
            x = Variable(x, requires_grad=True)
            x = checkpoint_sequential(functions=modules, segments=self.layers, input=x, use_reentrant=True)
        else:
            for mod in modules: x = mod(x)

        if self.residual: x /= self.layers

        out = utils.edge2node(x, self.edge_index, self.output_node_mask)

        # NOTE: returns only the "output" nodes 
        return out[:, self.output_node_mask]

