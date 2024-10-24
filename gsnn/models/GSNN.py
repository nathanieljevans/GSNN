import torch
import torch.nn.functional 
from gsnn.models import utils
from torch.utils.checkpoint import checkpoint #_sequential
from torch.autograd import Variable
from gsnn.models.ResBlock import ResBlock
from gsnn.models.SparseLinear import batch_graphs


class GSNN(torch.nn.Module): 

    def __init__(self, edge_index_dict, node_names_dict, channels, layers, residual=True, dropout=0., 
                            nonlin=torch.nn.GELU, bias=True, share_layers=True, fix_hidden_channels=True, two_layer_conv=False, 
                                add_function_self_edges=False, norm='layer', init='kaiming', verbose=False, edge_channels=1, checkpoint=False,
                                  fix_inputs=True, dropout_type='channel', latent_decay=1.):
        super().__init__()

        # Optional: add multiple latent edge features per edge
        # NOTE: this will significantly scale the total number of channels
        if edge_channels > 1:
            edge_index_dict['function', 'to', 'function'] = edge_index_dict['function', 'to', 'function'].repeat(1, edge_channels)

        edge_index, input_node_mask, output_node_mask, self.num_nodes = utils.hetero2homo(edge_index_dict, node_names_dict)
        self.share_layers = share_layers
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
        self.dropout_type = dropout_type
        self.norm = norm
        self.latent_decay = latent_decay

        function_nodes = (~(input_node_mask | output_node_mask)).nonzero(as_tuple=True)[0]
        if add_function_self_edges: 
            if verbose: print('Augmenting `edge index` with function node self-edges.')
            edge_index = torch.cat((edge_index, torch.stack((function_nodes, function_nodes), dim=0)), dim=1)
        self.register_buffer('edge_index', edge_index)
        self.E = self.edge_index.size(1)                             # number of edges 
        self.N = len(node_names_dict['input']) + len(node_names_dict['function']) + len(node_names_dict['output'])      # number of nodes

        func_edge_mask = torch.isin(edge_index[0], function_nodes) & torch.isin(edge_index[1], function_nodes) # edges from function -> function / e.g., not an input or output edge 
        self.register_buffer('function_edge_mask', func_edge_mask) 
        self.register_buffer('input_edge_mask', self.input_node_mask[self.edge_index[0]].type(torch.float32))

        self.dropout = dropout

        self.indices_params = utils.get_conv_indices(edge_index, channels, function_nodes, fix_hidden_channels)

        _n = 1 if self.share_layers else self.layers
        self.ResBlocks = torch.nn.ModuleList([ResBlock(bias, nonlin, residual=residual, two_layers=two_layer_conv, 
                                                       dropout=dropout, norm=norm, init=init, fix_inputs=fix_inputs, 
                                                       dropout_type=dropout_type, indices_params=self.indices_params) for i in range(_n)])
        
        self._B = None; self._batch_params = None

    def get_batch_params(self, B, device): 

        if (self._B == B) and (self._batch_params is not None): 
            # caching batch params
            return self._batch_params
        else: 
            self._B = B
            # precompute edge batching so it doesn't have to be done in every resblock 
            batched_edge_indices1 = batch_graphs(N=self.ResBlocks[0].lin1.N,
                                                            M=self.ResBlocks[0].lin1.M, 
                                                            edge_index = self.ResBlocks[0].lin1.indices, 
                                                            B=B, 
                                                            device=device)
            
            if self.two_layer_conv: 
                batched_edge_indices2 = batch_graphs(N=self.ResBlocks[0].lin2.N,
                                                                M=self.ResBlocks[0].lin2.M, 
                                                                edge_index = self.ResBlocks[0].lin2.indices, 
                                                                B=B, 
                                                                device=device)
            else: 
                batched_edge_indices2=None
            
            batched_edge_indices3 = batch_graphs(N=self.ResBlocks[0].lin3.N,
                                                            M=self.ResBlocks[0].lin3.M, 
                                                            edge_index = self.ResBlocks[0].lin3.indices, 
                                                            B=B, 
                                                            device=device)
            
            self._batch_params = (batched_edge_indices1, batched_edge_indices2, batched_edge_indices3)
        
        return self._batch_params

    def forward(self, x, node_mask=None, edge_mask=None, ret_edge_out=False):
        '''
        Assumes x is the values of the "input" nodes ONLY
        ''' 
        B = x.size(0)
        x_node = torch.zeros((B, self.num_nodes), device=x.device, dtype=torch.float32)
        #x_node[:, self.input_node_mask] = x
        idx = self.input_node_mask.nonzero(as_tuple=True)[0].unsqueeze(0).expand(B, -1)  # Shape: (B, num_input_nodes)
        x_node = x_node.scatter_add(1, idx, x)

        x = utils.node2edge(x_node, self.edge_index)  # convert x to edge-indexed
        
        if not self.residual: x0 = x.clone()#.to_sparse()

        if self.share_layers: 
            modules = [self.ResBlocks[0] for i in range(self.layers)]
        else: 
            modules = [blk for blk in self.ResBlocks]

        #for mod in modules: mod.set_edge_mask(edge_mask)

        # faster if we do this up front 
        if node_mask is not None: node_mask = torch.stack([torch.isin(modules[0].channel_groups, node_mask[i].nonzero(as_tuple=True)[0]) for i in range(node_mask.size(0))], dim=0)
        for mod in modules: mod.set_node_mask(node_mask)
        
        if not self.residual: 
            for mod in modules: mod.set_x0(x0)
        if (self.dropout > 0) & (self.dropout_type in ['node', 'edge']) & self.training: 
            # NOTE: edge and channel dropout increases batch runtime 2-3X 
            N = max(self.ResBlocks[0].channel_groups) + 1; B = x.size(0)
            if self.dropout_type == 'node': 
                do_mask = (torch.rand((B,N,1), device=x.device, dtype=torch.float32) > self.dropout)[:, self.ResBlocks[0].channel_groups, :]
            elif self.dropout_type == 'edge': 
                do_mask = torch.nn.functional.dropout(torch.ones((B,self.E,1), device=x.device, dtype=torch.float32), p=self.dropout)
            for mod in modules: 
                mod.set_dropout(do_mask)

        batch_params = self.get_batch_params(B, x.device)
        
        x = Variable(x, requires_grad=True)
        for mod in modules: 
            if self.checkpoint and self.training: 
                x = checkpoint(mod, x, batch_params)
            else: 
                x = mod(x, batch_params)
            x = x*self.latent_decay
            if edge_mask is not None: x = x*edge_mask

        if self.residual: 
            # under assumption that each layer output is iid unit normal (weak assumption since layer outputs will be correlated)
            # then x = N(0,1) + N(0,1) + ... + N(0,1) = N(0, sqrt(layers))
            x /= self.layers**(0.5)

        if ret_edge_out: return x

        out = utils.edge2node(x, self.edge_index, self.output_node_mask)

        # NOTE: returns only the "output" nodes 
        return out[:, self.output_node_mask]