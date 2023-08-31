import torch 
#from src.models.SparseLinear import SparseLinear
from src.models.SparseLinear2 import SparseLinear2 as SparseLinear
from src.models import utils

class GSNN(torch.nn.Module): 

    def __init__(self, edge_index, channels, input_node_mask, output_node_mask, layers, residual=True, dropout=0., 
                            nonlin=torch.nn.ELU, dropout_type='layerwise', norm='batch', bias=True, stochastic_depth=True): 
        super().__init__()

        self.stochastic_depth = stochastic_depth  # https://arxiv.org/abs/1603.09382
        self.dropout_type = dropout_type
        self.register_buffer('output_node_mask', output_node_mask)
        self.input_node_mask = input_node_mask
        self.layers = layers 
        self.register_buffer('edge_index', edge_index)
        function_nodes = (~(input_node_mask | output_node_mask)).nonzero(as_tuple=True)[0]
        self.residual = residual
        self.channels = channels

        self.E = edge_index.size(1)                             # number of edges 
        self.N = torch.unique(edge_index.view(-1)).size(0)      # number of nodes

        self.dropout = dropout
        self.lin1 = SparseLinear(indices=utils.get_W1_indices(edge_index, channels), size=(self.E,channels*self.N), d=channels, bias=bias)
        self.lin2 = SparseLinear(indices=utils.get_W2_indices(function_nodes, channels), size=(channels*self.N, channels*self.N), d=channels, bias=bias)
        self.lin3 = SparseLinear(indices=utils.get_W3_indices(edge_index, function_nodes, channels), size=(channels*self.N, self.E), d=1, bias=bias)
        
        # NOTE: only include affine on last layer to let predicted outputs scale appropriately. Note, affine is element wise for all. 
        self.norm_type = norm
        if norm == 'edge-batch': 
            self.norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.E, affine=False) for i in range(self.layers)])
        elif norm == 'layer-batch': 
            self.norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(1, affine=False) for i in range(self.layers)])
        elif norm =='layer': 
            self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(self.E, elementwise_affine=False) for i in range(self.layers)])
        elif norm =='group': 
            num_groups = utils.next_divisor(self.E, 100)
            print('group norm, # groups:', num_groups)
            self.norms = torch.nn.ModuleList([torch.nn.GroupNorm(num_channels=self.E, num_groups=num_groups, affine=False) for i in range(self.layers)])
        else: 
            raise ValueError('unrecognized `norm` type.')
        
        self.scale_out = torch.nn.Parameter(torch.ones(1,self.E, 1))
        self.bias_out = torch.nn.Parameter(torch.zeros(1,self.E, 1))
        
        self.nonlin = nonlin() 

        ## setup 
        self.register_buffer('input_edge_mask', self.input_node_mask[self.edge_index[0]].type(torch.float32))

    def edge_update(self, x, x0, layer): 

        x_last = x
        
        # batch ormalize all edges EXCEPT input edges, which are fixed
        # x (B, E, 1)
        # mask (E)
        #x = (1-self.input_edge_mask.view(1, self.E, 1))*self.norms[layer](x.squeeze(-1)).unsqueeze(-1) + self.input_edge_mask.view(1, self.E, 1)*x

        # edge update 
        x = self.lin1(x)        
        x = self.nonlin(x)      # latent node activations | (B, num_nodes*channels)
        x = self.lin2(x)
        x = self.nonlin(x)      # latent node activations | (B, num_nodes*channels)
        x = self.lin3(x)        # edge activations        | (B, E)

        if not hasattr(self, 'dropout_type'): 
            pass 
        elif self.dropout_type == 'layerwise': 
            # unique set of edges are dropped every layer. 
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        elif self.dropout_type == 'edgewise': 
            # the same set of edges are dropped every layer. 
            if self._edge_mask is None: self._edge_mask = (torch.rand(x.size(), device=x.device) > self.dropout).type(torch.float32)
            x *= self._edge_mask
            p = 1 - torch.mean(self._edge_mask).detach()
            if self.training: x *= 1/(1-p) # following pytorch convention (https://arxiv.org/abs/1207.0580)
        elif self.dropout_type == 'nodewise': 
            # all edges from a set of nodes will be dropped every layer. 
            if self._edge_mask is None: 
                drop_nodes = (torch.rand((self.N,), device=x.device) < self.dropout).nonzero(as_tuple=True)[0].view(-1)              # get node indices to drop 
                src, dst = self.edge_index 
                self._edge_mask = (~torch.isin(src, drop_nodes)).type(torch.float32).view(1, -1, 1) # drop edges from all dropped node
            x *= self._edge_mask
            p = 1 - torch.mean(self._edge_mask).detach()
            if self.training: x *= 1/(1-p) # following pytorch convention (https://arxiv.org/abs/1207.0580)
        else: 
            raise ValueError('unrecognized `dropout type`')
        
        if self.residual: x = x + x_last # residual connection 

        # NORMALIZE all edges EXCEPT input edges, which are fixed
        if hasattr(self, 'norm_type') and (self.norm_type == 'layer-batch'): 
            xnorm = self.norms[layer](x.view(-1, 1)).view(x.size())
        else: 
            xnorm = self.norms[layer](x.squeeze(-1)).unsqueeze(-1) 

        x = (1-self.input_edge_mask.view(1, self.E, 1))*xnorm + self.input_edge_mask.view(1, self.E, 1)*x0
        
        return x

    def forward(self, x, return_activations=False, mask=None, affine=True):
        '''
        Assumes x is `node` indexed 
        ''' 
        if return_activations: out = []  # convert x from node-indexed to edge-indexed
        if hasattr(self, 'dropout_type') and (self.dropout_type in ['edgewise', 'nodewise']): self._edge_mask = None  # reset dropout edge mask 

        x0 = utils.node2edge(x, self.edge_index)  # convert x to edge-indexed
        x  = x0
        if self.training and self.stochastic_depth: 
            num_layers = torch.randint(int(self.layers/2), self.layers + 1, size=(1,)).item() 
        else: 
            num_layers = self.layers 
        for l in range(num_layers): 
            x = self.edge_update(x, x0, l)
            if mask is not None: x = mask * x
            if return_activations: out.append(x)

        if affine: x = self.scale_out*x + self.bias_out  # edge values are normalized, add affine layer to predict unscaled output

        if return_activations: 
            return torch.stack(out, dim=0)
        else: 
            return utils.edge2node(x, self.edge_index, self.output_node_mask)  # convert x from edge-indexed to node-indexed

