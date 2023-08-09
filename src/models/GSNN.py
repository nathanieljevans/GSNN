import torch 
#from src.models.SparseLinear import SparseLinear
from src.models.SparseLinear2 import SparseLinear2 as SparseLinear
from src.models import utils

class GSNN(torch.nn.Module): 

    def __init__(self, edge_index, channels, input_node_mask, output_node_mask, layers, residual=True, dropout=0., nonlin=torch.nn.ELU): 
        super().__init__()

        self.output_node_mask = output_node_mask 
        self.input_node_mask = input_node_mask
        self.layers = layers 
        self.edge_index = edge_index
        function_nodes = (~(input_node_mask | output_node_mask)).nonzero(as_tuple=True)[0]
        self.residual = residual
        self.channels = channels

        E = edge_index.size(1)
        N = torch.unique(edge_index.view(-1)).size(0)

        self.dropout = torch.nn.Dropout(dropout)
        self.lin1 = SparseLinear(indices=utils.get_W1_indices(edge_index, channels), size=(E,channels*N), d=channels)
        self.lin2 = SparseLinear(indices=utils.get_W2_indices(function_nodes, channels), size=(channels*N, channels*N), d=channels)
        self.lin3 = SparseLinear(indices=utils.get_W3_indices(edge_index, function_nodes, channels), size=(channels*N, E), d=1)
        self.norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(1) for i in range(self.layers)])

        self.nonlin = nonlin() 

    def edge_update(self, x, x0, x_last, layer): 

        # edge update 
        x = self.lin1(x)
        x = self.nonlin(x)
        x = self.lin2(x)
        x = self.nonlin(x)
        x = self.lin3(x) 

        # NOTE: we don't want to scale the input edges (e.g., drug concs),
        # therefore we have to do this before adding residual layers
        # the drug concs will be primarily zero, and therefore will scale them abnormally large. 
        x = self.norms[layer](x.view(-1,1)).view(x.size())
        x = self.dropout(x)

        if self.residual: 
            x = x + x_last 
            x_last = x
        else: 
            x = x + x0  
        
        return x, x_last

    def forward(self, x, return_time_series=False):
        '''
        Assumes x is `node` indexed 
        ''' 
        #alpha = torch.sigmoid(self.alpha)
        # convert x from node-indexed to edge-indexed
        if return_time_series: out = [x]
        x0 = utils.node2edge(x, self.edge_index)
        x=x0
        if self.residual: x_last = x
        for l in range(self.layers): 
            x, x_last = self.edge_update(x, x0, x_last, l)
            if return_time_series: out.append(x)

        # scale by num layers 
        x /= self.layers

        # convert x from edge-indexed to node-indexed
        if return_time_series: 
            return torch.stack([utils.edge2node(x, self.edge_index, self.output_node_mask) for x in out], dim=0)
        else: 
            return utils.edge2node(x, self.edge_index, self.output_node_mask)

