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

        E = edge_index.size(1)
        N = torch.unique(edge_index.view(-1)).size(0)

        self.dropout = torch.nn.Dropout(dropout)
        self.lin1 = SparseLinear(indices=utils.get_W1_indices(edge_index, channels), size=(E,channels*N), d=channels)
        self.lin2 = SparseLinear(indices=utils.get_W2_indices(function_nodes, channels), size=(channels*N, channels*N), d=channels)
        self.lin3 = SparseLinear(indices=utils.get_W3_indices(edge_index, function_nodes, channels), size=(channels*N, E), d=1)
        self.norm1 = torch.nn.BatchNorm1d(channels*N)
        self.norm2 = torch.nn.BatchNorm1d(channels*N)

        self.nonlin = nonlin() 

        self.alpha = torch.nn.Parameter(torch.tensor([[[0.]]]))

    def forward(self, x):
        '''
        Assumes x is `node` indexed 
        ''' 
        alpha = torch.sigmoid(self.alpha)
        # convert x from node-indexed to edge-indexed
        x0 = utils.node2edge(x, self.edge_index)
        x=x0
        if self.residual: x_last = x
        for l in range(self.layers): 
            x = self.lin1(x)
            x = self.norm1(x)
            x = self.nonlin(x)
            #x = self.dropout(x)
            x = self.lin2(x)
            x = self.norm2(x)
            x = self.nonlin(x)
            #x = self.dropout(x)
            x = self.lin3(x) 
            x = self.dropout(x) # dropout on edges ? or dropout on node channels?
            x = x + x0 
            if self.residual: 
                x = (1-alpha)*x + alpha*x_last
                x_last = x

        # convert x from edge-indexed to node-indexed
        return utils.edge2node(x, self.edge_index, self.output_node_mask)

