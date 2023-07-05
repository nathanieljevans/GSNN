import torch 
from src.models.SparseLinear import SparseLinear
from src.models import utils

class GSNN(torch.nn.Module): 

    def __init__(self, edge_index, channels, input_node_mask, output_node_mask, layers, nonlin=torch.nn.ELU): 
        super().__init__()

        self.output_node_mask = output_node_mask 
        self.input_node_mask = input_node_mask
        self.layers = layers 
        self.edge_index = edge_index
        function_nodes = (~(input_node_mask | output_node_mask)).nonzero(as_tuple=True)[0]

        E = edge_index.size(1)
        N = torch.unique(edge_index.view(-1)).size(0)

        self.lin1 = SparseLinear(indices=utils.get_W1_indices(edge_index, channels), size=(E,channels*N), d=channels)
        self.lin2 = SparseLinear(indices=utils.get_W2_indices(function_nodes, channels), size=(channels*N, channels*N), d=channels)
        self.lin3 = SparseLinear(indices=utils.get_W3_indices(edge_index, function_nodes, channels), size=(channels*N, E), d=1)
        self.norm1 = torch.nn.BatchNorm1d(channels*N)
        self.norm2 = torch.nn.BatchNorm1d(channels*N)

        self.nonlin = nonlin() 

    def forward(self, x):
        '''
        Assumes x is `node` indexed 
        ''' 

        # convert x indexing from node to edge indices
        x0 = utils.node2edge(x, self.edge_index) 
        x = x0

        for l in range(self.layers): 

            x = self.lin1(x)
            x = self.norm1(x)
            x = self.nonlin(x)
            x = self.lin2(x)
            x = self.norm2(x)
            x = self.nonlin(x)
            x = self.lin3(x) 
            x = x + x0

        x = utils.edge2node(x, self.edge_index, self.output_node_mask)
        return x 

