import torch 
from src.models.Node import Node
from src.models import utils

class GSNN(torch.nn.Module): 
    '''Graph Structured Neural Network'''
    def __init__(self, edge_index, input_mask, output_mask, hidden_channels, layers, attr_dim=0, dropout=0., nonlin=torch.nn.ELU): 
        '''
        
        '''
        super().__init__()

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('input_mask', input_mask)
        self.register_buffer('output_mask', output_mask)
        self.layers = layers

        self.node_dict = self.build_nodes(edge_index=edge_index, 
                                          input_mask=input_mask, 
                                          output_mask=output_mask, 
                                          hidden_channels=hidden_channels, 
                                          nonlin=nonlin,
                                          attr_dim=attr_dim, 
                                          dropout=dropout)
        
    def _filter_to_function_nodes(self, function_evals): 
        return function_evals[torch.isin(function_evals, self.function_node_ixs)]
 
    def forward(self, x0, attr=None): 
        
        x = x0
        for l in range(self.layers): 
           
            x_new = torch.zeros_like(x0)
            for idx in self.function_node_ixs: 
                x_new += self.node_dict[str(idx.item())](x, attr=attr)

            x = x_new + x0

        return x


    def build_nodes(self, edge_index, input_mask, output_mask, hidden_channels, nonlin, dropout, attr_dim):
        
        # "function" nodes 
        # All nodes except the src in input edges and dst in output edges 
        src, dst = edge_index 
        self.register_buffer('function_node_ixs', torch.unique(torch.cat((src[~input_mask], dst[~output_mask]), dim=-1)))

        node_dict = {}
        for i in self.function_node_ixs: 
            in_degree, out_degree = utils.get_degree(edge_index, i)
            in_ixs, out_ixs = utils.get_in_out_ixs(edge_index, i)
            children = utils.get_children(edge_index, i)

            #if (in_degree == 0) or (out_degree == 0): continue

            node = Node(input_channels=in_degree, 
                        output_channels=out_degree, 
                        in_ixs = in_ixs, 
                        out_ixs = out_ixs,
                        idx = i,
                        children = children, 
                        attr_dim = attr_dim,
                        hidden_channels=hidden_channels,
                        nonlin = nonlin,
                        dropout = dropout)
            
            node_dict[str(i.item())] = node
        
        return torch.nn.ModuleDict(node_dict)