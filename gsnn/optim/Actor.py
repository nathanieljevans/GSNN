'''
'''

import torch 
from gsnn.models import NN
from gsnn.models.GSNN import GSNN
from torch_geometric.nn.models import GAT, GCN
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear



class GSNNActor(torch.nn.Module): 
    '''
    This enables the actor to learn correlation between actions. 
    '''
    def __init__(self, edge_index_dict, node_names_dict, channels=1, layers=10):
        super().__init__()
        self.model = GSNN(edge_index_dict, 
                          node_names_dict, 
                          channels=channels, 
                          layers=layers, 
                          norm='none', 
                          add_function_self_edges=False, 
                          share_layers=True, 
                          bias=False)
        
        self.register_buffer('x_inp', torch.ones((1, len(node_names_dict['input'])), dtype=torch.float32))


    def forward(self, x=None, edge_index=None): 
        out = self.model(self.x_inp, ret_edge_out=True)
        out = out[:, self.model.function_edge_mask]
        return out


class EmbeddedActor(torch.nn.Module): 
    '''
    This enables the actor to learn correlation between actions. 
    '''
    def __init__(self, num_actions, embed_dim=10):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.randn((1,embed_dim), dtype=torch.float32))
        self.lin = torch.nn.Linear(embed_dim, num_actions)

    def forward(self, x=None, edge_index=None): 
        return self.lin(self.embedding)
    

class Actor(torch.nn.Module): 
    def __init__(self, in_channels, bias=None, model='linear', hidden_channels=124): 
        '''
        N       num entities to select for 
        '''
        super().__init__()

        if model == 'linear':
            self.f = torch.nn.Linear(in_channels, 1, bias=False)

        elif model == 'nn': 
            self.f = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), 
                                         torch.nn.GELU(), 
                                         torch.nn.BatchNorm1d(hidden_channels),
                                         torch.nn.Linear(hidden_channels, hidden_channels), 
                                         torch.nn.GELU(), 
                                         torch.nn.BatchNorm1d(hidden_channels),
                                         torch.nn.Linear(hidden_channels,1))
            
        elif model == 'gcn': 
            self.f = GCN(in_channels         = in_channels, 
                         hidden_channels     = hidden_channels,
                         num_layers          = 3,  
                         out_channels        = 1, 
                         dropout             = 0, 
                         act                 = 'elu', 
                         norm                = 'layer', 
                         jk                  = 'last')
            
        elif model == 'gat': 
            self.f = GAT(in_channels         = in_channels, 
                         hidden_channels     = hidden_channels,
                         num_layers          = 3,  
                         out_channels        = 1, 
                         dropout             = 0, 
                         act                 = 'elu', 
                         norm                = 'layer', 
                         jk                  = 'last')
            
        elif model == 'hgnn':     
            self.f = HeteroGNN(hidden_channels, out_channels=1, num_layers=3)
        else: 
            raise Exception()

        if bias is not None: 
            self.bias = torch.nn.Parameter(torch.tensor([bias], dtype=torch.float32))
        else: 
            self.bias = 0

    def forward(self, x, edge_index=None): 

        if edge_index is not None: 
            return self.f(x, edge_index) + self.bias
        else: 
            return self.f(x) + self.bias



class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('input', 'to', 'function'): SAGEConv(-1, hidden_channels),
                ('function', 'to', 'input'): SAGEConv(-1, hidden_channels),
                ('function', 'to', 'function'): SAGEConv(-1, hidden_channels),
                ('output', 'to', 'function'): SAGEConv(-1, hidden_channels),
                ('function', 'to', 'output'): SAGEConv(-1, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['function'])