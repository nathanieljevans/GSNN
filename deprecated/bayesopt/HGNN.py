import torch 
import torch 
from gsnn.models import NN
from torch_geometric.nn.models import GAT, GCN
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, LayerNorm
from gsnn.bayesopt.MultiHeadAttentionPooling import MultiHeadAttentionPooling

class HGNN(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels=32, num_layers=5, eps=1e-2, n_heads=10):
        super().__init__()

        self.eps = eps

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

        self.norm = LayerNorm(hidden_channels)
        self.nonlin = torch.nn.GELU()
        self.pool = MultiHeadAttentionPooling(num_nodes=num_nodes, n_heads=n_heads)
        self.lin = Linear(hidden_channels*n_heads, 1)

    def forward(self, data, mask):

        mask = mask.view(-1,1)
        x_dict = data.x_dict

        x_dict['function'] = data.x_dict['function']*mask
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict['function'] = x_dict['function']*mask
            x_dict = {key: self.norm(self.nonlin(x)) for key, x in x_dict.items()}
        x_dict['function'] = x_dict['function']*mask

        #x, edge_index_pooled, batch_pooled, _, _, _ = self.pool(x_dict['function'], data.edge_index_dict['function', 'to', 'function'], batch=data['function'].batch) 

        #x = self.nonlin(global_mean_pool(x=x, batch=batch_pooled))
        x = self.nonlin(self.pool(x=x_dict['function'], batch=data['function'].batch)) #global_mean_pool(x=x_dict['function'], batch=data['function'].batch))

        out = self.lin(x)

        return out