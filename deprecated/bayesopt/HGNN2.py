import torch 
import torch 
from gsnn.models import NN
from torch_geometric.nn.models import GAT, GCN
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, LayerNorm, global_add_pool

class HGNN2(torch.nn.Module):
    def __init__(self, conv_type='gat', hidden_channels=32, num_layers=5, n_heads=5, eps=1e-2, dropout=0.1):
        super().__init__()

        self.eps = eps

        self.convs = torch.nn.ModuleList()

        if conv_type == 'gat': 
            conv_ = GATConv
            kwargs = {'add_self_loops':False}
        elif conv_type == 'sage': 
            conv_ = SAGEConv 
            kwargs = {'normalize':True}
        else:
            raise Exception()

        for _ in range(num_layers):
            conv = HeteroConv({
                ('input', 'to', 'function'): conv_((-1, -1), hidden_channels, **kwargs),
                ('function', 'to', 'function'): conv_(-1, hidden_channels, **kwargs),
                ('function', 'to', 'output'): conv_((-1,-1), hidden_channels, **kwargs),
                ('output', 'to', 'function'): conv_((-1,-1), hidden_channels, **kwargs),
                ('function', 'to', 'input'): conv_((-1,-1), hidden_channels, **kwargs),
            }, aggr='sum')
            self.convs.append(conv)

        self.dropout=torch.nn.Dropout(dropout)

        self.n_heads = n_heads
        self.att_lin = Linear(hidden_channels, n_heads)

        self.norm = LayerNorm(hidden_channels)
        self.nonlin = torch.nn.ELU()
        self.f_mu = torch.nn.Sequential(Linear(hidden_channels*n_heads, 50), 
                                         torch.nn.ELU(), 
                                         torch.nn.Dropout(dropout), 
                                         Linear(50, 10),
                                         torch.nn.ELU(), 
                                         torch.nn.Dropout(dropout), 
                                         Linear(10, 1))
        
        self.f_pi = torch.nn.Sequential(Linear(hidden_channels*n_heads, 50), 
                                         torch.nn.ELU(), 
                                         torch.nn.Dropout(dropout), 
                                         Linear(50, 10),
                                         torch.nn.ELU(), 
                                         torch.nn.Dropout(dropout), 
                                         Linear(10, 2))

    def forward(self, data):

        #x_dict = {k:self.embedding_dict[k].repeat(torch.max(data[k].batch) + 1, 1) for k in ['input', 'function', 'output']}
        x_dict = data.x_dict
        for i,conv in enumerate(self.convs):
            #x_dict_last = x_dict
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k:self.nonlin(v) for k,v in x_dict.items()}
            x_dict = {k:self.norm(v) for k,v in x_dict.items()}
            x_dict = {k:self.dropout(v) for k,v in x_dict.items()}
            #if i > 0: x_dict = {k:v+x_dict_last[k] for k,v in x_dict.items()}

        att = torch.softmax(self.dropout(self.att_lin(x_dict['function'])), dim=0)

        out = [] 
        for i in range(self.n_heads): 
            xx = x_dict['function']*att[:, [i]]
            xx = global_add_pool(x=xx, batch=data['function'].batch)
            out.append(xx)
        x = torch.cat(out, dim=-1)

        #out = self.f_out(x)

        mu = self.f_mu(x)
        pi = self.f_pi(x.detach())

        lcb = mu.detach().view(-1) - pi[:,0].exp()
        ucb = mu.detach().view(-1) + pi[:,1].exp()
        #ucb = lcb + out[:,1].relu() + 1e-1

        return lcb, ucb, mu