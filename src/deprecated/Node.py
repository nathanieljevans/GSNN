import torch

class Node(torch.nn.Module): 
    def __init__(self, input_channels, output_channels, in_ixs, out_ixs, idx, children, attr_dim=0, hidden_channels=2, nonlin=torch.nn.ELU, dropout=0., bias=True): 
        super().__init__()

        #self.idx = idx
        self.register_buffer('idx', idx)
        self.register_buffer('in_ixs', in_ixs)
        self.register_buffer('out_ixs', out_ixs)
        self.register_buffer('_children', children)

        self.nn = torch.nn.Sequential(torch.nn.Linear(input_channels + attr_dim, hidden_channels, bias=bias),
                                      torch.nn.BatchNorm1d(hidden_channels),
                                      nonlin(), 
                                      torch.nn.Dropout(dropout),
                                      torch.nn.Linear(hidden_channels, output_channels, bias=bias))
        
    def forward(self, x, attr=None): 


        x_in = x[:, self.in_ixs] # shape: (B, in_degree)

        if attr is not None: 
            x_in = torch.cat((x_in, attr[:, self.idx, :]), dim=-1) # shape: (B, in_degree + attr_dim)

        x_out = self.nn(x_in).tanh() # shape: (B, out_degree)

        # batch the indices
        B = x.size(0)
        indices = (torch.arange(B).unsqueeze(-1), self.out_ixs.unsqueeze(0).repeat(B, 1))

        x = torch.index_put(torch.zeros_like(x), indices=indices, values=x_out)

        return x