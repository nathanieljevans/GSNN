'''
Applies 1d layer normalization within each provided channel groups. 
'''

import torch 
import torch_geometric as pyg

class GroupLayerNorm(torch.nn.Module): 

    def __init__(self, channel_groups, eps=1e-2, affine=True): 
        '''

        Args: 
            channel_groups           tensor              specifies which group a channel belongs to; 
                                                         for instance given: [0,0, 1,1, 2,2] specifies 3 groups with 2 channels in each; 
                                                         the first two channels are assigned to group 0. 
        '''

        super().__init__()

        self.register_buffer('channel_groups', torch.tensor(channel_groups, dtype=torch.long))
        self.register_buffer('n_channels', torch.unique(self.channel_groups, return_counts=True)[1])

        self.eps = eps

        if affine: 
            N = torch.max(self.channel_groups).item() + 1
            self.gamma = torch.nn.Parameter(torch.ones(N))
            self.beta = torch.nn.Parameter(torch.zeros(N))

    def forward(self, x): 

        x = x.squeeze(-1)

        mean = pyg.utils.scatter(x, self.channel_groups, dim=1, reduce='mean')
        std = (pyg.utils.scatter((x - mean[:, self.channel_groups])**2, self.channel_groups, dim=1, reduce='sum') / (self.n_channels-1))**0.5
        mean = mean.detach()
        std = std.detach()    # BUG: introduces nan's after first gradient update if not detached 

        expanded_mean = mean.index_select(1, self.channel_groups)
        expanded_std = std.index_select(1, self.channel_groups)

        x = (x - expanded_mean) / (expanded_std + self.eps)

        if hasattr(self, 'gamma'): 
            x = x*self.gamma[self.channel_groups].unsqueeze(0) + self.beta[self.channel_groups].unsqueeze(0)

        return x.unsqueeze(-1)
