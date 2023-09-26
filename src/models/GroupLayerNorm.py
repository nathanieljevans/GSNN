'''
Applies 1d layer normalization within each provided channel groups. 
'''

import torch 
import torch_scatter 

class GroupLayerNorm(torch.nn.Module): 

    def __init__(self, channel_groups, eps=1e-5): 
        '''

        Args: 
            channel_groups           tensor              specifies which group a channel belongs to; for instance given: [0,0, 1,1, 2,2] specifies 3 groups with 2 channels in each; the first two channels are assigned to group 0. 
        '''

        super().__init__()

        self.register_buffer('channel_groups', torch.tensor(channel_groups, dtype=torch.long))

        self.eps = eps

    def forward(self, x): 

        x = x.squeeze()

        mean = torch_scatter.scatter_mean(x, self.channel_groups, dim=1)
        std = torch_scatter.scatter_std(x, self.channel_groups, dim=1).detach()     # BUG: introduces nan's after first gradient update if not detached 

        expanded_mean = mean.index_select(1, self.channel_groups)
        expanded_std = std.index_select(1, self.channel_groups)

        x = (x - expanded_mean) / (expanded_std + self.eps)

        x = x.unsqueeze(-1)

        return x






