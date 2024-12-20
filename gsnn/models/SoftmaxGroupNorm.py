import torch
import torch_geometric as pyg

class SoftmaxGroupNorm(torch.nn.Module):
    def __init__(self, channel_groups, eps=1e-8):
        """
        Args:
            channel_groups (tensor): Specifies which group each channel belongs to.
            eps (float): A small value to avoid division by zero.
            affine (bool): If True, includes learnable gamma and beta parameters per group.
        """
        super().__init__()
        
        self.register_buffer('channel_groups', torch.tensor(channel_groups, dtype=torch.long))
        unique_groups, counts = torch.unique(self.channel_groups, return_counts=True)
        self.register_buffer('n_channels', counts)
        
        self.eps = eps
        
    def forward(self, x):
        # If input has a trailing singleton dimension (e.g. shape (B, C, 1)), remove it
        if x.size(-1) == 1:
            x = x.squeeze(-1)

        # Compute per-group maxima for numerical stability (stable softmax)
        max_values = pyg.utils.scatter(x, self.channel_groups, dim=1, reduce='max')
        expanded_max_values = max_values.index_select(1, self.channel_groups)
        
        # Exponentiate shifted values
        exp_x = torch.exp(x - expanded_max_values)

        # Compute sum of exponentials per group
        sum_exp = pyg.utils.scatter(exp_x, self.channel_groups, dim=1, reduce='sum')
        expanded_sum = sum_exp.index_select(1, self.channel_groups) + self.eps

        # Compute stable softmax
        x = exp_x / expanded_sum

        # Restore trailing dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        return x
