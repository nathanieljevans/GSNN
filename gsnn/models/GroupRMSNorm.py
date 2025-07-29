"""
Group-wise Root Mean Square Layer Normalization.

RMSNorm is a simpler alternative to layer normalization that only uses the RMS 
for normalization without mean centering. It's particularly stable for small 
batch sizes and computationally more efficient than layer norm.
"""

import torch
import torch_geometric as pyg


class GroupRMSNorm(torch.nn.Module):
    """
    Applies Root Mean Square normalization within each channel group.
    
    RMSNorm normalizes using only the RMS (root mean square) without mean centering,
    making it simpler and more stable than layer normalization, especially for 
    small batch sizes.
    
    Args:
        channel_groups (list or tensor): Specifies which group each channel belongs to.
            For example, [0,0,1,1,2,2] specifies 3 groups with 2 channels each.
        eps (float): Small value to avoid division by zero. Default: 1e-6
        affine (bool): If True, applies learnable scale parameter. Default: True
    """

    def __init__(self, channel_groups, eps=1e-6, affine=True):
        super().__init__()

        self.register_buffer('channel_groups', torch.tensor(channel_groups, dtype=torch.long))
        self.register_buffer('n_channels', torch.unique(self.channel_groups, return_counts=True)[1])
        
        self.eps = eps

        if affine:
            # Only scale parameter (gamma), no bias since we don't center
            N = torch.max(self.channel_groups).item() + 1
            self.gamma = torch.nn.Parameter(torch.ones(N))
        else:
            self.register_parameter('gamma', None)

    def forward(self, x):
        # Handle input shape (B, C) or (B, C, 1)
        original_shape = x.shape
        x = x.squeeze(-1)  # Ensure (B, C) shape

        # Compute RMS per group
        # RMS = sqrt(mean(x^2))
        x_squared = x ** 2
        rms_squared = pyg.utils.scatter(x_squared, self.channel_groups, dim=1, reduce='mean')
        rms = torch.sqrt(rms_squared + self.eps)

        # Expand RMS to match input dimensions
        expanded_rms = rms.index_select(1, self.channel_groups)

        # Normalize by RMS
        x_normed = x / expanded_rms

        # Apply learnable scale if enabled
        if self.gamma is not None:
            x_normed = x_normed * self.gamma[self.channel_groups].unsqueeze(0)

        # Restore original shape
        if len(original_shape) == 3:
            x_normed = x_normed.unsqueeze(-1)

        return x_normed 