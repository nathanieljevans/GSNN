"""
Group-wise Exponential Moving Average Normalization.

EMANorm maintains running statistics using exponential moving averages but doesn't 
depend on current batch statistics during normalization, making it very robust 
for small and variable batch sizes.
"""

import torch
import torch.nn as nn
import torch_geometric as pyg


class GroupEMANorm(nn.Module):
    """
    Applies normalization within each channel group using exponential moving averages.
    
    This normalization maintains running statistics but doesn't use current batch
    statistics for normalization, making it very stable for small batch sizes.
    
    Args:
        channel_groups (list or tensor): Specifies which group each channel belongs to.
        eps (float): Small value to avoid division by zero. Default: 1e-5
        momentum (float): Momentum for updating running statistics. Default: 0.1
        affine (bool): If True, applies learnable scale and bias. Default: True
        track_running_stats (bool): If True, maintains running statistics. Default: True
    """

    def __init__(self, channel_groups, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.register_buffer('channel_groups', torch.tensor(channel_groups, dtype=torch.long))
        num_groups = self.channel_groups.max().item() + 1
        
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        # Learnable parameters per group
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_groups))
            self.beta = nn.Parameter(torch.zeros(num_groups))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            # Running statistics per group
            self.register_buffer('running_mean', torch.zeros(num_groups))
            self.register_buffer('running_var', torch.ones(num_groups))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def forward(self, x):
        # Handle input shape (B, C) or (B, C, 1)
        original_shape = x.shape
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)  # Ensure (B, C) shape
        
        B, C = x.shape
        
        # Update running statistics during training
        if self.training and self.track_running_stats:
            with torch.no_grad():
                # Flatten for group-wise statistics
                group_idx = self.channel_groups.unsqueeze(0).expand(B, -1).reshape(-1)
                x_flat = x.reshape(-1)
                
                # Compute current batch statistics per group
                batch_mean = pyg.utils.scatter(x_flat, group_idx, dim=0, reduce='mean')
                batch_var = pyg.utils.scatter((x_flat - batch_mean[group_idx])**2, group_idx, dim=0, reduce='mean')
                
                # Update running statistics with exponential moving average
                self.num_batches_tracked += 1
                momentum = self.momentum
                
                self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
                self.running_var = (1 - momentum) * self.running_var + momentum * batch_var

        # Always use running statistics for normalization (key difference from batch norm)
        if self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        else:
            # Fallback: compute statistics from current batch if no running stats
            group_idx = self.channel_groups.unsqueeze(0).expand(B, -1).reshape(-1)
            x_flat = x.reshape(-1)
            mean = pyg.utils.scatter(x_flat, group_idx, dim=0, reduce='mean')
            var = pyg.utils.scatter((x_flat - mean[group_idx])**2, group_idx, dim=0, reduce='mean')

        # Normalize using running statistics
        expanded_mean = mean.index_select(0, self.channel_groups).unsqueeze(0)
        expanded_var = var.index_select(0, self.channel_groups).unsqueeze(0)
        
        x_normed = (x - expanded_mean) / torch.sqrt(expanded_var + self.eps)

        # Apply learnable affine transformation
        if self.gamma is not None and self.beta is not None:
            gamma_expanded = self.gamma.index_select(0, self.channel_groups).unsqueeze(0)
            beta_expanded = self.beta.index_select(0, self.channel_groups).unsqueeze(0)
            x_normed = x_normed * gamma_expanded + beta_expanded

        # Restore original shape
        if len(original_shape) == 3:
            x_normed = x_normed.unsqueeze(-1)

        return x_normed 