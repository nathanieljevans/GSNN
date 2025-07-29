"""
Channel-wise Exponential Moving Average Normalization.

ChannelEMANorm maintains running statistics using exponential moving averages for each 
individual channel independently, making it very robust for small and variable batch sizes.
"""

import torch
import torch.nn as nn


class ChannelEMANorm(nn.Module):
    """
    Applies normalization per individual channel using exponential moving averages.
    
    This normalization maintains running statistics for each channel independently
    but doesn't use current batch statistics for normalization, making it very stable 
    for small batch sizes.
    
    Args:
        num_channels (int): Number of channels to normalize.
        eps (float): Small value to avoid division by zero. Default: 1e-5
        momentum (float): Momentum for updating running statistics. Default: 0.1
        affine (bool): If True, applies learnable scale and bias per channel. Default: True
        track_running_stats (bool): If True, maintains running statistics. Default: True
    """

    def __init__(self, num_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        # Learnable parameters per channel
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            # Running statistics per channel
            self.register_buffer('running_mean', torch.zeros(num_channels))
            self.register_buffer('running_var', torch.ones(num_channels))
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
        
        if C != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {C}")
        
        # Update running statistics during training
        if self.training and self.track_running_stats:
            with torch.no_grad():
                # Compute current batch statistics per channel
                batch_mean = x.mean(dim=0)  # Shape: (C,)
                batch_var = x.var(dim=0, unbiased=False)  # Shape: (C,)
                
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
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

        # Normalize using running statistics
        x_normed = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable affine transformation
        if self.gamma is not None and self.beta is not None:
            x_normed = x_normed * self.gamma + self.beta

        # Restore original shape
        if len(original_shape) == 3:
            x_normed = x_normed.unsqueeze(-1)

        return x_normed 