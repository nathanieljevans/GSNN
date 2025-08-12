import torch
import torch.nn as nn
import torch_geometric as pyg

class GroupBatchNorm(nn.Module):
    """
    A batch-norm style module that:
      - Partitions the C channels into groups via 'channel_groups'.
      - Computes mean/var for each group across the entire batch dimension.
      - Maintains running stats for inference (if track_running_stats=True).
    """
    def __init__(self, 
                 channel_groups, 
                 eps=1e-5, 
                 momentum=0.1,
                 affine=False,
                 track_running_stats=True):
        super().__init__()
        
        # channel_groups is a tensor of shape (C,) mapping each channel to a group index
        self.register_buffer('channel_groups', torch.tensor(channel_groups, dtype=torch.long))
        
        # Number of distinct groups:
        num_groups = self.channel_groups.max().item() + 1
        
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        # Optional learnable parameters gamma, beta per group
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_groups))
            self.beta = nn.Parameter(torch.zeros(num_groups))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            # Registers running mean and var for each group
            self.register_buffer('running_mean', torch.zeros(num_groups))
            self.register_buffer('running_var', torch.ones(num_groups))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        
    def forward(self, x):
        """
        x: (B, C) or (B, C, 1). We first squeeze the last dim if necessary.
        """
        # Squeeze last dim if (B, C, 1)
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)  # now (B, C)
        
        B, C = x.shape

        if self.training or (not self.track_running_stats):
            # ----- Compute batch mean & var for each group efficiently -----
            # Use scatter on channel dimension to avoid flattening
            group_means = pyg.utils.scatter(x, self.channel_groups, dim=1, reduce='mean')  # (B, num_groups)
            
            # Compute group means across batch dimension
            batch_group_means = group_means.mean(dim=0)  # (num_groups,)
            
            # For variance, we need to compute (x - group_mean)^2 for each group
            # Broadcast group means back to original shape efficiently
            expanded_means = batch_group_means.index_select(0, self.channel_groups).unsqueeze(0)  # (1, C)
            
            # Compute variance using the broadcast mean
            centered = x - expanded_means  # (B, C)
            group_vars_per_batch = pyg.utils.scatter(centered**2, self.channel_groups, dim=1, reduce='mean')  # (B, num_groups)
            batch_group_vars = group_vars_per_batch.mean(dim=0)  # (num_groups,)

            # If we're tracking running stats, update them
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    momentum = self.momentum
                    
                    self.running_mean = (1 - momentum)*self.running_mean + momentum*batch_group_means
                    self.running_var  = (1 - momentum)*self.running_var  + momentum*batch_group_vars

        else:
            # ----- Use running stats (inference mode) -----
            batch_group_means = self.running_mean
            batch_group_vars  = self.running_var
        
        # ----- Apply normalization efficiently -----
        # Broadcast means and vars back to (B, C) shape
        expanded_means = batch_group_means.index_select(0, self.channel_groups).unsqueeze(0)  # (1, C)
        expanded_vars = batch_group_vars.index_select(0, self.channel_groups).unsqueeze(0)    # (1, C)
        
        # Normalize
        x_normalized = (x - expanded_means) / torch.sqrt(expanded_vars + self.eps)

        # Apply learnable affine transform if provided
        if self.gamma is not None and self.beta is not None:
            gamma_expanded = self.gamma.index_select(0, self.channel_groups).unsqueeze(0)  # (1, C)
            beta_expanded = self.beta.index_select(0, self.channel_groups).unsqueeze(0)    # (1, C)
            x_normalized = x_normalized * gamma_expanded + beta_expanded
        
        return x_normalized.unsqueeze(-1)  # to match the original shape (B, C, 1) if needed
