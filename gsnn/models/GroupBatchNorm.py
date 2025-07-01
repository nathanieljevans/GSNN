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
        
        # We'll flatten out the batch to compute group stats across all (batch, group) items
        # group index for each of the B*C entries:
        #   e.g. if channel_groups = [0,0,1,1], repeated B times => shape(B*C,)
        group_idx = self.channel_groups.unsqueeze(0).expand(B, -1).reshape(-1)
        
        x_flat = x.reshape(-1)  # shape (B*C,)

        if self.training or (not self.track_running_stats):
            # ----- Compute batch mean & var for each group -----
            mean = pyg.utils.scatter(x_flat, group_idx, dim=0, reduce='mean')  # (num_groups,)
            
            # Use "mean[cg]" to broadcast for each item in x_flat, then recompute var
            var = pyg.utils.scatter((x_flat - mean[group_idx])**2, group_idx, dim=0, reduce='mean')  # (num_groups,)

            # If we're tracking running stats, update them
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                momentum = self.momentum
                
                self.running_mean = (1 - momentum)*self.running_mean + momentum*mean
                self.running_var  = (1 - momentum)*self.running_var  + momentum*var

        else:
            # ----- Use running stats (inference mode) -----
            mean = self.running_mean
            var  = self.running_var
        
        # Normalize
        # We must broadcast mean/var back to (B*C,)
        normed = x_flat - mean[group_idx]
        normed = normed / torch.sqrt(var[group_idx] + self.eps)

        # Apply learnable affine transform if provided
        if self.gamma is not None and self.beta is not None:
            normed = normed * self.gamma[group_idx] + self.beta[group_idx]
        
        # Reshape back to (B, C)
        x_out = normed.view(B, C)
        return x_out.unsqueeze(-1)  # to match the original shape (B, C, 1) if needed
