import torch
import torch_geometric as pyg


class SignedMessagePassing(pyg.nn.MessagePassing):
    """Aggregate scalar signals over function-function edges using stored signs (edge weights)."""

    def __init__(self, edge_weight, edge_index):
        """
        Args:
            edge_weight: Per-edge scalar weights (e.g. ±1), concatenated homogeneous ordering.
            edge_index: ``[2, E]`` homogeneous indices matching ``edge_weight``.
        """
        super().__init__(aggr='add')
        self.register_buffer('edge_weight', edge_weight)
        self.register_buffer('edge_index', edge_index)

    def forward(self, x):
        """Propagate ``x`` of shape ``(B, N_fn)`` over function-only subgraph; returns ``(B, N_fn)``."""
        B, N = x.shape

        # Create function node mask - function nodes are at indices 0 to N-1 in the homogeneous graph
        src_func = self.edge_index[0, :] < N
        dst_func = self.edge_index[1, :] < N
        func_mask = src_func & dst_func

        # Get function-only edges
        func_edge_index = self.edge_index[:, func_mask]
        func_edge_weight = self.edge_weight[func_mask]

        # Create batched edge indices for all batch items
        E = func_edge_index.size(1)
        batched_edge_indices = func_edge_index.repeat(1, B).contiguous()
        batch_idx = torch.repeat_interleave(torch.arange(B, dtype=torch.long, device=x.device), E).contiguous()

        # Add offsets for each batch item
        src_incr = batch_idx * N
        dst_incr = batch_idx * N
        incr = torch.stack((src_incr, dst_incr), dim=0)
        batched_edge_indices += incr

        # Create batched edge weights
        batched_edge_weights = func_edge_weight.repeat(B)

        # Reshape x to (B*N, 1) for batched processing
        x_flat = x.view(-1, 1)

        # Process all batch items at once
        out_flat = self.propagate(
            batched_edge_indices,
            x=x_flat,
            edge_weight=batched_edge_weights
        )

        # Reshape back to (B, N) - output should have same shape as input
        out = out_flat.view(B, N)

        return out

    def message(self, x_i, x_j, edge_weight):
        """Neighbor contribution weighted by ``edge_weight``."""
        return x_j.view(-1, 1) * edge_weight.view(-1, 1)
