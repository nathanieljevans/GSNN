import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, num_nodes, n_heads):
        super(MultiHeadAttentionPooling, self).__init__()
        self.n_heads = n_heads

        # For each head, define a linear layer to compute attention weights
        self.att = torch.nn.Parameter(torch.randn((n_heads, num_nodes), dtype=torch.float32))

    def forward(self, x, batch):
        """
        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels]
            batch (LongTensor): Batch vector mapping each node to its respective graph
                                in the batch, with shape [num_nodes]

        Returns:
            Tensor: Pooled representations of shape [num_graphs, in_channels * n_heads]
        """

        B = torch.max(batch) + 1 # assume all graphs have same number of nodes s
        pools = [] 
        for i in range(self.n_heads): 
            logi = torch.nn.functional.dropout(self.att[[i]], p=0.0, training=self.training)
            atti = torch.softmax(logi, dim=1).repeat(1, B).view(-1,1)
            pools.append( global_add_pool(x*atti, batch) )
        x = torch.cat(pools, dim=-1)

        return x