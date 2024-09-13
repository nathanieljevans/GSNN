

import torch
import pytest
from torch_geometric.utils import scatter

from gsnn.models.GroupLayerNorm import GroupLayerNorm


def test_group_layer_norm():
    # Set up the channel groups and input tensor
    channel_groups = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    x = torch.tensor([[0, 1, 1, 0, 2, 2, 0, 3, 3],
                       [0,-1,-1, 0, -2, -2, 0, -3, -3]], dtype=torch.float32)

    # Epsilon value for numerical stability
    eps = 1e-2
    
    # Instantiate GroupLayerNorm without affine transformation
    norm = GroupLayerNorm(channel_groups, eps=eps, affine=False)
    
    # Manually compute the expected output
    out = []
    for j in range(x.size(0)): 
        out2 = []
        for i in range(max(channel_groups) + 1):
            xx = x[j, torch.tensor(channel_groups) == i]
            xx = (xx - xx.mean()) / (xx.std() + eps)
            out2.append(xx)
        out.append(torch.cat(out2, dim=-1))
    
    # Concatenate the output to match the shape of the normalized output
    out = torch.stack(out, dim=0)

    # Run the GroupLayerNorm and check if the outputs match
    assert torch.allclose(out, norm(x).squeeze(-1)), "Manual normalization does not match GroupLayerNorm output."

if __name__ == "__main__":
    pytest.main([__file__])