"""Tests for gsnn.models.NodeActivity."""

import pytest
import torch

from gsnn.models.NodeActivity import NodeActivity


def test_node_activity_per_node_mode():
    groups = [0, 0, 0, 0]
    mod = NodeActivity(groups, activity_dim=1, channels=8, mode="per-node")
    x = torch.randn(3, 1)
    out = mod(x)
    assert out.shape == (3, 4)


def test_node_activity_per_channel_mode():
    groups = [0, 0, 1, 1]
    mod = NodeActivity(groups, activity_dim=2, channels=8, mode="per-channel")
    x = torch.randn(2, 2, 2)
    out = mod(x)
    assert out.shape == (2, 4)


def test_node_activity_2d_input():
    groups = [0, 0]
    mod = NodeActivity(groups, activity_dim=1, channels=4, mode="per-node")
    out = mod(torch.randn(2, 1))
    assert out.shape == (2, 2)


def test_node_activity_wrong_Nf():
    mod = NodeActivity([0, 0], activity_dim=1, mode="per-node")
    with pytest.raises(ValueError, match="Expected x with Nf"):
        mod(torch.randn(2, 3))


def test_node_activity_wrong_dim():
    mod = NodeActivity([0, 0], activity_dim=2, mode="per-node")
    with pytest.raises(ValueError, match="expected 3-D"):
        mod(torch.randn(2, 1))


def test_node_activity_temperature():
    groups = [0, 0]
    cold = NodeActivity(groups, activity_dim=1, temperature=0.1, mode="per-node")
    warm = NodeActivity(groups, activity_dim=1, temperature=10.0, mode="per-node")
    torch.manual_seed(0)
    x = torch.randn(5, 1)
    cold_out = cold(x)
    warm_out = warm(x)
    # colder temperature -> gates closer to 0/1 on average
    assert cold_out.std() > warm_out.std()


def test_get_alpha_mean_before_forward():
    mod = NodeActivity([0, 0], activity_dim=1, mode="per-node")
    with pytest.raises(ValueError, match="Alpha mean not stored"):
        mod.get_alpha_mean()
