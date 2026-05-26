"""Tests for gsnn.models.NodeAttention."""

import pytest
import torch

from gsnn.models.NodeAttention import NodeAttention


def test_node_attention_forward_shape():
    ch_groups = [0, 0, 1, 1]
    attn = NodeAttention(ch_groups, channels=8)
    x = torch.randn(4, 4)
    out = attn(x)
    assert out.shape == x.shape


def test_node_attention_return_alpha():
    ch_groups = [0, 0, 1, 1]
    attn = NodeAttention(ch_groups, channels=8)
    x = torch.randn(4, 4)
    out, alpha = attn(x, return_alpha=True)
    assert out.shape == x.shape
    assert alpha.shape == (4, 2)


def test_node_attention_wrong_channels():
    attn = NodeAttention([0, 0, 1, 1], channels=8)
    with pytest.raises(ValueError, match="Expected input"):
        attn(torch.randn(2, 3))


def test_node_attention_with_signed_mp():
    ch_groups = [0, 0, 1, 1]
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0])
    attn = NodeAttention(ch_groups, channels=8, edge_index=edge_index, edge_weight=edge_weight)
    x = torch.randn(2, 4)
    out = attn(x)
    assert out.shape == x.shape


def test_node_attention_stores_last_alpha():
    attn = NodeAttention([0, 0, 1, 1], channels=8)
    x = torch.randn(2, 4)
    attn(x)
    assert attn._last_alpha is not None
    assert attn._last_alpha.shape == (2, 2)
