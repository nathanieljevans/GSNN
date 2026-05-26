"""Tests for signed message passing."""

import pytest
import torch

from gsnn.models.SignedMessagePassing import SignedMessagePassing


def test_signed_mp_forward_shape():
    edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, -1.0])
    mp = SignedMessagePassing(edge_weight, edge_index)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = mp(x)
    assert out.shape == x.shape


def test_signed_mp_sign_flip():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_weight = torch.tensor([-1.0])
    mp = SignedMessagePassing(edge_weight, edge_index)
    x = torch.tensor([[1.0, 0.0]])
    out = mp(x)
    assert out[0, 1].item() == pytest.approx(-1.0)


def test_signed_mp_no_function_edges():
    # Only cross-partition edges (sources outside function node index range)
    edge_index = torch.tensor([[2, 3], [0, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0])
    mp = SignedMessagePassing(edge_weight, edge_index)
    x = torch.tensor([[1.0, 2.0]])
    out = mp(x)
    assert torch.allclose(out, torch.zeros_like(x))
