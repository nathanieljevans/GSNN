"""Tests for gsnn.models.SparseLinear."""

import pytest
import torch

from gsnn.models.SparseLinear import (
    Conv,
    SparseLinear,
    batch_graphs,
    kaiming_normal,
    kaiming_uniform,
    normal,
    uniform,
    xavier_normal,
    xavier_uniform,
)


def _toy_sparse_linear(bias=True, init="uniform"):
    # Bipartite: 2 sources -> 3 destinations
    indices = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long)
    size = (2, 3)
    return SparseLinear(indices, size, bias=bias, init=init)


def test_conv_message_passing():
    conv = Conv()
    edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    x = torch.tensor([[1.0], [2.0]])
    edge_weight = torch.tensor([1.0, 3.0])
    bias = torch.tensor([0.5, -0.5])
    out = conv(x, edge_index, edge_weight, bias, size=(2, 2))
    assert out.shape == (2, 1)
    assert torch.allclose(out[0], torch.tensor([1.0 + 0.5]))
    assert torch.allclose(out[1], torch.tensor([6.0 - 0.5]))


def test_batch_graphs_offsets():
    edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    batched = batch_graphs(N=2, M=2, edge_index=edge_index, B=3, device="cpu")
    assert batched.shape == (2, 6)
    # batch 1: src offset +2, dst offset +2
    assert batched[0, 2].item() == 2
    assert batched[1, 2].item() == 2
    # batch 2: src offset +4, dst offset +4
    assert batched[0, 4].item() == 4
    assert batched[1, 4].item() == 4


def test_sparse_linear_forward_shape():
    layer = _toy_sparse_linear()
    x = torch.randn(5, 2, 1)
    out = layer(x)
    assert out.shape == (5, 3, 1)


def test_sparse_linear_matches_dense():
    torch.manual_seed(0)
    indices = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long)
    layer = SparseLinear(indices, (2, 3), bias=True, init="uniform")
    B = 4
    x = torch.randn(B, 2, 1)

    out_sparse = layer(x).squeeze(-1)

    W = torch.zeros(2, 3)
    vals = layer.values.view(-1)
    for k in range(indices.size(1)):
        W[indices[0, k], indices[1, k]] = vals[k]
    bias = layer.bias.view(-1)
    out_dense = x.squeeze(-1) @ W + bias

    assert torch.allclose(out_sparse, out_dense, atol=1e-5)


def test_sparse_linear_no_bias():
    layer = _toy_sparse_linear(bias=False)
    assert not hasattr(layer, "bias")
    x = torch.randn(2, 2, 1)
    out = layer(x)
    assert out.shape == (2, 3, 1)


@pytest.mark.parametrize(
    "init",
    [
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
        "uniform",
        "normal",
        "degree_normalized",
        "zeros",
    ],
)
def test_init_schemes(init):
    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    layer = SparseLinear(indices, (2, 2), init=init)
    x = torch.randn(1, 2, 1)
    out = layer(x)
    assert out.shape == (1, 2, 1)
    assert torch.isfinite(out).all()


def test_init_invalid_raises():
    indices = torch.tensor([[0], [0]], dtype=torch.long)
    with pytest.raises(ValueError, match="unrecognized weight initialization"):
        SparseLinear(indices, (1, 1), init="invalid")


def test_sparse_linear_batched_indices_cache():
    layer = _toy_sparse_linear()
    x = torch.randn(3, 2, 1)
    auto = layer(x)
    batched = batch_graphs(N=2, M=3, edge_index=layer.indices, B=3, device=x.device)
    manual = layer(x, batched_indices=batched)
    assert torch.allclose(auto, manual)


def test_prune_reduces_edges():
    layer = _toy_sparse_linear()
    n_before = layer.indices.size(1)
    layer.prune(torch.tensor([0]))
    assert layer.indices.size(1) == 1
    assert layer.values.shape[0] == 1
    assert n_before == 3


def test_init_helpers_finite():
    n_in = torch.tensor([1.0, 2.0, 1.0, 2.0])
    n_out = torch.tensor([1.0, 1.0, 2.0, 1.0])
    for fn in (xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal):
        t = fn(4, n_in, n_out)
        assert t.numel() == 4
        assert torch.isfinite(t).all()
    for fn in (uniform, normal):
        t = fn(4)
        assert t.numel() == 4
        assert torch.isfinite(t).all()
