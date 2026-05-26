"""Tests for gsnn.models.ResBlock."""

import pytest
import torch
import torch.nn as nn

from gsnn.models.ResBlock import ResBlock
from gsnn.models.SparseLinear import SparseLinear
from gsnn.tests.helpers import WORKING_NORMS


def _make_resblock(resblock_indices_params, **kwargs):
    defaults = dict(
        bias=True,
        nonlin=nn.ELU,
        indices_params=resblock_indices_params,
        dropout=0.0,
        norm="none",
        init="uniform",
        node_mlp=False,
        node_attn=False,
        residual=True,
    )
    defaults.update(kwargs)
    return ResBlock(**defaults)


def _batch_params(block, B=2, device="cpu"):
    lin_in = block.lin_in
    lin_out = block.lin_out
    from gsnn.models.SparseLinear import batch_graphs

    batched_in = batch_graphs(lin_in.N, lin_in.M, lin_in.indices, B, device)
    batched_out = batch_graphs(lin_out.N, lin_out.M, lin_out.indices, B, device)
    return (batched_in, batched_out)


def _forward_resblock(block, x, batch_params, **kwargs):
    """ResBlock.forward reads self.node_mask; initialize when unset."""
    if not hasattr(block, "node_mask"):
        block.set_node_mask(None)
    out = block(x, batch_params, **kwargs)
    return out.squeeze(-1) if out.dim() == 3 else out


def test_resblock_forward_shape(resblock_indices_params):
    block = _make_resblock(resblock_indices_params)
    B, E = 3, resblock_indices_params[2][0]
    x = torch.randn(B, E)
    out = _forward_resblock(block, x, _batch_params(block, B))
    assert out.shape == (B, E)


def test_resblock_residual_off(resblock_indices_params):
    block = _make_resblock(resblock_indices_params, residual=False)
    B, E = 2, resblock_indices_params[2][0]
    x = torch.randn(B, E)
    out = _forward_resblock(block, x, _batch_params(block, B))
    assert out.shape == (B, E)


def test_resblock_node_mask(resblock_indices_params):
    block = _make_resblock(resblock_indices_params)
    B, E = 2, resblock_indices_params[2][0]
    x = torch.ones(B, E)
    n_channels = resblock_indices_params[4]
    mask = torch.zeros(B, len(n_channels))
    mask[:, 0] = 1.0
    block.set_node_mask(mask)
    out = _forward_resblock(block, x, _batch_params(block, B))
    assert out.shape == (B, E)


def test_resblock_fn_activity(resblock_indices_params):
    block = _make_resblock(resblock_indices_params)
    B, E = 2, resblock_indices_params[2][0]
    x = torch.randn(B, E)
    n_channels = len(resblock_indices_params[4])
    fn_activity = torch.ones(B, n_channels)
    out = _forward_resblock(block, x, _batch_params(block, B), fn_activity=fn_activity)
    assert out.shape == (B, E)


def test_resblock_node_err(resblock_indices_params):
    block = _make_resblock(resblock_indices_params)
    B, E = 2, resblock_indices_params[2][0]
    x = torch.randn(B, E)
    n_nodes = int(max(resblock_indices_params[4]) + 1)
    node_err = torch.randn(B, n_nodes)
    out = _forward_resblock(block, x, _batch_params(block, B), node_err=node_err)
    assert out.shape == (B, E)


def test_resblock_store_activations(resblock_indices_params):
    block = _make_resblock(resblock_indices_params)
    block._store_activations = True
    B, E = 2, resblock_indices_params[2][0]
    x = torch.randn(B, E)
    _forward_resblock(block, x, _batch_params(block, B))
    assert hasattr(block, "_last_activation")


@pytest.mark.parametrize("norm", WORKING_NORMS)
def test_resblock_norm_variants(resblock_indices_params, norm):
    block = _make_resblock(resblock_indices_params, norm=norm)
    B, E = 2, resblock_indices_params[2][0]
    x = torch.randn(B, E)
    out = _forward_resblock(block, x, _batch_params(block, B))
    assert out.shape == (B, E)
    assert torch.isfinite(out).all()


def test_resblock_invalid_norm(resblock_indices_params):
    with pytest.raises(ValueError, match="unrecognized norm type"):
        _make_resblock(resblock_indices_params, norm="bad_norm")


def test_resblock_node_mlp(resblock_indices_params):
    block = _make_resblock(resblock_indices_params, node_mlp=True, node_mlp_hidden=8)
    B, E = 2, resblock_indices_params[2][0]
    x = torch.randn(B, E)
    out = _forward_resblock(block, x, _batch_params(block, B))
    assert out.shape == (B, E)


def test_resblock_node_attn(resblock_indices_params, minimal_edge_index_dict, minimal_node_names_dict):
    from gsnn.models.utils import hetero2homo

    edge_index, in_mask, out_mask, _, _, edge_weight = hetero2homo(
        minimal_edge_index_dict, minimal_node_names_dict
    )
    function_nodes = (~(in_mask | out_mask)).nonzero(as_tuple=True)[0]
    edge_index = torch.cat(
        (edge_index, torch.stack((function_nodes, function_nodes), dim=0)), dim=1
    )
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))
    block = _make_resblock(
        resblock_indices_params,
        node_attn=True,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    B, E = 2, resblock_indices_params[2][0]
    x = torch.randn(B, E)
    out = _forward_resblock(block, x, _batch_params(block, B))
    assert out.shape == (B, E)


def test_resblock_shared_lin(resblock_indices_params):
    w_in, w_out, w_in_size, w_out_size, _ = resblock_indices_params
    shared_in = SparseLinear(w_in, w_in_size, init="uniform")
    shared_out = SparseLinear(w_out, w_out_size, init="uniform")
    block = _make_resblock(
        resblock_indices_params,
        lin_in=shared_in,
        lin_out=shared_out,
    )
    assert block.lin_in is shared_in
    assert block.lin_out is shared_out
