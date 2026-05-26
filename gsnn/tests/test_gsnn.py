"""Tests for gsnn.models.GSNN."""

import pytest
import torch

from gsnn.models.GSNN import GSNN
from gsnn.tests.helpers import WORKING_NORMS


def _build_gsnn(edge_index_dict, node_names_dict, **kwargs):
    defaults = dict(
        channels=4,
        layers=2,
        norm="none",
        share_layers=False,
        add_function_self_edges=True,
        node_mlp=False,
    )
    defaults.update(kwargs)
    return GSNN(
        edge_index_dict=edge_index_dict,
        node_names_dict=node_names_dict,
        **defaults,
    )


def test_gsnn_forward_shape(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(minimal_edge_index_dict, minimal_node_names_dict)
    x = torch.randn(8, 2)
    out = model(x)
    assert out.shape == (8, 1)


def test_gsnn_ret_edge_out(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(minimal_edge_index_dict, minimal_node_names_dict)
    x = torch.randn(4, 2)
    edge_out = model(x, ret_edge_out=True)
    assert edge_out.shape[0] == 4
    assert edge_out.shape[1] == model.E


def test_gsnn_share_layers(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(
        minimal_edge_index_dict, minimal_node_names_dict, share_layers=True, layers=3
    )
    w_in_0 = model.ResBlocks[0].lin_in.values
    w_in_2 = model.ResBlocks[2].lin_in.values
    assert w_in_0.data_ptr() == w_in_2.data_ptr()


def test_gsnn_unshared_layers(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(
        minimal_edge_index_dict, minimal_node_names_dict, share_layers=False, layers=2
    )
    assert model.ResBlocks[0].lin_in.values.data_ptr() != model.ResBlocks[1].lin_in.values.data_ptr()


def test_gsnn_self_edges(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(
        minimal_edge_index_dict, minimal_node_names_dict, add_function_self_edges=True
    )
    func_idx = 0
    self_edges = (model.edge_index[0] == func_idx) & (model.edge_index[1] == func_idx)
    assert self_edges.any()


def test_gsnn_edge_mask(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(minimal_edge_index_dict, minimal_node_names_dict)
    x = torch.randn(2, 2)
    edge_mask = torch.ones(2, model.E)
    edge_mask[:, 0] = 0.0
    out = model(x, edge_mask=edge_mask)
    assert out.shape == (2, 1)


def test_gsnn_node_mask(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(minimal_edge_index_dict, minimal_node_names_dict)
    x = torch.randn(2, 2)
    node_mask = torch.ones(2, model.num_nodes)
    out = model(x, node_mask=node_mask)
    assert out.shape == (2, 1)


def test_gsnn_node_activity_requires_x_fn(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(
        minimal_edge_index_dict,
        minimal_node_names_dict,
        node_activity=True,
        layers=1,
    )
    x = torch.randn(2, 2)
    with pytest.raises(ValueError, match="x_fn"):
        model(x)


def test_gsnn_node_activity_per_node(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(
        minimal_edge_index_dict,
        minimal_node_names_dict,
        node_activity=True,
        node_activity_mode="per-node",
        layers=1,
    )
    x = torch.randn(2, 2)
    x_fn = torch.randn(2, 1)
    out = model(x, x_fn=x_fn)
    assert out.shape == (2, 1)


def test_gsnn_node_activity_per_channel(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(
        minimal_edge_index_dict,
        minimal_node_names_dict,
        node_activity=True,
        node_activity_mode="per-channel",
        layers=1,
    )
    x = torch.randn(2, 2)
    x_fn = torch.randn(2, 1)
    out = model(x, x_fn=x_fn)
    assert out.shape == (2, 1)


def test_gsnn_get_batch_params_caching(minimal_gsnn):
    p1 = minimal_gsnn.get_batch_params(4, torch.device("cpu"))
    p2 = minimal_gsnn.get_batch_params(4, torch.device("cpu"))
    assert p1[0].data_ptr() == p2[0].data_ptr()


def test_gsnn_get_batch_params_different_B(minimal_gsnn):
    p4 = minimal_gsnn.get_batch_params(4, torch.device("cpu"))
    p8 = minimal_gsnn.get_batch_params(8, torch.device("cpu"))
    assert p4[0].shape != p8[0].shape


def test_gsnn_prune(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(minimal_edge_index_dict, minimal_node_names_dict, layers=1)
    with torch.no_grad():
        for mod in model.ResBlocks:
            mod.lin_in.values.fill_(1e-4)
            mod.lin_out.values.fill_(1.0)
    removed = model.prune(threshold=1e-2)
    assert removed >= 0


def test_gsnn_checkpoint_forward(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(
        minimal_edge_index_dict,
        minimal_node_names_dict,
        checkpoint=True,
        layers=1,
    )
    model.train()
    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    out = model(x)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_gsnn_node_errs_length(minimal_gsnn):
    x = torch.randn(2, 2)
    with pytest.raises(ValueError, match="node_errs"):
        minimal_gsnn(x, node_errs=[None])


@pytest.mark.parametrize("agg", ["sum", "mean", "max", "last", "all"])
def test_gsnn_get_node_activations_agg(minimal_gsnn, agg):
    x = torch.randn(2, 2)
    acts = minimal_gsnn.get_node_activations(x, agg=agg)
    assert "func0" in acts
    if agg == "all":
        assert acts["func0"].dim() == 2
        assert acts["func0"].shape[0] == 2  # batch
    elif agg == "last":
        assert acts["func0"].dim() == 2


def test_gsnn_get_node_attention(minimal_edge_index_dict, minimal_node_names_dict):
    model = _build_gsnn(
        minimal_edge_index_dict,
        minimal_node_names_dict,
        node_attn=True,
        layers=2,
    )
    x = torch.randn(2, 2)
    attn = model.get_node_attention(x)
    assert "func0" in attn
    assert attn["func0"].shape[0] == 2  # layers


def test_gsnn_edge_weight_dict(minimal_edge_index_dict, minimal_node_names_dict):
    edge_weight_dict = {
        k: torch.ones(v.size(1)) for k, v in minimal_edge_index_dict.items()
    }
    model = _build_gsnn(
        minimal_edge_index_dict,
        minimal_node_names_dict,
        edge_weight_dict=edge_weight_dict,
    )
    x = torch.randn(2, 2)
    out = model(x)
    assert out.shape == (2, 1)


@pytest.mark.parametrize("norm", WORKING_NORMS)
def test_gsnn_norm_variants(minimal_edge_index_dict, minimal_node_names_dict, norm):
    model = _build_gsnn(
        minimal_edge_index_dict, minimal_node_names_dict, norm=norm, layers=1
    )
    x = torch.randn(2, 2)
    out = model(x)
    assert torch.isfinite(out).all()


def test_gsnn_gradient_flow(minimal_gsnn):
    minimal_gsnn.train()
    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    out = minimal_gsnn(x)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    grads = [p.grad for mod in minimal_gsnn.ResBlocks for p in mod.lin_in.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
