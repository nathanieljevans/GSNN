"""Tests for gsnn.models.PathwayLatentRegularizer."""

import pytest
import torch

from gsnn.models.GSNN import GSNN
from gsnn.models.PathwayLatentRegularizer import PathwayLatentRegularizer


def _model_and_membership(minimal_edge_index_dict, minimal_node_names_dict):
    model = GSNN(
        edge_index_dict=minimal_edge_index_dict,
        node_names_dict=minimal_node_names_dict,
        channels=4,
        layers=1,
        norm="none",
        node_mlp=False,
    )
    M = torch.eye(1)  # one pathway, one function node
    return model, M


def test_enable_disable_hooks(minimal_edge_index_dict, minimal_node_names_dict):
    model, M = _model_and_membership(minimal_edge_index_dict, minimal_node_names_dict)
    reg = PathwayLatentRegularizer(model, pathway_membership=M, lambda_sim=0.1)
    assert model.ResBlocks[0]._store_activations is True
    reg.disable(model)
    assert model.ResBlocks[0]._store_activations is False
    reg.enable(model)
    assert model.ResBlocks[0]._store_activations is True


def test_loss_nonnegative(minimal_edge_index_dict, minimal_node_names_dict):
    model, M = _model_and_membership(minimal_edge_index_dict, minimal_node_names_dict)
    reg = PathwayLatentRegularizer(model, pathway_membership=M, lambda_sim=0.1)
    model.train()
    x = torch.randn(4, 2)
    model(x)
    l_sim, l_dis = reg.loss(model)
    assert torch.isfinite(l_sim)
    assert torch.isfinite(l_dis)


def test_loss_requires_forward(minimal_edge_index_dict, minimal_node_names_dict):
    model, M = _model_and_membership(minimal_edge_index_dict, minimal_node_names_dict)
    reg = PathwayLatentRegularizer(model, pathway_membership=M, lambda_sim=0.1)
    reg.disable(model)
    model.train()
    with pytest.raises(RuntimeError, match="_last_activation"):
        reg.loss(model)
