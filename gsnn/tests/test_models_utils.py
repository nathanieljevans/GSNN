"""Tests for gsnn.models.utils."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from gsnn.models import utils


@pytest.fixture
def hetero_fixture():
    edge_index_dict = {
        ("input", "to", "function"): torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        ("function", "to", "function"): torch.tensor([[0], [0]], dtype=torch.long),
        ("function", "to", "output"): torch.tensor([[0], [0]], dtype=torch.long),
    }
    node_names_dict = {
        "input": ["in0", "in1"],
        "function": ["func0"],
        "output": ["out0"],
    }
    return edge_index_dict, node_names_dict


def test_hetero2homo_offsets(hetero_fixture):
    edge_index_dict, node_names_dict = hetero_fixture
    edge_index, _, _, num_nodes, homo_names, _ = utils.hetero2homo(
        edge_index_dict, node_names_dict
    )
    assert num_nodes == 4
    assert homo_names == ["func0", "in0", "in1", "out0"]
    assert edge_index.size(1) == 4
    # input edges: src offset by N_function (=1)
    assert edge_index[0, 1].item() == 1  # in1 -> func0


def test_hetero2homo_masks(hetero_fixture):
    edge_index_dict, node_names_dict = hetero_fixture
    _, in_mask, out_mask, _, _, _ = utils.hetero2homo(edge_index_dict, node_names_dict)
    assert in_mask.sum().item() == 2
    assert out_mask.sum().item() == 1


def test_hetero2homo_edge_weights(hetero_fixture):
    edge_index_dict, node_names_dict = hetero_fixture
    edge_weight_dict = {
        k: torch.arange(edge_index_dict[k].size(1), dtype=torch.float32)
        for k in edge_index_dict
    }
    _, _, _, _, _, edge_weight = utils.hetero2homo(
        edge_index_dict, node_names_dict, edge_weight_dict
    )
    assert edge_weight is not None
    assert edge_weight.numel() == 4


def test_get_Win_indices():
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 1]], dtype=torch.long)
    function_nodes = torch.tensor([0, 1])
    indices, counts = utils.get_Win_indices(edge_index, channels=2, function_nodes=function_nodes)
    assert indices.shape[0] == 2
    assert counts[0] == 2
    assert counts[1] == 2


def test_get_Wout_indices():
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 1]], dtype=torch.long)
    function_nodes = torch.tensor([0, 1])
    channels = np.array([2, 2, 0, 0])
    indices = utils.get_Wout_indices(edge_index, function_nodes, channels)
    assert indices.shape[0] == 2
    assert indices.size(1) > 0


def test_get_conv_indices():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    function_nodes = torch.tensor([0])
    w_in, w_out, w_in_size, w_out_size, groups = utils.get_conv_indices(
        edge_index, channels=3, function_nodes=function_nodes
    )
    assert w_in_size[0] == 2
    assert w_out_size[1] == 2
    assert len(groups) == 3


def test_node2edge():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 0]])
    out = utils.node2edge(x, edge_index)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0, 2.0]]))


def test_edge2node_single_in_degree():
    x = torch.tensor([[1.0, 2.0]])
    edge_index = torch.tensor([[0, 1], [2, 3]])
    output_mask = torch.tensor([False, False, True, True])
    out = utils.edge2node(x, edge_index, output_mask)
    assert out.shape == (1, 4)
    assert out[0, 2].item() == pytest.approx(1.0)
    assert out[0, 3].item() == pytest.approx(2.0)


def test_edge2node_multi_in_degree():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    edge_index = torch.tensor([[0, 1, 1], [2, 2, 3]])
    output_mask = torch.tensor([False, False, True, True])
    out = utils.edge2node(x, edge_index, output_mask)
    # node 2 receives edges 0 and 1 (values 1 and 2) -> (1+2)/sqrt(2)
    expected = (1.0 + 2.0) / np.sqrt(2)
    assert out[0, 2].item() == pytest.approx(expected)


def test_apply_norm_and_nonlin_order():
    norm = nn.Identity()
    nonlin = nn.ReLU()

    class Track(nn.Module):
        def __init__(self, mod, name):
            super().__init__()
            self.mod = mod
            self.name = name
            self.order = []

        def forward(self, x):
            self.order.append(self.name)
            return self.mod(x)

    t_norm = Track(norm, "norm")
    t_nonlin = Track(nonlin, "nonlin")
    x = torch.tensor([[-1.0, 2.0]])

    utils.apply_norm_and_nonlin(t_norm, t_nonlin, x, norm_first=True)
    assert t_norm.order == ["norm"]
    assert t_nonlin.order == ["nonlin"]

    t_norm.order.clear()
    t_nonlin.order.clear()
    utils.apply_norm_and_nonlin(t_norm, t_nonlin, x, norm_first=False)
    assert t_nonlin.order == ["nonlin"]
    assert t_norm.order == ["norm"]


def test_corr_score_pearson():
    y = np.array([[0.0, 1.0, 2.0, 3.0]]).T
    yhat = np.array([[0.1, 1.1, 2.1, 3.1]]).T
    score = utils.corr_score(y, yhat, method="pearson")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_corr_score_constant_column():
    y = np.ones((10, 1))
    yhat = np.random.randn(10, 1)
    score = utils.corr_score(y, yhat)
    assert score == 0.0


def test_corr_score_invalid_method():
    with pytest.raises(ValueError, match="unrecognized metric"):
        utils.corr_score(np.zeros((2, 1)), np.zeros((2, 1)), method="bad")


def test_predict_gsnn(minimal_gsnn):
    class MockLoader:
        def __init__(self):
            self.data = [
                (torch.randn(2, 2), torch.randn(2, 1), "a"),
                (torch.randn(3, 2), torch.randn(3, 1), "b"),
            ]

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

    y, yhat, sig_ids = utils.predict_gsnn(MockLoader(), minimal_gsnn, "cpu", verbose=False)
    assert y.shape[0] == 5
    assert yhat.shape[0] == 5
    assert sig_ids == ["a", "b"]
