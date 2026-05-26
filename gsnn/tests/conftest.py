"""Shared pytest fixtures for the GSNN test suite."""

import networkx as nx
import pytest
import torch

from gsnn.models.GSNN import GSNN
from gsnn.models.utils import get_conv_indices, hetero2homo


WORKING_NORMS = ["batch", "groupbatch", "layer", "none", "rms", "softmax"]
ALL_NORMS = WORKING_NORMS


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def minimal_edge_index_dict():
    return {
        ("input", "to", "function"): torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        ("function", "to", "function"): torch.tensor([[0], [0]], dtype=torch.long),
        ("function", "to", "output"): torch.tensor([[0], [0]], dtype=torch.long),
    }


@pytest.fixture
def minimal_node_names_dict():
    return {
        "input": ["in0", "in1"],
        "function": ["func0"],
        "output": ["out0"],
    }


@pytest.fixture
def tiny_edge_index_dict():
    """Single input / function / output graph."""
    return {
        ("input", "to", "function"): torch.tensor([[0], [0]], dtype=torch.long),
        ("function", "to", "function"): torch.empty((2, 0), dtype=torch.long),
        ("function", "to", "output"): torch.tensor([[0], [0]], dtype=torch.long),
    }


@pytest.fixture
def tiny_node_names_dict():
    return {
        "input": ["inp"],
        "function": ["func"],
        "output": ["out"],
    }


@pytest.fixture
def minimal_gsnn(minimal_edge_index_dict, minimal_node_names_dict):
    return GSNN(
        edge_index_dict=minimal_edge_index_dict,
        node_names_dict=minimal_node_names_dict,
        channels=4,
        layers=2,
        norm="none",
        share_layers=False,
        add_function_self_edges=True,
        node_mlp=False,
    )


@pytest.fixture
def tiny_gsnn(tiny_edge_index_dict, tiny_node_names_dict):
    return GSNN(
        edge_index_dict=tiny_edge_index_dict,
        node_names_dict=tiny_node_names_dict,
        channels=4,
        layers=1,
        norm="none",
        share_layers=False,
        add_function_self_edges=True,
        node_mlp=False,
    )


@pytest.fixture
def resblock_indices_params(minimal_edge_index_dict, minimal_node_names_dict):
    edge_index, input_mask, output_mask, num_nodes, _, _ = hetero2homo(
        minimal_edge_index_dict, minimal_node_names_dict
    )
    function_nodes = (~(input_mask | output_mask)).nonzero(as_tuple=True)[0]
    edge_index = torch.cat(
        (edge_index, torch.stack((function_nodes, function_nodes), dim=0)), dim=1
    )
    return get_conv_indices(edge_index, channels=4, function_nodes=function_nodes)


@pytest.fixture
def toy_nx_graph():
    G = nx.DiGraph()
    input_nodes = ["in0", "in1"]
    function_nodes = ["func0", "func1"]
    output_nodes = ["out0"]
    G.add_edges_from([("in0", "func0"), ("in1", "func1"), ("func0", "func1"), ("func1", "out0")])
    return G, input_nodes, function_nodes, output_nodes


class DummyData:
    """Minimal graph metadata container for explainers and extract_entity_function."""

    def __init__(self, node_names_dict):
        self.node_names_dict = node_names_dict


@pytest.fixture
def tiny_data(tiny_node_names_dict):
    return DummyData(tiny_node_names_dict)


@pytest.fixture
def minimal_data(minimal_node_names_dict):
    return DummyData(minimal_node_names_dict)


@pytest.fixture
def tiny_gsnn_trained(tiny_gsnn, tiny_edge_index_dict, tiny_node_names_dict):
    """Few-step trained model for explainer smoke tests."""
    model = GSNN(
        edge_index_dict=tiny_edge_index_dict,
        node_names_dict=tiny_node_names_dict,
        channels=4,
        layers=1,
        norm="none",
        share_layers=False,
        add_function_self_edges=True,
        node_mlp=False,
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(16, 1)
    y = torch.randn(16, 1)
    for _ in range(8):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model
