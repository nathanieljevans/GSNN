"""Extended tests for extract_entity_function."""

import pytest
import torch

from gsnn.interpret.extract_entity_function import dense_func_node, extract_entity_function
from gsnn.models.GSNN import GSNN
from gsnn.tests.helpers import EXTRACT_ENTITY_NORMS


def build_dummy_graph():
    edge_index_dict = {
        ("input", "to", "function"): torch.tensor([[0], [0]], dtype=torch.long),
        ("function", "to", "function"): torch.empty((2, 0), dtype=torch.long),
        ("function", "to", "output"): torch.tensor([[0], [0]], dtype=torch.long),
    }
    node_names_dict = {
        "input": ["inp"],
        "function": ["func"],
        "output": ["out"],
    }
    return edge_index_dict, node_names_dict


@pytest.mark.parametrize("norm", EXTRACT_ENTITY_NORMS)
def test_extract_entity_function_runs(norm):
    edge_index_dict, node_names_dict = build_dummy_graph()
    model = GSNN(
        edge_index_dict=edge_index_dict,
        node_names_dict=node_names_dict,
        channels=4,
        layers=1,
        norm=norm,
        share_layers=False,
        add_function_self_edges=True,
        node_mlp=False,
    )

    class DummyData:
        pass

    data = DummyData()
    data.node_names_dict = node_names_dict

    func, meta = extract_entity_function("func", model, data, layer=0)
    batch = 3
    x = torch.randn(batch, len(meta["input_edge_names"]))
    out = func(x)
    assert out.shape == (batch, len(meta["output_edge_names"]))


def test_extract_invalid_node(tiny_gsnn, tiny_data):
    with pytest.raises(ValueError):
        extract_entity_function("missing", tiny_gsnn, tiny_data, layer=0)


def test_extract_layer_index(tiny_edge_index_dict, tiny_node_names_dict, tiny_data):
    model = GSNN(
        edge_index_dict=tiny_edge_index_dict,
        node_names_dict=tiny_node_names_dict,
        channels=4,
        layers=2,
        norm="none",
        share_layers=False,
        node_mlp=False,
    )
    func0, _ = extract_entity_function("func", model, tiny_data, layer=0)
    func1, _ = extract_entity_function("func", model, tiny_data, layer=1)
    w0 = func0.lin_in.weight.data.clone()
    w1 = func1.lin_in.weight.data.clone()
    assert not torch.allclose(w0, w1)


def test_dense_func_node_forward():
    lin_in = torch.nn.Linear(2, 4)
    lin_out = torch.nn.Linear(4, 1)
    mod = dense_func_node(lin_in, lin_out, torch.nn.ELU(), norm="none")
    x = torch.randn(3, 2)
    out = mod(x)
    assert out.shape == (3, 1)


def test_extract_unsupported_norm_raises(tiny_data):
    edge_index_dict, node_names_dict = build_dummy_graph()
    model = GSNN(
        edge_index_dict=edge_index_dict,
        node_names_dict=node_names_dict,
        channels=4,
        layers=1,
        norm="groupbatch",
        share_layers=False,
        node_mlp=False,
    )
    with pytest.raises(NotImplementedError):
        extract_entity_function("func", model, tiny_data, layer=0)
