import torch
import pytest

from gsnn.models.GSNN import GSNN
from gsnn.interpret.extract_entity_function import extract_entity_function


def build_dummy_graph():
    edge_index_dict = {
        ('input', 'to', 'function'): torch.tensor([[0], [0]], dtype=torch.long),
        ('function', 'to', 'function'): torch.empty((2, 0), dtype=torch.long),
        ('function', 'to', 'output'): torch.tensor([[0], [0]], dtype=torch.long),
    }
    node_names_dict = {
        'input': ['inp'],
        'function': ['func'],
        'output': ['out'],
    }
    return edge_index_dict, node_names_dict


@pytest.mark.parametrize("norm", ["layer", "batch", "none"])
def test_extract_entity_function_runs(norm):
    edge_index_dict, node_names_dict = build_dummy_graph()

    # Create a tiny GSNN model with the desired norm
    model = GSNN(
        edge_index_dict=edge_index_dict,
        node_names_dict=node_names_dict,
        channels=4,
        layers=1,
        norm=norm,
        share_layers=False,
        add_function_self_edges=True,
    )

    # Minimal mock for the `data` argument
    class DummyData:
        pass

    data = DummyData()
    data.node_names_dict = node_names_dict

    func, meta = extract_entity_function("func", model, data, layer=0)

    # Sanity-check shapes
    batch = 3
    x = torch.randn(batch, len(meta["input_edge_names"]))
    out = func(x)

    assert out.shape == (batch, len(meta["output_edge_names"])) 