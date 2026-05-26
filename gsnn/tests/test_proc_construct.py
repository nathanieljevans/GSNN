"""Tests for gsnn.proc.construct."""

import pandas as pd

from gsnn.proc.construct import GSNNNetworkConstructor


def _edge_tables():
    input_edges = pd.DataFrame({"src": ["A"], "dst": ["X"]})
    function_edges = pd.DataFrame({"src": ["X"], "dst": ["Y"]})
    output_edges = pd.DataFrame({"src": ["Y"], "dst": ["O"]})
    return input_edges, function_edges, output_edges


def test_gsnn_network_constructor_build():
    builder = GSNNNetworkConstructor(depth=5, verbose=False)
    data = builder.build(*_edge_tables())
    assert "input" in data.node_names_dict
    assert ("input", "to", "function") in data.edge_index_dict


def test_constructor_force_include_names():
    builder = GSNNNetworkConstructor(depth=5, verbose=False)
    data = builder.build(
        *_edge_tables(),
        input_names=["A", "EXTRA_IN"],
        function_names=["X", "Y", "ORPHAN"],
        output_names=["O"],
    )
    assert "EXTRA_IN" in data.node_names_dict["input"]
    assert "ORPHAN" in data.node_names_dict["function"]


def test_constructor_prunes_unreachable():
    import pandas as pd

    builder = GSNNNetworkConstructor(depth=5, verbose=False)
    input_edges = pd.DataFrame({"src": ["A"], "dst": ["X"]})
    function_edges = pd.DataFrame({"src": ["X"], "dst": ["Y"]})
    output_edges = pd.DataFrame({"src": ["Y"], "dst": ["O"]})
    # orphan function node with no path to output
    function_edges = pd.concat(
        [function_edges, pd.DataFrame({"src": ["Z"], "dst": ["W"]})], ignore_index=True
    )
    data = builder.build(input_edges, function_edges, output_edges)
    assert "Z" not in data.node_names_dict["function"] or "W" not in data.node_names_dict["function"]


def test_constructor_graph_summary():
    builder = GSNNNetworkConstructor(depth=5, verbose=False)
    data = builder.build(*_edge_tables())
    assert hasattr(data, "graph_summary")
    assert "function" in data.graph_summary or isinstance(data.graph_summary, dict)
