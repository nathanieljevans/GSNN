"""Tests for gsnn.proc.coarsen."""

import networkx as nx

from gsnn.proc.coarsen import diff_equivalence, diff_io_equivalence, io_equivalence


def _toy_graph():
    G = nx.DiGraph()
    G.add_edges_from([("in0", "f0"), ("in1", "f1"), ("f0", "out0"), ("f1", "out0")])
    return G


def test_io_equivalence():
    G = _toy_graph()
    in_eq, out_eq = io_equivalence(
        G, input_nodes=["in0", "in1"], function_nodes=["f0", "f1"], output_nodes=["out0"]
    )
    assert in_eq.shape == (2, 2)
    assert out_eq.shape == (2, 2)


def test_diff_equivalence():
    G = _toy_graph()
    scores = diff_equivalence(G, sources=["in0", "in1"], nodes=["f0", "f1"], iters=3)
    assert scores.shape == (2, 2)


def test_diff_io_equivalence():
    G = _toy_graph()
    scores = diff_io_equivalence(
        G,
        input_nodes=["in0", "in1"],
        function_nodes=["f0", "f1"],
        output_nodes=["out0"],
        iters=3,
    )
    assert scores.shape == (2, 3)
