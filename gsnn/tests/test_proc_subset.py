"""Tests for gsnn.proc.subset."""

import networkx as nx
import numpy as np
import pytest

from gsnn.proc.subset import bfs_distance, build_nx, get_all_possible_paths_set, subset_graph


def test_bfs_distance():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (0, 3), (3, 2)])
    node_names = [0, 1, 2, 3]
    dist = bfs_distance(G, 0, depth=3, node_names=node_names)
    assert dist[0] == 0
    assert dist[1] == 1
    assert dist[2] == 2
    assert dist[3] == 1


def test_get_all_possible_paths_set():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (0, 3), (3, 2)])
    rG = G.reverse()
    node_names = [0, 1, 2, 3]
    spl, _, _ = get_all_possible_paths_set(G, rG, 0, 2, 3, {}, {}, node_names)
    assert spl[2] == pytest.approx(2.0)
    assert spl[1] == 2


def test_subset_graph_keeps_viable_paths():
    G = nx.DiGraph()
    G.add_edges_from([("in", "f"), ("f", "out"), ("dead", "nowhere")])
    sub = subset_graph(G, depth=5, roots=["in"], leafs=["out"], verbose=False)
    assert "f" in sub.nodes
    assert "in" in sub.nodes
    assert "out" in sub.nodes


def test_subset_graph_prunes_dead_ends():
    G = nx.DiGraph()
    G.add_edges_from([("in", "f"), ("f", "out"), ("dead", "nowhere")])
    sub = subset_graph(G, depth=5, roots=["in"], leafs=["out"], verbose=False)
    assert "dead" not in sub.nodes
    assert "nowhere" not in sub.nodes


def test_build_nx():
    import pandas as pd

    func_df = pd.DataFrame(
        {"source": ["PROTEIN__a"], "target": ["PROTEIN__b"]}
    )
    targets = pd.DataFrame({"pert_id": ["drug1"], "target": ["a"]})
    outputs = ["gene1"]
    G = build_nx(func_df, targets, outputs)
    assert G.number_of_edges() >= 2
