"""Tests for gsnn.simulate.nx2pyg."""

import networkx as nx
import torch

from gsnn.simulate.nx2pyg import nx2pyg, pyg2nx


def test_nx2pyg_edge_categorization(toy_nx_graph):
    G, input_nodes, function_nodes, output_nodes = toy_nx_graph
    data = nx2pyg(G, input_nodes, function_nodes, output_nodes)
    assert data.edge_index_dict[("input", "to", "function")].shape[1] == 2
    assert data.edge_index_dict[("function", "to", "function")].shape[1] == 1
    assert data.edge_index_dict[("function", "to", "output")].shape[1] == 1


def test_nx2pyg_with_weights(toy_nx_graph):
    G, input_nodes, function_nodes, output_nodes = toy_nx_graph
    for u, v in G.edges():
        G[u][v]["w"] = float(len(u))
    data = nx2pyg(G, input_nodes, function_nodes, output_nodes, weight_attr="w")
    assert hasattr(data, "edge_weight_dict")
    assert data.edge_weight_dict[("input", "to", "function")].numel() == 2


def test_pyg2nx_roundtrip(toy_nx_graph):
    G, input_nodes, function_nodes, output_nodes = toy_nx_graph
    data = nx2pyg(G, input_nodes, function_nodes, output_nodes)
    G2 = pyg2nx(data)
    assert set(G2.edges()) == set(G.edges())
