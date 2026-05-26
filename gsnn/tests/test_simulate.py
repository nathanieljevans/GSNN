"""Tests for gsnn.simulate.simulate."""

import networkx as nx
import numpy as np
import pytest

from gsnn.simulate.simulate import simulate, simulate_sde


def _simple_dag():
    G = nx.DiGraph()
    input_nodes = ["in0"]
    output_nodes = ["out0"]
    G.add_edges_from([("in0", "func0"), ("func0", "out0")])
    return G, input_nodes, output_nodes


def test_simulate_shapes():
    G, input_nodes, output_nodes = _simple_dag()
    x_train, x_test, y_train, y_test = simulate(
        G, n_train=10, n_test=5, input_nodes=input_nodes, output_nodes=output_nodes
    )
    assert x_train.shape == (10, 1)
    assert x_test.shape == (5, 1)
    assert y_train.shape == (10, 1)
    assert y_test.shape == (5, 1)


def test_simulate_sde_shapes():
    G, input_nodes, output_nodes = _simple_dag()
    result = simulate_sde(
        G,
        n_train=8,
        n_test=4,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        seed=42,
    )
    assert len(result) == 4
    x_train, y_train, x_test, y_test = result
    assert x_train.shape == (8, 1)
    assert y_train.shape == (8, 1)
    assert x_test.shape == (4, 1)
    assert y_test.shape == (4, 1)


def test_simulate_sde_seed():
    G, input_nodes, output_nodes = _simple_dag()
    r1 = simulate_sde(
        G, 5, 3, input_nodes, output_nodes, seed=123, dt=0.1, t_final=1.0
    )
    r2 = simulate_sde(
        G, 5, 3, input_nodes, output_nodes, seed=123, dt=0.1, t_final=1.0
    )
    assert np.allclose(r1[0], r2[0])


def test_simulate_sde_return_order():
    """Document actual return order: (x_train, y_train, x_test, y_test)."""
    G, input_nodes, output_nodes = _simple_dag()
    x_train, y_train, x_test, y_test = simulate_sde(
        G, 4, 2, input_nodes, output_nodes, seed=0
    )
    assert x_train.shape[1] == len(input_nodes)
    assert y_train.shape[1] == len(output_nodes)
    assert x_test.shape[0] == 2
    assert y_test.shape[0] == 2

