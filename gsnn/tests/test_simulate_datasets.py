"""Tests for gsnn.simulate.datasets."""

import networkx as nx
import pytest
import torch

from gsnn.models.GSNN import GSNN
from gsnn.simulate.datasets import simulate_10_in_25_func_10_out_cyclic, simulate_3_in_3_out
from gsnn.simulate.nx2pyg import nx2pyg


def test_simulate_3_in_3_out():
    G, pos, x_train, x_test, y_train, y_test, inp, func, out = simulate_3_in_3_out(
        n_train=16, n_test=8, device="cpu"
    )
    assert len(inp) == 3
    assert len(out) == 3
    assert x_train.shape == (16, 3)
    assert y_test.shape == (8, 3)
    assert isinstance(G, nx.DiGraph)
    assert isinstance(pos, dict)


def test_simulate_3_in_3_out_zscorey():
    _, _, _, _, y_tr, y_te, _, _, _ = simulate_3_in_3_out(
        n_train=32, n_test=8, zscorey=True, device="cpu"
    )
    assert y_tr.std(dim=0).mean().item() == pytest.approx(1.0, abs=0.3)


@pytest.mark.slow
@pytest.mark.skip(reason="simulate_sde requires acyclic graph (topological_sort)")
def test_simulate_10_in_25_func_10_out_cyclic():
    G, pos, x_train, x_test, y_train, y_test, inp, func, out = (
        simulate_10_in_25_func_10_out_cyclic(
            n_train=8, n_test=4, device="cpu", seed=0, t_final=1.0, dt=0.1
        )
    )
    assert len(inp) == 10
    assert len(func) == 25
    assert len(out) == 10
    assert x_train.shape == (8, 10)
    assert y_test.shape == (4, 10)


@pytest.mark.slow
def test_gsnn_trainable_on_simulated_data():
    G, _, x_train, _, y_train, _, input_nodes, function_nodes, output_nodes = (
        simulate_3_in_3_out(n_train=32, n_test=8, device="cpu")
    )
    data = nx2pyg(G, input_nodes, function_nodes, output_nodes)
    model = GSNN(
        edge_index_dict=data.edge_index_dict,
        node_names_dict=data.node_names_dict,
        channels=4,
        layers=1,
        norm="none",
        share_layers=False,
        node_mlp=False,
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = x_train[:16]
    y = y_train[:16]
    loss_before = ((model(x) - y) ** 2).mean().item()
    for _ in range(5):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    loss_after = ((model(x) - y) ** 2).mean().item()
    assert loss_after <= loss_before
