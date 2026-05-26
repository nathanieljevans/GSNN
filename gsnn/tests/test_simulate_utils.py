"""Tests for gsnn.simulate.utils."""

import networkx as nx
import pytest
import torch

from gsnn.simulate.utils import nx_to_pyro_model


def _linear_dag():
    G = nx.DiGraph()
    G.add_edges_from([("in0", "func0"), ("func0", "out0")])
    return G, ["in0"], ["out0"]


def test_nx_to_pyro_model_samples():
    G, inputs, outputs = _linear_dag()
    model = nx_to_pyro_model(G, inputs, outputs, noise_scale=0.01)
    x_values = {"in0": torch.tensor(1.0)}
    with torch.no_grad():
        out = model(x_values)
    assert "out0" in out
    assert isinstance(out["out0"], torch.Tensor)


def test_nx_to_pyro_special_functions():
    G, inputs, outputs = _linear_dag()
    special = {"func0": lambda parents: sum(parents) ** 2}
    model = nx_to_pyro_model(G, inputs, outputs, special_functions=special, noise_scale=0.01)
    out = model({"in0": torch.tensor(2.0)})
    assert out["out0"].item() == pytest.approx(4.0, abs=0.5)


def test_nx_to_pyro_signed_edges():
    G, inputs, outputs = _linear_dag()
    signed = {("in0", "func0"): -1}
    model = nx_to_pyro_model(G, inputs, outputs, signed_edges=signed, noise_scale=0.01)
    out = model({"in0": torch.tensor(3.0)})
    assert out["out0"].item() == pytest.approx(-3.0, abs=0.5)
