"""Tests for miscellaneous interpret modules."""

import pandas as pd
import pytest
import torch

from gsnn.interpret.CounterfactualExplainer import CounterfactualExplainer
from gsnn.interpret.NoiseTunnel import NoiseTunnel


def test_counterfactual_explainer(tiny_gsnn_trained, tiny_data):
    model = tiny_gsnn_trained
    x = torch.randn(1, 1)
    explainer = CounterfactualExplainer(model, tiny_data, ignore_cuda=True)
    target = model(x).detach()[0, 0].item() + 0.5
    df = explainer.explain(
        x.squeeze(0), target_value=target, target_idx=0, max_iter=20, verbose=False
    )
    assert "perturbation" in df.columns
    assert "counterfactual" in df.columns


def test_noise_tunnel_wraps_explainer(tiny_gsnn_trained, tiny_data):
    from gsnn.interpret.IGExplainer import IGExplainer

    base = IGExplainer(tiny_gsnn_trained, tiny_data, ignore_cuda=True, n_steps=3)
    tunnel = NoiseTunnel(base, n_samples=2, noise_std=0.01)
    x = torch.randn(1, 1)
    df = tunnel.explain(x.squeeze(0), target_idx=0, target="edge")
    assert isinstance(df, pd.DataFrame)
    assert "score" in df.columns


def test_plot_edge_importance_smoke(tmp_path):
    pytest.importorskip("pydot")
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from gsnn.interpret import utils as iutils

    res = pd.DataFrame(
        {"source": ["a", "b"], "target": ["c", "d"], "score": [0.5, -0.5]}
    )
    save = tmp_path / "edge.png"
    iutils.plot_edge_importance(res, title="test", save=str(save))
    assert save.exists()


def test_plot_node_importance_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from gsnn.interpret import utils as iutils

    G = nx.DiGraph()
    G.add_edges_from([("a", "b"), ("b", "c")])
    res = pd.DataFrame({"node": ["a", "b", "c"], "score": [0.1, 0.5, 0.9]})
    pos = {"a": (0, 0), "b": (1, 0), "c": (2, 0)}
    iutils.plot_node_importance(res, G, pos=pos, title="test")
    save = tmp_path / "node.png"
    plt.gcf().savefig(save)
    plt.close()
    assert save.exists()
