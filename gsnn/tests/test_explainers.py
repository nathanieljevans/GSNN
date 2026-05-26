"""Tests for gsnn.interpret explainers."""

import pytest
import torch

from gsnn.interpret.GSNNExplainer import GSNNExplainer
from gsnn.interpret.IGExplainer import IGExplainer
from gsnn.interpret.OcclusionExplainer import OcclusionExplainer


@pytest.fixture
def explainer_setup(tiny_gsnn_trained, tiny_data):
    x = torch.randn(1, 1)
    return tiny_gsnn_trained, tiny_data, x


def test_gsnn_explainer_edge_scores(explainer_setup):
    model, data, x = explainer_setup
    explainer = GSNNExplainer(model, data, ignore_cuda=True, iters=5, verbose=False)
    df = explainer.explain(x, target_idx=[0], target="edge")
    assert "score" in df.columns
    assert len(df) == model.E


def test_gsnn_explainer_node_scores(explainer_setup):
    model, data, x = explainer_setup
    explainer = GSNNExplainer(model, data, ignore_cuda=True, iters=5, verbose=False)
    df = explainer.explain(x, target_idx=[0], target="node")
    assert "score" in df.columns
    assert "node" in df.columns


def test_gsnn_explainer_model_frozen(explainer_setup):
    model, data, x = explainer_setup
    before = {n: p.clone() for n, p in model.named_parameters()}
    explainer = GSNNExplainer(model, data, ignore_cuda=True, iters=3, verbose=False)
    explainer.explain(x, target_idx=[0], target="edge")
    for n, p in model.named_parameters():
        assert torch.allclose(before[n], p)


def test_gsnn_explainer_with_model_kwargs(tiny_edge_index_dict, tiny_node_names_dict, tiny_data):
    from gsnn.models.GSNN import GSNN

    model = GSNN(
        edge_index_dict=tiny_edge_index_dict,
        node_names_dict=tiny_node_names_dict,
        channels=4,
        layers=1,
        norm="none",
        node_activity=True,
        share_layers=False,
    )
    model.eval()
    x = torch.randn(1, 1)
    x_fn = torch.randn(1, 1)
    explainer = GSNNExplainer(model, tiny_data, ignore_cuda=True, iters=3, verbose=False)
    df = explainer.explain(x, target_idx=[0], target="edge", model_kwargs={"x_fn": x_fn})
    assert len(df) == model.E


def test_ig_explainer_edge_attributions(explainer_setup):
    model, data, x = explainer_setup
    explainer = IGExplainer(model, data, ignore_cuda=True, n_steps=4)
    df = explainer.explain(x.squeeze(0), target_idx=0, target="edge")
    assert "score" in df.columns
    assert len(df) == model.E


def test_ig_explainer_node_attributions(explainer_setup):
    model, data, x = explainer_setup
    explainer = IGExplainer(model, data, ignore_cuda=True, n_steps=4)
    df = explainer.explain(x.squeeze(0), target_idx=0, target="node")
    assert "score" in df.columns


def test_ig_explainer_baseline(explainer_setup):
    model, data, x = explainer_setup
    baseline = torch.ones(1, model.E)
    explainer = IGExplainer(model, data, ignore_cuda=True, n_steps=4, baseline=baseline)
    df = explainer.explain(x.squeeze(0), target_idx=0, target="edge")
    assert df["score"].notna().any()


def test_ig_explainer_model_kwargs(tiny_edge_index_dict, tiny_node_names_dict, tiny_data):
    from gsnn.models.GSNN import GSNN

    model = GSNN(
        edge_index_dict=tiny_edge_index_dict,
        node_names_dict=tiny_node_names_dict,
        channels=4,
        layers=1,
        norm="none",
        node_activity=True,
        share_layers=False,
    )
    model.eval()
    x = torch.randn(1, 1)
    x_fn = torch.randn(1, 1)
    explainer = IGExplainer(model, tiny_data, ignore_cuda=True, n_steps=3)
    df = explainer.explain(
        x.squeeze(0), target_idx=0, target="edge", model_kwargs={"x_fn": x_fn}
    )
    assert len(df) == model.E


def test_occlusion_edge_importance(explainer_setup):
    model, data, x = explainer_setup
    explainer = OcclusionExplainer(model, data, ignore_cuda=True, batch_size=8)
    df = explainer.explain(x.squeeze(0), target_idx=0, target="edge")
    assert "score" in df.columns


def test_occlusion_node_importance(explainer_setup):
    model, data, x = explainer_setup
    explainer = OcclusionExplainer(model, data, ignore_cuda=True, batch_size=8)
    df = explainer.explain(x.squeeze(0), target_idx=0, target="node")
    assert "score" in df.columns


def test_occlusion_element_mask(explainer_setup):
    model, data, x = explainer_setup
    mask = torch.zeros(model.E, dtype=torch.bool)
    mask[0] = True
    explainer = OcclusionExplainer(model, data, ignore_cuda=True, batch_size=4)
    df = explainer.explain(
        x.squeeze(0), target_idx=0, target="edge", element_mask=mask.numpy()
    )
    assert len(df) == model.E
