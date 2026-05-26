"""Tests for contrastive interpret explainers."""

import torch

from gsnn.interpret.ContrastiveGSNNExplainer import ContrastiveGSNNExplainer
from gsnn.interpret.ContrastiveIGExplainer import ContrastiveIGExplainer
from gsnn.interpret.ContrastiveOcclusionExplainer import ContrastiveOcclusionExplainer


def test_contrastive_gsnn_explainer_edges(tiny_gsnn_trained, tiny_data):
    model = tiny_gsnn_trained
    x1 = torch.randn(1, 1)
    x2 = torch.randn(1, 1)
    explainer = ContrastiveGSNNExplainer(
        model, tiny_data, ignore_cuda=True, iters=5, verbose=False
    )
    df = explainer.explain(x1, x2, target_idx=[0], target="edge")
    assert "score" in df.columns
    assert len(df) == model.E


def test_contrastive_gsnn_explainer_nodes(tiny_gsnn_trained, tiny_data):
    model = tiny_gsnn_trained
    x1 = torch.randn(1, 1)
    x2 = torch.randn(1, 1)
    explainer = ContrastiveGSNNExplainer(
        model, tiny_data, ignore_cuda=True, iters=5, verbose=False
    )
    df = explainer.explain(x1, x2, target_idx=[0], target="node")
    assert "node" in df.columns


def test_contrastive_ig_explainer(tiny_gsnn_trained, tiny_data):
    model = tiny_gsnn_trained
    x1 = torch.randn(1, 1)
    x2 = torch.randn(1, 1)
    explainer = ContrastiveIGExplainer(model, tiny_data, ignore_cuda=True, n_steps=4)
    df = explainer.explain(x1, x2, target_idx=[0], target="edge")
    assert "score" in df.columns


def test_contrastive_occlusion_explainer(tiny_gsnn_trained, tiny_data):
    model = tiny_gsnn_trained
    x1 = torch.randn(1, 1)
    x2 = torch.randn(1, 1)
    explainer = ContrastiveOcclusionExplainer(
        model, tiny_data, ignore_cuda=True, batch_size=4
    )
    df = explainer.explain(x1, x2, target_idx=[0], target="edge")
    assert "score" in df.columns
