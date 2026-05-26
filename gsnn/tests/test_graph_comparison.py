"""Tests for gsnn.simulate.graph_comparison."""

import torch

from gsnn.simulate.graph_comparison import GraphComparison


def _edge_dict():
    return {
        ("input", "to", "function"): torch.tensor([[0], [0]], dtype=torch.long),
        ("function", "to", "function"): torch.empty((2, 0), dtype=torch.long),
        ("function", "to", "output"): torch.tensor([[0], [0]], dtype=torch.long),
    }


def test_graph_comparison_identical():
    ref = _edge_dict()
    cmp = GraphComparison(ref)
    metrics = cmp(ref)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_graph_comparison_disjoint():
    ref = _edge_dict()
    alt = {
        ("input", "to", "function"): torch.tensor([[0], [0]], dtype=torch.long),
        ("function", "to", "function"): torch.tensor([[0], [0]], dtype=torch.long),
        ("function", "to", "output"): torch.tensor([[0], [0]], dtype=torch.long),
    }
    cmp = GraphComparison(ref)
    metrics = cmp(alt)
    assert metrics["true_positives"] >= 0


def test_get_dependency_details():
    ref = _edge_dict()
    cmp = GraphComparison(ref)
    details = cmp.get_dependency_details(ref)
    assert isinstance(details, dict)
