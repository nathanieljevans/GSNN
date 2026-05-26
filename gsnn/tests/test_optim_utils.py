"""Tests for gsnn.optim.utils."""

import torch

from gsnn.optim.utils import compute_ECE, compute_picp


def test_compute_picp():
    dist = torch.distributions.Normal(torch.zeros(100), torch.ones(100))
    y_true = torch.zeros(100)
    picp = compute_picp(dist, y_true, alpha=0.05, N=200)
    assert 0.0 <= picp <= 1.0


def test_compute_ECE():
    dist = torch.distributions.Normal(torch.zeros(50), torch.ones(50))
    y_true = torch.randn(50)
    ece = compute_ECE(dist, y_true, num_intervals=5)
    assert 0.0 <= ece <= 1.0
