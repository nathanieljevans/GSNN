"""Tests for gsnn.models.NN baseline."""

import torch
import torch.nn as nn

from gsnn.models.NN import NN


def test_nn_forward_shape():
    model = NN(in_channels=5, hidden_channels=8, out_channels=3, layers=2)
    x = torch.randn(4, 5)
    out = model(x)
    assert out.shape == (4, 3)


def test_nn_layers_depth():
    shallow = NN(in_channels=4, hidden_channels=8, out_channels=2, layers=1)
    deep = NN(in_channels=4, hidden_channels=8, out_channels=2, layers=3)
    assert sum(p.numel() for p in deep.parameters()) > sum(p.numel() for p in shallow.parameters())


def test_nn_no_norm():
    model = NN(in_channels=4, hidden_channels=8, out_channels=2, layers=2, norm=None)
    x = torch.randn(2, 4)
    out = model(x)
    assert out.shape == (2, 2)
