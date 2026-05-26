"""Tests for gsnn.models.NodeMLP."""

import torch
import torch.nn as nn

from gsnn.models.NodeMLP import NodeMLP


def test_node_mlp_forward_shape():
    mlp = NodeMLP(in_features=4, hidden_features=8, nonlin=nn.ELU, dropout=0.0)
    x = torch.randn(3, 2, 4)
    out = mlp(x)
    assert out.shape == x.shape


def test_node_mlp_dropout_train_eval():
    mlp = NodeMLP(in_features=4, hidden_features=8, nonlin=nn.ELU, dropout=0.5)
    x = torch.randn(10, 2, 4)
    mlp.eval()
    out1 = mlp(x)
    out2 = mlp(x)
    assert torch.allclose(out1, out2)
