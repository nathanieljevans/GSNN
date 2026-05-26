"""Tests for training diagnostics utilities."""

import torch
import torch.nn as nn

from gsnn.models.GSNN import GSNN
from gsnn.optim.GradDiagnostics import GradDiagnostics


def test_grad_diagnostics_analyze(minimal_gsnn):
    model = minimal_gsnn
    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    diag = GradDiagnostics(verbose=False)
    summary = diag.analyze(model)
    assert "per_layer" in summary
    assert "overall" in summary


def test_grad_diagnostics_update_reset(minimal_gsnn):
    model = minimal_gsnn
    diag = GradDiagnostics(verbose=False)
    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    diag.update(model, loss.item(), step=0)
    assert len(diag.history) == 1
    summary = diag.get_summary()
    assert summary is not None
    diag.reset()
    assert len(diag.history) == 0


def test_training_diagnostics_smoke(minimal_gsnn):
    from gsnn.optim.TrainingDiagnostics import TrainingDiagnostics

    diag = TrainingDiagnostics(minimal_gsnn, track_every=1, verbose=False)
    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    loss = ((minimal_gsnn(x) - y) ** 2).mean()
    loss.backward()
    diag.update(minimal_gsnn, loss.item(), step=0)
    summary = diag.get_summary()
    assert isinstance(summary, dict)
    diag.reset()
