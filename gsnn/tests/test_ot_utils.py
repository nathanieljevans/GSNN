"""Tests for gsnn.ot MMD utilities."""

import pytest
import torch

from gsnn.ot import mmd as ot_mmd


def test_mmd_distance():
    x = torch.randn(20, 4)
    y = torch.randn(20, 4)
    d = ot_mmd.mmd_distance(x, y, gamma=1.0)
    assert d.item() >= 0


def test_mmd_identical_samples():
    x = torch.randn(10, 3)
    d = ot_mmd.mmd_distance(x, x, gamma=1.0)
    assert d.item() == pytest.approx(0.0, abs=1e-5)


def test_compute_scalar_mmd():
    target = torch.randn(15, 2)
    transport = torch.randn(15, 2)
    val = ot_mmd.compute_scalar_mmd(target, transport)
    assert isinstance(val, float)
