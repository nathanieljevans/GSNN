"""Tests for gsnn.optim.EarlyStopper."""

from gsnn.optim.EarlyStopper import EarlyStopper


def test_early_stop_improvement_resets():
    es = EarlyStopper(patience=2, min_delta=0.0)
    assert es.early_stop(1.0) is False
    assert es.counter == 0
    assert es.early_stop(0.5) is False
    assert es.counter == 0


def test_early_stop_triggers():
    es = EarlyStopper(patience=2, min_delta=0.0)
    es.early_stop(1.0)
    assert es.early_stop(1.0) is False
    assert es.early_stop(1.0) is False
    assert es.early_stop(1.0) is True


def test_early_stop_min_delta():
    es = EarlyStopper(patience=1, min_delta=0.1)
    es.early_stop(1.0)
    # improvement of 0.05 is below min_delta
    assert es.early_stop(0.95) is False
    assert es.counter == 1
