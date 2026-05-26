"""Tests for gsnn.optim.RewardScaler."""

import numpy as np

from gsnn.optim.RewardScaler import RewardScaler, edw


def test_edw_weights():
    w = edw(5, alpha=0.5)
    assert len(w) == 5
    assert np.all(w >= 0)


def test_reward_scaler_warmup():
    rs = RewardScaler(warmup=3)
    for r in [1.0, 2.0, 3.0]:
        rs.update(r)
    params = rs.get_params()
    assert "mean" in params or params is not None


def test_reward_scaler_scale_clip():
    rs = RewardScaler(clip=2, warmup=1)
    rs.update(0.0)
    scaled = rs.scale(100.0)
    assert abs(scaled) <= 2.0 + 1e-6
