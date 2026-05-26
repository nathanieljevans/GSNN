"""Tests for gsnn.interpret._kwargs_utils."""

import pytest
import torch

from gsnn.interpret._kwargs_utils import (
    concat_pair,
    normalize_model_kwargs,
    repeat_batch,
    slice_per_sample,
    tile_for_grid,
)


def test_normalize_model_kwargs_none():
    assert normalize_model_kwargs(None) == {}


def test_normalize_model_kwargs_none():
    assert normalize_model_kwargs(None) == {}
    assert normalize_model_kwargs({"a": 1}) == {"a": 1}


def test_slice_per_sample():
    kw = {"x_fn": torch.arange(6).view(3, 2).float(), "scalar": 1}
    out = slice_per_sample(kw, 1)
    assert out["x_fn"].shape == (1, 2)
    assert out["scalar"] == 1


def test_repeat_batch_from_1():
    kw = {"x_fn": torch.ones(1, 4)}
    out = repeat_batch(kw, 3)
    assert out["x_fn"].shape == (3, 4)


def test_repeat_batch_passthrough():
    kw = {"x_fn": torch.ones(3, 4)}
    out = repeat_batch(kw, 3)
    assert out["x_fn"].shape == (3, 4)


def test_repeat_batch_invalid_dim():
    with pytest.raises(ValueError, match="repeat_batch"):
        repeat_batch({"x_fn": torch.ones(2, 4)}, 3)


def test_tile_for_grid():
    kw = {"x_fn": torch.arange(4).view(2, 2).float()}
    out = tile_for_grid(kw, outer=3, inner_B=2)
    assert out["x_fn"].shape == (6, 2)


def test_concat_pair():
    left = {"x_fn": torch.zeros(2, 3)}
    right = {"x_fn": torch.ones(2, 3)}
    out = concat_pair(left, right)
    assert out["x_fn"].shape == (4, 3)
