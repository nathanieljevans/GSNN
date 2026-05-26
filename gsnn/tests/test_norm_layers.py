"""Tests for normalization layers."""

import pytest
import torch

from gsnn.models.ChannelEMANorm import ChannelEMANorm
from gsnn.models.GroupBatchNorm import GroupBatchNorm
from gsnn.models.GroupEMANorm import GroupEMANorm
from gsnn.models.GroupLayerNorm import GroupLayerNorm
from gsnn.models.GroupRMSNorm import GroupRMSNorm
from gsnn.models.SoftmaxGroupNorm import SoftmaxGroupNorm


CHANNEL_GROUPS = [0, 0, 1, 1]


def test_group_batch_norm_forward_shape():
    norm = GroupBatchNorm(CHANNEL_GROUPS, affine=True)
    x = torch.randn(4, 4, 1)
    out = norm(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_group_rms_norm_forward_shape():
    norm = GroupRMSNorm(CHANNEL_GROUPS, affine=True)
    x = torch.randn(4, 4, 1)
    out = norm(x)
    assert out.shape == x.shape


def test_softmax_group_norm_weights_sum():
    norm = SoftmaxGroupNorm(CHANNEL_GROUPS)
    x = torch.randn(2, 4, 1)
    out = norm(x).squeeze(-1)
    # each group softmax over its channels
    g0 = out[0, :2]
    assert torch.allclose(g0.sum(), torch.tensor(1.0), atol=1e-5)


def test_group_ema_norm_train_eval():
    norm = GroupEMANorm(CHANNEL_GROUPS, affine=True)
    x = torch.randn(8, 4, 1)
    norm.train()
    out_train = norm(x)
    norm.eval()
    out_eval = norm(x)
    assert out_train.shape == out_eval.shape == x.shape


def test_channel_ema_norm_forward():
    norm = ChannelEMANorm(4, affine=True)
    x = torch.randn(3, 4, 1)
    out = norm(x)
    assert out.shape == x.shape


def test_group_layer_norm_forward_shape():
    norm = GroupLayerNorm(CHANNEL_GROUPS)
    x = torch.randn(4, 4, 1)
    out = norm(x)
    assert out.shape == x.shape
