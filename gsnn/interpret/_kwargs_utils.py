"""Internal helpers for reshaping ``model_kwargs`` inside the explainers.

Some explainers (``IGExplainer``, ``OcclusionExplainer``, the contrastive
variants) reshape the input ``x`` before calling the underlying model — they
slice per-sample, replicate across mask perturbations, etc. When the model
also expects per-sample side inputs (e.g. ``x_fn`` for GSNNs trained with
``node_activity=True``), those tensors must be reshaped in lock-step with
``x``.  These helpers centralize that bookkeeping so every explainer applies
the same convention:

* The user passes per-sample tensors whose leading dim matches ``x``'s
  leading dim (``B``) -- or 1, which broadcasts to any batch size.
* :func:`slice_per_sample` extracts the i-th row when the explainer loops
  per sample.
* :func:`repeat_batch` expands an already-sliced tensor across replicated
  forward calls (e.g. the ``n_steps+1`` IG path or the ``BB`` occluded-edge
  batch).
* :func:`tile_for_grid` expands an unsliced ``(B, ...)`` tensor across a
  ``(BB, B, ...)`` grid (used by ``OcclusionExplainer`` which forwards all
  ``BB*B`` perturbed inputs in one call).

When ``model_kwargs`` is ``None`` (or empty) all helpers are no-ops and the
explainers' call sites remain byte-identical to the legacy code path.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch


def normalize_model_kwargs(model_kwargs: Optional[Mapping]) -> dict:
    """Return a fresh dict (possibly empty) suitable for **kwargs splatting."""
    return {} if model_kwargs is None else dict(model_kwargs)


def slice_per_sample(model_kwargs: Optional[Mapping], i: int) -> dict:
    """Pick row ``i`` from every batched tensor value.

    Non-tensor values, scalars, and tensors whose leading dim is 1 are
    passed through unchanged (the latter broadcast naturally).
    """
    if not model_kwargs:
        return {}
    out = {}
    for k, v in model_kwargs.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) > 1:
            out[k] = v[i : i + 1]
        else:
            out[k] = v
    return out


def repeat_batch(model_kwargs: Optional[Mapping], n: int) -> dict:
    """Expand the leading dim of every tensor value to ``n``.

    * Leading dim ``1`` → repeated ``n`` times.
    * Leading dim ``n`` → passed through.
    * Any other leading dim raises ``ValueError`` (caller bug).
    """
    if not model_kwargs:
        return {}
    out = {}
    for k, v in model_kwargs.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            if v.size(0) == n:
                out[k] = v
            elif v.size(0) == 1:
                shape = [n] + [1] * (v.dim() - 1)
                out[k] = v.repeat(*shape)
            else:
                raise ValueError(
                    f"repeat_batch: tensor '{k}' has leading dim {v.size(0)} "
                    f"which is neither 1 nor the target {n}."
                )
        else:
            out[k] = v
    return out


def tile_for_grid(model_kwargs: Optional[Mapping], outer: int, inner_B: int) -> dict:
    """Tile a ``(B=inner_B, ...)`` tensor into a flat ``(outer*B, ...)`` batch.

    Mirrors the ``x_batch = x.unsqueeze(0).repeat(outer, 1, 1).view(-1, ...)``
    pattern that :class:`OcclusionExplainer` uses to broadcast a per-edge
    perturbation grid over all input samples.
    """
    if not model_kwargs:
        return {}
    out = {}
    for k, v in model_kwargs.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            if v.size(0) == 1:
                v = v.expand(inner_B, *v.shape[1:])
            elif v.size(0) != inner_B:
                raise ValueError(
                    f"tile_for_grid: tensor '{k}' has leading dim {v.size(0)} "
                    f"which is neither 1 nor inner batch size {inner_B}."
                )
            # (B, ...) -> (outer, B, ...) -> (outer*B, ...)
            tiled = v.unsqueeze(0).expand(outer, *v.shape).reshape(outer * inner_B, *v.shape[1:])
            out[k] = tiled.contiguous()
        else:
            out[k] = v
    return out


def concat_pair(left: Optional[Mapping], right: Optional[Mapping]) -> dict:
    """Concatenate matching tensor values from ``left`` and ``right`` along dim 0.

    Used by the contrastive explainers that issue a single joint
    ``model(cat([x1, x2], dim=0))`` call.  Non-tensor entries are required to
    agree (and the right value wins on equality).  Tensor entries are
    concatenated; keys present in only one side are passed through as-is.
    """
    if not left and not right:
        return {}
    if not left:
        return dict(right)
    if not right:
        return dict(left)
    out = {}
    keys = set(left.keys()) | set(right.keys())
    for k in keys:
        lv = left.get(k)
        rv = right.get(k)
        if lv is None:
            out[k] = rv
            continue
        if rv is None:
            out[k] = lv
            continue
        if isinstance(lv, torch.Tensor) and isinstance(rv, torch.Tensor):
            out[k] = torch.cat([lv, rv], dim=0)
        else:
            out[k] = rv
    return out
