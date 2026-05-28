'''
Online Tier-0 edge inference via auxiliary linear regression during GSNN training.

For each adjacent layer pair (n-1, n) and source aggregator k, fit a shared
(N, N) weight matrix W so that activation magnitudes at layer n-1 predict
gradient magnitudes at layer n:

    Y_hat[:, j] = sum_i W[i, j] * Xtilde[:, i]

Magnitudes are taken from ``ResBlock._last_pre_norm_activation`` (post-``lin_in``,
pre-norm) and corresponding activation gradients, matching
``MagnitudeEdgeInferer`` information flow.

The regressor trains jointly with the GSNN (detached features, separate optimizer).
Held-out validation edges drive best-checkpoint selection, mitigating gradient
absorption at equilibrium.

See ``docs/notes/edge_inference_notes.md`` section 4 and tutorial 14.
'''

from __future__ import annotations

import copy
import math
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.special
import torch
import torch.nn as nn

from gsnn.optim.MagnitudeEdgeInferer import MagnitudeEdgeInferer


class MagnitudeEdgeRegressor(nn.Module):
    '''
    Online auxiliary linear regressor for function -> function edge inference.

    Learns a single shared weight matrix ``W`` of shape ``(N, N)`` during GSNN
    training. Source activations (layer n-1) predict target gradient magnitudes
    (layer n) across adjacent ResBlock pairs and multiple source aggregators.

    Parameters
    ----------
    model : GSNN
        Model being trained. Must have ``checkpoint=False``.
    data : HeteroData-like
        Graph container with ``node_names_dict`` and ``edge_index_dict``.
    aggregators : sequence of str
        Source-side channel reductions: ``'sum'``, ``'max'``, ``'mean'``, ``'l2'``.
        Target gradients always use L1 (sum of absolute values).
    use_pre_norm : bool
        If True (default), use post-``lin_in`` pre-norm activations.
    standardize : bool
        If True (default), EMA z-score features per (pair, aggregator).
    lr, weight_decay : float
        AdamW hyperparameters for ``W`` only.
    ridge : float
        Additional L2 penalty on ``W`` beyond ``weight_decay``.
    score_mode : {'abs', 'relu', 'signed'}
        How to convert ``W`` entries into edge scores for ranking.
    ema_momentum : float
        Momentum for running mean/variance updates during standardization.
    '''

    _AGG_FNS = {
        'sum': lambda v: v.abs().sum(dim=-1),
        'max': lambda v: v.abs().max(dim=-1).values,
        'mean': lambda v: v.abs().mean(dim=-1),
        'l2': lambda v: v.pow(2).sum(dim=-1).sqrt(),
    }

    def __init__(
        self,
        model,
        data,
        *,
        aggregators: Sequence[str] = ('sum', 'max'),
        use_pre_norm: bool = True,
        standardize: bool = True,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        dropout: float = 0.0,
        ridge: float = 0.0,
        score_mode: Literal['abs', 'relu', 'signed'] = 'abs',
        ema_momentum: float = 0.1,
    ):
        super().__init__()

        if getattr(model, 'checkpoint', False):
            raise ValueError(
                'MagnitudeEdgeRegressor requires model.checkpoint=False; '
                'gradient checkpointing recomputes activations during backward.'
            )

        if not use_pre_norm:
            norm = getattr(model, 'norm', None)
            if norm not in ('batch', 'groupbatch', 'none'):
                raise ValueError(
                    'MagnitudeEdgeRegressor with use_pre_norm=False requires '
                    'model.norm to be "batch", "groupbatch", or "none"; '
                    f'got {norm!r}. Use use_pre_norm=True (default) instead.'
                )

        for agg in aggregators:
            if agg not in self._AGG_FNS:
                raise ValueError(
                    f'Unknown aggregator {agg!r}. Use one of {tuple(self._AGG_FNS)}.'
                )

        self.__dict__['model'] = model
        self.data = data
        self.aggregators = tuple(aggregators)
        self.use_pre_norm = use_pre_norm
        self.standardize = standardize
        self.ridge = float(ridge)
        self.score_mode = score_mode
        self.ema_momentum = float(ema_momentum)
        self.dropout = torch.nn.Dropout(dropout)

        self.function_nodes = list(data.node_names_dict['function'])
        self.N = len(self.function_nodes)
        self.L = len(model.ResBlocks)
        self.n_pairs = max(self.L - 1, 0)
        self.n_aggs = len(self.aggregators)

        block0 = model.ResBlocks[0]
        self.channel_groups = block0.channel_groups.detach().cpu()

        function_mask = ~(model.input_node_mask | model.output_node_mask)
        self.function_homo_idxs = function_mask.nonzero(as_tuple=True)[0].cpu().tolist()
        if len(self.function_homo_idxs) != self.N:
            raise ValueError(
                f'Expected {self.N} function nodes in homogeneous graph, '
                f'got {len(self.function_homo_idxs)}.'
            )

        self._channel_ixs_by_fn = []
        for homo_idx in self.function_homo_idxs:
            ixs = (self.channel_groups == homo_idx).nonzero(as_tuple=True)[0]
            if ixs.numel() == 0:
                raise ValueError(
                    f'Function node homo index {homo_idx} has no channels in channel_groups.'
                )
            self._channel_ixs_by_fn.append(ixs)

        edge_index = data.edge_index_dict['function', 'to', 'function']
        if hasattr(edge_index, 'detach'):
            edge_arr = edge_index.T.detach().cpu().numpy()
        else:
            edge_arr = np.asarray(edge_index.T)
        self.edges = {
            (self.function_nodes[i], self.function_nodes[j])
            for i, j in edge_arr
        }

        # Shared (N, N) weight matrix; columns predict each target gradient magnitude.
        self.W = nn.Parameter(torch.zeros(self.N, self.N))
        nn.init.xavier_uniform_(self.W, gain=0.1)

        self.optim = torch.optim.AdamW(
            [self.W], lr=lr, weight_decay=weight_decay,
        )

        P, K, N = self.n_pairs, self.n_aggs, self.N
        self.register_buffer('running_mean_x', torch.zeros(P, K, N))
        self.register_buffer('running_var_x', torch.ones(P, K, N))
        self.register_buffer('running_mean_y', torch.zeros(P, N))
        self.register_buffer('running_var_y', torch.ones(P, N))
        self.register_buffer('n_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Streaming Gram for p-value computation in evaluate().
        self.register_buffer('gram', torch.zeros(N, N))
        self.register_buffer('n_samples', torch.tensor(0, dtype=torch.long))
        self.register_buffer('residual_sum_sq', torch.tensor(0.0))

        self._act_attr = '_last_pre_norm_activation' if use_pre_norm else '_last_activation'
        self._acts: list[torch.Tensor] | None = None
        self._best_state: dict | None = None
        self._best_metric: float | None = None

    def pre_forward(self) -> None:
        '''Enable activation caching on all ResBlocks before ``model(x)``.'''
        for mod in self.model.ResBlocks:
            mod._store_activations = True
        self._acts = None

    def arm_retained_grads(self) -> None:
        '''Call after ``model(x)`` and before ``loss.backward()`` to retain grads.'''
        acts = []
        for mod in self.model.ResBlocks:
            act = getattr(mod, self._act_attr, None)
            if act is None:
                raise RuntimeError(
                    f'ResBlock.{self._act_attr} is None; call pre_forward() first.'
                )
            act.retain_grad()
            acts.append(act)
        self._acts = acts

    def _cleanup_hooks(self) -> None:
        for mod in self.model.ResBlocks:
            mod._store_activations = False
            for attr in ('_last_activation', '_last_pre_norm_activation'):
                if hasattr(mod, attr):
                    delattr(mod, attr)
        self._acts = None

    def _reduce_channels(
        self,
        tensor: torch.Tensor,
        aggregator: str,
    ) -> torch.Tensor:
        '''Reduce (B, C_total) to (B, N) function-node magnitudes.'''
        if tensor.dim() == 3 and tensor.size(-1) == 1:
            tensor = tensor.squeeze(-1)
        if tensor.dim() != 2:
            raise ValueError(
                f'Expected activation/grad shape (B, C) or (B, C, 1), got {tuple(tensor.shape)}'
            )

        B = tensor.size(0)
        fn = self._AGG_FNS[aggregator]
        out = torch.empty(B, self.N, device=tensor.device, dtype=tensor.dtype)
        for fn_idx, ch_ixs in enumerate(self._channel_ixs_by_fn):
            ch_ixs = ch_ixs.to(tensor.device)
            out[:, fn_idx] = fn(tensor[:, ch_ixs])
        return out

    def _standardize(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pair_idx: int,
        agg_idx: int,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.standardize:
            return x, y

        eps = 1e-8
        if training:
            batch_mean_x = x.mean(dim=0)
            batch_var_x = x.var(dim=0, unbiased=False)
            batch_mean_y = y.mean(dim=0)
            batch_var_y = y.var(dim=0, unbiased=False)

            m = self.ema_momentum
            self.running_mean_x[pair_idx, agg_idx].lerp_(batch_mean_x, m)
            self.running_var_x[pair_idx, agg_idx].lerp_(batch_var_x, m)
            self.running_mean_y[pair_idx].lerp_(batch_mean_y, m)
            self.running_var_y[pair_idx].lerp_(batch_var_y, m)
            self.n_batches_tracked += 1

            mean_x, var_x = batch_mean_x, batch_var_x
            mean_y, var_y = batch_mean_y, batch_var_y
        else:
            mean_x = self.running_mean_x[pair_idx, agg_idx]
            var_x = self.running_var_x[pair_idx, agg_idx]
            mean_y = self.running_mean_y[pair_idx]
            var_y = self.running_var_y[pair_idx]

        x_std = (x - mean_x) / torch.sqrt(var_x + eps)
        y_std = (y - mean_y) / torch.sqrt(var_y + eps)
        return x_std, y_std

    def _update_gram(self, x_std: torch.Tensor, y_hat: torch.Tensor, y_std: torch.Tensor) -> None:
        '''Accumulate streaming Gram matrix and residual variance for p-values.'''
        B = x_std.size(0)
        with torch.no_grad():
            self.gram.add_(x_std.T @ x_std)
            self.n_samples += B
            self.residual_sum_sq += ((y_hat - y_std) ** 2).sum().item()

    def aux_step(self) -> dict[str, float]:
        '''
        Build features from cached activations/grads, update ``W``, return metrics.

        Must be called after ``loss.backward()`` so activation gradients exist.
        Features are detached — no gradient flows into the GSNN from this step.
        '''
        if self._acts is None:
            raise RuntimeError(
                'No cached activations; call pre_forward(), model(x), '
                'arm_retained_grads(), then loss.backward() before aux_step().'
            )

        try:
            grads = []
            for act in self._acts:
                if act.grad is None:
                    raise RuntimeError(
                        'Activation gradient is None after backward; '
                        'call arm_retained_grads() before loss.backward().'
                    )
                grads.append(act.grad)

            y_layers = [self._reduce_channels(g.detach(), 'sum') for g in grads]

            total_loss = torch.tensor(0.0, device=self.W.device)
            n_terms = 0
            gram_updates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

            for p in range(self.n_pairs):
                y = y_layers[p + 1]
                for k, agg in enumerate(self.aggregators):
                    x = self._reduce_channels(self._acts[p].detach(), agg)
                    x_std, y_std = self._standardize(
                        x, y, pair_idx=p, agg_idx=k, training=self.training,
                    )
                    y_hat = x_std @ self.dropout(self.W)
                    mse = ((y_hat - y_std) ** 2).mean()
                    total_loss = total_loss + mse
                    n_terms += 1
                    gram_updates.append((x_std.detach(), y_hat.detach(), y_std.detach()))

            if self.ridge > 0:
                total_loss = total_loss + self.ridge * (self.W ** 2).sum()

            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            with torch.no_grad():
                for x_std, y_hat, y_std in gram_updates:
                    self._update_gram(x_std, y_hat, y_std)

            avg_loss = (total_loss / max(n_terms, 1)).item()
            return {'aux_loss': avg_loss, 'n_terms': float(n_terms)}
        finally:
            self._cleanup_hooks()

    def score_matrix(self) -> np.ndarray:
        """Return (N, N) edge score matrix derived from ``W``."""
        w = self.W.detach().cpu().numpy()
        if self.score_mode == 'abs':
            return np.abs(w)
        if self.score_mode == 'relu':
            return np.maximum(w, 0.0)
        if self.score_mode == 'signed':
            return w
        raise ValueError(f'Unknown score_mode: {self.score_mode!r}')

    @staticmethod
    def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
        pvals = pvals.astype(float)
        m = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]
        bh = ranked * m / (np.arange(1, m + 1))
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        bh = np.clip(bh, 0.0, 1.0)
        q_values = np.empty_like(bh)
        q_values[order] = bh
        return q_values

    def _coefficient_pvalues(self) -> np.ndarray:
        '''One-sided p-values testing W_ij > 0 via Gram-based SE approximation.'''
        N = self.N
        pvals = np.ones((N, N), dtype=np.float64)
        n = int(self.n_samples.item())
        if n < N + 2:
            return pvals

        w = self.W.detach().cpu().numpy()
        gram = self.gram.detach().cpu().numpy() / max(n, 1)
        ridge = self.ridge + 1e-8
        try:
            gram_inv = np.linalg.inv(gram + ridge * np.eye(N))
        except np.linalg.LinAlgError:
            return pvals

        rss = float(self.residual_sum_sq.item())
        sigma2 = rss / max(n * self.n_pairs * self.n_aggs * N - N, 1)

        for i in range(N):
            se_i = math.sqrt(max(sigma2 * gram_inv[i, i], 1e-12))
            z = w[i, :] / se_i
            p = 0.5 * (1.0 - scipy.special.erf(z / math.sqrt(2.0)))
            pvals[i, :] = np.clip(p, 0.0, 1.0)

        return pvals

    def evaluate(self, *, exclude_self: bool = True) -> pd.DataFrame:
        '''
        Build edge score DataFrame from current ``W``.

        Returns columns compatible with ``MagnitudeEdgeInferer.evaluate``:
        ``src_func, dst_func, src_idx, dst_idx, score, has_edge, p_value, q_value``.
        '''
        scores = self.score_matrix()
        pvals = self._coefficient_pvalues()

        rows = []
        for i, src in enumerate(self.function_nodes):
            for j, dst in enumerate(self.function_nodes):
                if exclude_self and i == j:
                    continue
                rows.append({
                    'src_func': src,
                    'dst_func': dst,
                    'src_idx': i,
                    'dst_idx': j,
                    'score': scores[i, j],
                    'has_edge': (src, dst) in self.edges,
                    'p_value': pvals[i, j],
                })

        res = pd.DataFrame(rows)
        res['q_value'] = self._bh_fdr(res['p_value'].to_numpy())
        res = res.sort_values('score', ascending=False, na_position='last').reset_index(drop=True)
        return res

    def evaluate_against(
        self,
        positive_edges: set[tuple[str, str]] | list[tuple[str, str]],
        *,
        top_k: tuple[int, ...] = (1, 3, 5),
    ) -> dict[str, float]:
        '''
        Score held-out edges against non-edges using current ``W``.

        Returns global ROC-AUC plus within-target MRR and top@k rates.
        '''
        from sklearn.metrics import roc_auc_score

        pos = {(s, d) for s, d in positive_edges}
        if not pos:
            raise ValueError('positive_edges is empty.')

        res = self.evaluate(exclude_self=True)
        pos = {(s, d) for s, d in positive_edges}

        def _category(row):
            pair = (row['src_func'], row['dst_func'])
            if pair in pos:
                return 'held_out'
            if pair in self.edges:
                return 'in_graph'
            return 'non_edge'

        res = res.assign(edge_category=res.apply(_category, axis=1))

        pos_scores = res.loc[res.edge_category == 'held_out', 'score'].dropna().values
        neg_scores = res.loc[res.edge_category == 'non_edge', 'score'].dropna().values

        out: dict[str, float] = {}
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
            y_score = np.concatenate([pos_scores, neg_scores])
            out['auc'] = float(roc_auc_score(y_true, y_score))
        else:
            out['auc'] = float('nan')

        _, summary = self.evaluate_target_ranking(
            res, positive_edges=pos, score_col='score', top_k=top_k,
        )
        out.update(summary)
        return out

    @staticmethod
    def evaluate_target_ranking(
        res: pd.DataFrame,
        positive_edges: set[tuple[str, str]] | list[tuple[str, str]],
        score_col: str = 'score',
        top_k: tuple[int, ...] = (1, 3, 5),
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        '''Delegate to ``MagnitudeEdgeInferer.evaluate_target_ranking``.'''
        return MagnitudeEdgeInferer.evaluate_target_ranking(
            res, positive_edges=positive_edges, score_col=score_col, top_k=top_k,
        )

    def maybe_save_best(self, metric: float, mode: str = 'max') -> bool:
        '''Save ``state_dict`` if ``metric`` improves over the previous best.'''
        if metric is None or (isinstance(metric, float) and math.isnan(metric)):
            return False

        improved = (
            self._best_metric is None
            or (mode == 'max' and metric > self._best_metric)
            or (mode == 'min' and metric < self._best_metric)
        )
        if improved:
            self._best_metric = float(metric)
            self._best_state = copy.deepcopy(self.state_dict())
            return True
        return False

    def load_best(self) -> None:
        '''Restore weights from the best validation checkpoint.'''
        if self._best_state is None:
            raise RuntimeError('No best checkpoint saved; call maybe_save_best() first.')
        self.load_state_dict(self._best_state)
