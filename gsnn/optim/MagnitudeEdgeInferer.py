'''
Tier-0 edge inference via magnitude correlation between node activations and
node gradients.

For each function node i (source) and j (target), per adjacent layer pair (n-1, n):

    x_i^{n-1} = ||z_i^{n-1}||_p          (activation magnitude at layer n-1)
    y_j^n     = ||grad z_j^n L||_p        (gradient magnitude at layer n)

Score (``score='corr'``):
    s_n(i -> j) = corr_b(x_i^{n-1}, y_j^n)

Partial-correlation score (``score='partial'``) controls for the kept parents
of ``j`` in the partial graph ``G_partial``:

    S_j = parents_{G_partial}(j)
    s_n(i -> j) = pcorr_b( x_i^{n-1},  y_j^n  |  { x_k^{n-1} : k in S_j } )

This removes the contribution of edges the model already uses, mitigating
transitive / common-cause confounds that inflate the plain correlation score.

This matches information flow: activations propagate forward n-1 -> n, gradients
propagate backward n -> n-1. Pairs are aggregated (mean / max) and tested with
Fisher-z p-values + BH-FDR (``df = n - 3 - |S_j|`` for partial correlation).

Requires a trained GSNN with gradient checkpointing disabled (``checkpoint=False``).
Uses ``ResBlock._last_pre_norm_activation`` (post-``lin_in``, pre-norm) so magnitude
scores remain meaningful even with layer / RMS normalization.

See ``docs/notes/edge_inference_notes.md`` section 4 for design rationale.
'''

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.special
import torch


class MagnitudeEdgeInferer:
    '''
    Post-hoc inferrer for function -> function edges via activation/gradient
    magnitude correlation across adjacent layers.

    For each pair of consecutive ResBlocks, correlates activation magnitudes at
    layer n-1 with gradient magnitudes at layer n (matching forward/backward
    information flow), then aggregates across pairs.

    Parameters
    ----------
    model : GSNN
        Trained model. Must have ``checkpoint=False``.
    data : HeteroData-like
        Graph container with ``node_names_dict`` and ``edge_index_dict``.
    reduction : {'l1', 'l2'}
        Norm used to reduce per-node channel vectors to scalars.
    use_pre_norm : bool
        If True (default), score magnitudes from post-``lin_in`` activations before
        the ResBlock normalization layer. This preserves cross-sample magnitude
        variation when ``norm`` is layer, RMS, etc.
    '''

    def __init__(
        self,
        model,
        data,
        reduction: Literal['l1', 'l2'] = 'l1',
        use_pre_norm: bool = True,
    ):
        if getattr(model, 'checkpoint', False):
            raise ValueError(
                'MagnitudeEdgeInferer requires model.checkpoint=False; '
                'gradient checkpointing recomputes activations during backward.'
            )

        self.model = model
        self.data = data
        self.reduction = reduction
        self.use_pre_norm = use_pre_norm

        if not use_pre_norm:
            norm = getattr(model, 'norm', None)
            if norm not in ('batch', 'groupbatch', 'none'):
                raise ValueError(
                    'MagnitudeEdgeInferer with use_pre_norm=False requires '
                    'model.norm to be "batch", "groupbatch", or "none"; '
                    f'got {norm!r}. Layer/RMS norms invalidate post-norm magnitude '
                    'scores — use use_pre_norm=True (default) instead.'
                )

        self.function_nodes = list(data.node_names_dict['function'])
        self.N = len(self.function_nodes)
        self.L = len(model.ResBlocks)
        self.n_pairs = max(self.L - 1, 0)

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

        # Kept parents of each target j (function-function edges in G_partial).
        # Used as the conditioning set for partial-correlation scoring.
        self.parents_by_target: list[np.ndarray] = [
            np.zeros(0, dtype=np.int64) for _ in range(self.N)
        ]
        if edge_arr.size > 0:
            buf: list[list[int]] = [[] for _ in range(self.N)]
            for i, j in edge_arr:
                if 0 <= i < self.N and 0 <= j < self.N and i != j:
                    buf[int(j)].append(int(i))
            self.parents_by_target = [
                np.asarray(sorted(set(p)), dtype=np.int64) for p in buf
            ]

        self.reset_stats()

    def reset_stats(self) -> None:
        '''Clear accumulated streaming statistics.

        Memory: O(P * N^2) for both ``sum_xy`` and ``sum_xx`` (the latter is
        only required for ``score='partial'`` but is always tracked since the
        per-batch cost is one extra ``N x N`` matmul).
        '''
        P, N = self.n_pairs, self.N
        self.n: int = 0
        self.sum_x = np.zeros((P, N), dtype=np.float64)
        self.sum_y = np.zeros((P, N), dtype=np.float64)
        self.sum_x2 = np.zeros((P, N), dtype=np.float64)
        self.sum_y2 = np.zeros((P, N), dtype=np.float64)
        self.sum_xy = np.zeros((P, N, N), dtype=np.float64)
        self.sum_xx = np.zeros((P, N, N), dtype=np.float64)

    def _reduce_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Reduce (B, C_total) activations/gradients to (B, N) function-node magnitudes.
        '''
        if tensor.dim() == 3 and tensor.size(-1) == 1:
            tensor = tensor.squeeze(-1)
        if tensor.dim() != 2:
            raise ValueError(
                f'Expected activation/grad shape (B, C) or (B, C, 1), got {tuple(tensor.shape)}'
            )

        B = tensor.size(0)
        out = torch.empty(B, self.N, device=tensor.device, dtype=tensor.dtype)
        for fn_idx, ch_ixs in enumerate(self._channel_ixs_by_fn):
            ch_ixs = ch_ixs.to(tensor.device)
            vals = tensor[:, ch_ixs]
            if self.reduction == 'l1':
                out[:, fn_idx] = vals.abs().sum(dim=-1)
            elif self.reduction == 'l2':
                out[:, fn_idx] = vals.pow(2).sum(dim=-1).sqrt()
            else:
                raise ValueError(f'Unknown reduction: {self.reduction}')
        return out

    def _per_batch_magnitudes(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        crit: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward + backward pass; return activation and gradient magnitudes.

        Returns
        -------
        x_mag, y_mag : Tensor
            Shapes (L, B, N) for activation magnitudes and gradient magnitudes.
        '''
        model = self.model
        was_training = model.training
        model.eval()

        for mod in model.ResBlocks:
            mod._store_activations = True

        act_attr = '_last_pre_norm_activation' if self.use_pre_norm else '_last_activation'

        try:
            yhat = model(x)
            loss = crit(yhat, y)

            acts = [getattr(mod, act_attr, None) for mod in model.ResBlocks]
            for act in acts:
                if act is None:
                    raise RuntimeError(
                        f'ResBlock.{act_attr} is None; ensure _store_activations=True.'
                    )
                act.retain_grad()

            model.zero_grad(set_to_none=True)
            loss.backward()

            grads = []
            for act in acts:
                if act.grad is None:
                    raise RuntimeError(
                        'Activation gradient is None after backward; '
                        'check that checkpoint=False and retain_grad() was called.'
                    )
                grads.append(act.grad)

            x_mag = torch.stack([self._reduce_channels(a) for a in acts], dim=0)
            y_mag = torch.stack([self._reduce_channels(g) for g in grads], dim=0)
        finally:
            for mod in model.ResBlocks:
                mod._store_activations = False
                for attr in ('_last_activation', '_last_pre_norm_activation'):
                    if hasattr(mod, attr):
                        delattr(mod, attr)
            model.train(was_training)

        return x_mag, y_mag

    def _update_stats(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> None:
        '''Update streaming sufficient statistics from one batch.

        Correlates activation magnitudes at layer n-1 with gradient magnitudes
        at layer n for each adjacent pair.
        '''
        x_np = x_mag.detach().cpu().numpy()
        y_np = y_mag.detach().cpu().numpy()
        B = x_np.shape[1]

        self.n += B
        for p in range(self.n_pairs):
            act_l = x_np[p]
            grad_l = y_np[p + 1]
            self.sum_x[p] += act_l.sum(axis=0)
            self.sum_y[p] += grad_l.sum(axis=0)
            self.sum_x2[p] += (act_l ** 2).sum(axis=0)
            self.sum_y2[p] += (grad_l ** 2).sum(axis=0)
            self.sum_xy[p] += act_l.T @ grad_l
            self.sum_xx[p] += act_l.T @ act_l

    def fit(
        self,
        dataloader,
        crit: Optional[torch.nn.Module] = None,
        device: str = 'cpu',
        verbose: bool = True,
    ) -> int:
        '''
        Accumulate magnitude statistics over ``dataloader``.

        Parameters
        ----------
        dataloader : Iterable
            Yields ``(x, y)`` batches.
        crit : callable, optional
            Loss function. Defaults to ``MSELoss``.
        device : str
            Device for forward/backward passes.
        verbose : bool
            Print batch progress.

        Returns
        -------
        int
            Total number of samples processed.
        '''
        if crit is None:
            crit = torch.nn.MSELoss()

        self.reset_stats()
        self.model.to(device)

        n_batches = len(dataloader) if hasattr(dataloader, '__len__') else None
        for bi, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            x_mag, y_mag = self._per_batch_magnitudes(x, y, crit)
            self._update_stats(x_mag, y_mag)

            if verbose and n_batches is not None:
                print(f'[batch {bi + 1}/{n_batches}] n={self.n}', end='\r')
        if verbose:
            print()

        return self.n

    def _check_ready(self) -> None:
        if self.n_pairs == 0:
            raise RuntimeError(
                f'Need at least 2 ResBlocks to form activation/gradient pairs; got L={self.L}.'
            )
        if self.n < 3:
            raise RuntimeError(
                f'Need at least 3 samples to compute correlations; got n={self.n}.'
            )

    def _compute_correlations(self) -> np.ndarray:
        '''Return per-pair correlation tensor of shape (L-1, N, N).

        Pair p compares activation magnitudes at layer p with gradient magnitudes
        at layer p+1.
        '''
        self._check_ready()

        n = float(self.n)
        eps = 1e-12

        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        var_x = np.maximum(self.sum_x2 / n - mean_x ** 2, 0.0)
        var_y = np.maximum(self.sum_y2 / n - mean_y ** 2, 0.0)

        corr = np.full((self.n_pairs, self.N, self.N), np.nan, dtype=np.float64)
        for p in range(self.n_pairs):
            cov_xy = self.sum_xy[p] / n - mean_x[p][:, None] * mean_y[p][None, :]
            denom = np.sqrt(var_x[p][:, None] * var_y[p][None, :])
            with np.errstate(divide='ignore', invalid='ignore'):
                corr_p = cov_xy / np.maximum(denom, eps)
            corr_p[(var_x[p][:, None] <= eps) | (var_y[p][None, :] <= eps)] = np.nan
            corr[p] = np.clip(corr_p, -1.0, 1.0)

        return corr

    def _compute_partial_correlations(self, ridge: float = 1e-8) -> np.ndarray:
        '''Per-pair partial correlation tensor of shape (L-1, N, N).

        Pair p partials activation magnitudes at layer p (source) and gradient
        magnitudes at layer p+1 (target) on the kept parents of the target in
        ``G_partial``:

            S_j = parents_{G_partial}(j)
            pcorr(x_i, y_j | x_S_j)

        Computed via Schur complement on the joint covariance of
        (x_S, x_i, y_j) using the streamed sufficient statistics.

        Entries with ``i in S_j`` are set to NaN (partial correlation undefined
        / trivially 0; these are kept edges, not candidates). Targets with
        empty ``S_j`` reduce to plain Pearson correlation.

        Parameters
        ----------
        ridge : float
            Tikhonov regularizer added to the diagonal of ``Sigma_SS`` for
            numerical stability when conditioning sets are nearly collinear.
        '''
        self._check_ready()

        n = float(self.n)
        eps = 1e-12

        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        var_x_diag = np.maximum(self.sum_x2 / n - mean_x ** 2, 0.0)
        var_y_diag = np.maximum(self.sum_y2 / n - mean_y ** 2, 0.0)

        pcorr = np.full((self.n_pairs, self.N, self.N), np.nan, dtype=np.float64)
        for p in range(self.n_pairs):
            Sigma_xx = self.sum_xx[p] / n - np.outer(mean_x[p], mean_x[p])
            Sigma_xy = self.sum_xy[p] / n - np.outer(mean_x[p], mean_y[p])
            var_x = var_x_diag[p]
            var_y = var_y_diag[p]

            for j in range(self.N):
                S = self.parents_by_target[j]
                vy_j = var_y[j]

                if S.size == 0:
                    if vy_j <= eps:
                        continue
                    denom = np.sqrt(np.maximum(var_x, 0.0) * vy_j)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        row = Sigma_xy[:, j] / np.maximum(denom, eps)
                    row[var_x <= eps] = np.nan
                    pcorr[p, :, j] = np.clip(row, -1.0, 1.0)
                    continue

                Sigma_SS = Sigma_xx[np.ix_(S, S)].copy()
                if ridge > 0:
                    Sigma_SS.flat[:: Sigma_SS.shape[0] + 1] += ridge
                Sigma_Sj = Sigma_xy[S, j]

                try:
                    alpha_j = np.linalg.solve(Sigma_SS, Sigma_Sj)
                except np.linalg.LinAlgError:
                    continue

                var_y_partial = vy_j - Sigma_Sj @ alpha_j
                if not np.isfinite(var_y_partial) or var_y_partial <= eps:
                    continue

                Sigma_iS = Sigma_xx[:, S]
                try:
                    M = np.linalg.solve(Sigma_SS, Sigma_iS.T)
                except np.linalg.LinAlgError:
                    continue

                var_x_partial = var_x - np.einsum('ik,ki->i', Sigma_iS, M)
                num = Sigma_xy[:, j] - Sigma_iS @ alpha_j

                denom = np.sqrt(np.maximum(var_x_partial, 0.0) * var_y_partial)
                with np.errstate(divide='ignore', invalid='ignore'):
                    row = num / np.maximum(denom, eps)

                bad = (var_x_partial <= eps) | ~np.isfinite(row)
                row[bad] = np.nan
                # i in S_j: x_i is in the conditioning set, partial corr undefined.
                row[S] = np.nan
                # i == j: covered by exclude_self in evaluate, but mark explicitly.
                row[j] = np.nan

                pcorr[p, :, j] = np.clip(row, -1.0, 1.0)

        return pcorr

    @staticmethod
    def _aggregate_pairs(corr: np.ndarray, layer_agg: str) -> np.ndarray:
        if layer_agg == 'mean':
            with np.errstate(all='ignore'):
                return np.nanmean(corr, axis=0)
        if layer_agg == 'max':
            abs_corr = np.abs(corr)
            all_nan = np.all(np.isnan(abs_corr), axis=0)
            # nanargmax raises on all-NaN slices; use -inf sentinel instead.
            abs_filled = abs_corr.copy()
            abs_filled[np.isnan(abs_filled)] = -np.inf
            idx = np.argmax(abs_filled, axis=0)
            P, N, _ = corr.shape
            rows = np.arange(N)[:, None]
            cols = np.arange(N)[None, :]
            out = corr[idx, rows, cols]
            out[all_nan] = np.nan
            return out
        raise ValueError(f"Unknown layer_agg: {layer_agg}. Use 'mean' or 'max'.")

    @staticmethod
    def _fisher_pvalues(
        corr: np.ndarray,
        n: int,
        cond_size: int | np.ndarray = 0,
    ) -> np.ndarray:
        '''One-sided p-values testing corr > 0 (H0: r <= 0).

        For partial correlation, ``cond_size`` is the number of conditioning
        variables (df penalty); for plain Pearson, leave at 0.
        ``cond_size`` may be a scalar or per-element array broadcastable
        against ``corr``.
        '''
        df = (n - 3) - np.asarray(cond_size, dtype=np.float64)
        safe_df = np.maximum(df, 1.0)
        z = np.arctanh(np.clip(corr, -0.9999, 0.9999)) * np.sqrt(safe_df)
        p = 0.5 * (1.0 - scipy.special.erf(z / math.sqrt(2.0)))
        p = np.where(np.isnan(corr), 1.0, p)
        p = np.where(df < 2, 1.0, p)
        return p

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

    def evaluate(
        self,
        layer_agg: Literal['mean', 'max'] = 'mean',
        exclude_self: bool = True,
        score: Literal['corr', 'partial'] = 'corr',
        ridge: float = 1e-8,
    ) -> pd.DataFrame:
        '''
        Compute correlation scores, p-values, and q-values from accumulated stats.

        Parameters
        ----------
        layer_agg : {'mean', 'max'}
            How to aggregate adjacent-layer-pair correlations into one score matrix.
        exclude_self : bool
            If True, omit diagonal i->i pairs from the output.
        score : {'corr', 'partial'}
            ``'corr'``: Pearson correlation between activation magnitudes at
            layer n-1 and gradient magnitudes at layer n.
            ``'partial'``: partial correlation conditioning on the kept parents
            of the target in ``G_partial`` (function-function edges in
            ``data.edge_index_dict``). Removes contributions from edges the
            model already uses.
        ridge : float
            Tikhonov regularizer for the conditioning-set covariance matrix
            when ``score='partial'``. Only used in that mode.

        Returns
        -------
        pandas.DataFrame
            Columns: src_func, dst_func, src_idx, dst_idx, corr, corr_a*_g*,
            p_value, q_value, has_edge. When ``score='partial'``, additionally
            ``n_cond`` (size of the conditioning set for that target). For
            kept edges (``has_edge=True``) the partial correlation is NaN
            since the source is in the conditioning set.
        '''
        if score == 'corr':
            corr_pairs = self._compute_correlations()
        elif score == 'partial':
            corr_pairs = self._compute_partial_correlations(ridge=ridge)
        else:
            raise ValueError(f"Unknown score: {score!r}. Use 'corr' or 'partial'.")

        corr = self._aggregate_pairs(corr_pairs, layer_agg)
        cond_sizes = np.asarray(
            [self.parents_by_target[j].size for j in range(self.N)],
            dtype=np.int64,
        )

        rows = []
        for i, src in enumerate(self.function_nodes):
            for j, dst in enumerate(self.function_nodes):
                if exclude_self and i == j:
                    continue
                row = {
                    'src_func': src,
                    'dst_func': dst,
                    'src_idx': i,
                    'dst_idx': j,
                    'corr': corr[i, j],
                    'has_edge': (src, dst) in self.edges,
                }
                if score == 'partial':
                    row['n_cond'] = int(cond_sizes[j])
                for p in range(self.n_pairs):
                    row[f'corr_a{p}_g{p + 1}'] = corr_pairs[p, i, j]
                rows.append(row)

        res = pd.DataFrame(rows)
        cs = cond_sizes[res['dst_idx'].to_numpy()] if score == 'partial' else 0
        pvals = self._fisher_pvalues(res['corr'].to_numpy(), self.n, cond_size=cs)
        res['p_value'] = pvals
        res['q_value'] = self._bh_fdr(pvals)
        res = res.sort_values('corr', ascending=False, na_position='last').reset_index(drop=True)
        return res

    @staticmethod
    def evaluate_target_ranking(
        res: pd.DataFrame,
        positive_edges: set[tuple[str, str]] | list[tuple[str, str]],
        score_col: str = 'corr',
        top_k: tuple[int, ...] = (1, 3, 5),
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        '''Within-target ranking metrics for edge recovery.

        For each positive edge ``(src, dst)`` in ``positive_edges``, rank all
        candidate sources for ``dst`` by ``score_col`` (descending) and record
        the rank of the true missing parent. This answers: *for each target
        with a held-out edge, is the correct source ranked highest among
        competitors for that target?*

        Parameters
        ----------
        res : pandas.DataFrame
            Output of :meth:`evaluate`.
        positive_edges : set or list of (src, dst)
            Ground-truth edges to recover (typically held-out edges). ``src``
            and ``dst`` must match ``src_func`` / ``dst_func`` in ``res``.
        score_col : str
            Column to rank on (default ``'corr'``).
        top_k : tuple of int
            Compute ``top@k`` hit rate for each ``k``.

        Returns
        -------
        detail : pandas.DataFrame
            One row per positive edge with columns ``src_func``, ``dst_func``,
            ``score``, ``rank`` (1 = best), ``n_candidates``, ``reciprocal_rank``,
            and ``top@{k}`` boolean flags.
        summary : dict
            ``n_positives``, ``mrr``, and ``top@{k}`` rates.
        '''
        pos = {(s, d) for s, d in positive_edges}
        if not pos:
            raise ValueError('positive_edges is empty.')

        ranked = res.dropna(subset=[score_col]).copy()
        ranked['rank'] = (
            ranked.groupby('dst_func', sort=False)[score_col]
            .rank(ascending=False, method='first')
            .astype(int)
        )
        ranked['n_candidates'] = ranked.groupby('dst_func', sort=False)['dst_func'].transform('count')

        rows = []
        for src, dst in sorted(pos):
            sub = ranked[(ranked['src_func'] == src) & (ranked['dst_func'] == dst)]
            if sub.empty:
                rows.append({
                    'src_func': src,
                    'dst_func': dst,
                    'score': np.nan,
                    'rank': np.nan,
                    'n_candidates': int((ranked['dst_func'] == dst).sum()),
                    'reciprocal_rank': 0.0,
                    **{f'top@{k}': False for k in top_k},
                })
                continue

            row = sub.iloc[0]
            rank = int(row['rank'])
            rr = 1.0 / rank
            rows.append({
                'src_func': src,
                'dst_func': dst,
                'score': row[score_col],
                'rank': rank,
                'n_candidates': int(row['n_candidates']),
                'reciprocal_rank': rr,
                **{f'top@{k}': rank <= k for k in top_k},
            })

        detail = pd.DataFrame(rows)
        n = len(detail)
        summary: dict[str, float] = {'n_positives': float(n), 'mrr': detail['reciprocal_rank'].mean()}
        for k in top_k:
            col = f'top@{k}'
            summary[col] = detail[col].mean() if n else float('nan')
        return detail, summary
