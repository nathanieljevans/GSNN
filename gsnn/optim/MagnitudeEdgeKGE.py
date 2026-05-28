'''
Post-hoc Tier-0 edge inference via node2vec on the augmented function graph.

After training a GSNN, ``MagnitudeEdgeInferer`` accumulates activation/gradient
magnitude correlations. High-scoring non-edges are mined as inferred positives
and pooled with the kept-graph edges into a single augmented graph. A single
shared embedding table is learned by skip-gram with negative sampling on
random walks, so all nodes live in the same space - kept and inferred edges
contribute equally to neighborhood structure.

Held-out edges are scored by ``<emb[i], emb[j]>``.

See tutorial 17 and ``docs/notes/edge_inference_notes.md`` section 4.
'''

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from gsnn.optim.MagnitudeEdgeInferer import MagnitudeEdgeInferer


class MagnitudeEdgeKGE(nn.Module):
    '''
    Post-hoc node2vec edge inferrer for function -> function edges.

    Consumes a fitted ``MagnitudeEdgeInferer``, mines inferred positive edges
    from its correlation scores, builds an augmented directed graph from
    (kept + inferred) edges, and trains a single node embedding table by
    skip-gram with negative sampling on random walks. Parameter count is
    ``O(N * d)``.

    Parameters
    ----------
    mei : MagnitudeEdgeInferer
        Fitted inferrer with accumulated statistics (``mei.n >= 3``).
    embedding_dim : int
        Embedding dimension.
    score : {'corr', 'partial'}
        MEI score column used to mine inferred positives.
    layer_agg : {'mean', 'max'}
        MEI layer aggregation for the score matrix.
    mining_strategy : {'fdr', 'topk_per_target'}
        How to select inferred positives from the MEI score table.
    fdr_alpha : float
        BH-FDR threshold when ``mining_strategy='fdr'``.
    top_k_per_target : int
        Top sources per target when ``mining_strategy='topk_per_target'``.
    exclude_edges : iterable of (src, dst)
        Held-out val/test edges to remove from inferred positives (anti-leakage).
    walks_per_node : int
        Number of random walks starting from each node per epoch.
    walk_length : int
        Length of each random walk (number of nodes).
    window_size : int
        Skip-gram context window (sliding distance within a walk).
    n_negatives : int
        Negative samples per positive (center, context) pair.
    walk_undirected : bool
        If True, treat the augmented graph as undirected for walk traversal.
        Walks rarely die at sinks, so coverage is better. Skip-gram positives
        are still emitted symmetrically. Default True.
    walk_corr_weighted : bool
        If True, transition probability ``P(j | i)`` along the walk is
        proportional to ``max(corr[i, j], 0) ** walk_alpha`` rather than
        uniform over neighbors. Brings back MEI's continuous signal that
        the binary mining step otherwise discards. Default True.
    walk_alpha : float
        Power applied to ``max(corr, 0)`` before normalizing into transition
        probabilities. ``alpha=1.0`` is linear; larger values concentrate
        walks on high-correlation edges; smaller values flatten toward
        uniform. Default 1.0.
    kept_edge_weight : float or None
        Walk weight assigned to kept (true) function-function edges. If None,
        defaults to the maximum inferred-edge weight, so kept edges are at
        least as likely to be traversed as the strongest inferred edge.
    lr, weight_decay : float
        Optimizer settings.
    '''

    _NODE_TYPE = 'function'

    def __init__(
        self,
        mei: MagnitudeEdgeInferer,
        *,
        embedding_dim: int = 64,
        score: Literal['corr', 'partial'] = 'corr',
        layer_agg: Literal['mean', 'max'] = 'max',
        mining_strategy: Literal['fdr', 'topk_per_target'] = 'fdr',
        fdr_alpha: float = 0.05,
        top_k_per_target: int = 5,
        exclude_edges: Iterable[tuple[str, str]] = (),
        walks_per_node: int = 20,
        walk_length: int = 10,
        window_size: int = 5,
        n_negatives: int = 5,
        walk_undirected: bool = True,
        walk_corr_weighted: bool = True,
        walk_alpha: float = 1.0,
        kept_edge_weight: float | None = None,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
    ):
        super().__init__()

        if score not in ('corr', 'partial'):
            raise ValueError(f"Unknown score {score!r}. Use 'corr' or 'partial'.")
        if layer_agg not in ('mean', 'max'):
            raise ValueError(f"Unknown layer_agg {layer_agg!r}. Use 'mean' or 'max'.")
        if mining_strategy not in ('fdr', 'topk_per_target'):
            raise ValueError(
                f"Unknown mining_strategy {mining_strategy!r}. "
                "Use 'fdr' or 'topk_per_target'."
            )
        if mei.n < 3:
            raise RuntimeError(
                f'MagnitudeEdgeKGE requires a fitted MEI with n >= 3; got n={mei.n}. '
                'Call mei.fit(dataloader) first.'
            )

        self.__dict__['mei'] = mei
        self.data = mei.data
        self.function_nodes = list(mei.function_nodes)
        self.N = mei.N
        self.edges = set(mei.edges)

        self.score = score
        self.layer_agg = layer_agg
        self.mining_strategy = mining_strategy
        self.fdr_alpha = float(fdr_alpha)
        self.top_k_per_target = int(top_k_per_target)
        self.exclude_edges = {(s, d) for s, d in exclude_edges}
        self.embedding_dim = int(embedding_dim)
        self.walks_per_node = int(walks_per_node)
        self.walk_length = int(walk_length)
        self.window_size = int(window_size)
        self.n_negatives = int(n_negatives)
        self.walk_undirected = bool(walk_undirected)
        self.walk_corr_weighted = bool(walk_corr_weighted)
        self.walk_alpha = float(walk_alpha)
        self.kept_edge_weight = (
            float(kept_edge_weight) if kept_edge_weight is not None else None
        )

        self.emb = nn.Embedding(self.N, self.embedding_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.1)

        self.optim = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay,
        )

        self._mei_df = mei.evaluate(layer_agg=layer_agg, score=score)
        self._prepare_positives()
        self._build_adjacency()

        self._best_state: dict | None = None
        self._best_metric: float | None = None

    def _prepare_positives(self) -> None:
        '''Mine inferred positives from MEI scores; materialize edge tensors.'''
        df = self._mei_df.copy()
        df = df.dropna(subset=['corr'])
        df = df[df['src_idx'] != df['dst_idx']]

        kept_mask = df.apply(
            lambda r: (r['src_func'], r['dst_func']) in self.edges, axis=1,
        )
        exclude_mask = df.apply(
            lambda r: (r['src_func'], r['dst_func']) in self.exclude_edges, axis=1,
        )
        candidates = df[~kept_mask & ~exclude_mask].copy()

        if self.mining_strategy == 'fdr':
            inferred = candidates[candidates['q_value'] <= self.fdr_alpha].copy()
        else:
            inferred = (
                candidates
                .sort_values('corr', ascending=False)
                .groupby('dst_func', sort=False, group_keys=False)
                .head(self.top_k_per_target)
            )

        inferred = inferred.sort_values('corr', ascending=False).reset_index(drop=True)
        self._inferred_df = inferred

        inf_heads = inferred['src_idx'].to_numpy(dtype=np.int64)
        inf_tails = inferred['dst_idx'].to_numpy(dtype=np.int64)
        self.inferred_heads = torch.tensor(inf_heads, dtype=torch.long)
        self.inferred_tails = torch.tensor(inf_tails, dtype=torch.long)

        true_pairs = sorted(self.edges)
        true_heads = [self.function_nodes.index(s) for s, _ in true_pairs]
        true_tails = [self.function_nodes.index(d) for _, d in true_pairs]
        self.true_heads = torch.tensor(true_heads, dtype=torch.long)
        self.true_tails = torch.tensor(true_tails, dtype=torch.long)

        self.pos_heads = torch.cat([self.true_heads, self.inferred_heads])
        self.pos_tails = torch.cat([self.true_tails, self.inferred_tails])

    def _build_adjacency(self) -> None:
        '''Build per-node weighted neighbor lists for walk generation.

        Each edge weight is ``max(MEI corr, 0) ** walk_alpha`` for inferred
        edges; kept edges receive ``kept_edge_weight`` (defaulting to the
        max inferred-edge weight). Per-node weights are normalized into
        transition probabilities; nodes with all-zero weights fall back to
        uniform.
        '''
        N = self.N

        corr_lookup: dict[tuple[int, int], float] = {}
        if self.walk_corr_weighted:
            mei_df = self._mei_df.dropna(subset=['corr'])
            corr_lookup = {
                (int(r['src_idx']), int(r['dst_idx'])): float(r['corr'])
                for _, r in mei_df.iterrows()
            }

        def _w_from_corr(c: float) -> float:
            return float(max(c, 0.0)) ** self.walk_alpha

        inferred_pairs = list(zip(
            self.inferred_heads.tolist(), self.inferred_tails.tolist(),
        ))
        true_pairs = list(zip(
            self.true_heads.tolist(), self.true_tails.tolist(),
        ))

        if self.walk_corr_weighted:
            inferred_w = [_w_from_corr(corr_lookup.get((i, j), 0.0))
                          for i, j in inferred_pairs]
            if self.kept_edge_weight is not None:
                kept_w_default = self.kept_edge_weight
            elif inferred_w:
                kept_w_default = max(inferred_w)
            else:
                kept_w_default = 1.0
            true_w = [
                max(_w_from_corr(corr_lookup.get((i, j), 0.0)), kept_w_default)
                for i, j in true_pairs
            ]
        else:
            inferred_w = [1.0] * len(inferred_pairs)
            true_w = [1.0] * len(true_pairs)

        nbr_dicts: list[dict[int, float]] = [dict() for _ in range(N)]
        for (h, t), w in zip(true_pairs + inferred_pairs, true_w + inferred_w):
            nbr_dicts[h][t] = nbr_dicts[h].get(t, 0.0) + w
            if self.walk_undirected:
                nbr_dicts[t][h] = nbr_dicts[t].get(h, 0.0) + w

        self._adj: list[np.ndarray] = []
        self._adj_p: list[np.ndarray] = []
        for i in range(N):
            d = nbr_dicts[i]
            if not d:
                self._adj.append(np.empty(0, dtype=np.int64))
                self._adj_p.append(np.empty(0, dtype=np.float64))
                continue
            keys = np.asarray(sorted(d.keys()), dtype=np.int64)
            weights = np.asarray([d[k] for k in keys], dtype=np.float64)
            total = weights.sum()
            if total > 0:
                p = weights / total
            else:
                p = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
            self._adj.append(keys)
            self._adj_p.append(p)

        self._has_neighbors = np.asarray(
            [arr.size > 0 for arr in self._adj], dtype=bool,
        )

    def _generate_walks(self, rng: np.random.Generator) -> np.ndarray:
        '''Random walks of shape ``(walks_per_node * N, walk_length)``.'''
        N = self.N
        L = self.walk_length
        K = self.walks_per_node

        starts = np.tile(np.arange(N, dtype=np.int64), K)
        rng.shuffle(starts)

        walks = np.full((starts.size, L), -1, dtype=np.int64)
        walks[:, 0] = starts

        for step in range(1, L):
            current = walks[:, step - 1]
            valid = current >= 0
            for w_idx in np.where(valid)[0]:
                cur = current[w_idx]
                nbrs = self._adj[cur]
                if nbrs.size == 0:
                    continue
                p = self._adj_p[cur]
                walks[w_idx, step] = rng.choice(nbrs, p=p)

        return walks

    def _walks_to_pairs(self, walks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Convert walks to ``(center, context)`` skip-gram pairs.'''
        L = walks.shape[1]
        w = self.window_size
        centers = []
        contexts = []
        for offset in range(1, w + 1):
            c = walks[:, : L - offset]
            ct = walks[:, offset:]
            valid = (c >= 0) & (ct >= 0) & (c != ct)
            centers.append(c[valid])
            contexts.append(ct[valid])
            centers.append(ct[valid])
            contexts.append(c[valid])

        if not centers:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        return (
            np.concatenate(centers).astype(np.int64),
            np.concatenate(contexts).astype(np.int64),
        )

    def _skipgram_loss(
        self,
        centers: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        '''Skip-gram with negative sampling: log-sigmoid losses.'''
        device = centers.device
        c = self.emb(centers)
        p = self.emb(contexts)
        pos_score = (c * p).sum(dim=-1)
        pos_loss = -F.logsigmoid(pos_score)

        if self.n_negatives > 0:
            neg = torch.randint(
                0, self.N, (centers.size(0), self.n_negatives), device=device,
            )
            n = self.emb(neg)
            neg_score = torch.einsum('bd,bkd->bk', c, n)
            neg_loss = -F.logsigmoid(-neg_score).mean(dim=-1)
        else:
            neg_loss = torch.zeros_like(pos_loss)

        return (pos_loss + neg_loss).mean()

    def fit(
        self,
        n_epochs: int = 100,
        batch_size: int = 2048,
        validation_edges: Iterable[tuple[str, str]] | None = None,
        verbose: bool = True,
        seed: int = 0,
    ) -> dict[str, list[float]]:
        '''
        Train embeddings via skip-gram with negative sampling.

        Walks are regenerated each epoch.

        Returns
        -------
        history : dict
            ``train_loss`` per epoch, optional ``val_auc`` per epoch.
        '''
        if not self._has_neighbors.any():
            raise RuntimeError('Augmented graph has no edges; cannot train.')

        device = next(self.parameters()).device
        val_edges = list(validation_edges) if validation_edges is not None else None
        rng = np.random.default_rng(seed)

        history: dict[str, list[float]] = {'train_loss': []}
        if val_edges is not None:
            history['val_auc'] = []

        for epoch in range(n_epochs):
            self.train()
            walks = self._generate_walks(rng)
            centers_np, contexts_np = self._walks_to_pairs(walks)
            n_pairs = centers_np.size

            if n_pairs == 0:
                history['train_loss'].append(0.0)
                if val_edges is not None:
                    history['val_auc'].append(float('nan'))
                continue

            perm = rng.permutation(n_pairs)
            centers_np = centers_np[perm]
            contexts_np = contexts_np[perm]

            centers = torch.from_numpy(centers_np).to(device)
            contexts = torch.from_numpy(contexts_np).to(device)

            epoch_loss = 0.0
            n_steps = 0
            for start in range(0, n_pairs, batch_size):
                end = min(start + batch_size, n_pairs)
                loss = self._skipgram_loss(centers[start:end], contexts[start:end])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                epoch_loss += float(loss.item())
                n_steps += 1

            history['train_loss'].append(epoch_loss / max(n_steps, 1))

            if val_edges is not None:
                val_metrics = self.evaluate_against(val_edges)
                val_auc = val_metrics['auc']
                history['val_auc'].append(val_auc)
                self.maybe_save_best(val_auc)

            if verbose and (
                epoch == 0
                or (epoch + 1) % max(1, n_epochs // 10) == 0
                or epoch == n_epochs - 1
            ):
                msg = (
                    f'epoch {epoch + 1:4d}/{n_epochs} | pairs {n_pairs:6d} | '
                    f'loss {history["train_loss"][-1]:.4f}'
                )
                if val_edges is not None:
                    msg += f' | val AUC {history["val_auc"][-1]:.3f}'
                print(msg)

        return history

    def score_matrix(self) -> np.ndarray:
        '''Return (N, N) edge score matrix from node embeddings.'''
        self.eval()
        with torch.inference_mode():
            scores = self.emb.weight @ self.emb.weight.T
        return scores.detach().cpu().numpy().astype(np.float64)

    def evaluate(self, *, exclude_self: bool = True) -> pd.DataFrame:
        '''
        Build edge score DataFrame from node embeddings.

        Columns: ``src_func, dst_func, src_idx, dst_idx, score, has_edge``.
        '''
        score_mat = self.score_matrix()

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
                    'score': score_mat[i, j],
                    'has_edge': (src, dst) in self.edges,
                })

        res = pd.DataFrame(rows)
        res = res.sort_values('score', ascending=False, na_position='last').reset_index(drop=True)
        return res

    def evaluate_against(
        self,
        positive_edges: set[tuple[str, str]] | list[tuple[str, str]],
        *,
        top_k: tuple[int, ...] = (1, 3, 5),
    ) -> dict[str, float]:
        '''ROC-AUC and within-target ranking metrics on held-out edges.'''
        from sklearn.metrics import roc_auc_score

        pos = {(s, d) for s, d in positive_edges}
        if not pos:
            raise ValueError('positive_edges is empty.')

        res = self.evaluate(exclude_self=True)

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
        """Delegate to ``MagnitudeEdgeInferer.evaluate_target_ranking``."""
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
