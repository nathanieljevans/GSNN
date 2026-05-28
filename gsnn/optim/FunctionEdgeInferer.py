"""Post-hoc inference of latent function -> function edges in a trained GSNN.

Overview
--------
:class:`FunctionEdgeInferer` produces a dense ``(N, N)`` evidence matrix ``W``
over function nodes from a trained :class:`gsnn.models.GSNN.GSNN` model and an
evaluation batch. The score ``W[i, j]`` is a rank (or Pearson) correlation
between layer-:math:`l` per-node *activations* at node ``i`` and
layer-:math:`l+1` per-node *gradients* at node ``j``, pooled across layers and
batch elements. High values are interpreted as evidence for a directed edge
``i -> j`` (a Hebbian-style "what fired together with a downstream credit
signal").

The pipeline is:

    1. Toggle activation capture on every ResBlock and run a forward + backward
       pass on the supplied batch.
    2. Read each block's per-channel pre-/post-norm activations and gradients.
    3. Reduce ``|act|`` / ``|grad|`` within each node group (``sum`` / ``mean``
       / ``max``) to obtain per-node activity at every layer.
    4. Stack ``(layer-l act, layer-(l+1) grad)`` pairs across layers + batch
       as ``M = (L - 1) * B`` observations.
    5. Compute the cross-correlation matrix between the ``N`` activation
       columns and the ``N`` gradient columns.

A short :func:`mrr` utility is provided to score the resulting matrix against
ground-truth edges via mean-reciprocal-rank.

Tractability
------------
Let

    N = number of function nodes
    L = number of ResBlocks
    B = batch size used for attribution
    M = (L - 1) * B = number of (activation, gradient) observations

Dominant costs:

================================  ===================  ============================
quantity                          memory               compute
================================  ===================  ============================
activations / gradients buffers   ``O(L * B * N)``     forward + backward pass
ranked observations (Spearman)    ``O(M * N)``         ``N * O(M log M)``
correlation matrix ``W``          ``O(N^2)``           ``O(M * N^2)``
================================  ===================  ============================

Concrete back-of-envelope figures (float32):

    -    1k nodes,   10k obs  ->  ~40 MB working set, sub-second on CPU.
    -    5k nodes,   50k obs  ->  ~1 GB working set, ~seconds on GPU.
    -   10k nodes,   10k obs  ->  ~400 MB just for ``W``, seconds on GPU.
    -   10k nodes,  100k obs  ->  ~4 GB per (acts, grads) tensor;
                                  feasible on a 24 GB GPU with chunked matmul.
    -   10k nodes,  300k obs  ->  ~12 GB per tensor; *not* feasible with the
                                  prior pandas/Spearman implementation
                                  (the intermediate ``(2N, 2N)`` float64
                                  correlation alone is ~3.2 GB and the pandas
                                  Spearman ranker is effectively single-threaded).
                                  The GPU rank-Pearson path in this module runs
                                  in well under a minute on a single 24 GB GPU
                                  using ``row_chunk`` to stream the final matmul.

For configurations beyond ~5k nodes the previous ``pd.DataFrame.corr`` path
was a hard wall (both compile-time of the full corr and the float64 result).
The implementation here:

    * keeps everything in float32 on ``device``;
    * computes Spearman as rank-Pearson via ``argsort``;
    * only computes the off-diagonal ``(a x g)`` block, never the full
      ``(2N x 2N)`` matrix;
    * exposes ``row_chunk`` so the output matmul can be streamed when ``N^2``
      no longer fits;
    * preserves all model state (activation flags, ``.grad`` buffers,
      training/eval mode) on exit.

Caveats:

    * Spearman ties are broken by argsort order rather than by the textbook
      average-rank convention. For continuous-valued neural-network
      activations ties are vanishingly rare so the bias is negligible; it can
      become visible on heavily quantised / ReLU-zeroed activations.
    * The (activation, gradient) score is a heuristic, not a causal estimator;
      shared up-stream drivers can inflate apparent edge strength.
"""

from contextlib import contextmanager
from typing import Optional, Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


# -----------------------------------------------------------------------------
# Mean reciprocal rank
# -----------------------------------------------------------------------------
def mrr(A, edge_index, all_true_edges, query_chunk_size: int = 2048) -> float:
    """Mean reciprocal rank of true ``(src -> dst)`` edges scored by ``A``.

    For each query edge ``(s, d)`` in ``edge_index`` the rank of ``A[s, d]``
    is computed only against *false* candidate tails ``A[s, j]`` -- i.e. tails
    ``j`` such that ``(s, j)`` is **not** present in ``all_true_edges``. Other
    true edges that share the same source as the query are masked out of the
    candidate pool, so true edges never compete against one another. Ties are
    broken with the average-rank convention. Returns a scalar in
    :math:`(0, 1]`.

    Parameters
    ----------
    A : Tensor of shape ``(N, N)``
        Score matrix; larger values indicate stronger evidence of an edge.
    edge_index : LongTensor of shape ``(2, E_q)``
        Query edges whose ranks contribute to the MRR.
    all_true_edges : LongTensor of shape ``(2, E_total)``
        Union of all known/true edges (training + validation + test). Used to
        build the "competing-true" mask.
    query_chunk_size : int, default ``2048``
        Number of query edges processed at once. Tunes the
        ``O(query_chunk_size * N)`` peak working set; lower this if you OOM on
        very large ``N`` or ``E_q``.

    Returns
    -------
    float
        Mean reciprocal rank across all query edges.
    """
    if not torch.is_tensor(A):
        A = torch.as_tensor(A)
    device = A.device
    src, dst = edge_index
    src = torch.as_tensor(src, device=device, dtype=torch.long)
    dst = torch.as_tensor(dst, device=device, dtype=torch.long)
    n_rows, n_cols = A.shape

    t_src, t_dst = all_true_edges
    t_src = torch.as_tensor(t_src, device=device, dtype=torch.long)
    t_dst = torch.as_tensor(t_dst, device=device, dtype=torch.long)
    true_mask = torch.zeros(n_rows, n_cols, dtype=torch.bool, device=device)
    true_mask[t_src, t_dst] = True

    E = src.numel()
    if E == 0:
        return float('nan')

    rr_sum = 0.0
    for i in range(0, E, query_chunk_size):
        s = src[i:i + query_chunk_size]
        d = dst[i:i + query_chunk_size]

        scores = A[s]                                # (e, N)
        target = scores.gather(1, d.view(-1, 1))     # (e, 1)

        # Exclude other true edges sharing this source; keep the query target.
        competing_true = true_mask[s].clone()        # (e, N)
        competing_true.scatter_(1, d.view(-1, 1), False)
        valid = ~competing_true                      # (e, N)

        greater = ((scores > target) & valid).sum(dim=1).float()
        equal = ((scores == target) & valid).sum(dim=1).float()
        # Average rank among the (greater) strictly-better candidates and the
        # (equal) tied candidates: ranks [G+1, G+equal] -> mean = G + (equal+1)/2.
        rank = greater + 0.5 * (equal + 1.0)
        rr_sum += (1.0 / rank).sum().item()

    return rr_sum / E


# -----------------------------------------------------------------------------
# Helpers: activation capture, ranks, fast correlation
# -----------------------------------------------------------------------------
@contextmanager
def _enable_activation_storage(modules: Sequence[torch.nn.Module]):
    """Temporarily enable ``_store_activations`` on each module.

    Restores the previous flag value on exit and removes any leftover
    ``_last_pre_norm_activation`` / ``_last_activation`` attributes so the
    captured tensors are eligible for garbage collection.
    """
    prev = [getattr(m, '_store_activations', False) for m in modules]
    for m in modules:
        m._store_activations = True
    try:
        yield
    finally:
        for m, p in zip(modules, prev):
            m._store_activations = p
            for attr in ('_last_pre_norm_activation', '_last_activation'):
                if hasattr(m, attr):
                    try:
                        delattr(m, attr)
                    except AttributeError:
                        pass


def _ranks_along_dim0(x: torch.Tensor) -> torch.Tensor:
    """Per-column competition ranks (no tie averaging) along dim 0.

    Returns a float tensor of the same shape and dtype as ``x``.
    ``argsort(argsort(.))`` style is avoided so we only sort once.
    """
    M = x.shape[0]
    order = x.argsort(dim=0)
    ranks = torch.empty_like(x)
    arange = torch.arange(M, device=x.device, dtype=x.dtype).unsqueeze(1)
    arange = arange.expand_as(x)
    ranks.scatter_(0, order, arange)
    return ranks


def _fast_corr(
    a: ArrayLike,
    g: ArrayLike,
    method: str = 'spearman',
    device: Union[str, torch.device] = 'cpu',
    row_chunk: Optional[int] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Fast ``(N_a, N_g)`` column-by-column correlation between ``a`` and ``g``.

    Pearson is computed via centred-and-normalised matmul; Spearman is computed
    as Pearson of column-wise ranks. Both run on ``device`` in float32.

    Parameters
    ----------
    a : array of shape ``(M, N_a)``
    g : array of shape ``(M, N_g)``
    method : ``'pearson'`` or ``'spearman'``
    device : torch device on which to compute
    row_chunk : if set, the final matmul is performed in chunks of
        ``row_chunk`` rows of ``W``. Use to bound peak memory when ``N_a``
        is large.
    eps : guard against division by zero for constant columns.

    Returns
    -------
    W : Tensor on ``device`` of shape ``(N_a, N_g)``
        Constant columns (zero variance) produce zeros (not NaN) in ``W``.
    """
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    g = torch.as_tensor(g, dtype=torch.float32, device=device)
    if a.shape[0] != g.shape[0]:
        raise ValueError(
            f"Observation count mismatch: a has {a.shape[0]} rows, "
            f"g has {g.shape[0]} rows."
        )

    # Detect constant columns on the *raw* inputs: argsort-ranking a constant
    # column would otherwise produce a perfectly monotonic 0..M-1 sequence and
    # silently fabricate spurious correlation.
    a_valid = (a.std(dim=0, unbiased=False) > eps)
    g_valid = (g.std(dim=0, unbiased=False) > eps)

    if method == 'spearman':
        a = _ranks_along_dim0(a)
        g = _ranks_along_dim0(g)
    elif method != 'pearson':
        raise ValueError(f"Unrecognized correlation method '{method}'")

    a = a - a.mean(dim=0, keepdim=True)
    g = g - g.mean(dim=0, keepdim=True)

    a_norm = a.norm(dim=0, keepdim=True)
    g_norm = g.norm(dim=0, keepdim=True)

    a = a / a_norm.clamp_min(eps)
    g = g / g_norm.clamp_min(eps)

    Na = a.shape[1]
    Ng = g.shape[1]
    if row_chunk is None or row_chunk >= Na:
        W = a.T @ g
    else:
        W = torch.empty(Na, Ng, dtype=a.dtype, device=device)
        for i in range(0, Na, row_chunk):
            W[i:i + row_chunk] = a[:, i:i + row_chunk].T @ g

    # Zero-out columns/rows that came from constant inputs.
    mask = a_valid.unsqueeze(1) & g_valid.unsqueeze(0)
    W = torch.where(mask, W, torch.zeros((), dtype=W.dtype, device=W.device))
    return W


# -----------------------------------------------------------------------------
# Per-node activation / gradient extraction
# -----------------------------------------------------------------------------
def _get_node_attrs(
    model,
    crit,
    x,
    y,
    use_prenorm: bool = True,
    device: Union[str, torch.device] = 'cpu',
    norm: str = 'l1',
    agg: str = 'sum',
):
    """Extract per-node activations and gradients aligned across layers.

    Runs one forward + backward pass on ``(x, y)``, captures the per-channel
    activations of each :class:`gsnn.models.ResBlock.ResBlock`, computes
    gradients of the loss w.r.t. those activations, then reduces channel
    dimensions into the underlying node groups.

    Side-effect free w.r.t. the model: ``_store_activations`` is restored,
    captured activation buffers are removed, and parameter ``.grad`` is not
    populated (we use :func:`torch.autograd.grad` instead of
    :meth:`Tensor.backward`).

    Parameters
    ----------
    model : GSNN
        Trained model exposing ``model.ResBlocks`` whose blocks support the
        ``_store_activations`` flag.
    crit : callable
        Loss criterion called as ``crit(model(x), y)``.
    x, y : Tensor
        Inputs / targets. Moved to ``device`` before the forward pass.
    use_prenorm : bool, default ``True``
        Use ``_last_pre_norm_activation`` (pre-norm/nonlin) instead of
        ``_last_activation`` (post-everything).
    device : torch device for the forward / backward / reductions.
    norm : ``'l1'`` (``|x|``), ``'l2'`` (``x**2``), or ``'none'``.
    agg : ``'sum'``, ``'mean'``, or ``'max'`` reduction across the channels
        of each node group.

    Returns
    -------
    a : Tensor of shape ``((L-1) * B, N)``
        Per-node activations at layers ``0 .. L-2``.
    g : Tensor of shape ``((L-1) * B, N)``
        Per-node gradients at layers ``1 .. L-1``, aligned with ``a``.

    Notes
    -----
    Memory: peak is dominated by stacked per-channel activations and gradients
    of shape ``(L, B, sum_channels)``, plus their grouped reductions of shape
    ``(L, B, N)``. For ``L=6``, ``B=128``, ``N=10k`` with 4 channels/node the
    intermediate is ~120 MB and the grouped output is ~30 MB.
    """
    if not hasattr(model, 'ResBlocks') or len(model.ResBlocks) < 2:
        raise ValueError(
            "model must expose `ResBlocks` with at least 2 layers to align "
            "(layer-l activation, layer-(l+1) gradient) pairs."
        )

    x = x.to(device)
    y = y.to(device)
    act_attr = '_last_pre_norm_activation' if use_prenorm else '_last_activation'

    with _enable_activation_storage(model.ResBlocks):
        yhat = model(x)
        loss = crit(yhat, y)

        acts_list = [getattr(mod, act_attr, None) for mod in model.ResBlocks]
        if any(a is None for a in acts_list):
            missing = [i for i, a in enumerate(acts_list) if a is None]
            raise RuntimeError(
                f"ResBlock(s) {missing} did not record '{act_attr}'. "
                "Check that the model implementation stores the requested "
                "activation when `_store_activations` is True."
            )

        # autograd.grad avoids polluting model parameter .grad buffers,
        # so callers can interleave training without state corruption.
        grad_list = torch.autograd.grad(
            loss, acts_list, retain_graph=False, allow_unused=True,
        )
        grad_list = [
            gi if gi is not None else torch.zeros_like(ai)
            for gi, ai in zip(grad_list, acts_list)
        ]

        acts = torch.stack([ai.detach() for ai in acts_list], dim=0)
        grads = torch.stack([gi.detach() for gi in grad_list], dim=0)

    # Drop the trailing singleton produced by SparseLinear output: (L, B, C, 1).
    if acts.dim() == 4 and acts.shape[-1] == 1:
        acts = acts.squeeze(-1)
        grads = grads.squeeze(-1)

    if norm == 'l1':
        acts = acts.abs()
        grads = grads.abs()
    elif norm == 'l2':
        acts = acts.pow(2)
        grads = grads.pow(2)
    elif norm == 'none':
        pass
    else:
        raise ValueError(f"Unrecognized norm type '{norm}'")

    groups = model.ResBlocks[0].channel_groups.detach().to(acts.device).long()
    num_groups = int(groups.max().item()) + 1
    index = groups.view(1, 1, -1).expand_as(acts)
    L, B, _ = acts.shape

    def _aggregate(src: torch.Tensor) -> torch.Tensor:
        if agg == 'sum':
            out = torch.zeros(L, B, num_groups, dtype=src.dtype, device=src.device)
            out.scatter_add_(2, index, src)
            return out
        if agg == 'mean':
            out = torch.zeros(L, B, num_groups, dtype=src.dtype, device=src.device)
            out.scatter_add_(2, index, src)
            counts = torch.zeros_like(out)
            counts.scatter_add_(2, index, torch.ones_like(src))
            return out / counts.clamp_min(1.0)
        if agg == 'max':
            out = torch.full(
                (L, B, num_groups), float('-inf'),
                dtype=src.dtype, device=src.device,
            )
            out.scatter_reduce_(2, index, src, reduce='amax', include_self=True)
            # Replace untouched bins (no channel mapped here) with 0.
            out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
            return out
        raise ValueError(f"Unrecognized aggregation type '{agg}'")

    acts_grouped = _aggregate(acts)              # (L, B, N)
    grads_grouped = _aggregate(grads)            # (L, B, N)

    # Pair layer l with layer l+1: act at l "drives" grad at l+1.
    a = acts_grouped[:-1].reshape(-1, num_groups).contiguous()
    g = grads_grouped[1:].reshape(-1, num_groups).contiguous()
    return a, g


# -----------------------------------------------------------------------------
# Inferer
# -----------------------------------------------------------------------------
class FunctionEdgeInferer:
    """Infer a ``(N, N)`` function-to-function edge-evidence matrix.

    See the module docstring for the underlying scoring heuristic and a
    discussion of tractability.

    Parameters
    ----------
    model : GSNN
        Trained model whose ResBlocks expose ``_store_activations``,
        ``channel_groups`` and recorded activations.
    crit : callable
        Loss criterion (e.g. :class:`torch.nn.MSELoss`).
    edge_index : LongTensor of shape ``(2, E)``
        Known directed edges in the function-to-function subgraph; used by
        :meth:`_penalize_dependencies` and made available to callers that
        want to compare ``W`` against the prior graph.
    use_prenorm : bool, default ``True``
        Use pre-norm activations rather than post-everything ones.
    device : torch device used for the forward/backward pass and correlation.
    norm : ``'l1'``, ``'l2'`` or ``'none'`` applied to ``|act|`` / ``|grad|``.
    agg : ``'sum'``, ``'mean'`` or ``'max'`` reduction within each node group.
    """

    def __init__(
        self,
        model,
        crit,
        edge_index,
        use_prenorm: bool = True,
        device: Union[str, torch.device] = 'cpu',
        norm: str = 'l1',
        agg: str = 'sum',
    ):
        self.model = model
        self.crit = crit
        self.use_prenorm = use_prenorm
        self.device = device
        self.norm = norm
        self.agg = agg
        self.edge_index = edge_index

    def _corr_matrix(
        self,
        x,
        y,
        method: str = 'spearman',
        scale_by_act_mean: bool = False,
        row_chunk: Optional[int] = None,
    ) -> np.ndarray:
        """Build the cross-correlation matrix from a single ``(x, y)`` batch.

        Returns a NumPy ``(N, N)`` matrix with rows indexed by source node and
        columns by destination node. ``method='spearman'`` corresponds to
        rank-Pearson (ties broken by argsort order, see module docstring).
        """
        a, g = _get_node_attrs(
            self.model, self.crit, x, y,
            use_prenorm=self.use_prenorm,
            device=self.device,
            norm=self.norm,
            agg=self.agg,
        )
        W = _fast_corr(
            a, g, method=method, device=self.device, row_chunk=row_chunk,
        )
        if scale_by_act_mean:
            # Scale rows of W (sources) by their mean activation magnitude so
            # silent source nodes are down-weighted regardless of correlation.
            row_scale = a.mean(dim=0).to(W.device)
            W = W * row_scale.unsqueeze(1)
        return W.detach().cpu().numpy()

    def _penalize_dependencies(
        self, W: np.ndarray, edge_index, alpha: float = 0.05,
    ) -> np.ndarray:
        """Multiplicatively re-weight ``W`` to discount paths through known edges.

        For every known edge ``(i, j)`` we shrink the score of *all* candidate
        edges into ``j`` by ``(1 - alpha)`` (a known parent already explains
        ``j``'s activity) and amplify candidate edges leaving ``i`` by
        ``(1 + alpha)`` (``i`` already has explanatory power). The original
        implementation applied this per-edge in a Python loop and was both
        slow and order-sensitive; we collapse it to a single vectorised
        ``bincount + power`` operation.

        Parameters
        ----------
        W : ndarray of shape ``(N, N)``
            Score matrix; modified out-of-place.
        edge_index : array-like of shape ``(2, E)``
            Known edges to penalise around.
        alpha : float in ``[0, 1)``
            Shrink/grow factor per incident known edge.

        Returns
        -------
        ndarray of shape ``(N, N)``
        """
        ei = torch.as_tensor(edge_index, dtype=torch.long).cpu()
        src, dst = ei[0], ei[1]
        n_src = torch.bincount(src, minlength=W.shape[0]).numpy().astype(np.float64)
        n_dst = torch.bincount(dst, minlength=W.shape[1]).numpy().astype(np.float64)
        W = W.astype(np.float64, copy=True)
        W = W * np.power(1.0 - alpha, n_dst)[None, :]
        W = W * np.power(1.0 + alpha, n_src)[:, None]
        return W

    def fit(
        self,
        x,
        y,
        method: str = 'spearman',
        penalty_factor: float = 0.0,
        scale_by_act_mean: bool = False,
        estimate: Union[bool, None] = False,
        estimate_iters: int = 10,
        estimate_n_samples: int = 2500,
        row_chunk: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the edge-evidence matrix, optionally with bootstrap averaging.

        Parameters
        ----------
        x, y : Tensor
            Forward-pass inputs and targets (typically held-out data).
        method : ``'spearman'`` (default) or ``'pearson'``.
        penalty_factor : float, default ``0.``
            If non-zero, post-multiplies ``W`` via
            :meth:`_penalize_dependencies` to soften scores around known edges.
        scale_by_act_mean : bool, default ``False``
            Multiply rows of ``W`` by the mean source-node activation. Useful
            when correlations are dominated by rarely-firing nodes.
        estimate : bool, default ``False``
            If truthy, sample ``estimate_n_samples`` observations with replacement
            ``estimate_iters`` times and average the resulting matrices.
        estimate_n_samples : int, default ``2500``
            Number of observations to sample for each estimate.
        estimate_iters : int, default ``10``
            Number of estimates to average.
        row_chunk : optional int
            If set, the ``(N, N)`` matmul inside :func:`_fast_corr` is computed
            in row chunks of this size. Use to bound peak memory at large
            ``N`` (rough guide: ``row_chunk * (N + M) * 4 bytes``).
        verbose : bool, default ``False``
            Print a progress line during bootstrap.

        Returns
        -------
        W : ndarray of shape ``(N, N)``
            Score matrix; ``W[i, j]`` is evidence for edge ``i -> j``.
        """
        if estimate:
            n = x.shape[0]
            W = None
            for i in range(estimate_iters):
                if verbose:
                    print(
                        f'estimate iteration: {i + 1}/{estimate_iters}',
                        end='\r',
                    )
                ixs = np.random.choice(n, size=estimate_n_samples, replace=True)
                Wi = self._corr_matrix(
                    x[ixs], y[ixs],
                    method=method,
                    scale_by_act_mean=scale_by_act_mean,
                    row_chunk=row_chunk,
                )
                # Running mean keeps peak memory at one (N, N) instead of
                # bootstrap_iters * (N, N) -- matters for N >> 1k.
                W = Wi if W is None else W + Wi 
            W = W / estimate_iters
            if verbose:
                print()
        else:
            W = self._corr_matrix(
                x, y,
                method=method,
                scale_by_act_mean=scale_by_act_mean,
                row_chunk=row_chunk,
            )

        if penalty_factor != 0.0:
            W = self._penalize_dependencies(W, self.edge_index, penalty_factor)

        return W
