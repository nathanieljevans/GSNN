r"""Pathway latent-factor regularizer for :class:`gsnn.models.GSNN`.

Implements the latent-factor objective described in
``docs/notes/functional_dis_and_similarity.md`` (section 5).

Given a pathway membership matrix ``M`` of shape ``(P, N_func)`` over the
function nodes of a :class:`GSNN` model, the regularizer encourages members of
the same pathway to *co-vary across the batch dimension* with a shared
per-pathway latent score, and (optionally) penalizes correlation between
designated dissimilar pathway pairs. This biases the model toward solutions
consistent with prior pathway co-membership without modifying the network
topology or adding hub nodes.

The regularizer is opt-in and fully backward compatible: it does not modify
:class:`GSNN`. It only toggles the existing :pyobj:`ResBlock._store_activations`
flag and reads :pyobj:`ResBlock._last_activation` after a training-mode forward
pass.

Example
-------
>>> from gsnn.models import GSNN, PathwayLatentRegularizer
>>> model = GSNN(edge_index_dict, node_names_dict, channels=8, layers=3)
>>> reg = PathwayLatentRegularizer(model, pathway_membership=M, lambda_sim=0.1)
>>> for x, y in loader:
...     yhat = model(x)
...     L_main = mse(y, yhat)
...     L_sim, L_dis = reg.loss(model)
...     (L_main + L_sim + L_dis).backward()
"""

import torch
import torch.nn as nn


class PathwayLatentRegularizer(nn.Module):
    r"""Per-pathway latent-factor auxiliary loss for GSNN.

    For each :class:`ResBlock` :math:`\ell` and minibatch of size :math:`B`:

    1. Reduce the cached activation
       :math:`A_\ell \in \mathbb{R}^{B \times N_{\text{func}} \times C_{pn}}`
       to per-node scalars
       :math:`s_\ell \in \mathbb{R}^{B \times N_{\text{func}}}` via
       :math:`\phi`.
    2. Compute per-pathway scores
       :math:`S_\ell = \mathrm{normalize}(M\, s_\ell^\top)^\top \in
       \mathbb{R}^{B \times P}`.
    3. Standardize across the batch dimension and compute the
       member-by-pathway correlation matrix
       :math:`C \in \mathbb{R}^{N_{\text{func}} \times P}`.
    4. Add the negative member-side correlation to :math:`L_{\text{sim}}`,
       and (if ``dissim_pairs`` is provided) the squared score-correlation of
       dissimilar pairs to :math:`L_{\text{dis}}`.

    Cost is :math:`O(L \cdot B \cdot (N_{\text{func}} \cdot C_{pn} + P \cdot
    N_{\text{func}}))` per minibatch and parameter count is either zero
    (``phi='mean'``) or :math:`C_{pn}` (``phi='learned'``).

    Args:
        model (GSNN): Reference model. Used only for shape introspection
            (``ResBlocks[0].channel_groups``) and to toggle
            :pyobj:`_store_activations`.
        pathway_membership (Tensor): Float tensor of shape
            :obj:`(P, N_func)`. Real-valued entries are allowed and treated as
            soft membership weights; binary 0/1 entries give classical hard
            membership. Rows correspond to pathways, columns to function-node
            indices in the same order as the model's ``function`` node names.
        dissim_pairs (Tensor, optional): :obj:`(M, 2)` LongTensor of pathway
            index pairs whose scores should be encouraged to be uncorrelated.
            (default: :obj:`None`)
        lambda_sim (float, optional): Scaling for the similarity term.
            (default: :obj:`0.1`)
        lambda_dis (float, optional): Scaling for the dissimilarity term.
            (default: :obj:`0.0`)
        phi (str or nn.Module, optional): Per-node scalar reduction.
            ``'mean'`` averages across channels (no parameters). ``'learned'``
            uses a single :class:`nn.Linear` projection ``Linear(C_pn, 1)``
            shared across layers. A custom :class:`nn.Module` may also be
            passed; it must accept a tensor of shape
            :obj:`(B, N_func, C_pn)` and return :obj:`(B, N_func, 1)` or
            :obj:`(B, N_func)`. (default: :obj:`'mean'`)
        eps (float, optional): Small constant for numerical stability in the
            standardization step. (default: :obj:`1e-6`)
    """

    def __init__(
        self,
        model,
        pathway_membership,
        dissim_pairs=None,
        lambda_sim=0.1,
        lambda_dis=0.0,
        phi="mean",
        eps=1e-6,
    ):
        super().__init__()

        if not hasattr(model, "ResBlocks") or len(model.ResBlocks) == 0:
            raise ValueError(
                "model must be a GSNN instance with at least one ResBlock."
            )

        # Derive per-node and per-channel sizes from the first block. GSNN
        # currently uses uniform channel counts across function nodes; this is
        # the same assumption made by ResBlock and NodeAttention internally.
        channel_groups = model.ResBlocks[0].channel_groups
        self.n_func = int(channel_groups.max().item() + 1)
        self.c_pn = int(channel_groups.numel() // self.n_func)

        if not isinstance(pathway_membership, torch.Tensor):
            pathway_membership = torch.as_tensor(pathway_membership)
        if pathway_membership.dim() != 2:
            raise ValueError(
                "pathway_membership must be 2-D (P, N_func); got "
                f"shape {tuple(pathway_membership.shape)}."
            )
        if pathway_membership.size(1) != self.n_func:
            raise ValueError(
                f"pathway_membership has {pathway_membership.size(1)} columns; "
                f"expected {self.n_func} (= number of function nodes)."
            )

        self.register_buffer("M", pathway_membership.float())

        if dissim_pairs is not None:
            if not isinstance(dissim_pairs, torch.Tensor):
                dissim_pairs = torch.as_tensor(dissim_pairs)
            if dissim_pairs.dim() != 2 or dissim_pairs.size(-1) != 2:
                raise ValueError(
                    "dissim_pairs must be of shape (M, 2); got "
                    f"shape {tuple(dissim_pairs.shape)}."
                )
            self.register_buffer("dissim_pairs", dissim_pairs.long())
        else:
            self.dissim_pairs = None

        self.lambda_sim = float(lambda_sim)
        self.lambda_dis = float(lambda_dis)
        self.eps = float(eps)

        if phi == "mean":
            self.phi = None
        elif phi == "learned":
            self.phi = nn.Linear(self.c_pn, 1, bias=False)
        elif isinstance(phi, nn.Module):
            self.phi = phi
        else:
            raise ValueError(
                "phi must be 'mean', 'learned', or an nn.Module; got "
                f"{phi!r}."
            )

        self.enable(model)

    def enable(self, model):
        r"""Enable activation caching on every ResBlock of ``model``.

        Idempotent. Called automatically at construction time. Returns
        ``self`` for chaining.
        """
        for blk in model.ResBlocks:
            blk._store_activations = True
        return self

    def disable(self, model):
        r"""Disable activation caching and clear cached tensors on ``model``.

        Useful at evaluation time to avoid retaining graph references. Returns
        ``self`` for chaining.
        """
        for blk in model.ResBlocks:
            blk._store_activations = False
            if hasattr(blk, "_last_activation"):
                blk._last_activation = None
        return self

    def _reduce(self, A):
        r"""Reduce a cached activation tensor to per-(sample, node) scalars.

        Args:
            A (Tensor): Cached activation of shape :obj:`(B, N_func * C_pn)`
                or :obj:`(B, N_func * C_pn, 1)`.

        Returns:
            Tensor: :obj:`(B, N_func)`.
        """
        A = A.squeeze(-1).reshape(A.size(0), self.n_func, self.c_pn)
        if self.phi is None:
            return A.mean(-1)
        out = self.phi(A)
        if out.dim() == 3 and out.size(-1) == 1:
            out = out.squeeze(-1)
        return out

    def loss(self, model):
        r"""Compute the auxiliary similarity / dissimilarity losses.

        Must be called *after* a training-mode forward pass so that each
        :class:`ResBlock` has populated its :pyobj:`_last_activation`.

        Args:
            model (GSNN): The same model passed to :py:meth:`__init__`.

        Returns:
            tuple: ``(L_sim, L_dis)`` — both already scaled by their
            respective :obj:`lambda_*`. ``L_dis`` is :obj:`0.0` if no
            ``dissim_pairs`` were provided.
        """
        L_sim = torch.zeros((), device=self.M.device)
        L_dis = torch.zeros((), device=self.M.device)
        n_layers = 0

        # Member mask reused across layers.
        member_mask = self.M.T  # (N_func, P)
        member_norm = member_mask.sum().clamp_min(1)
        pathway_size = self.M.sum(1).clamp_min(1)  # (P,)

        for blk in model.ResBlocks:
            A = getattr(blk, "_last_activation", None)
            if A is None:
                raise RuntimeError(
                    "ResBlock._last_activation is missing. Run a forward "
                    "pass with the model in training mode before calling "
                    "loss(); the regularizer's enable() must have set "
                    "_store_activations=True (this happens automatically "
                    "at construction)."
                )
            n_layers += 1

            s = self._reduce(A)                         # (B, N_func)
            S = (self.M @ s.t()).t() / pathway_size      # (B, P)

            s_std = (s - s.mean(0, keepdim=True)) / (
                s.std(0, unbiased=False, keepdim=True) + self.eps
            )
            S_std = (S - S.mean(0, keepdim=True)) / (
                S.std(0, unbiased=False, keepdim=True) + self.eps
            )

            # member <-> pathway-score correlation
            corr = (s_std.t() @ S_std) / s_std.size(0)   # (N_func, P)
            L_sim = L_sim - (corr * member_mask).sum() / member_norm

            if self.dissim_pairs is not None and self.dissim_pairs.numel() > 0:
                Sp = S_std[:, self.dissim_pairs[:, 0]]
                Sq = S_std[:, self.dissim_pairs[:, 1]]
                pair_corr = (Sp * Sq).mean(0)            # (M,)
                L_dis = L_dis + (pair_corr ** 2).mean()

        if n_layers > 1:
            L_sim = L_sim / n_layers
            L_dis = L_dis / n_layers

        return self.lambda_sim * L_sim, self.lambda_dis * L_dis
