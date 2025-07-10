import copy
from typing import Union, List

import numpy as np
import pandas as pd
import torch


class ContrastiveIGExplainer:
    r"""Edge-level Integrated-Gradients explainer for *contrastive* questions.

    This module attributes the prediction **difference**::

        Δf = f(x1)[target_idx] - f(x2)[target_idx]

    to every edge *e* in the graph by integrating along a straight-line **mask
    path** *m(α)=α·1, α∈[0,1]* while keeping the two inputs ``x1`` and ``x2``
    fixed.  The attribution for an edge equals

    .. math::

        \mathrm{IG}_e = \int_0^1 \frac{\partial}{\partial m_e}
                         |f(x_1;m(α)) - f(x_2;m(α))|\,dα.

    Interpretation
    --------------
    * ``IG_e > 0``  the presence of edge *e* increases |Δf|.
    * ``IG_e < 0``  the presence of edge *e* decreases |Δf|.
    * ``IG_e ≈ 0``  edge *e* is irrelevant to the gap.

    By construction :math:`\sum_e \mathrm{IG}_e = |Δf|` (completeness).

    Parameters
    ----------
    model : torch.nn.Module
        Trained GSNN model (evaluation mode is enforced internally).
    data : torch_geometric.data.Data
        Graph data object; only used for human-readable edge names.
    n_steps : int, optional (default=50)
        Number of interpolation points along the mask path (baseline included).
    ignore_cuda : bool, optional (default=False)
        Force the explainer to run on CPU even if CUDA is available.

    Example
    -------
    >>> explainer = ContrastiveIGExplainer(model, data, n_steps=64)
    >>> df = explainer.explain(x1, x2, target_idx=0)
    >>> df.sort_values('score', ascending=False).head()
    source target   score
    in0    func0    0.42
    func0  func3    0.40
    func3  out0     0.38
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data,
        n_steps: int = 50,
        ignore_cuda: bool = False,
    ) -> None:
        self.data = data
        self.n_steps = n_steps
        self.device = (
            "cuda" if (torch.cuda.is_available() and not ignore_cuda) else "cpu"
        )

        # Work on a frozen copy of the model to avoid side-effects.
        model = copy.deepcopy(model).eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

        # Constant edge-mask variable (value = 1); gradients will flow through.
        self.E = model.edge_index.size(1)
        self._edge_mask = torch.ones((1, self.E), device=self.device, requires_grad=True)


    def explain(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_idx: Union[int, List[int]],
    ) -> pd.DataFrame:
        """Compute edge attributions for *f(x₁) − f(x₂)*.

        Parameters
        ----------
        x1, x2 : torch.Tensor  (shape: [B, N_in])
            Two input feature tensors.  They must have identical batch size and
            ordering of nodes.
        target_idx : int or list[int]
            Output dimension(s) to explain.  If a list is provided the
            attributions refer to the **sum** of those outputs.
        """

        # ------------------------------------------------------------------
        # 1.  Build mask path  m(α) = α · 1  ,  α ∈ [0,1]
        # ------------------------------------------------------------------
        T = self.n_steps + 1  # number of points along the path (baseline included)
        alphas = torch.linspace(0.0, 1.0, T, device=self.device).view(-1, 1)  # (T,1)

        # mask_path shape: (T , E)  –  each row is α * 1
        mask_path = (alphas * torch.ones((1, self.E), device=self.device)).requires_grad_(True)

        # ------------------------------------------------------------------
        # 2.  Prepare constant feature batches (x1, x2) replicated along path
        # ------------------------------------------------------------------
        x1, x2 = x1.to(self.device), x2.to(self.device)
        x1_batch = x1.repeat(T, 1)  # (T , N_in)
        x2_batch = x2.repeat(T, 1)  # (T , N_in)

        # Concatenate so we can run them through the model in one call – this
        # guarantees identical dropout/mask noise across the two inputs.
        x_batch = torch.cat([x1_batch, x2_batch], dim=0)  # (2T , N_in)

        # Repeat masks for the two halves (x1 and x2)
        mask_batch = mask_path.repeat(2, 1)  # (2T , E)

        # ------------------------------------------------------------------
        # 3.  Forward pass and construct the **absolute** prediction difference
        # ------------------------------------------------------------------
        if isinstance(target_idx, int):
            target_idx = [target_idx]

        preds = self.model(x_batch, edge_mask=mask_batch)[:, target_idx]  # (2T , |T|)
        preds = preds.sum(dim=1)  # (2T ,)

        preds_x1 = preds[:T]
        preds_x2 = preds[T:]

        diff_abs = (preds_x1 - preds_x2).abs()  # (T ,)

        # ------------------------------------------------------------------
        # 4.  Back-propagate through mask_path
        # ------------------------------------------------------------------
        grads = torch.autograd.grad(diff_abs.sum(), mask_path)[0]  # (T , E)

        # Trapezoidal rule over the path
        trap = (grads[:-1] + grads[1:]) / 2.0  # (T-1 , E)
        avg_grad = trap.mean(dim=0)            # (E ,)
        ig_scores = avg_grad  # Δmask = 1, so multiply by 1

        # 5.  Package as DataFrame ------------------------------------------------
        src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        return pd.DataFrame({
            "source": src,
            "target": dst,
            "score": ig_scores.detach().cpu().numpy(),
        }) 