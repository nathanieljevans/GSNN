import copy
from typing import Union, List, Optional

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
    
    >>> # Compute IG for only a subset of edges
    >>> edge_mask = np.array([True, False, True, False, True])  # Only integrate edges 0, 2, 4
    >>> df = explainer.explain(x1, x2, target_idx=0, element_mask=edge_mask)
    >>> # Edges 1 and 3 will have None scores; edges 0, 2, 4 have IG attributions
    >>> # Note: Completeness axiom won't hold when using element_mask
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
        *,
        jitter: Optional[torch.Tensor] = None,
        element_mask=None,
        target: str = 'edge',
        reduction: str = 'mean',
    ) -> pd.DataFrame:
        """Compute attributions for *f(x₁) − f(x₂)*.

        Parameters
        ----------
        x1, x2 : torch.Tensor  (shape: [N_in], [1, N_in], or [B, N_in] for batch)
            Two input feature tensors.  They must have identical batch size and
            ordering of nodes. Each pair (x1[i], x2[i]) is explained.
        target_idx : int or list[int]
            Output dimension(s) to explain.  If a list is provided the
            attributions refer to the **sum** of those outputs.
        jitter : torch.Tensor, optional
            Optional noise to perturb the mask path.
        element_mask : torch.Tensor or np.ndarray, optional (shape: [E] or [N])
            Boolean mask indicating which elements to compute IG attributions for.
            If None, all elements are integrated. If provided:
            - True/nonzero elements: integrate from 0 to 1 (normal IG path)
            - False/zero elements: fixed at 1 throughout the path (no integration)
            Elements not in the mask will have None scores in the output.
            
            Note: When using element_mask, the completeness axiom (attributions sum
            to |Δf|) will not hold since only a subset of elements are integrated.
            The attributions measure "contribution to |Δf| while holding other 
            elements fixed at full strength".
        target : str, optional (default='edge')
            Whether to return 'edge' or 'node' level attributions.
        reduction : str, optional (default='mean')
            How to aggregate attributions across batch samples:
            - 'mean': average attributions across samples (default)
            - 'sum': sum attributions across samples
            - 'none': return all per-sample attributions (adds 'sample_idx' column)

        Returns
        -------
        pd.DataFrame
            If target='edge': columns ['source', 'target', 'score'] for edge attributions.
            If target='node': columns ['node', 'score'] for node attributions.
            If reduction='none': additional 'sample_idx' column for batch dimension.
            Elements not in element_mask will have None scores.
        """
        if target not in ['edge', 'node']:
            raise ValueError(f"target must be 'edge' or 'node', got '{target}'")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

        if target == 'edge':
            return self._compute_edge_attributions(x1, x2, target_idx, jitter, element_mask, reduction)
        else:
            return self._compute_node_attributions(x1, x2, target_idx, jitter, element_mask, reduction)

    def _compute_edge_attributions(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_idx: Union[int, List[int]],
        jitter: Optional[torch.Tensor] = None,
        element_mask=None,
        reduction: str = 'mean',
    ) -> pd.DataFrame:
        """Compute edge-level attributions for *f(x₁) − f(x₂)*."""
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        # Ensure batch dimension
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        
        B = x1.size(0)  # batch size
        T = self.n_steps + 1  # number of points along the path (baseline included)
        
        if isinstance(target_idx, int):
            target_idx = [target_idx]

        # -------------------------------------------------------------
        # Process element_mask
        # -------------------------------------------------------------
        if element_mask is not None:
            if isinstance(element_mask, np.ndarray):
                element_mask = torch.from_numpy(element_mask)
            element_mask = element_mask.to(self.device).bool()
            mask_float = element_mask.float().unsqueeze(0)  # (1, E)
        else:
            element_mask = None
            mask_float = None

        # -------------------------------------------------------------
        # Build base mask (optionally perturbed with jitter)
        # -------------------------------------------------------------
        base_mask = torch.ones((1, self.E), device=self.device)
        if jitter is not None:
            jitter = jitter.to(self.device)
            if jitter.dim() == 1:
                jitter = jitter.unsqueeze(0)
            base_mask = torch.clamp(base_mask * (1.0 + jitter), 0.0, 1.0)

        alphas = torch.linspace(0.0, 1.0, T, device=self.device).view(-1, 1)  # (T,1)
        
        # -------------------------------------------------------------
        # Build interpolated mask path
        # -------------------------------------------------------------
        # Standard interpolation from 0 to base_mask
        interpolated = alphas * base_mask  # (T, E)
        
        if mask_float is not None:
            # For masked edges (True): use interpolated values (0 -> base_mask)
            # For unmasked edges (False): fix at 1.0 throughout
            mask_path_template = mask_float * interpolated + (1.0 - mask_float) * 1.0
        else:
            mask_path_template = interpolated

        # ------------------------------------------------------------------
        # Process each sample pair
        # ------------------------------------------------------------------
        all_ig = []
        
        for sample_idx in range(B):
            x1i = x1[sample_idx:sample_idx+1]  # (1, N_in)
            x2i = x2[sample_idx:sample_idx+1]  # (1, N_in)
            
            # Need fresh tensor for gradient computation
            mask_path = mask_path_template.clone().requires_grad_(True)

            # Prepare feature batches replicated along path
            x1_batch = x1i.repeat(T, 1)  # (T , N_in)
            x2_batch = x2i.repeat(T, 1)  # (T , N_in)

            # Concatenate for single forward pass
            x_batch = torch.cat([x1_batch, x2_batch], dim=0)  # (2T , N_in)
            mask_batch = mask_path.repeat(2, 1)  # (2T , E)

            # Forward pass
            preds = self.model(x_batch, edge_mask=mask_batch)[:, target_idx]  # (2T , |T|)
            preds = preds.sum(dim=1)  # (2T ,)

            preds_x1 = preds[:T]
            preds_x2 = preds[T:]

            diff_abs = (preds_x1 - preds_x2).abs()  # (T ,)

            # Back-propagate through mask_path
            grads = torch.autograd.grad(diff_abs.sum(), mask_path)[0]  # (T , E)

            # Trapezoidal rule over the path
            trap = (grads[:-1] + grads[1:]) / 2.0  # (T-1 , E)
            avg_grad = trap.mean(dim=0)            # (E ,)
            ig_scores = avg_grad * base_mask.squeeze(0)  # multiply by Δmask (base_mask - 0)
            
            # Set unmasked edges to NaN
            if element_mask is not None:
                ig_scores = torch.where(element_mask, ig_scores, torch.tensor(float('nan'), device=self.device))
            
            all_ig.append(ig_scores)
        
        all_ig = torch.stack(all_ig, dim=0)  # (B, E)

        # ------------------------------------------------------------------
        # Package results with reduction
        # ------------------------------------------------------------------
        src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        
        if reduction == 'none':
            dfs = []
            for i in range(B):
                scores = all_ig[i].detach().cpu().numpy()
                scores = [None if np.isnan(score) else score for score in scores]
                df = pd.DataFrame({
                    "sample_idx": i,
                    "source": src,
                    "target": dst,
                    "score": scores,
                })
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        elif reduction == 'sum':
            scores_agg = torch.nansum(all_ig, dim=0) if element_mask is not None else all_ig.sum(dim=0)
        else:  # mean
            scores_agg = torch.nanmean(all_ig, dim=0) if element_mask is not None else all_ig.mean(dim=0)
        
        # Convert NaN to None for edges not in mask
        scores = scores_agg.detach().cpu().numpy()
        if element_mask is not None:
            scores = [None if np.isnan(score) else score for score in scores]
        
        return pd.DataFrame({
            "source": src,
            "target": dst,
            "score": scores,
        })

    def _compute_node_attributions(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_idx: Union[int, List[int]],
        jitter: Optional[torch.Tensor] = None,
        element_mask=None,
        reduction: str = 'mean',
    ) -> pd.DataFrame:
        """Compute node-level attributions for *f(x₁) − f(x₂)*."""
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        # Ensure batch dimension
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        
        B = x1.size(0)  # batch size
        T = self.n_steps + 1  # number of points along the path (baseline included)
        N = self.model.num_nodes
        
        if isinstance(target_idx, int):
            target_idx = [target_idx]

        # -------------------------------------------------------------
        # Process element_mask
        # -------------------------------------------------------------
        if element_mask is not None:
            if isinstance(element_mask, np.ndarray):
                element_mask = torch.from_numpy(element_mask)
            element_mask = element_mask.to(self.device).bool()
            mask_float = element_mask.float().unsqueeze(0)  # (1, N)
        else:
            element_mask = None
            mask_float = None

        # -------------------------------------------------------------
        # Build base mask (optionally perturbed with jitter)
        # -------------------------------------------------------------
        base_mask = torch.ones((1, N), device=self.device)
        if jitter is not None:
            jitter = jitter.to(self.device)
            if jitter.dim() == 1:
                jitter = jitter.unsqueeze(0)
            base_mask = torch.clamp(base_mask * (1.0 + jitter), 0.0, 1.0)

        alphas = torch.linspace(0.0, 1.0, T, device=self.device).view(-1, 1)  # (T,1)
        
        # -------------------------------------------------------------
        # Build interpolated mask path
        # -------------------------------------------------------------
        # Standard interpolation from 0 to base_mask
        interpolated = alphas * base_mask  # (T, N)
        
        if mask_float is not None:
            # For masked nodes (True): use interpolated values (0 -> base_mask)
            # For unmasked nodes (False): fix at 1.0 throughout
            mask_path_template = mask_float * interpolated + (1.0 - mask_float) * 1.0
        else:
            mask_path_template = interpolated

        # ------------------------------------------------------------------
        # Process each sample pair
        # ------------------------------------------------------------------
        all_ig = []
        
        for sample_idx in range(B):
            x1i = x1[sample_idx:sample_idx+1]  # (1, N_in)
            x2i = x2[sample_idx:sample_idx+1]  # (1, N_in)
            
            # Need fresh tensor for gradient computation
            mask_path = mask_path_template.clone().requires_grad_(True)

            # Prepare feature batches replicated along path
            x1_batch = x1i.repeat(T, 1)  # (T , N_in)
            x2_batch = x2i.repeat(T, 1)  # (T , N_in)

            # Concatenate for single forward pass
            x_batch = torch.cat([x1_batch, x2_batch], dim=0)  # (2T , N_in)
            mask_batch = mask_path.repeat(2, 1)  # (2T , N)

            # Forward pass
            preds = self.model(x_batch, node_mask=mask_batch)[:, target_idx]  # (2T , |T|)
            preds = preds.sum(dim=1)  # (2T ,)

            preds_x1 = preds[:T]
            preds_x2 = preds[T:]

            diff_abs = (preds_x1 - preds_x2).abs()  # (T ,)

            # Back-propagate through mask_path
            grads = torch.autograd.grad(diff_abs.sum(), mask_path)[0]  # (T , N)

            # Trapezoidal rule over the path
            trap = (grads[:-1] + grads[1:]) / 2.0  # (T-1 , N)
            avg_grad = trap.mean(dim=0)            # (N ,)
            ig_scores = avg_grad * base_mask.squeeze(0)  # multiply by Δmask (base_mask - 0)
            
            # Set unmasked nodes to NaN
            if element_mask is not None:
                ig_scores = torch.where(element_mask, ig_scores, torch.tensor(float('nan'), device=self.device))
            
            all_ig.append(ig_scores)
        
        all_ig = torch.stack(all_ig, dim=0)  # (B, N)

        # ------------------------------------------------------------------
        # Package results with reduction
        # ------------------------------------------------------------------
        node_names = np.array(self.model.homo_names)
        
        if reduction == 'none':
            dfs = []
            for i in range(B):
                scores = all_ig[i].detach().cpu().numpy()
                scores = [None if np.isnan(score) else score for score in scores]
                df = pd.DataFrame({
                    "sample_idx": i,
                    "node": node_names,
                    "score": scores,
                })
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        elif reduction == 'sum':
            scores_agg = torch.nansum(all_ig, dim=0) if element_mask is not None else all_ig.sum(dim=0)
        else:  # mean
            scores_agg = torch.nanmean(all_ig, dim=0) if element_mask is not None else all_ig.mean(dim=0)
        
        # Convert NaN to None for nodes not in mask
        scores = scores_agg.detach().cpu().numpy()
        if element_mask is not None:
            scores = [None if np.isnan(score) else score for score in scores]
        
        return pd.DataFrame({
            "node": node_names,
            "score": scores,
        }) 