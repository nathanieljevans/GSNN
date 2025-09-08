import copy
from typing import Union, List

import numpy as np
import pandas as pd
import torch


class ContrastiveOcclusionExplainer:
    r"""Simple batched edge occlusion explainer for *contrastive* questions.

    This module attributes the prediction **difference**::

        Δf = f(x1)[target_idx] - f(x2)[target_idx]

    to every edge *e* by systematically removing each edge and measuring 
    the change in the absolute prediction difference:

    .. math::

        \mathrm{Occ}_e = |Δf_{\text{baseline}}| - |Δf_{\text{without } e}|

    where the baseline uses all edges present and the occluded version 
    removes edge *e* completely (edge_mask = 0).

    Interpretation
    --------------
    * ``Occ_e > 0``  removing edge *e* decreases |Δf| (edge contributes to difference).
    * ``Occ_e < 0``  removing edge *e* increases |Δf| (edge reduces difference).
    * ``Occ_e ≈ 0``  edge *e* has no impact on the prediction difference.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GSNN model (evaluation mode is enforced internally).
    data : torch_geometric.data.Data
        Graph data object; only used for human-readable edge names.
    batch_size : int, optional (default=32)
        Number of edge occlusions to process in parallel.
    ignore_cuda : bool, optional (default=False)
        Force the explainer to run on CPU even if CUDA is available.
    verbose : bool, optional (default=False)
        Print progress information during explanation computation.

    Example
    -------
    >>> explainer = ContrastiveOcclusionExplainer(model, data, batch_size=64)
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
        batch_size: int = 32,
        ignore_cuda: bool = False,
        verbose: bool = False,
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = (
            "cuda" if (torch.cuda.is_available() and not ignore_cuda) else "cpu"
        )

        # Work on a frozen copy of the model to avoid side-effects.
        model = copy.deepcopy(model).eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

        # Store number of edges
        self.E = model.edge_index.size(1)
        
        if self.verbose:
            print(f"ContrastiveOcclusionExplainer initialized:")
            print(f"  Device: {self.device}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Total edges: {self.E}")


    def explain(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_idx: Union[int, List[int]],
    ) -> pd.DataFrame:
        """Compute edge occlusion attributions for *f(x₁) − f(x₂)*.

        Parameters
        ----------
        x1, x2 : torch.Tensor  (shape: [B, N_in])
            Two input feature tensors.  They must have identical batch size and
            ordering of nodes.
        target_idx : int or list[int]
            Output dimension(s) to explain.  If a list is provided the
            attributions refer to the **sum** of those outputs.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'source', 'target', 'score' containing
            edge attributions.
        """

        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        if isinstance(target_idx, int):
            target_idx = [target_idx]

        if self.verbose:
            print(f"\nStarting contrastive occlusion explanation:")
            print(f"  Input shapes: x1={x1.shape}, x2={x2.shape}")
            print(f"  Target indices: {target_idx}")
            print(f"  Total batches to process: {((self.E - 1) // self.batch_size) + 1}")

        # ------------------------------------------------------------------
        # 1. Compute baseline difference (all edges present)
        # ------------------------------------------------------------------
        if self.verbose:
            print("Computing baseline prediction difference...")
            
        baseline_mask = torch.ones((1, self.E), device=self.device)
        baseline_diff = self._compute_diff(x1, x2, target_idx, baseline_mask)
        
        if self.verbose:
            print(f"  Baseline |Δf| = {baseline_diff:.6f}")

        # ------------------------------------------------------------------
        # 2. Compute occlusion scores in batches
        # ------------------------------------------------------------------
        if self.verbose:
            print("Computing edge occlusion scores...")
            
        occlusion_scores = torch.zeros(self.E, device=self.device)
        
        for start_idx in range(0, self.E, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.E)
            batch_size_actual = end_idx - start_idx
            
            if self.verbose:
                print(f"Processing batch {start_idx // self.batch_size + 1}/{((self.E - 1) // self.batch_size) + 1}", end='\r')

            # Create batch of masks with one edge removed per mask
            batch_masks = torch.ones((batch_size_actual, self.E), device=self.device)
            for i, edge_idx in enumerate(range(start_idx, end_idx)):
                batch_masks[i, edge_idx] = 0.0

            # Replicate inputs for batch processing
            x1_batch = x1.repeat(batch_size_actual, 1)
            x2_batch = x2.repeat(batch_size_actual, 1)
            x_batch = torch.cat([x1_batch, x2_batch], dim=0)
            mask_batch = batch_masks.repeat(2, 1)

            # Forward pass
            preds = self.model(x_batch, edge_mask=mask_batch)[:, target_idx].sum(dim=1)
            preds_x1 = preds[:batch_size_actual]
            preds_x2 = preds[batch_size_actual:]
            
            # Compute occlusion effects
            occluded_diffs = (preds_x1 - preds_x2).abs()
            batch_scores = baseline_diff - occluded_diffs
            occlusion_scores[start_idx:end_idx] = batch_scores

        if self.verbose:
            print("Edge occlusion explanation complete.")

        # ------------------------------------------------------------------
        # 3. Package results
        # ------------------------------------------------------------------
        src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        
        return pd.DataFrame({
            "source": src,
            "target": dst,
            "score": occlusion_scores.detach().cpu().numpy(),
        })


    def _compute_diff(self, x1: torch.Tensor, x2: torch.Tensor, target_idx: List[int], mask: torch.Tensor) -> float:
        """Compute absolute prediction difference with given edge mask."""
        pred1 = self.model(x1, edge_mask=mask)[:, target_idx].sum(dim=1)
        pred2 = self.model(x2, edge_mask=mask)[:, target_idx].sum(dim=1)
        return (pred1 - pred2).abs().item() 