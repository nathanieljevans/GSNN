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
        *,
        element_mask=None,
        target: str = 'edge',
        reduction: str = 'mean',
    ) -> pd.DataFrame:
        """Compute occlusion attributions for *f(x₁) − f(x₂)*.

        Parameters
        ----------
        x1, x2 : torch.Tensor  (shape: [N_in], [1, N_in], or [B, N_in] for batch)
            Two input feature tensors.  They must have identical batch size and
            ordering of nodes. Each pair (x1[i], x2[i]) is explained.
        target_idx : int or list[int]
            Output dimension(s) to explain.  If a list is provided the
            attributions refer to the **sum** of those outputs.
        element_mask : torch.Tensor or np.ndarray, optional (shape: [E] or [N])
            Boolean mask indicating which elements to compute occlusion for.
            If None, all elements are considered. If provided, only elements where
            element_mask[i] is True will have occlusion scores computed.
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
            return self._compute_edge_attributions(x1, x2, target_idx, element_mask, reduction)
        else:
            return self._compute_node_attributions(x1, x2, target_idx, element_mask, reduction)

    def _compute_edge_attributions(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_idx: Union[int, List[int]],
        element_mask=None,
        reduction: str = 'mean',
    ) -> pd.DataFrame:
        """Compute edge-level occlusion attributions for *f(x₁) − f(x₂)*."""
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        # Ensure batch dimension
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        
        B = x1.size(0)  # batch size
        
        if isinstance(target_idx, int):
            target_idx = [target_idx]

        # ------------------------------------------------------------------
        # Process element_mask
        # ------------------------------------------------------------------
        if element_mask is not None:
            if isinstance(element_mask, np.ndarray):
                element_mask = torch.from_numpy(element_mask)
            element_mask = element_mask.to(self.device).bool()
            edges_to_occlude = torch.where(element_mask)[0]
        else:
            edges_to_occlude = torch.arange(self.E, device=self.device)

        num_edges_to_occlude = len(edges_to_occlude)

        if self.verbose:
            print(f"\nStarting contrastive edge occlusion explanation:")
            print(f"  Input shapes: x1={x1.shape}, x2={x2.shape}")
            print(f"  Batch size: {B}")
            print(f"  Target indices: {target_idx}")
            print(f"  Edges to occlude: {num_edges_to_occlude} / {self.E}")
            print(f"  Total edge batches to process: {((num_edges_to_occlude - 1) // self.batch_size) + 1 if num_edges_to_occlude > 0 else 0}")

        # ------------------------------------------------------------------
        # Process each sample pair
        # ------------------------------------------------------------------
        all_scores = []
        
        for sample_idx in range(B):
            x1i = x1[sample_idx:sample_idx+1]  # (1, N_in)
            x2i = x2[sample_idx:sample_idx+1]  # (1, N_in)
            
            if self.verbose and B > 1:
                print(f"Processing sample {sample_idx + 1}/{B}")

            # Compute baseline difference (all edges present)
            baseline_mask = torch.ones((1, self.E), device=self.device)
            baseline_diff = self._compute_diff_edge(x1i, x2i, target_idx, baseline_mask)
            
            if self.verbose:
                print(f"  Baseline |Δf| = {baseline_diff:.6f}")

            # Compute occlusion scores in batches
            occlusion_scores = torch.full((self.E,), float('nan'), device=self.device)
            
            if num_edges_to_occlude > 0:
                for start_idx in range(0, num_edges_to_occlude, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_edges_to_occlude)
                    batch_size_actual = end_idx - start_idx
                    
                    if self.verbose:
                        print(f"Processing edge batch {start_idx // self.batch_size + 1}/{((num_edges_to_occlude - 1) // self.batch_size) + 1}", end='\r')

                    # Get the actual edge indices for this batch
                    batch_edge_indices = edges_to_occlude[start_idx:end_idx]

                    # Create batch of masks with one edge removed per mask
                    batch_masks = torch.ones((batch_size_actual, self.E), device=self.device)
                    for i, edge_idx in enumerate(batch_edge_indices):
                        batch_masks[i, edge_idx] = 0.0

                    # Replicate inputs for batch processing
                    x1_batch = x1i.repeat(batch_size_actual, 1)
                    x2_batch = x2i.repeat(batch_size_actual, 1)
                    x_batch = torch.cat([x1_batch, x2_batch], dim=0)
                    mask_batch = batch_masks.repeat(2, 1)

                    # Forward pass
                    preds = self.model(x_batch, edge_mask=mask_batch)[:, target_idx].sum(dim=1)
                    preds_x1 = preds[:batch_size_actual]
                    preds_x2 = preds[batch_size_actual:]
                    
                    # Compute occlusion effects
                    occluded_diffs = (preds_x1 - preds_x2).abs()
                    batch_scores = baseline_diff - occluded_diffs
                    occlusion_scores[batch_edge_indices] = batch_scores
            
            all_scores.append(occlusion_scores)

        if self.verbose:
            print("\nEdge occlusion explanation complete.")

        all_scores = torch.stack(all_scores, dim=0)  # (B, E)

        # ------------------------------------------------------------------
        # Package results with reduction
        # ------------------------------------------------------------------
        src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        
        if reduction == 'none':
            dfs = []
            for i in range(B):
                scores = all_scores[i].detach().cpu().numpy()
                scores = [None if np.isnan(score) else score for score in scores]
                df = pd.DataFrame({
                    "sample_idx": i,
                    "source": src,
                    "target": dst,
                    "score": scores,
                })
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        
        # For mean/sum, handle NaN values properly
        if reduction == 'sum':
            scores_agg = torch.nansum(all_scores, dim=0)
        else:  # mean
            scores_agg = torch.nanmean(all_scores, dim=0)
        
        # Convert NaN to None for edges not in mask
        scores = scores_agg.detach().cpu().numpy()
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
        element_mask=None,
        reduction: str = 'mean',
    ) -> pd.DataFrame:
        """Compute node-level occlusion attributions for *f(x₁) − f(x₂)*."""
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        # Ensure batch dimension
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        
        B = x1.size(0)  # batch size
        
        if isinstance(target_idx, int):
            target_idx = [target_idx]

        N = self.model.num_nodes

        # ------------------------------------------------------------------
        # Process element_mask
        # ------------------------------------------------------------------
        if element_mask is not None:
            if isinstance(element_mask, np.ndarray):
                element_mask = torch.from_numpy(element_mask)
            element_mask = element_mask.to(self.device).bool()
            nodes_to_occlude = torch.where(element_mask)[0]
        else:
            nodes_to_occlude = torch.arange(N, device=self.device)

        num_nodes_to_occlude = len(nodes_to_occlude)

        if self.verbose:
            print(f"\nStarting contrastive node occlusion explanation:")
            print(f"  Input shapes: x1={x1.shape}, x2={x2.shape}")
            print(f"  Batch size: {B}")
            print(f"  Target indices: {target_idx}")
            print(f"  Nodes to occlude: {num_nodes_to_occlude} / {N}")
            print(f"  Total node batches to process: {((num_nodes_to_occlude - 1) // self.batch_size) + 1 if num_nodes_to_occlude > 0 else 0}")

        # ------------------------------------------------------------------
        # Process each sample pair
        # ------------------------------------------------------------------
        all_scores = []
        
        for sample_idx in range(B):
            x1i = x1[sample_idx:sample_idx+1]  # (1, N_in)
            x2i = x2[sample_idx:sample_idx+1]  # (1, N_in)
            
            if self.verbose and B > 1:
                print(f"Processing sample {sample_idx + 1}/{B}")

            # Compute baseline difference (all nodes present)
            baseline_mask = torch.ones((1, N), device=self.device)
            baseline_diff = self._compute_diff_node(x1i, x2i, target_idx, baseline_mask)
            
            if self.verbose:
                print(f"  Baseline |Δf| = {baseline_diff:.6f}")

            # Compute occlusion scores in batches
            occlusion_scores = torch.full((N,), float('nan'), device=self.device)
            
            if num_nodes_to_occlude > 0:
                for start_idx in range(0, num_nodes_to_occlude, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_nodes_to_occlude)
                    batch_size_actual = end_idx - start_idx
                    
                    if self.verbose:
                        print(f"Processing node batch {start_idx // self.batch_size + 1}/{((num_nodes_to_occlude - 1) // self.batch_size) + 1}", end='\r')

                    # Get the actual node indices for this batch
                    batch_node_indices = nodes_to_occlude[start_idx:end_idx]

                    # Create batch of masks with one node removed per mask
                    batch_masks = torch.ones((batch_size_actual, N), device=self.device)
                    for i, node_idx in enumerate(batch_node_indices):
                        batch_masks[i, node_idx] = 0.0

                    # Replicate inputs for batch processing
                    x1_batch = x1i.repeat(batch_size_actual, 1)
                    x2_batch = x2i.repeat(batch_size_actual, 1)
                    x_batch = torch.cat([x1_batch, x2_batch], dim=0)
                    mask_batch = batch_masks.repeat(2, 1)

                    # Forward pass
                    preds = self.model(x_batch, node_mask=mask_batch)[:, target_idx].sum(dim=1)
                    preds_x1 = preds[:batch_size_actual]
                    preds_x2 = preds[batch_size_actual:]
                    
                    # Compute occlusion effects
                    occluded_diffs = (preds_x1 - preds_x2).abs()
                    batch_scores = baseline_diff - occluded_diffs
                    occlusion_scores[batch_node_indices] = batch_scores
            
            all_scores.append(occlusion_scores)

        if self.verbose:
            print("\nNode occlusion explanation complete.")

        all_scores = torch.stack(all_scores, dim=0)  # (B, N)

        # ------------------------------------------------------------------
        # Package results with reduction
        # ------------------------------------------------------------------
        node_names = np.array(self.model.homo_names)
        
        if reduction == 'none':
            dfs = []
            for i in range(B):
                scores = all_scores[i].detach().cpu().numpy()
                scores = [None if np.isnan(score) else score for score in scores]
                df = pd.DataFrame({
                    "sample_idx": i,
                    "node": node_names,
                    "score": scores,
                })
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        
        # For mean/sum, handle NaN values properly
        if reduction == 'sum':
            scores_agg = torch.nansum(all_scores, dim=0)
        else:  # mean
            scores_agg = torch.nanmean(all_scores, dim=0)
        
        # Convert NaN to None for nodes not in mask
        scores = scores_agg.detach().cpu().numpy()
        scores = [None if np.isnan(score) else score for score in scores]
        
        return pd.DataFrame({
            "node": node_names,
            "score": scores,
        })

    def _compute_diff_edge(self, x1: torch.Tensor, x2: torch.Tensor, target_idx: List[int], mask: torch.Tensor) -> float:
        """Compute absolute prediction difference with given edge mask."""
        pred1 = self.model(x1, edge_mask=mask)[:, target_idx].sum(dim=1)
        pred2 = self.model(x2, edge_mask=mask)[:, target_idx].sum(dim=1)
        return (pred1 - pred2).abs().item()

    def _compute_diff_node(self, x1: torch.Tensor, x2: torch.Tensor, target_idx: List[int], mask: torch.Tensor) -> float:
        """Compute absolute prediction difference with given node mask."""
        pred1 = self.model(x1, node_mask=mask)[:, target_idx].sum(dim=1)
        pred2 = self.model(x2, node_mask=mask)[:, target_idx].sum(dim=1)
        return (pred1 - pred2).abs().item() 