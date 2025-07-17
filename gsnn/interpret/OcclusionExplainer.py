import numpy as np
import torch
import copy
import pandas as pd


class OcclusionExplainer:
    r"""Edge occlusion explainer for single observations.

    Computes per-edge attributions for a prediction *f(x)[target_idx]* by
    systematically removing each edge and measuring the change in prediction::

        Occ_e = f(x; mask_baseline) - f(x; mask_e_removed)

    where *mask_baseline* uses all edges present and *mask_e_removed* removes
    only edge *e* (sets edge_mask[e] = 0).

    **Interpretation of scores**
    ----------------------------
    * ``Occ_e > 0``  edge *e* contributes positively to the prediction
    * ``Occ_e < 0``  edge *e* inhibits the prediction (removing it increases output)  
    * ``Occ_e ≈ 0``  edge *e* has no impact on the prediction

    The occlusion approach provides a direct, model-agnostic measure of edge
    importance by directly measuring the effect of completely removing each edge.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GSNN model (copied and frozen internally).
    data : torch_geometric.data.Data
        Graph data object; only used for edge names.
    ignore_cuda : bool, optional (default=False)
        Force the explainer to run on CPU even if CUDA is available.
    batch_size : int, optional (default=32)
        Number of edge occlusions to process in parallel.

    Example
    -------
    >>> explainer = OcclusionExplainer(model, data, batch_size=64)
    >>> df = explainer.explain(x, target_idx=0)
    >>> df.nlargest(5, 'score')
    source target   score
    in0    func0    0.42
    func0  func3    0.40
    func3  out0     0.38
    """

    def __init__(self, model, data, ignore_cuda=False, batch_size=32):
        """Create a new OcclusionExplainer instance."""
        self.data = data
        self.device = 'cuda' if (torch.cuda.is_available() and not ignore_cuda) else 'cpu'

        model = copy.deepcopy(model)
        model = model.eval()
        model = model.to(self.device)
        self.model = model

        self.batch_size = batch_size
        self.E = model.edge_index.size(1)


    def explain(self, x, target_idx):
        """Compute edge occlusion attributions for *f(x)[target_idx]*.

        Parameters
        ----------
        x : torch.Tensor  (shape: [1, N_in] or [N_in])
            Input feature tensor. Will be moved to appropriate device.
        target_idx : int
            Output dimension to explain.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'source', 'target', 'score' containing
            edge attributions sorted by edge index.
        """
        
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension

        # ------------------------------------------------------------------
        # 1. Compute baseline prediction (all edges present)
        # ------------------------------------------------------------------
        baseline_mask = torch.ones((1, self.E), device=self.device)
        baseline_pred = self.model(x, edge_mask=baseline_mask)[:, target_idx].item()

        # ------------------------------------------------------------------
        # 2. Compute occlusion scores in batches
        # ------------------------------------------------------------------
        occlusion_scores = torch.zeros(self.E, device=self.device)
        
        for start_idx in range(0, self.E, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.E)
            batch_size_actual = end_idx - start_idx
            
            # Create batch of masks with one edge removed per mask
            batch_masks = torch.ones((batch_size_actual, self.E), device=self.device)
            for i, edge_idx in enumerate(range(start_idx, end_idx)):
                batch_masks[i, edge_idx] = 0.0

            # Replicate input for batch processing
            x_batch = x.repeat(batch_size_actual, 1)

            # Forward pass
            preds = self.model(x_batch, edge_mask=batch_masks)[:, target_idx]
            
            # Compute occlusion effects
            batch_scores = baseline_pred - preds
            occlusion_scores[start_idx:end_idx] = batch_scores

        # ------------------------------------------------------------------
        # 3. Package results
        # ------------------------------------------------------------------
        src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        
        return pd.DataFrame({
            'source': src,
            'target': dst,
            'score': occlusion_scores.detach().cpu().numpy()
        }) 