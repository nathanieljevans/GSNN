import numpy as np
import torch
import copy
import pandas as pd


class OcclusionExplainer:
    r"""Edge/node occlusion explainer for single observations.

    Computes per-edge or per-node attributions for a prediction *f(x)[target_idx]* by
    systematically removing each element and measuring the change in prediction.

    For edge-level attributions::

        Occ_e = f(x; mask_baseline) - f(x; mask_e_removed)

    For node-level attributions::

        Occ_n = f(x; mask_baseline) - f(x; mask_n_removed)

    where *mask_baseline* uses all elements present and *mask_element_removed* removes
    only the specified element (sets mask[element] = 0).

    **Interpretation of scores**
    ----------------------------
    * ``Occ > 0``  element contributes positively to the prediction
    * ``Occ < 0``  element inhibits the prediction (removing it increases output)  
    * ``Occ ≈ 0``  element has no impact on the prediction

    The occlusion approach provides a direct, model-agnostic measure of element
    importance by directly measuring the effect of completely removing each element.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GSNN model (copied and frozen internally).
    data : torch_geometric.data.Data
        Graph data object; only used for element names.
    ignore_cuda : bool, optional (default=False)
        Force the explainer to run on CPU even if CUDA is available.
    batch_size : int, optional (default=32)
        Number of element occlusions to process in parallel.

    Example
    -------
    >>> explainer = OcclusionExplainer(model, data, batch_size=64)
    >>> # Edge-level attributions
    >>> edge_df = explainer.explain(x, target_idx=0, target='edge')
    >>> edge_df.nlargest(5, 'score')
    source target   score
    in0    func0    0.42
    func0  func3    0.40
    func3  out0     0.38
    
    >>> # Node-level attributions
    >>> node_df = explainer.explain(x, target_idx=0, target='node')
    >>> node_df.nlargest(5, 'score')
    
    >>> # Occlude only a subset of edges
    >>> edge_mask = np.array([True, False, True, False, True])  # Only occlude edges 0, 2, 4
    >>> edge_df = explainer.explain(x, target_idx=0, target='edge', element_mask=edge_mask)
    >>> # Edges 1 and 3 will have None scores
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
        self.N = model.num_nodes


    def explain(self, x, target_idx, element_mask=None, target='edge', reduction='mean'):
        """Compute edge or node occlusion attributions for *f(x)[target_idx]*.

        Parameters
        ----------
        x : torch.Tensor  (shape: [N_in], [1, N_in], or [B, N_in] for batch)
            Input feature tensor. Will be moved to appropriate device.
        target_idx : int
            Output dimension to explain.
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
            return self._explain_edges(x, target_idx, element_mask, reduction)
        else:
            return self._explain_nodes(x, target_idx, element_mask, reduction)

    def _explain_edges(self, x, target_idx, element_mask=None, reduction='mean'):
        """
        Compute edge-level attributions using occlusion.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (N_in,), (1, N_in), or (B, N_in).
        target_idx : int
            Index of the target output node to explain.
        element_mask : torch.Tensor or np.ndarray, optional
            Boolean mask indicating which edges to compute occlusion for.
        reduction : str
            How to aggregate across batch: 'mean', 'sum', or 'none'.
            
        Returns
        -------
        pd.DataFrame
            Columns ['source', 'target', 'score'] for edge attributions.
            If reduction='none': additional 'sample_idx' column.
        """
        
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension
        
        B = x.size(0)  # batch size

        # ------------------------------------------------------------------
        # 1. Process element_mask
        # ------------------------------------------------------------------
        if element_mask is not None:
            if isinstance(element_mask, np.ndarray):
                element_mask = torch.from_numpy(element_mask)
            element_mask = element_mask.to(self.device).bool()
            edges_to_occlude = torch.where(element_mask)[0]
        else:
            edges_to_occlude = torch.arange(self.E, device=self.device)

        # ------------------------------------------------------------------
        # 2. Process each sample
        # ------------------------------------------------------------------
        all_scores = []
        
        for sample_idx in range(B):
            xi = x[sample_idx:sample_idx+1]  # (1, N_in)
            
            # Compute baseline prediction (all edges present)
            baseline_mask = torch.ones((1, self.E), device=self.device)
            baseline_pred = self.model(xi, edge_mask=baseline_mask)[:, target_idx].item()

            # Compute occlusion scores in batches
            occlusion_scores = torch.full((self.E,), float('nan'), device=self.device)
            
            if len(edges_to_occlude) > 0:
                for start_idx in range(0, len(edges_to_occlude), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(edges_to_occlude))
                    batch_size_actual = end_idx - start_idx
                    
                    # Create batch of masks with one edge removed per mask
                    batch_masks = torch.ones((batch_size_actual, self.E), device=self.device)
                    batch_edge_indices = edges_to_occlude[start_idx:end_idx]
                    
                    for i, edge_idx in enumerate(batch_edge_indices):
                        batch_masks[i, edge_idx] = 0.0

                    # Replicate input for batch processing
                    x_batch = xi.repeat(batch_size_actual, 1)

                    # Forward pass
                    preds = self.model(x_batch, edge_mask=batch_masks)[:, target_idx]
                    
                    # Compute occlusion effects
                    batch_scores = baseline_pred - preds
                    occlusion_scores[batch_edge_indices] = batch_scores
            
            all_scores.append(occlusion_scores)
        
        all_scores = torch.stack(all_scores, dim=0)  # (B, E)

        # ------------------------------------------------------------------
        # 3. Package results with reduction
        # ------------------------------------------------------------------
        src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        
        if reduction == 'none':
            # Return per-sample attributions
            dfs = []
            for i in range(B):
                scores = all_scores[i].detach().cpu().numpy()
                scores = [None if np.isnan(score) else score for score in scores]
                df = pd.DataFrame({
                    'sample_idx': i,
                    'source': src,
                    'target': dst,
                    'score': scores
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
            'source': src,
            'target': dst,
            'score': scores
        })

    def _explain_nodes(self, x, target_idx, element_mask=None, reduction='mean'):
        """
        Compute node-level attributions using occlusion.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (N_in,), (1, N_in), or (B, N_in).
        target_idx : int
            Index of the target output node to explain.
        element_mask : torch.Tensor or np.ndarray, optional
            Boolean mask indicating which nodes to compute occlusion for.
        reduction : str
            How to aggregate across batch: 'mean', 'sum', or 'none'.
            
        Returns
        -------
        pd.DataFrame
            Columns ['node', 'score'] for node attributions.
            If reduction='none': additional 'sample_idx' column.
        """
        
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension
        
        B = x.size(0)  # batch size

        # ------------------------------------------------------------------
        # 1. Process element_mask
        # ------------------------------------------------------------------
        if element_mask is not None:
            if isinstance(element_mask, np.ndarray):
                element_mask = torch.from_numpy(element_mask)
            element_mask = element_mask.to(self.device).bool()
            nodes_to_occlude = torch.where(element_mask)[0]
        else:
            nodes_to_occlude = torch.arange(self.N, device=self.device)

        # ------------------------------------------------------------------
        # 2. Process each sample
        # ------------------------------------------------------------------
        all_scores = []
        
        for sample_idx in range(B):
            xi = x[sample_idx:sample_idx+1]  # (1, N_in)
            
            # Compute baseline prediction (all nodes present)
            baseline_mask = torch.ones((1, self.N), device=self.device)
            baseline_pred = self.model(xi, node_mask=baseline_mask)[:, target_idx].item()

            # Compute occlusion scores in batches
            occlusion_scores = torch.full((self.N,), float('nan'), device=self.device)
            
            if len(nodes_to_occlude) > 0:
                for start_idx in range(0, len(nodes_to_occlude), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(nodes_to_occlude))
                    batch_size_actual = end_idx - start_idx
                    
                    # Create batch of masks with one node removed per mask
                    batch_masks = torch.ones((batch_size_actual, self.N), device=self.device)
                    batch_node_indices = nodes_to_occlude[start_idx:end_idx]
                    
                    for i, node_idx in enumerate(batch_node_indices):
                        batch_masks[i, node_idx] = 0.0

                    # Replicate input for batch processing
                    x_batch = xi.repeat(batch_size_actual, 1)

                    # Forward pass
                    preds = self.model(x_batch, node_mask=batch_masks)[:, target_idx]
                    
                    # Compute occlusion effects
                    batch_scores = baseline_pred - preds
                    occlusion_scores[batch_node_indices] = batch_scores
            
            all_scores.append(occlusion_scores)
        
        all_scores = torch.stack(all_scores, dim=0)  # (B, N)

        # ------------------------------------------------------------------
        # 3. Package results with reduction
        # ------------------------------------------------------------------
        node_names = np.array(self.model.homo_names)
        
        if reduction == 'none':
            # Return per-sample attributions
            dfs = []
            for i in range(B):
                scores = all_scores[i].detach().cpu().numpy()
                scores = [None if np.isnan(score) else score for score in scores]
                df = pd.DataFrame({
                    'sample_idx': i,
                    'node': node_names,
                    'score': scores
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
            'node': node_names,
            'score': scores
        }) 