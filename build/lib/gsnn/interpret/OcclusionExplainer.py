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


    def explain(self, x, target_idx, element_mask=None, target='edge'):
        """Compute edge or node occlusion attributions for *f(x)[target_idx]*.

        Parameters
        ----------
        x : torch.Tensor  (shape: [1, N_in] or [N_in])
            Input feature tensor. Will be moved to appropriate device.
        target_idx : int
            Output dimension to explain.
        element_mask : torch.Tensor or np.ndarray, optional (shape: [E] or [N])
            Boolean mask indicating which elements to compute occlusion for.
            If None, all elements are considered. If provided, only elements where
            element_mask[i] is True will have occlusion scores computed.
        target : str, optional (default='edge')
            Whether to return 'edge' or 'node' level attributions.

        Returns
        -------
        pd.DataFrame
            If target='edge': columns ['source', 'target', 'score'] for edge attributions.
            If target='node': columns ['node', 'score'] for node attributions.
            Elements not in element_mask will have None scores.
        """
        
        if target not in ['edge', 'node']:
            raise ValueError(f"target must be 'edge' or 'node', got '{target}'")

        if target == 'edge':
            return self._explain_edges(x, target_idx, element_mask)
        elif target == 'node':
            return self._explain_nodes(x, target_idx, element_mask)

    def _explain_edges(self, x, target_idx, element_mask=None):
        """
        Compute edge-level attributions using occlusion.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (N_in,) or (1, N_in).
        target_idx : int
            Index of the target output node to explain.
        element_mask : torch.Tensor or np.ndarray, optional
            Boolean mask indicating which edges to compute occlusion for.
            
        Returns
        -------
        pd.DataFrame
            Columns ['source', 'target', 'score'] for edge attributions.
        """
        
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension

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
        # 2. Compute baseline prediction (all edges present)
        # ------------------------------------------------------------------
        baseline_mask = torch.ones((1, self.E), device=self.device)
        baseline_pred = self.model(x, edge_mask=baseline_mask)[:, target_idx].item()

        # ------------------------------------------------------------------
        # 3. Compute occlusion scores in batches
        # ------------------------------------------------------------------
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
                x_batch = x.repeat(batch_size_actual, 1)

                # Forward pass
                preds = self.model(x_batch, edge_mask=batch_masks)[:, target_idx]
                
                # Compute occlusion effects
                batch_scores = baseline_pred - preds
                occlusion_scores[batch_edge_indices] = batch_scores

        # ------------------------------------------------------------------
        # 4. Package results
        # ------------------------------------------------------------------
        src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        
        # Convert NaN to None for edges not in mask
        scores = occlusion_scores.detach().cpu().numpy()
        scores = [None if np.isnan(score) else score for score in scores]
        
        return pd.DataFrame({
            'source': src,
            'target': dst,
            'score': scores
        })

    def _explain_nodes(self, x, target_idx, element_mask=None):
        """
        Compute node-level attributions using occlusion.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (N_in,) or (1, N_in).
        target_idx : int
            Index of the target output node to explain.
        element_mask : torch.Tensor or np.ndarray, optional
            Boolean mask indicating which nodes to compute occlusion for.
            
        Returns
        -------
        pd.DataFrame
            Columns ['node', 'score'] for node attributions.
        """
        
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension

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
        # 2. Compute baseline prediction (all nodes present)
        # ------------------------------------------------------------------
        baseline_mask = torch.ones((1, self.N), device=self.device)
        baseline_pred = self.model(x, node_mask=baseline_mask)[:, target_idx].item()

        # ------------------------------------------------------------------
        # 3. Compute occlusion scores in batches
        # ------------------------------------------------------------------
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
                x_batch = x.repeat(batch_size_actual, 1)

                # Forward pass
                preds = self.model(x_batch, node_mask=batch_masks)[:, target_idx]
                
                # Compute occlusion effects
                batch_scores = baseline_pred - preds
                occlusion_scores[batch_node_indices] = batch_scores

        # ------------------------------------------------------------------
        # 4. Package results
        # ------------------------------------------------------------------
        node_names = np.array(self.model.homo_names)
        
        # Convert NaN to None for nodes not in mask
        scores = occlusion_scores.detach().cpu().numpy()
        scores = [None if np.isnan(score) else score for score in scores]
        
        return pd.DataFrame({
            'node': node_names,
            'score': scores
        }) 