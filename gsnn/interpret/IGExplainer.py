import numpy as np
import torch
import copy
import pandas as pd
from typing import Optional


class IGExplainer:
    r"""Integrated-Gradients explainer for GSNN models (non-contrastive).

    Computes per-edge or per-node attributions for a prediction *f(x)[target_idx]* by
    integrating the gradient along a straight-line path in **feature space**
    from a baseline input *x′* (default all zeros) to the observation *x*.

    For edge-level attributions::

        IG_e = (x - x′) · \int_0^1 ∂f(x′ + α(x-x′))/∂m_e dα.

    For node-level attributions::

        IG_n = (x - x′) · \int_0^1 ∂f(x′ + α(x-x′))/∂n_n dα.

    When the baseline masks are zero this reduces to the EdgeIG/NodeIG variants.  
    The attributions satisfy the completeness axiom for their respective domains.

    Node-level and edge-level attributions are computed independently using 
    separate masking mechanisms in the GSNN model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GSNN model (copied and frozen internally).
    data : torch_geometric.data.Data
        Graph data object; only used for edge names.
    ignore_cuda : bool, optional (default=False)
        Force the explainer to run on CPU even if CUDA is available.
    n_steps : int, optional (default=50)
        Number of points on the IG path (baseline included).
    baseline : torch.Tensor or None, optional
        Custom baseline edge-mask of shape ``(1,E)``.  ``None`` defaults to
        an all-zeros mask.

    Example
    -------
    >>> explainer = IGExplainer(model, data, n_steps=64)
    >>> # Edge-level attributions
    >>> df_edge = explainer.explain(x, target_idx=0, target='edge')
    >>> df_edge.nlargest(5, 'score')
    >>> # Node-level attributions  
    >>> df_node = explainer.explain(x, target_idx=0, target='node')
    >>> df_node.nlargest(5, 'score')
    """

    def __init__(self, model, data, ignore_cuda=False, n_steps=50, baseline=None): 
        """Create a new IGExplainer instance."""
        self.data = data 
        self.device = 'cuda' if (torch.cuda.is_available() and not ignore_cuda) else 'cpu'

        model = copy.deepcopy(model)
        model = model.eval()
        model = model.to(self.device)
        self.model = model 

        self.n_steps = n_steps
        self.E = model.edge_index.size(1)

        # Baseline mask (shape: 1 x E). Default is all-zeros if none provided.
        self.baseline = torch.zeros((1, self.E), device=self.device) if baseline is None else baseline.to(self.device)
        

    def explain(self, x, target_idx, *, jitter: Optional[torch.Tensor] = None, target='edge', reduction='mean'):
        '''
        Compute integrated gradients attributions for GSNN predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (N_in,), (1, N_in), or (B, N_in) for batch.
        target_idx : int
            Index of the target output node to explain.
        jitter : torch.Tensor, optional
            Optional noise to add to baseline, shape (E,) or (1, E) for edge target,
            shape (N,) or (1, N) for node target.
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

        Following approach and style from: https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py
        
        Reference: 
        @article{DBLP:journals/corr/SundararajanTY17,
                author       = {Mukund Sundararajan and
                                Ankur Taly and
                                Qiqi Yan},
                title        = {Axiomatic Attribution for Deep Networks},
                journal      = {CoRR},
                volume       = {abs/1703.01365},
                year         = {2017},
                url          = {http://arxiv.org/abs/1703.01365},
                eprinttype    = {arXiv},
                eprint       = {1703.01365},
                timestamp    = {Mon, 13 Aug 2018 16:48:32 +0200},
                biburl       = {https://dblp.org/rec/journals/corr/SundararajanTY17.bib},
                bibsource    = {dblp computer science bibliography, https://dblp.org}
                }
        '''

        if target not in ['edge', 'node']:
            raise ValueError(f"target must be 'edge' or 'node', got '{target}'")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

        if target == 'edge':
            return self._compute_edge_attributions(x, target_idx, jitter, reduction)
        else:
            return self._compute_node_attributions(x, target_idx, jitter, reduction)

    def _compute_edge_attributions(self, x, target_idx, jitter=None, reduction='mean'):
        '''
        Compute edge-level attributions using integrated gradients on edge_mask.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (N_in,), (1, N_in), or (B, N_in).
        target_idx : int
            Index of the target output node to explain.
        jitter : torch.Tensor, optional
            Optional noise to add to baseline, shape (E,) or (1, E).
        reduction : str
            How to aggregate across batch: 'mean', 'sum', or 'none'.
            
        Returns
        -------
        pd.DataFrame
            Columns ['source', 'target', 'score'] for edge attributions.
            If reduction='none': additional 'sample_idx' column.
        '''
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, N_in)
        
        B = x.size(0)  # batch size
        
        # -------------------------------------------------------------
        # 0.  Optionally perturb the baseline with *jitter*
        # -------------------------------------------------------------
        if jitter is not None:
            jitter = jitter.to(self.device)
            if jitter.dim() == 1:
                jitter = jitter.unsqueeze(0)  # make shape (1,E)
            baseline_ = torch.clamp(self.baseline + jitter, 0.0, 1.0)
        else:
            baseline_ = self.baseline

        # alphas: 0 … 1 (inclusive). We include both baseline (0) and full input (1).
        alphas = torch.linspace(0.0, 1.0, self.n_steps + 1, device=self.device).view(-1, 1)

        # Build a stack of interpolated edge-masks along the batch dimension
        edge_masks_template = baseline_ + alphas * (1.0 - baseline_)  # (n_steps+1 , E)

        # Process each sample and collect IG scores
        all_ig = []
        for i in range(B):
            xi = x[i:i+1]  # (1, N_in)
            
            # Need fresh tensor for gradient computation
            edge_masks = edge_masks_template.clone().requires_grad_(True)
            
            x_batch = xi.repeat(self.n_steps + 1, 1)  # (n_steps+1 , N_in)
            preds = self.model(x_batch, edge_mask=edge_masks)[:, target_idx]  # (n_steps+1,)

            # d(pred)/d(edge_mask)
            grads = torch.autograd.grad(preds.sum(), edge_masks)[0]  # (n_steps+1 , E)

            # Trapezoidal rule approximation of the path integral
            trap_grads = (grads[:-1] + grads[1:]) / 2.0  # (n_steps , E)
            avg_grads = trap_grads.mean(dim=0)  # (E,)

            ig = avg_grads * (1. - baseline_.squeeze(0))  # (E,)
            all_ig.append(ig)
        
        all_ig = torch.stack(all_ig, dim=0)  # (B, E)
        
        # Apply reduction
        src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        
        if reduction == 'none':
            # Return per-sample attributions
            dfs = []
            for i in range(B):
                df = pd.DataFrame({
                    'sample_idx': i,
                    'source': src,
                    'target': dst,
                    'score': all_ig[i].detach().cpu().numpy()
                })
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        elif reduction == 'sum':
            ig_agg = all_ig.sum(dim=0)
        else:  # mean
            ig_agg = all_ig.mean(dim=0)
        
        return pd.DataFrame({
            'source': src,
            'target': dst,
            'score': ig_agg.detach().cpu().numpy()
        })

    def _compute_node_attributions(self, x, target_idx, jitter=None, reduction='mean'):
        '''
        Compute node-level attributions using integrated gradients on node_mask.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (N_in,), (1, N_in), or (B, N_in).
        target_idx : int
            Index of the target output node to explain.
        jitter : torch.Tensor, optional
            Optional noise to add to baseline, shape (N,) or (1, N).
        reduction : str
            How to aggregate across batch: 'mean', 'sum', or 'none'.
            
        Returns
        -------
        pd.DataFrame
            Columns ['node', 'score'] for node attributions.
            If reduction='none': additional 'sample_idx' column.
        '''
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, N_in)
        
        B = x.size(0)  # batch size
        N = self.model.num_nodes
        
        # Baseline node mask (shape: 1 x N). Default is all-zeros (no nodes active) if none provided.
        baseline_node = torch.zeros((1, N), device=self.device)
        
        # -------------------------------------------------------------
        # 0.  Optionally perturb the baseline with *jitter*
        # -------------------------------------------------------------
        if jitter is not None:
            jitter = jitter.to(self.device)
            if jitter.dim() == 1:
                jitter = jitter.unsqueeze(0)  # make shape (1,N)
            baseline_node = torch.clamp(baseline_node + jitter, 0.0, 1.0)

        # alphas: 0 … 1 (inclusive). We include both baseline (0) and full input (1).
        alphas = torch.linspace(0.0, 1.0, self.n_steps + 1, device=self.device).view(-1, 1)

        # Full node mask (all nodes active) as the target
        full_node_mask = torch.ones((1, N), device=self.device)
        node_masks_template = baseline_node + alphas * (full_node_mask - baseline_node)  # (n_steps+1 , N)

        # Process each sample and collect IG scores
        all_ig = []
        for i in range(B):
            xi = x[i:i+1]  # (1, N_in)
            
            # Need fresh tensor for gradient computation
            node_masks = node_masks_template.clone().requires_grad_(True)
            
            x_batch = xi.repeat(self.n_steps + 1, 1)  # (n_steps+1 , N_in)
            preds = self.model(x_batch, node_mask=node_masks)[:, target_idx]  # (n_steps+1,)

            # d(pred)/d(node_mask)
            grads = torch.autograd.grad(preds.sum(), node_masks)[0]  # (n_steps+1 , N)

            # Trapezoidal rule approximation of the path integral
            trap_grads = (grads[:-1] + grads[1:]) / 2.0  # (n_steps , N)
            avg_grads = trap_grads.mean(dim=0)  # (N,)

            ig = avg_grads * (full_node_mask.squeeze(0) - baseline_node.squeeze(0))  # (N,)
            all_ig.append(ig)
        
        all_ig = torch.stack(all_ig, dim=0)  # (B, N)
        
        # Apply reduction
        node_names = np.array(self.model.homo_names)
        
        if reduction == 'none':
            # Return per-sample attributions
            dfs = []
            for i in range(B):
                df = pd.DataFrame({
                    'sample_idx': i,
                    'node': node_names,
                    'score': all_ig[i].detach().cpu().numpy()
                })
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        elif reduction == 'sum':
            ig_agg = all_ig.sum(dim=0)
        else:  # mean
            ig_agg = all_ig.mean(dim=0)
        
        return pd.DataFrame({
            'node': node_names,
            'score': ig_agg.detach().cpu().numpy()
        })
