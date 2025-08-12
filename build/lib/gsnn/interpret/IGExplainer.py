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
        

    def explain(self, x, target_idx, *, jitter: Optional[torch.Tensor] = None, target='edge'):
        '''
        Compute integrated gradients attributions for GSNN predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (N_in,) or (1, N_in).
        target_idx : int
            Index of the target output node to explain.
        jitter : torch.Tensor, optional
            Optional noise to add to baseline, shape (E,) or (1, E) for edge target,
            shape (N,) or (1, N) for node target.
        target : str, optional (default='edge')
            Whether to return 'edge' or 'node' level attributions.

        Returns
        -------
        pd.DataFrame
            If target='edge': columns ['source', 'target', 'score'] for edge attributions.
            If target='node': columns ['node', 'score'] for node attributions.

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

        if target == 'edge':
            return self._compute_edge_attributions(x, target_idx, jitter)
        elif target == 'node':
            return self._compute_node_attributions(x, target_idx, jitter)

    def _compute_edge_attributions(self, x, target_idx, jitter=None):
        '''
        Compute edge-level attributions using integrated gradients on edge_mask.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (N_in,) or (1, N_in).
        target_idx : int
            Index of the target output node to explain.
        jitter : torch.Tensor, optional
            Optional noise to add to baseline, shape (E,) or (1, E).
            
        Returns
        -------
        pd.DataFrame
            Columns ['source', 'target', 'score'] for edge attributions.
        '''
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
        edge_masks = baseline_ + alphas * (1.0 - baseline_)  # (n_steps+1 , E)
        edge_masks.requires_grad_(True)

        x_batch = x.repeat(self.n_steps + 1, 1)  # (n_steps+1 , N_in)

        preds = self.model(x_batch, edge_mask=edge_masks)[:, target_idx]  # (n_steps+1,)

        # d(pred)/d(edge_mask)
        grads = torch.autograd.grad(preds.sum(), edge_masks)[0]  # (n_steps+1 , E)

        # Trapezoidal rule approximation of the path integral
        trap_grads = (grads[:-1] + grads[1:]) / 2.0  # (n_steps , E)
        avg_grads = trap_grads.mean(dim=0)  # (E,)

        ig = avg_grads * (1. - self.baseline.squeeze(0))  # (E,)
        
        # Return edge-level attributions
        src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()] 
        res = pd.DataFrame({'source': src, 'target': dst, 'score': ig.detach().cpu().numpy()})
        
        return res

    def _compute_node_attributions(self, x, target_idx, jitter=None):
        '''
        Compute node-level attributions using integrated gradients on node_mask.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (N_in,) or (1, N_in).
        target_idx : int
            Index of the target output node to explain.
        jitter : torch.Tensor, optional
            Optional noise to add to baseline, shape (N,) or (1, N).
            
        Returns
        -------
        pd.DataFrame
            Columns ['node', 'score'] for node attributions.
        '''
        # Baseline node mask (shape: 1 x N). Default is all-zeros (no nodes active) if none provided.
        baseline_node = torch.zeros((1, self.model.num_nodes), device=self.device)
        
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

        # Build a stack of interpolated node-masks along the batch dimension
        # Full node mask (all nodes active) as the target
        full_node_mask = torch.ones((1, self.model.num_nodes), device=self.device)
        node_masks = baseline_node + alphas * (full_node_mask - baseline_node)  # (n_steps+1 , N)
        node_masks.requires_grad_(True)

        x_batch = x.repeat(self.n_steps + 1, 1)  # (n_steps+1 , N_in)

        preds = self.model(x_batch, node_mask=node_masks)[:, target_idx]  # (n_steps+1,)

        # d(pred)/d(node_mask)
        grads = torch.autograd.grad(preds.sum(), node_masks)[0]  # (n_steps+1 , N)

        # Trapezoidal rule approximation of the path integral
        trap_grads = (grads[:-1] + grads[1:]) / 2.0  # (n_steps , N)
        avg_grads = trap_grads.mean(dim=0)  # (N,)

        ig = avg_grads * (full_node_mask.squeeze(0) - baseline_node.squeeze(0))  # (N,)
        
        # Create result DataFrame
        node_names = np.array(self.model.homo_names)
        res = pd.DataFrame({
            'node': node_names, 
            'score': ig.detach().cpu().numpy()
        })

        return res
