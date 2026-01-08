import copy
from typing import Union, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score


class ContrastiveGSNNExplainer:
    r"""Edge/node mask optimiser for *contrastive* explanations.

    This explainer learns a binary mask *m∈{0,1}^{E|N}* that maximises fidelity
    between the **prediction difference** on the masked graph and the difference
    on the full graph, while simultaneously penalising mask size::

        Δf(m) = f(x₁; m)[target_idx] − f(x₂; m)[target_idx]   (multivariate)
        
        L = MSE(Δf(m), Δf(1))      # over all B×T elements
            + β max(0, ‖m‖₁ − free_elements)
            − λ H(m)               (optional entropy term)

    Here *m* is obtained via a differentiable Gumbel-Softmax relaxation so the
    optimisation can be performed with vanilla back-prop. After convergence
    the importance score is the softmax probability *p_i = P(m_i=1)*.

    Interpretation
    --------------
    * ``score_i → 1``  element i is essential for reproducing the prediction difference.
    * ``score_i → 0``  element i can be removed without affecting the difference.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GSNN model (its parameters are *frozen* during explanation).
    data : torch_geometric.data.Data
        Graph data object (only metadata are used).
    ignore_cuda : bool, optional (default=False)
        Force CPU even if CUDA is available.
    gumbel_softmax : bool, optional (default=True)
        Use the Gumbel-Softmax re-parameterisation; otherwise plain Softmax.
    hard : bool, optional (default=False)
        Use the straight-through estimator to obtain discrete masks at test
        time while keeping gradients continuous.
    tau0 : float, optional (default=3.0)
        Initial temperature for the (hard) Gumbel-Softmax.
    min_tau : float, optional (default=0.5)
        Minimum temperature reached after exponential decay.
    prior : float, optional (default=1.0)
        Initial bias added to the positive/negative logits.
    iters : int, optional (default=250)
        Number of optimisation steps.
    lr : float, optional (default=1e-2)
        Learning rate for the optimiser.
    weight_decay : float, optional (default=1e-5)
        Weight decay applied to the mask logits.
    free_edges : int, optional (default=0)
        Number of elements allowed before the sparsity penalty activates.
    beta : float, optional (default=1.0)
        Coefficient of the sparsity term.
    entropy : float, optional (default=0.0)
        Strength of the entropy bonus (encourages exploration).
    verbose : bool, optional (default=True)
        Print progress information during optimisation.

    Example
    -------
    >>> explainer = ContrastiveGSNNExplainer(model, data, iters=400, beta=5)
    >>> # Edge-level attributions
    >>> edge_df = explainer.explain(x1, x2, target_idx=0, target='edge')
    >>> edge_df.sort_values('score', ascending=False).head()
    >>> # Node-level attributions
    >>> node_df = explainer.explain(x1, x2, target_idx=0, target='node')
    >>> node_df.sort_values('score', ascending=False).head()
    """

    def __init__(
        self,
        model,
        data,
        ignore_cuda: bool = False,
        gumbel_softmax: bool = True,
        hard: bool = False,
        tau0: float = 3.0,
        min_tau: float = 0.5,
        prior: float = 1.0,
        iters: int = 250,
        lr: float = 1e-2,
        weight_decay: float = 1e-5,
        free_edges: int = 0,
        beta: float = 1.0,
        verbose: bool = True,
        optimizer=torch.optim.Adam,
        entropy: float = 0.0,
    ) -> None:
        """
        Contrastive version of GSNNExplainer for explaining prediction differences.
        
        Adapted from the methods presented in `GNNExplainer` (https://arxiv.org/abs/1903.03894).
        """
        self.free_edges = free_edges
        self.iters = iters
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.verbose = verbose
        self.optimizer = optimizer
        self.gumbel_softmax = gumbel_softmax
        self.prior = prior
        self.hard = hard
        self.min_tau = min_tau
        self.tau0 = tau0
        self.data = data
        self.device = 'cuda' if (torch.cuda.is_available() and not ignore_cuda) else 'cpu'
        self.entropy = entropy

        model = copy.deepcopy(model)
        model = model.eval()
        model = model.to(self.device)

        # freeze model parameters
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.E = model.edge_index.size(1)
        self.N = model.num_nodes

    def explain(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_idx: Union[int, List[int]],
        *,
        return_weights: bool = False,
        target: str = 'edge',
    ) -> pd.DataFrame:
        """Compute attributions for *f(x₁) − f(x₂)*.

        Initializes and runs gradient descent to select a minimal subset of
        elements that preserve the prediction difference between x1 and x2.
        
        When given multiple pairs (batch), learns ONE mask that works well
        across ALL pairs by treating the differences as a multi-output objective.
        This is much faster than per-sample optimization.

        Parameters
        ----------
        x1, x2 : torch.Tensor  (shape: [N_in], [1, N_in], or [B, N_in] for batch)
            Two input feature tensors. They must have identical batch size.
            When B > 1, learns a single mask that preserves the prediction
            difference across all pairs simultaneously.
        target_idx : int or list[int]
            Output dimension(s) to explain. If a list is provided the
            attributions refer to the **sum** of those outputs.
        return_weights : bool, optional (default=False)
            Whether to return raw weights along with the DataFrame.
        target : str, optional (default='edge')
            Whether to return 'edge' or 'node' level attributions.

        Returns
        -------
        pd.DataFrame
            If target='edge': columns ['source', 'target', 'score'] for edge attributions.
            If target='node': columns ['node', 'score'] for node attributions.
        """
        if target not in ['edge', 'node']:
            raise ValueError(f"target must be 'edge' or 'node', got '{target}'")

        if target == 'edge':
            return self._explain_edges(x1, x2, target_idx, return_weights)
        else:
            return self._explain_nodes(x1, x2, target_idx, return_weights)

    def _explain_edges(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_idx: Union[int, List[int]],
        return_weights: bool = False,
    ) -> pd.DataFrame:
        """Compute edge-level attributions for *f(x₁) − f(x₂)*.
        
        Learns ONE mask across all sample pairs by treating the differences
        as a multi-output objective. This is much faster than per-sample optimization.
        """
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        # Ensure batch dimension
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        
        B = x1.size(0)  # batch size
        
        if isinstance(target_idx, int):
            target_idx = [target_idx]

        # Initialize edge parameters (single mask for all samples)
        weights = torch.stack((
            self.prior * torch.ones(self.E, dtype=torch.float32, device=self.device),
            -self.prior * torch.ones(self.E, dtype=torch.float32, device=self.device)
        ), dim=0)
        edge_params = torch.nn.Parameter(weights)
        
        # Optimizer and loss
        crit = torch.nn.MSELoss()
        optim = self.optimizer([edge_params], lr=self.lr, weight_decay=self.weight_decay)
        
        # Calculate tau decay rate
        tau_decay_rate = (self.min_tau / self.tau0) ** (1 / self.iters)
        
        # Get target prediction differences for ALL pairs (baseline) - keep as multivariate
        with torch.no_grad():
            pred1_full = self.model(x1)[:, target_idx]  # (B, T)
            pred2_full = self.model(x2)[:, target_idx]  # (B, T)
            target_diffs = pred1_full - pred2_full  # (B, T) - multivariate differences
        
        if self.verbose:
            print(f"Batch size: {B}, Target dims: {len(target_idx)}")
            print(f"Target Δf mean: {target_diffs.mean().item():.6f}, std: {target_diffs.std().item():.6f}")
        
        # Optimization loop - learns ONE mask for all pairs
        for iter in range(self.iters):
            optim.zero_grad()
            
            tau = max(self.tau0 * tau_decay_rate ** iter, self.min_tau)
            
            edge_weight, _ = torch.nn.functional.gumbel_softmax(edge_params, dim=0, hard=self.hard, tau=tau)
            
            # Broadcast mask to all samples: (1, E) -> used for all B samples
            edge_mask_batch = edge_weight.view(1, -1).expand(B, -1)  # (B, E)
            
            # Forward pass for all pairs at once - keep as multivariate
            pred1 = self.model(x1, edge_mask=edge_mask_batch)[:, target_idx]  # (B, T)
            pred2 = self.model(x2, edge_mask=edge_mask_batch)[:, target_idx]  # (B, T)
            masked_diffs = pred1 - pred2  # (B, T) - multivariate differences
            
            # MSE over all B*T elements
            mse = crit(masked_diffs, target_diffs)
            
            edge_probs, _ = torch.nn.functional.softmax(edge_params, dim=0)
            m = torch.distributions.Bernoulli(probs=edge_probs)
            ent = m.entropy().mean()
            
            loss = mse \
                + self.beta * torch.maximum(torch.tensor([0.], device=self.device), edge_weight.sum() - self.free_edges) \
                - self.entropy * ent
            
            loss.backward()
            optim.step()
            
            if self.verbose:
                with torch.no_grad():
                    r2 = r2_score(
                        target_diffs.detach().cpu().numpy().ravel(),
                        masked_diffs.detach().cpu().numpy().ravel()
                    ) if target_diffs.numel() > 1 else -666
                print(f'iter: {iter} | loss: {loss.item():.4f} | mse: {mse.item():.4f} | r2: {r2:.3f} | active edges: {(edge_weight > 0.5).sum().item()} / {self.E} | entropy: {ent.item():.4f}', end='\r')
        
        # Post-training evaluation
        if self.verbose:
            print()
            with torch.no_grad():
                final_edge_probs, _ = torch.nn.functional.softmax(edge_params.data, dim=0)
                subset_mask = (final_edge_probs > 0.5).float()
                subset_mask_batch = subset_mask.view(1, -1).expand(B, -1)
                
                pred1_sub = self.model(x1, edge_mask=subset_mask_batch)[:, target_idx]  # (B, T)
                pred2_sub = self.model(x2, edge_mask=subset_mask_batch)[:, target_idx]  # (B, T)
                subset_diffs = pred1_sub - pred2_sub  # (B, T)
                
                subset_mse = torch.nn.functional.mse_loss(subset_diffs, target_diffs).item()
                subset_r2 = r2_score(
                    target_diffs.detach().cpu().numpy().ravel(),
                    subset_diffs.detach().cpu().numpy().ravel()
                ) if target_diffs.numel() > 1 else -666
                
                num_selected = (subset_mask > 0.5).sum().item()
                
                print("=" * 50)
                print("POST-TRAINING EVALUATION (edges > 0.5)")
                print("=" * 50)
                print(f"Selected edges: {num_selected} / {self.E} ({100 * num_selected / self.E:.1f}%)")
                print(f"Target Δf mean: {target_diffs.mean().item():.6f}")
                print(f"Subset Δf mean: {subset_diffs.mean().item():.6f}")
                print(f"MSE: {subset_mse:.6f}")
                print(f"R² (across {B}x{len(target_idx)} elements): {subset_r2:.4f}")
                print("=" * 50)
        
        edge_scores, _ = torch.nn.functional.softmax(edge_params.data, dim=0)
        
        # Package results - single set of scores for all pairs
        src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        
        result_df = pd.DataFrame({
            "source": src,
            "target": dst,
            "score": edge_scores.detach().cpu().numpy(),
        })
        
        if return_weights:
            return result_df, edge_scores.detach().cpu().numpy()
        return result_df

    def _explain_nodes(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_idx: Union[int, List[int]],
        return_weights: bool = False,
    ) -> pd.DataFrame:
        """Compute node-level attributions for *f(x₁) − f(x₂)*.
        
        Learns ONE mask across all sample pairs by treating the differences
        as a multi-output objective. This is much faster than per-sample optimization.
        """
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        # Ensure batch dimension
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        
        B = x1.size(0)  # batch size
        
        if isinstance(target_idx, int):
            target_idx = [target_idx]

        # Initialize node parameters (single mask for all samples)
        weights = torch.stack((
            self.prior * torch.ones(self.N, dtype=torch.float32, device=self.device),
            -self.prior * torch.ones(self.N, dtype=torch.float32, device=self.device)
        ), dim=0)
        node_params = torch.nn.Parameter(weights)
        
        # Optimizer and loss
        crit = torch.nn.MSELoss()
        optim = self.optimizer([node_params], lr=self.lr, weight_decay=self.weight_decay)
        
        # Calculate tau decay rate
        tau_decay_rate = (self.min_tau / self.tau0) ** (1 / self.iters)
        
        # Get target prediction differences for ALL pairs (baseline) - keep as multivariate
        with torch.no_grad():
            pred1_full = self.model(x1)[:, target_idx]  # (B, T)
            pred2_full = self.model(x2)[:, target_idx]  # (B, T)
            target_diffs = pred1_full - pred2_full  # (B, T) - multivariate differences
        
        if self.verbose:
            print(f"Batch size: {B}, Target dims: {len(target_idx)}")
            print(f"Target Δf mean: {target_diffs.mean().item():.6f}, std: {target_diffs.std().item():.6f}")
        
        # Optimization loop - learns ONE mask for all pairs
        for iter in range(self.iters):
            optim.zero_grad()
            
            tau = max(self.tau0 * tau_decay_rate ** iter, self.min_tau)
            
            node_weight, _ = torch.nn.functional.gumbel_softmax(node_params, dim=0, hard=self.hard, tau=tau)
            
            # Broadcast mask to all samples: (1, N) -> used for all B samples
            node_mask_batch = node_weight.view(1, -1).expand(B, -1)  # (B, N)
            
            # Forward pass for all pairs at once - keep as multivariate
            pred1 = self.model(x1, node_mask=node_mask_batch)[:, target_idx]  # (B, T)
            pred2 = self.model(x2, node_mask=node_mask_batch)[:, target_idx]  # (B, T)
            masked_diffs = pred1 - pred2  # (B, T) - multivariate differences
            
            # MSE over all B*T elements
            mse = crit(masked_diffs, target_diffs)
            
            node_probs, _ = torch.nn.functional.softmax(node_params, dim=0)
            m = torch.distributions.Bernoulli(probs=node_probs)
            ent = m.entropy().mean()
            
            loss = mse \
                + self.beta * torch.maximum(torch.tensor([0.], device=self.device), node_weight.sum() - self.free_edges) \
                - self.entropy * ent
            
            loss.backward()
            optim.step()
            
            if self.verbose:
                with torch.no_grad():
                    r2 = r2_score(
                        target_diffs.detach().cpu().numpy().ravel(),
                        masked_diffs.detach().cpu().numpy().ravel()
                    ) if target_diffs.numel() > 1 else -666
                print(f'iter: {iter} | loss: {loss.item():.4f} | mse: {mse.item():.4f} | r2: {r2:.3f} | active nodes: {(node_weight > 0.5).sum().item()} / {self.N} | entropy: {ent.item():.4f}', end='\r')
        
        # Post-training evaluation
        if self.verbose:
            print()
            with torch.no_grad():
                final_node_probs, _ = torch.nn.functional.softmax(node_params.data, dim=0)
                subset_mask = (final_node_probs > 0.5).float()
                subset_mask_batch = subset_mask.view(1, -1).expand(B, -1)
                
                pred1_sub = self.model(x1, node_mask=subset_mask_batch)[:, target_idx]  # (B, T)
                pred2_sub = self.model(x2, node_mask=subset_mask_batch)[:, target_idx]  # (B, T)
                subset_diffs = pred1_sub - pred2_sub  # (B, T)
                
                subset_mse = torch.nn.functional.mse_loss(subset_diffs, target_diffs).item()
                subset_r2 = r2_score(
                    target_diffs.detach().cpu().numpy().ravel(),
                    subset_diffs.detach().cpu().numpy().ravel()
                ) if target_diffs.numel() > 1 else -666
                
                num_selected = (subset_mask > 0.5).sum().item()
                
                print("=" * 50)
                print("POST-TRAINING EVALUATION (nodes > 0.5)")
                print("=" * 50)
                print(f"Selected nodes: {num_selected} / {self.N} ({100 * num_selected / self.N:.1f}%)")
                print(f"Target Δf mean: {target_diffs.mean().item():.6f}")
                print(f"Subset Δf mean: {subset_diffs.mean().item():.6f}")
                print(f"MSE: {subset_mse:.6f}")
                print(f"R² (across {B}x{len(target_idx)} elements): {subset_r2:.4f}")
                print("=" * 50)
        
        node_scores, _ = torch.nn.functional.softmax(node_params.data, dim=0)
        
        # Package results - single set of scores for all pairs
        node_names = np.array(self.model.homo_names)
        
        result_df = pd.DataFrame({
            "node": node_names,
            "score": node_scores.detach().cpu().numpy(),
        })
        
        if return_weights:
            return result_df, node_scores.detach().cpu().numpy()
        return result_df

    def tune(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_idx: Union[int, List[int]] = None,
        min_fidelity: float = 0.9,
        beta_step: float = 1.5,
        max_trials: int = 20,
        verbose: bool = True,
        target: str = 'edge',
        **explain_kwargs,
    ):
        """
        Tune beta parameter to find maximum sparsity while maintaining fidelity.
        
        For contrastive explanations, fidelity is measured as how well the subset
        preserves the prediction difference |f(x1) - f(x2)| across all pairs.
        
        Parameters
        ----------
        x1, x2 : torch.Tensor  (shape: [N_in], [1, N_in], or [B, N_in])
            Input data pairs for explanation. When B > 1, learns a single mask
            that works well across all pairs simultaneously.
        target_idx : int or list[int], optional
            Target output indices to explain.
        min_fidelity : float, optional (default=0.9)
            Minimum fidelity threshold (1 - mean_relative_error) to maintain.
        beta_step : float, optional (default=1.5)
            Multiplicative step size for beta adjustment.
        max_trials : int, optional (default=20)
            Maximum number of beta adjustments to try.
        verbose : bool, optional (default=True)
            Whether to print search progress.
        target : str, optional (default='edge')
            Whether to tune for 'edge' or 'node' level attributions.
        **explain_kwargs : dict, optional
            Override any explainer parameters during tuning.
            
        Returns
        -------
        dict
            Results containing optimal beta, achieved fidelity, number of elements,
            and final DataFrame.
        """
        if target not in ['edge', 'node']:
            raise ValueError(f"target must be 'edge' or 'node', got '{target}'")
        
        x1, x2 = x1.to(self.device), x2.to(self.device)
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        
        B = x1.size(0)
        
        if isinstance(target_idx, int):
            target_idx = [target_idx]
            
        if verbose:
            print("=" * 60)
            print("BETA TUNING - Contrastive Explainer")
            print("=" * 60)
            print(f"Target: Find max beta with fidelity >= {min_fidelity:.3f}")
            print(f"Explanation target: {target}")
            print(f"Batch size: {B}")
            print(f"Starting beta: {self.beta:.4f}")
            print(f"Step size: {beta_step:.2f}x")
            print("=" * 60)
        
        # Store original settings
        original_settings = {
            'beta': self.beta,
            'iters': self.iters,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'free_edges': self.free_edges,
            'prior': self.prior,
            'tau0': self.tau0,
            'min_tau': self.min_tau,
            'hard': self.hard,
            'entropy': self.entropy,
            'verbose': self.verbose
        }
        
        # Apply parameter overrides
        for param, value in explain_kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        def evaluate_beta(beta_val):
            """Evaluate performance for a given beta value using batch optimization."""
            num_elements = self.E if target == 'edge' else self.N
            
            weights = torch.stack((
                self.prior * torch.ones(num_elements, dtype=torch.float32, device=self.device),
                -self.prior * torch.ones(num_elements, dtype=torch.float32, device=self.device)
            ), dim=0)
            params = torch.nn.Parameter(weights)
            
            crit = torch.nn.MSELoss()
            optim = self.optimizer([params], lr=self.lr, weight_decay=self.weight_decay)
            tau_decay_rate = (self.min_tau / self.tau0) ** (1 / self.iters)
            
            # Get target differences for ALL pairs - keep as multivariate
            with torch.no_grad():
                pred1_full = self.model(x1)[:, target_idx] if target_idx else self.model(x1)  # (B, T)
                pred2_full = self.model(x2)[:, target_idx] if target_idx else self.model(x2)  # (B, T)
                target_diffs = pred1_full - pred2_full  # (B, T) - multivariate
            
            # Training loop - single mask for all pairs
            for iter in range(self.iters):
                optim.zero_grad()
                tau = max(self.tau0 * tau_decay_rate ** iter, self.min_tau)
                weight, _ = torch.nn.functional.gumbel_softmax(params, dim=0, hard=self.hard, tau=tau)
                
                # Broadcast mask to all samples
                mask_batch = weight.view(1, -1).expand(B, -1)
                
                if target == 'edge':
                    pred1 = self.model(x1, edge_mask=mask_batch)[:, target_idx] if target_idx else self.model(x1, edge_mask=mask_batch)  # (B, T)
                    pred2 = self.model(x2, edge_mask=mask_batch)[:, target_idx] if target_idx else self.model(x2, edge_mask=mask_batch)  # (B, T)
                else:
                    pred1 = self.model(x1, node_mask=mask_batch)[:, target_idx] if target_idx else self.model(x1, node_mask=mask_batch)  # (B, T)
                    pred2 = self.model(x2, node_mask=mask_batch)[:, target_idx] if target_idx else self.model(x2, node_mask=mask_batch)  # (B, T)
                
                masked_diffs = pred1 - pred2  # (B, T) - multivariate
                mse = crit(masked_diffs, target_diffs)
                
                probs, _ = torch.nn.functional.softmax(params, dim=0)
                m = torch.distributions.Bernoulli(probs=probs)
                ent = m.entropy().mean()
                
                loss = mse + beta_val * torch.maximum(torch.tensor([0.], device=self.device), weight.sum() - self.free_edges) - self.entropy * ent
                loss.backward()
                optim.step()
            
            # Evaluate final performance
            with torch.no_grad():
                final_probs, _ = torch.nn.functional.softmax(params.data, dim=0)
                subset_mask = (final_probs > 0.5).float()
                subset_mask_batch = subset_mask.view(1, -1).expand(B, -1)
                
                if target == 'edge':
                    pred1_sub = self.model(x1, edge_mask=subset_mask_batch)[:, target_idx] if target_idx else self.model(x1, edge_mask=subset_mask_batch)  # (B, T)
                    pred2_sub = self.model(x2, edge_mask=subset_mask_batch)[:, target_idx] if target_idx else self.model(x2, edge_mask=subset_mask_batch)  # (B, T)
                else:
                    pred1_sub = self.model(x1, node_mask=subset_mask_batch)[:, target_idx] if target_idx else self.model(x1, node_mask=subset_mask_batch)  # (B, T)
                    pred2_sub = self.model(x2, node_mask=subset_mask_batch)[:, target_idx] if target_idx else self.model(x2, node_mask=subset_mask_batch)  # (B, T)
                
                subset_diffs = pred1_sub - pred2_sub  # (B, T) - multivariate
                # Fidelity based on MSE (lower is better, so 1 - normalized_mse)
                mse_val = torch.nn.functional.mse_loss(subset_diffs, target_diffs).item()
                target_var = target_diffs.var().item() + 1e-8
                fidelity = 1.0 - mse_val / target_var  # R²-like metric
                num_selected = (subset_mask > 0.5).sum().item()
            
            return fidelity, num_selected, params
        
        # Adaptive search
        current_beta = self.beta
        best_beta = current_beta
        best_fidelity = 0.0
        total_elements = self.E if target == 'edge' else self.N
        best_elements = total_elements
        best_params = None
        
        if verbose:
            print(f"\nStep 1: Evaluating starting beta = {current_beta:.4f}")
        
        try:
            initial_fidelity, initial_elements, initial_params = evaluate_beta(current_beta)
            element_type = "Edges" if target == 'edge' else "Nodes"
            
            if verbose:
                print(f"  → Fidelity = {initial_fidelity:.4f}, {element_type} = {initial_elements}")
            
            best_beta = current_beta
            best_fidelity = initial_fidelity
            best_elements = initial_elements
            best_params = initial_params
            
            # Determine search direction
            if initial_fidelity >= min_fidelity:
                search_direction = "up"
                if verbose:
                    print(f"  ✓ Good fidelity! Searching upward for more sparsity...")
            else:
                search_direction = "down"
                if verbose:
                    print(f"  ✗ Poor fidelity! Searching downward...")
            
            # Search
            for trial in range(max_trials):
                if search_direction == "up":
                    test_beta = current_beta * beta_step
                else:
                    test_beta = current_beta / beta_step
                
                if verbose:
                    print(f"\nTrial {trial + 1}: Testing beta = {test_beta:.4f}")
                
                try:
                    test_fidelity, test_elements, test_params = evaluate_beta(test_beta)
                    
                    if verbose:
                        print(f"  → Fidelity = {test_fidelity:.4f}, {element_type} = {test_elements}")
                    
                    if search_direction == "up":
                        if test_fidelity >= min_fidelity:
                            best_beta = test_beta
                            best_fidelity = test_fidelity
                            best_elements = test_elements
                            best_params = test_params
                            current_beta = test_beta
                            if verbose:
                                print(f"  ✓ Still good! New best: β={best_beta:.4f}")
                        else:
                            if verbose:
                                print(f"  ✗ Fidelity dropped, boundary found!")
                            break
                    else:
                        if test_fidelity >= min_fidelity:
                            best_beta = test_beta
                            best_fidelity = test_fidelity
                            best_elements = test_elements
                            best_params = test_params
                            if verbose:
                                print(f"  ✓ Fidelity recovered! Optimal: β={best_beta:.4f}")
                            break
                        else:
                            current_beta = test_beta
                            if verbose:
                                print(f"  ✗ Still poor, continuing...")
                    
                    if test_beta > 100 or test_beta < 0.001:
                        if verbose:
                            print(f"  ⚠ Beta limit reached, stopping")
                        break
                        
                except Exception as e:
                    if verbose:
                        print(f"  Error: {e}")
                    break
                    
        except Exception as e:
            if verbose:
                print(f"Error with initial beta: {e}")
            best_beta = self.beta
        
        # Restore original settings
        for param, value in original_settings.items():
            setattr(self, param, value)
        
        self.beta = best_beta
        
        # Create final dataframe
        final_df = None
        if best_params is not None:
            scores, _ = torch.nn.functional.softmax(best_params.data, dim=0).detach().cpu().numpy()
            if target == 'edge':
                src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
                final_df = pd.DataFrame({'source': src, 'target': dst, 'score': scores})
            else:
                node_names = np.array(self.model.homo_names)
                final_df = pd.DataFrame({'node': node_names, 'score': scores})
        
        if verbose:
            print("\n" + "=" * 60)
            print("TUNING COMPLETE")
            print("=" * 60)
            print(f"Starting beta: {original_settings['beta']:.4f}")
            print(f"Optimal beta: {best_beta:.4f}")
            print(f"Final fidelity (across {B} pairs): {best_fidelity:.4f}")
            element_type_lower = "edges" if target == 'edge' else "nodes"
            print(f"Selected {element_type_lower}: {best_elements} / {total_elements} ({100 * best_elements / total_elements:.1f}%)")
            print("=" * 60)
        
        return {
            'starting_beta': original_settings['beta'],
            'optimal_beta': best_beta,
            'beta_change_factor': best_beta / original_settings['beta'],
            'achieved_fidelity': best_fidelity,
            'num_elements': best_elements,
            'total_elements': total_elements,
            'sparsity_ratio': best_elements / total_elements,
            'result_df': final_df,
            'target': target,
            'batch_size': B
        }

