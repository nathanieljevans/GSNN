import copy
from typing import Union, List, Optional, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class CounterfactualExplainer:
    r"""Feature-level counterfactual explainer using gradient descent.

    This module learns a minimal perturbation **δ** to an input **x** such that::

        f(x + δ) ≈ target_value

    The perturbation is learned via gradient descent with L2 regularization to
    enforce minimality. The optimization objective is:

    .. math::

        \min_δ \|f(x + δ) - \text{target}\|^2 + λ\|δ\|^2

    where λ is the weight decay parameter controlling the trade-off between
    achieving the target and minimizing the perturbation.

    Interpretation
    --------------
    * ``δ_i > 0``  feature *i* needs to be increased to reach the target.
    * ``δ_i < 0``  feature *i* needs to be decreased to reach the target.
    * ``δ_i ≈ 0``  feature *i* is irrelevant for the counterfactual.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GSNN model (evaluation mode is enforced internally).
    data : torch_geometric.data.Data, optional
        Graph data object; used for human-readable feature names.
    ignore_cuda : bool, optional (default=False)
        Force the explainer to run on CPU even if CUDA is available.

    Example
    -------
    >>> explainer = CounterfactualExplainer(model, data)
    >>> # Single observation
    >>> df = explainer.explain(x, target_value=0.8, target_idx=0, max_iter=500)
    >>> # Multiple observations (same perturbation applied to all)
    >>> df = explainer.explain(x_batch, target_value=0.8, target_idx=0, max_iter=500)
    >>> df.sort_values('perturbation', key=abs, ascending=False).head()
    feature    original  perturbation  counterfactual
    in0        0.12      0.45          0.57
    in1        0.89     -0.23          0.66
    in2        0.34      0.11          0.45
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data=None,
        ignore_cuda: bool = False,
    ) -> None:
        self.data = data
        self.device = (
            "cuda" if (torch.cuda.is_available() and not ignore_cuda) else "cpu"
        )

        # Work on a frozen copy of the model to avoid side-effects.
        model = copy.deepcopy(model).eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

    def explain(
        self,
        x: torch.Tensor,
        target_value: Union[float, torch.Tensor],
        target_idx: Optional[Union[int, List[int]]] = None,
        trainable_mask: Optional[torch.Tensor] = None,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        dropout: float = 0.0,
        min_iter: int = 25,
        max_iter: int = 1000,
        tolerance: float = 1e-5,
        verbose: bool = True,
        transform: Optional[Callable] = torch.nn.Identity(),
    ) -> pd.DataFrame:
        """Learn minimal perturbation to achieve target model output.

        Parameters
        ----------
        x : torch.Tensor  (shape: [N_in] or [B, N_in])
            Input feature tensor. If 1D, it will be unsqueezed to batch size 1.
            For multiple observations, the same perturbation will be applied to all.
        target_value : float or torch.Tensor
            Desired model output. If target_idx is specified, this should be a
            scalar or tensor matching the number of target indices. If target_idx
            is None, this should match the full output dimension. The same target
            value is used for all observations in the batch.
        target_idx : int, list[int], or None
            Output dimension(s) to target. If None, targets all outputs.
        trainable_mask : torch.Tensor, optional (shape: [N_in])
            Boolean mask specifying which features can be perturbed. If None,
            all features are trainable.
        lr : float, optional (default=0.01)
            Learning rate for gradient descent.
        weight_decay : float, optional (default=0.01)
            L2 regularization coefficient for minimizing perturbation magnitude.
        dropout : float, optional (default=0.0)
            Dropout rate for the model.
        min_iter : int, optional (default=25)
            Minimum number of optimization iterations.
        max_iter : int, optional (default=1000)
            Maximum number of optimization iterations.
        tolerance : float, optional (default=1e-6)
            Convergence tolerance for loss change between iterations.
        verbose : bool, optional (default=False)
            Print optimization progress.
        transform : Callable, optional
            Transform the perturbation, must be differentiable. E.g., relu(), tanh()

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'feature', 'original', 'perturbation', 'counterfactual'
            showing the learned perturbations for each input feature.
        """
        
        # ------------------------------------------------------------------
        # 1. Setup and validation
        # ------------------------------------------------------------------
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size, n_features = x.shape
        

        
        # Handle target_idx
        target_idx_tensor = None
        if target_idx is not None:
            if isinstance(target_idx, int):
                target_idx_list = [target_idx]
            else:
                target_idx_list = target_idx
            target_idx_tensor = torch.tensor(target_idx_list, device=self.device)
        
        # Setup trainable mask
        if trainable_mask is not None:
            trainable_mask = trainable_mask.to(self.device).bool()
            if trainable_mask.shape != (n_features,):
                raise ValueError(f"trainable_mask shape {trainable_mask.shape} doesn't match input features {n_features}")
        else:
            trainable_mask = torch.ones(n_features, device=self.device, dtype=torch.bool)

        # ------------------------------------------------------------------
        # 2. Initialize perturbation and optimizer
        # ------------------------------------------------------------------
        # Use a single perturbation vector that will be broadcast across all batch examples
        x_attack = torch.zeros(1, n_features, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([x_attack], lr=lr, weight_decay=weight_decay)
        
        # Ensure target_value is properly shaped for batch operations
        if not isinstance(target_value, torch.Tensor):
            target_value = torch.tensor(target_value, device=self.device, dtype=x.dtype)
        else:
            target_value = target_value.to(self.device)
        
        # Expand target_value to match batch size if needed
        if target_idx_tensor is not None:
            target_shape = (batch_size, len(target_idx_tensor))
        else:
            # Get output size from a test forward pass
            with torch.no_grad():
                test_output = self.model(x[:1])  # Use first sample to get output shape
                target_shape = (batch_size, test_output.shape[1])
        
        if target_value.dim() == 0:  # scalar
            target_value = target_value.expand(target_shape)
        elif target_value.dim() == 1 and target_value.shape[0] == target_shape[1]:
            target_value = target_value.unsqueeze(0).expand(target_shape)
        
        # Store original prediction for reference
        with torch.no_grad():
            original_pred = self.model(x)
            if target_idx_tensor is not None:
                original_pred = original_pred[:, target_idx_tensor]

        # ------------------------------------------------------------------
        # 3. Gradient descent optimization
        # ------------------------------------------------------------------
        prev_loss = float('inf')
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # Forward pass with perturbation
            x_perturbed = x + F.dropout(transform(x_attack), p=dropout)
            pred = self.model(x_perturbed)
            
            # Select target dimensions if specified
            if target_idx_tensor is not None:
                pred = pred[:, target_idx_tensor]
            
            # Compute loss (MSE between prediction and target)
            loss = F.mse_loss(pred, target_value)
            
            # Backward pass
            loss.backward()
            
            # Apply trainable mask by zeroing gradients of non-trainable features
            if x_attack.grad is not None:
                x_attack.grad[:, ~trainable_mask] = 0.0
            
            # Optimization step
            optimizer.step()
            
            # Apply trainable mask to perturbation itself (hard constraint)
            with torch.no_grad():
                x_attack[:, ~trainable_mask] = 0.0
            
            # Check convergence
            loss_val = loss.item()
            if verbose: print(f"Iteration {iteration}: Loss = {loss_val:.6f}", end='\r')
            
            if (abs(prev_loss - loss_val) < tolerance) and (iteration > min_iter):
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                    print(f"Final loss: {loss_val:.6f}")
                break
            
            prev_loss = loss_val


        # ------------------------------------------------------------------
        # 4. Package results as DataFrame
        # ------------------------------------------------------------------
        with torch.no_grad():
            x_final = x + transform(x_attack)
            final_pred = self.model(x_final)
            if target_idx_tensor is not None:
                final_pred = final_pred[:, target_idx_tensor]
        
        # Extract numpy arrays - for multiple observations, we show the average original and counterfactual
        if batch_size == 1:
            x_np = x.squeeze(0).detach().cpu().numpy()
            x_final_np = x_final.squeeze(0).detach().cpu().numpy()
        else:
            x_np = x.mean(dim=0).detach().cpu().numpy()  # Average across batch
            x_final_np = x_final.mean(dim=0).detach().cpu().numpy()  # Average across batch
        
        x_attack_np = transform(x_attack).squeeze(0).detach().cpu().numpy()  # Same perturbation for all
        
        # Create feature names
        if self.data is not None and hasattr(self.data, 'node_names_dict'):
            feature_names = self.data.node_names_dict['input']
        else:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Create DataFrame
        df = pd.DataFrame({
            "feature": feature_names,
            "original": x_np,
            "perturbation": x_attack_np,
            "counterfactual": x_final_np,
        })
        
        # Add metadata as attributes
        df.attrs['converged_loss'] = loss_val
        df.attrs['iterations'] = iteration + 1
        df.attrs['batch_size'] = batch_size
        df.attrs['original_prediction'] = original_pred.detach().cpu().numpy()
        df.attrs['final_prediction'] = final_pred.detach().cpu().numpy()
        df.attrs['target_value'] = target_value.detach().cpu().numpy()
        
        return df 