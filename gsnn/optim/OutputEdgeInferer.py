'''
Lightweight optimizer to infer output edges from intermediate GSNN node activations.

This module estimates a per-function-node linear mapping from channel activations to
each output node using a simple batched regression. The learned weights can be
interpreted as evidence for candidate edges from function nodes to output nodes.

Assumptions:
- The GSNN `model` exposes `get_node_activations(x, agg=...)` returning a dict
  mapping function node names to tensors of shape (B, C), where B is batch size
  and C is the channel dimension for that node.
- The target `y` has shape (B, O), where O is the number of output nodes.

Notes:
- Provides `fit(dataloader, model, epochs=...)` for training.
'''

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score
import math
import scipy.special


def _safe_corrcoef(a, b):
            if a.size == 0 or b.size == 0:
                return np.nan
            if np.std(a) == 0 or np.std(b) == 0:
                return np.nan
            return float(np.corrcoef(a, b)[0, 1])

def _safe_evs(y_true, y_pred):
    try:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        # If y_true is constant, EVS is undefined; return 0 to be conservative
        if np.allclose(np.std(y_true), 0.0):
            return 0.0
        return float(explained_variance_score(y_true, y_pred))
    except Exception:
        return np.nan


class OutputEdgeInferer(torch.nn.Module):
    '''
    Learns per-function-node linear mappings from channel activations to outputs.

    Each function node i has weights `W[i]` with shape (C, O), producing per-node
    predictions that can be compared to ground truth outputs to score candidate edges.
    '''

    def __init__(self, data, channels, lr=1e-2, wd=1e-2, epochs=100, agg='last',
                 use_batchnorm=False, bn_affine=False, tol=1e-6, patience=10):
        '''
        Initialize the edge inferrer.

        Args:
            data: Dataset/graph container exposing `node_names_dict` and `edge_index_dict`.
            channels: Channel dimension C for function node activations.
            lr: Learning rate for Adam optimizer.
            wd: Weight decay (L2) for Adam optimizer.
            epochs: Number of epochs to fit over the provided dataloader.
            agg: Aggregation key passed to `model.get_node_activations`.
            use_batchnorm: If True, apply vectorized per-node, per-channel normalization with
                running mean/variance (BatchNorm-like behavior) using a single fused op.
            bn_affine: If True, learn a per-node, per-channel scale/shift.
            tol: Minimum improvement in epoch loss to reset patience (early stopping).
            patience: Number of epochs without sufficient improvement before stopping.
        '''

        super().__init__()

        self.data = data
        self.agg = agg

        self.epochs = epochs
        self.use_batchnorm = use_batchnorm
        self.bn_affine = bn_affine
        self.tol = float(tol)
        self.patience = int(patience)

        num_function_nodes = len(data.node_names_dict['function'])

        N = num_function_nodes  # number of function nodes
        O = len(data.node_names_dict['output'])
        C = channels

        # Each function node i has a weight matrix W[i] of shape (C, O)
        self.W = torch.nn.Parameter(torch.empty(N, C, O))
        for i in range(N):
            torch.nn.init.xavier_uniform_(self.W[i], gain=1.0)

        # Optional per-node normalization across channels with running stats (vectorized)
        if self.use_batchnorm:
            self.register_buffer('running_mean', torch.zeros(C, N))  # (C, N)
            self.register_buffer('running_var', torch.ones(C, N))    # (C, N)
            self.register_buffer('bn_num_batches_tracked', torch.tensor(0, dtype=torch.long))
            self.bn_momentum = 0.1
            self.bn_eps = 1e-5
            if self.bn_affine:
                self.bn_gamma = torch.nn.Parameter(torch.ones(C, N))
                self.bn_beta = torch.nn.Parameter(torch.zeros(C, N))
            else:
                self.register_buffer('bn_gamma', torch.ones(C, N))
                self.register_buffer('bn_beta', torch.zeros(C, N))
        else:
            self.running_mean = None
            self.running_var = None

        # Optimizer after all parameters are registered (including BN if affine)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        # Build existing edge set for reference; handle torch or numpy index types
        edge_index = data.edge_index_dict['function', 'to', 'output']
        if hasattr(edge_index, 'detach'):
            edge_arr = edge_index.T.detach().cpu().numpy()
        else:
            edge_arr = np.asarray(edge_index.T)
        self.edges = set(
            [
                (data.node_names_dict['function'][i], data.node_names_dict['output'][j])
                for (i, j) in edge_arr
            ]
        )


    def fit(self, dataloader, model, epochs=None, device='cpu', verbose=True):
        '''
        Fit the per-node linear mappings using batches from a dataloader.

        Args:
            dataloader: Iterable yielding tuples (x, y) with shapes x=?, y=(B, O).
            model: GSNN model exposing `get_node_activations(x, agg=...)`.
            epochs: Optional override for number of epochs. Defaults to `self.epochs`.

        Returns:
            List of average epoch losses.
        '''

        if epochs is None:
            epochs = self.epochs

        loss_history = []

        best_loss = float('inf')
        epochs_without_improve = 0

        # Ensure BN runs in training mode to update running stats
        self.train() 
        self = self.to(device)

        if verbose: print(f"Fitting OutputEdgeInferer on {device}...")
        if verbose: print('# parameters: ', sum(p.numel() for p in self.parameters()))

        for _ in range(epochs):
            epoch_losses = []
            for i, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)

                with torch.no_grad():
                    a_dict = model.get_node_activations(x, agg=self.agg)

                # Stack function node activations: list of (B, C) -> (B, C, N)
                a = torch.stack(
                    [a_dict[node] for node in self.data.node_names_dict['function']],
                    dim=-1,
                ).to(device)

                if self.use_batchnorm:
                    a = self._normalize(a, training=True)

                # Forward: (B, C, N) -> (N, B, O)
                yhat = self.forward(a)

                # Expand targets to (N, B, O) to match per-node predictions
                y_expanded = y.unsqueeze(0).expand_as(yhat)

                # Mean over batch, sum over nodes and outputs
                mse = torch.mean((yhat - y_expanded) ** 2, dim=1).mean()

                self.optim.zero_grad()
                mse.backward()
                self.optim.step()

                epoch_losses.append(mse.detach().item())
            
                if verbose: print(f"[batch {i}/{len(dataloader)} loss: {mse.detach().item()}]", end='\r')

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            loss_history.append(epoch_loss)

            # Early stopping using training loss
            if best_loss - epoch_loss > self.tol:
                best_loss = epoch_loss
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= self.patience:
                    break

            if verbose: print(f'epoch {_} loss: {epoch_loss}')

        return loss_history


    def forward(self, a):
        '''
        Compute per-function-node linear maps to outputs.

        Args:
            a: Activation tensor of shape (B, C, N), where:
               - B: batch size
               - C: channels
               - N: number of function nodes

        Returns:
            Tensor of shape (N, B, O): per-node predictions for each output.
        '''
        # Ensure activations are on same device as parameters
        a = a.to(self.W.device)
        # (B, C, N) -> (N, B, C)
        a = a.permute(2, 0, 1)
        # Batched matmul over nodes: (N, B, C) @ (N, C, O) -> (N, B, O)
        out = torch.bmm(a, self.W)
        return out

    def evaluate(self, dataloader, model, device='cpu', verbose=True):
        '''
        Evaluate per-node predictive power across a full dataset using streaming statistics.

        Args:
            dataloader: Iterable yielding tuples (x, y) with shapes x=?, y=(B, O).
            model: GSNN model exposing `get_node_activations(x, agg=...)`.

        Returns:
            pandas.DataFrame with columns:
            - func_node, output_node, mse, r2, r, has_edge
            - model_mse, model_r2, model_r
            - r2_gain, r_gain, mse_gain
            - p_value: one-sided p-value testing improvement (r2_gain > 0), via paired
              mean-squared-error test with normal approximation over samples.
            - q_value: Benjamini-Hochberg FDR-adjusted p-value.
            - snr: Signal-to-Noise Ratio (Var(predictions) / MSE). Higher values indicate
              stronger signal from function node to output.
            - l1_norm: L1 norm of weights (sparsity-promoting). Lower values = sparser model.
            - l2_norm: L2 norm of weights (regularization). Lower values = smaller weights.
            - sparsity: Fraction of weights close to zero. Higher values = sparser model.
            - eff_rank: Effective rank measure. Lower values = simpler model.

        p-value meaning:
        - Null hypothesis: the edge-specific predictor does not reduce expected MSE vs the
          baseline model for this output (i.e., r2_gain <= 0).
        - Alternative: the edge-specific predictor reduces expected MSE (r2_gain > 0).
        - We compute per-sample squared-error differences and apply a one-sided normal
          approximation to the mean difference. This is tractable and aligns with r2_gain
          since r2_gain = (mse_baseline - mse_node) / Var(y).

        FDR: We report q-values (BH-adjusted p-values) over all (func, output) pairs.
        '''

        # Use running stats for normalization during evaluation
        self.eval() 
        self = self.to(device)

        if verbose: print(f"Evaluating OutputEdgeInferer on {device}...")

        function_nodes = self.data.node_names_dict['function']
        output_nodes = self.data.node_names_dict['output']
        N = len(function_nodes)
        O = len(output_nodes)

        # Initialize vectorized streaming statistics 
        # Shape: (N, O) for node pairs, (O,) for model stats
        node_n = np.zeros((N, O), dtype=np.int64)
        node_sum_x = np.zeros((N, O), dtype=np.float64)
        node_sum_y = np.zeros((N, O), dtype=np.float64)
        node_sum_x2 = np.zeros((N, O), dtype=np.float64)
        node_sum_y2 = np.zeros((N, O), dtype=np.float64)
        node_sum_xy = np.zeros((N, O), dtype=np.float64)
        node_sum_se = np.zeros((N, O), dtype=np.float64)
        node_sum_diff = np.zeros((N, O), dtype=np.float64)
        node_sum_diff2 = np.zeros((N, O), dtype=np.float64)

        # Model statistics (O,)
        model_n = np.zeros(O, dtype=np.int64)
        model_sum_x = np.zeros(O, dtype=np.float64)
        model_sum_y = np.zeros(O, dtype=np.float64)
        model_sum_x2 = np.zeros(O, dtype=np.float64)
        model_sum_y2 = np.zeros(O, dtype=np.float64)
        model_sum_xy = np.zeros(O, dtype=np.float64)
        model_sum_se = np.zeros(O, dtype=np.float64)

        for bi, (x, y) in enumerate(dataloader):

            x = x.to(device)
            y = y.to(device)

            with torch.inference_mode():
                a_dict = model.get_node_activations(x, agg=self.agg)
                a = torch.stack([a_dict[node] for node in function_nodes], dim=-1).to(device)
                if self.use_batchnorm:
                    a = self._normalize(a, training=False)
                yhat_nodes = self.forward(a)  # (N, B, O)
                yhat_model = model(x)  # (B, O)

            y_np = y.detach().cpu().numpy()  # (B, O)
            yhat_model_np = yhat_model.detach().cpu().numpy()  # (B, O)
            yhat_nodes_np = yhat_nodes.detach().cpu().numpy()  # (N, B, O)

            B = y_np.shape[0]

            # Vectorized model statistics update
            model_n += B
            model_sum_x += np.sum(yhat_model_np, axis=0)  # (O,)
            model_sum_y += np.sum(y_np, axis=0)  # (O,)
            model_sum_x2 += np.sum(yhat_model_np ** 2, axis=0)  # (O,)
            model_sum_y2 += np.sum(y_np ** 2, axis=0)  # (O,)
            model_sum_xy += np.sum(yhat_model_np * y_np, axis=0)  # (O,)
            model_sum_se += np.sum((yhat_model_np - y_np) ** 2, axis=0)  # (O,)

            # Vectorized node statistics update using broadcasting
            # Reshape for broadcasting: y_np (B, O) -> (1, B, O), yhat_model_np (B, O) -> (1, B, O)
            y_broadcast = y_np[np.newaxis, :, :]  # (1, B, O)
            yhat_model_broadcast = yhat_model_np[np.newaxis, :, :]  # (1, B, O)
            
            # yhat_nodes_np is already (N, B, O)
            # Compute all statistics using broadcasting
            node_n += B  # (N, O) += scalar
            node_sum_x += np.sum(yhat_nodes_np, axis=1)  # (N, O)
            node_sum_y += np.sum(y_broadcast, axis=1)  # (N, O)
            node_sum_x2 += np.sum(yhat_nodes_np ** 2, axis=1)  # (N, O)
            node_sum_y2 += np.sum(y_broadcast ** 2, axis=1)  # (N, O)
            node_sum_xy += np.sum(yhat_nodes_np * y_broadcast, axis=1)  # (N, O)
            node_sum_se += np.sum((yhat_nodes_np - y_broadcast) ** 2, axis=1)  # (N, O)
            
            # Compute paired differences for p-value: (node_pred - true)² - (model_pred - true)²
            node_errors = (yhat_nodes_np - y_broadcast) ** 2  # (N, B, O)
            model_errors = (yhat_model_broadcast - y_broadcast) ** 2  # (1, B, O)
            diff = node_errors - model_errors  # (N, B, O)
            
            node_sum_diff += np.sum(diff, axis=1)  # (N, O)
            node_sum_diff2 += np.sum(diff ** 2, axis=1)  # (N, O)

            if verbose: print(f"[batch {bi}/{len(dataloader)}]", end='\r')

        # Compute final metrics from accumulated statistics
        
        # Vectorized model metrics computation
        model_mse = np.where(model_n > 0, model_sum_se / model_n, np.nan)
        
        # Vectorized correlation and explained variance for model
        model_mean_x = np.where(model_n > 0, model_sum_x / model_n, 0)
        model_mean_y = np.where(model_n > 0, model_sum_y / model_n, 0)
        model_var_x = np.where(model_n > 0, (model_sum_x2 / model_n) - model_mean_x ** 2, 0)
        model_var_y = np.where(model_n > 0, (model_sum_y2 / model_n) - model_mean_y ** 2, 0)
        model_cov_xy = np.where(model_n > 0, (model_sum_xy / model_n) - model_mean_x * model_mean_y, 0)
        
        # Correlation coefficient
        model_r = np.where(
            (model_n > 1) & (model_var_x > 0) & (model_var_y > 0),
            model_cov_xy / np.sqrt(model_var_x * model_var_y),
            np.where(model_n > 0, 0, np.nan)
        )
        
        # Explained variance score
        model_r2 = np.where(
            (model_n > 1) & (model_var_y > 0),
            np.maximum(0.0, 1.0 - model_mse / model_var_y),
            np.where(model_n > 0, 0, np.nan)
        )

        # Build model dataframe
        rdf = pd.DataFrame({
            "output_node": output_nodes,
            "model_r2": model_r2,
            "model_r": model_r,
            "model_mse": model_mse
        })

        # Vectorized node metrics computation
        node_mse = np.where(node_n > 0, node_sum_se / node_n, np.nan)
        
        # Vectorized correlation and explained variance for nodes
        node_mean_x = np.where(node_n > 0, node_sum_x / node_n, 0)
        node_mean_y = np.where(node_n > 0, node_sum_y / node_n, 0)
        node_var_x = np.where(node_n > 0, (node_sum_x2 / node_n) - node_mean_x ** 2, 0)
        node_var_y = np.where(node_n > 0, (node_sum_y2 / node_n) - node_mean_y ** 2, 0)
        node_cov_xy = np.where(node_n > 0, (node_sum_xy / node_n) - node_mean_x * node_mean_y, 0)
        
        # Correlation coefficient
        node_r = np.where(
            (node_n > 1) & (node_var_x > 0) & (node_var_y > 0),
            node_cov_xy / np.sqrt(node_var_x * node_var_y),
            np.where(node_n > 0, 0, np.nan)
        )
        
        # Explained variance score
        node_r2 = np.where(
            (node_n > 1) & (node_var_y > 0),
            np.maximum(0.0, 1.0 - node_mse / node_var_y),
            np.where(node_n > 0, 0, np.nan)
        )
        
        # Signal-to-Noise Ratio (SNR) computation for model selection
        # SNR = Var(predicted_output) / Var(residuals) = Var(predicted) / MSE
        # Higher SNR indicates stronger signal from function to output
        node_snr = np.where(
            (node_n > 1) & (node_mse > 0) & (node_var_x > 0),
            node_var_x / node_mse,
            0.0
        )
        
        # Model complexity metrics using trained weights
        # Get weights on CPU for computation: self.W shape is (N, C, O)
        W_cpu = self.W.detach().cpu().numpy()  # (N, C, O)
        
        # L1 norm (sparsity-promoting): sum of absolute weights per (function, output)
        node_l1_norm = np.sum(np.abs(W_cpu), axis=1)  # (N, O)
        
        # L2 norm (weight magnitude): Euclidean norm per (function, output)  
        node_l2_norm = np.sqrt(np.sum(W_cpu ** 2, axis=1))  # (N, O)
        
        # Weight sparsity: fraction of weights close to zero (< 1e-6)
        sparsity_threshold = 1e-6
        node_sparsity = np.mean(np.abs(W_cpu) < sparsity_threshold, axis=1)  # (N, O)
        
        # Effective rank: number of significant singular values (> 1% of max)
        node_eff_rank = np.zeros((N, O))
        for i in range(N):
            for j in range(O):
                w_vec = W_cpu[i, :, j]  # (C,)
                if np.any(np.abs(w_vec) > 1e-12):  # avoid zero vectors
                    # For 1D weight vector, effective rank is just whether it's non-zero
                    node_eff_rank[i, j] = 1.0 if np.std(w_vec) > 1e-6 else 0.0
                else:
                    node_eff_rank[i, j] = 0.0
        
        # Vectorized p-value computation
        node_d_mean = np.where(node_n > 0, node_sum_diff / node_n, 0)
        node_d_var = np.where(node_n > 0, (node_sum_diff2 / node_n) - node_d_mean ** 2, 0)
        node_d_std = np.sqrt(np.maximum(0, node_d_var))
        
        # Compute z-scores
        node_z = np.where(
            (node_n >= 5) & (node_d_std > 0),
            node_d_mean / (node_d_std / np.sqrt(node_n)),
            0
        )
        
        # One-sided normal CDF for alternative mean<0
        # Using vectorized error function
        node_pval = np.where(
            (node_n >= 5) & (node_d_std > 0),
            0.5 * (1.0 + scipy.special.erf(node_z / np.sqrt(2.0))),
            1.0
        )
        node_pval = np.where(node_n > 0, node_pval, 1.0)
        
        # Build node dataframe using vectorized operations
        func_nodes_flat = []
        output_nodes_flat = []
        mse_flat = []
        r2_flat = []
        r_flat = []
        has_edge_flat = []
        pval_flat = []
        snr_flat = []
        l1_norm_flat = []
        l2_norm_flat = []
        sparsity_flat = []
        eff_rank_flat = []
        
        for i, fi in enumerate(function_nodes):
            for j, oj in enumerate(output_nodes):
                func_nodes_flat.append(fi)
                output_nodes_flat.append(oj)
                mse_flat.append(node_mse[i, j])
                r2_flat.append(node_r2[i, j])
                r_flat.append(node_r[i, j])
                has_edge_flat.append((fi, oj) in self.edges)
                pval_flat.append(node_pval[i, j])
                snr_flat.append(node_snr[i, j])
                l1_norm_flat.append(node_l1_norm[i, j])
                l2_norm_flat.append(node_l2_norm[i, j])
                sparsity_flat.append(node_sparsity[i, j])
                eff_rank_flat.append(node_eff_rank[i, j])
        
        res = pd.DataFrame({
            "func_node": func_nodes_flat,
            "output_node": output_nodes_flat,
            "mse": mse_flat,
            "r2": r2_flat,
            "r": r_flat,
            "has_edge": has_edge_flat,
            "p_value": pval_flat,
            "snr": snr_flat,
            "l1_norm": l1_norm_flat,
            "l2_norm": l2_norm_flat,
            "sparsity": sparsity_flat,
            "eff_rank": eff_rank_flat
        })



        res = res.merge(rdf, on='output_node', how='left')

        res = res.assign(
            r2_gain=lambda x: x.r2 - x.model_r2,
            r_gain=lambda x: x.r - x.model_r,
            mse_gain=lambda x: x.mse - x.model_mse,
        )

        # Benjamini–Hochberg q-values across all pairs (monotone BH)
        pvals = res["p_value"].values.astype(float)
        m = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]
        bh = ranked * m / (np.arange(1, m + 1))
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        bh = np.clip(bh, 0.0, 1.0)
        q_values = np.empty_like(bh)
        q_values[order] = bh
        res["q_value"] = q_values

        # Add SNR-based ranking within each output (1 = highest SNR/strongest signal)
        res = res.sort_values(['output_node', 'snr'], ascending=[True, False]).reset_index(drop=True)
        res = res.assign(snr_rank=lambda x: x.groupby('output_node').cumcount() + 1)
        
        # Add sparsity-based ranking within each output (1 = most sparse/simplest)
        res = res.sort_values(['output_node', 'sparsity'], ascending=[True, False]).reset_index(drop=True)
        res = res.assign(sparsity_rank=lambda x: x.groupby('output_node').cumcount() + 1)
        
        # Add within output rank based on q-value (1 = most significant)
        res = res.sort_values(['output_node', 'q_value'], ascending=[True, True]).reset_index(drop=True)
        res = res.assign(within_output_rank=lambda x: x.groupby('output_node').cumcount() + 1)
        
        # Sort by r2_gain for final output
        res = res.sort_values(by='r2_gain', ascending=False).reset_index(drop=True)

        return res

    def _normalize(self, a, training=True):
        '''
        Vectorized per-node, per-channel normalization with running stats.

        Args:
            a: Tensor (B, C, N)
            training: If True, update running stats using batch mean/var; else use running stats.

        Returns:
            Normalized tensor (B, C, N).
        '''
        if not self.use_batchnorm:
            return a
        device = a.device
        dtype = a.dtype
        # Ensure buffers/params on device/dtype
        self.running_mean = self.running_mean.to(device=device, dtype=dtype)
        self.running_var = self.running_var.to(device=device, dtype=dtype)
        self.bn_gamma = self.bn_gamma.to(device=device, dtype=dtype)
        self.bn_beta = self.bn_beta.to(device=device, dtype=dtype)

        if training:
            batch_mean = a.mean(dim=0)  # (C, N)
            batch_var = a.var(dim=0, unbiased=False)  # (C, N)
            momentum = self.bn_momentum
            # Update running stats in-place
            self.running_mean.lerp_(batch_mean, momentum)
            self.running_var.lerp_(batch_var, momentum)
            if self.bn_num_batches_tracked is not None:
                self.bn_num_batches_tracked = self.bn_num_batches_tracked.to(device)
                self.bn_num_batches_tracked += 1
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        a = (a - mean) / torch.sqrt(var + self.bn_eps)
        a = a * self.bn_gamma + self.bn_beta
        return a
