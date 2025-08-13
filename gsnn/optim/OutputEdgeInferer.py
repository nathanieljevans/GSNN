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


class OutputEdgeInferrer(torch.nn.Module):
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


    def fit(self, dataloader, model, epochs=None):
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
        super().train(True)

        for _ in range(epochs):
            epoch_losses = []
            for x, y in dataloader:
                with torch.no_grad():
                    a_dict = model.get_node_activations(x, agg=self.agg)

                # Stack function node activations: list of (B, C) -> (B, C, N)
                a = torch.stack(
                    [a_dict[node] for node in self.data.node_names_dict['function']],
                    dim=-1,
                )
                a = a.to(self.W.device)
                if self.use_batchnorm:
                    a = self._normalize(a, training=True)
                y = y.to(self.W.device)

                # Forward: (B, C, N) -> (N, B, O)
                yhat = self.forward(a)

                # Expand targets to (N, B, O) to match per-node predictions
                y_expanded = y.unsqueeze(0).expand_as(yhat)

                # Mean over batch, sum over nodes and outputs
                mse = torch.mean((yhat - y_expanded) ** 2, dim=1).sum()

                self.optim.zero_grad()
                mse.backward()
                self.optim.step()

                epoch_losses.append(mse.detach().item())

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

    def evaluate(self, dataloader, model):
        '''
        Evaluate per-node predictive power across a full dataset.

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
            - q_value: Benjamini–Hochberg FDR-adjusted p-value.

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
        super().eval()

        function_nodes = self.data.node_names_dict['function']
        output_nodes = self.data.node_names_dict['output']
        N = len(function_nodes)
        O = len(output_nodes)

        preds = [[[] for _ in range(O)] for _ in range(N)]  # per (i,j) -> list of arrays (B,)
        trues = [[[] for _ in range(O)] for _ in range(N)]

        model_preds = [[] for _ in range(O)]  # per j -> list of arrays (B,)
        model_trues = [[] for _ in range(O)]

        for x, y in dataloader:
            with torch.inference_mode():
                a_dict = model.get_node_activations(x, agg=self.agg)
                a = torch.stack([a_dict[node] for node in function_nodes], dim=-1).to(self.W.device)
                if self.use_batchnorm:
                    a = self._normalize(a, training=False)
                yhat_nodes = self.forward(a)  # (N, B, O)
                yhat_model = model(x)  # (B, O)

            y_np = y.detach().cpu().numpy()
            yhat_model_np = yhat_model.detach().cpu().numpy()

            for j in range(O):
                model_trues[j].append(y_np[:, j])
                model_preds[j].append(yhat_model_np[:, j])

            yhat_nodes_np = yhat_nodes.detach().cpu().numpy()  # (N, B, O)
            for i in range(N):
                for j in range(O):
                    preds[i][j].append(yhat_nodes_np[i, :, j])
                    trues[i][j].append(y_np[:, j])

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

        # Aggregate metrics per (func, output)
        res = {"func_node": [], "output_node": [], "mse": [], "r2": [], "r": [], "has_edge": [], "p_value": []}
        for i, fi in enumerate(function_nodes):
            for j, oj in enumerate(output_nodes):
                y_true = np.concatenate(trues[i][j]) if trues[i][j] else np.array([])
                y_pred = np.concatenate(preds[i][j]) if preds[i][j] else np.array([])
                y_pred_model = np.concatenate(model_preds[j]) if model_preds[j] else np.array([])

                if y_true.size > 0:
                    mse = float(np.mean((y_pred - y_true) ** 2))
                    r2 = _safe_evs(y_true, y_pred)
                    r = _safe_corrcoef(y_pred, y_true)
                    # One-sided p-value for improvement: mean((e_node - e_model)) < 0
                    n = min(y_true.size, y_pred_model.size)
                    if n >= 5:
                        yt = y_true[:n]
                        yn = y_pred[:n]
                        yb = y_pred_model[:n]
                        d = (yn - yt) ** 2 - (yb - yt) ** 2
                        d_mean = float(np.mean(d))
                        d_std = float(np.std(d, ddof=1)) if n > 1 else 0.0
                        if d_std > 0:
                            z = d_mean / (d_std / np.sqrt(n))
                            # One-sided normal CDF for alternative mean<0
                            pval = float(0.5 * (1.0 + math.erf(z / np.sqrt(2.0))))
                        else:
                            pval = 1.0
                    else:
                        pval = 1.0
                else:
                    mse = np.nan
                    r2 = np.nan
                    r = np.nan
                    pval = 1.0

                res["func_node"].append(fi)
                res["output_node"].append(oj)
                res["mse"].append(mse)
                res["r2"].append(r2)
                res["r"].append(r)
                res["has_edge"].append((fi, oj) in self.edges)
                res["p_value"].append(pval)

        # Baseline model metrics per output
        rdf = {"output_node": [], "model_r2": [], "model_r": [], "model_mse": []}
        for j, oj in enumerate(output_nodes):
            y_true = np.concatenate(model_trues[j]) if model_trues[j] else np.array([])
            y_pred = np.concatenate(model_preds[j]) if model_preds[j] else np.array([])

            if y_true.size > 0:
                mse = float(np.mean((y_pred - y_true) ** 2))
                r2 = _safe_evs(y_true, y_pred)
                r = _safe_corrcoef(y_pred, y_true)
            else:
                mse = np.nan
                r2 = np.nan
                r = np.nan

            rdf["output_node"].append(oj)
            rdf["model_r2"].append(r2)
            rdf["model_r"].append(r)
            rdf["model_mse"].append(mse)

        rdf = pd.DataFrame(rdf)
        res = pd.DataFrame(res)

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

        res = res.sort_values(by='r2_gain', ascending=False).reset_index(drop=True)

        # add within output rank (most likely per output node)
        res = res.sort_values(by='q_value', ascending=True).reset_index(drop=True)
        res = res.assign(within_output_rank=lambda x: x.groupby('output_node').cumcount() + 1)

        return res

    # Backwards-compatible alias to the new evaluation API (now expects a dataloader)
    def eval(self, dataloader, model):
        return self.evaluate(dataloader, model)

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
