import numpy as np 
import torch 
import pandas as pd 
from sklearn.metrics import r2_score
import copy 
 

class GSNNExplainer:
    r"""Edge/node mask optimiser that produces *sparse* explanations.

    The explainer learns a binary mask *m∈\{0,1\}^{E|N}* that maximises fidelity
    between the model's prediction on the **masked** graph and the prediction
    on the *full* graph while simultaneously penalising mask size::

        L = MSE\bigl(f(x; m), f(x; 1)\bigr)
            + β \max(0, \|m\|₁ − free_elements)
            − λ H(m)            (optional entropy term)

    Here *m* is obtained via a differentiable Gumbel-Softmax relaxation so the
    optimisation can be performed with vanilla back-prop.  After convergence
    the importance score is the softmax probability *p_i = P(m_i=1)*.

    Interpretation
    --------------
    * ``score_i → 1``   element i is essential for reproducing the original prediction.
    * ``score_i → 0``   element i can be removed with little impact.

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

    Example
    -------
    >>> explainer = GSNNExplainer(model, data, iters=400, beta=5)
    >>> # Edge-level attributions
    >>> edge_df = explainer.explain(x, targets=[0], target='edge')
    >>> edge_df.sort_values('score', ascending=False).head()
    >>> # Node-level attributions
    >>> node_df = explainer.explain(x, targets=[0], target='node') 
    >>> node_df.sort_values('score', ascending=False).head()
    """

    def __init__(self, model, data, ignore_cuda=False, gumbel_softmax=True, hard=False, tau0=3, min_tau=0.5, 
                            prior=1, iters=250, lr=1e-2, weight_decay=1e-5, free_edges=0, grad_norm_clip=0,
                                    beta=1, verbose=True, optimizer=torch.optim.Adam, entropy=0): 
        '''
        Adapted from the methods presented in `GNNExplainer` (https://arxiv.org/abs/1903.03894). 

        Args: 
            model           torch.nn.Module             GSNN Model 
            data            pyg.Data                    GSNN processed graph data
            beta            float                       regularization scalar encouraging a minimal subset of edges
            ignore_cuda     bool                        whether to use cuda if available 
            hard            bool                        discrete forward operation for gumbel-softmax 
            tau0            float                       initial temperature value for gumbel-softmax 
            min_tau         float                       minimum temperature value for gumbel-softmax 
            prior           float                       prior strength to initialize theta; value of 0 will make each element 0.5 prob of being selecting, value > 0 will make it more likely to be selected. 
            grad_norm_clip  float                       gradient norm clipping value
            verbose         bool                        whether to print progress information during optimisation
            optimizer       torch.optim.Optimizer       optimizer to use for training
            entropy         float                       entropy bonus strength
            iters           int                         number of optimisation steps
            lr              float                       learning rate for the optimiser
            weight_decay    float                       weight decay for the optimiser
            free_edges      int                         number of edges allowed before the sparsity penalty activates

        Returns 
            None 
        '''
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
        self.grad_norm_clip = grad_norm_clip
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

    def explain(self, x, target_idx=None, return_weights=False, target='edge'): 
        '''
        Initializes and runs gradient descent to select a minimal subset of edges or nodes that produce comparable predictions 
        to the full graph. 
        
        Parameters
        ----------
        x : torch.tensor
            Input features to explain; in shape (B, I).
        targets : list, optional
            Target output indices to explain.
        return_weights : bool, optional (default=False)
            Whether to return raw weights along with the DataFrame.
        target : str, optional (default='edge')
            Whether to return 'edge' or 'node' level attributions.

        Returns 
        -------
        pd.DataFrame
            If target='edge': columns ['source', 'target', 'score'] for edge attributions.
            If target='node': columns ['node', 'score'] for node attributions.
        '''
        
        if target not in ['edge', 'node']:
            raise ValueError(f"target must be 'edge' or 'node', got '{target}'")

        if target == 'edge':
            return self._explain_edges(x, target_idx, return_weights)
        elif target == 'node':
            return self._explain_nodes(x, target_idx, return_weights)

    def _explain_edges(self, x, targets=None, return_weights=False):
        '''
        Compute edge-level attributions using gradient descent optimization.
        
        Parameters
        ----------
        x : torch.tensor
            Input features to explain; in shape (B, I).
        targets : list, optional
            Target output indices to explain.
        return_weights : bool, optional (default=False)
            Whether to return raw weights along with the DataFrame.
            
        Returns
        -------
        pd.DataFrame
            Columns ['source', 'target', 'score'] for edge attributions.
        '''
        
        weights = torch.stack((self.prior*torch.ones(self.model.edge_index.size(1), dtype=torch.float32, device=self.device, requires_grad=True), 
                                -self.prior*torch.ones(self.model.edge_index.size(1), dtype=torch.float32, device=self.device, requires_grad=True)), dim=0)

        edge_params = torch.nn.Parameter(weights)
        
        # optimize parameter mask with objective 
        crit = torch.nn.MSELoss()
        optim = self.optimizer([edge_params], lr=self.lr, weight_decay=self.weight_decay)

        # calculate tau decay rate
        tau_decay_rate = (self.min_tau / self.tau0) ** (1 / self.iters)

        # get target predictions 
        with torch.no_grad():
            target_preds = self.model(x)

        if targets is not None: 
            target_preds = target_preds[:, targets]

        for iter in range(self.iters):    
            optim.zero_grad()

            tau = max(self.tau0 * tau_decay_rate**iter, self.min_tau)

            edge_weight, _ = torch.nn.functional.gumbel_softmax(edge_params, dim=0, hard=self.hard, tau=tau)

            out = self.model(x, edge_mask=edge_weight.view(1, -1))

            if targets is not None: 
                out = out[:, targets]

            mse = crit(out, target_preds)

            edge_probs, _ = torch.nn.functional.softmax(edge_params, dim=0)
            m = torch.distributions.Bernoulli(probs=edge_probs)
            ent = m.entropy().mean()

            loss = mse \
                + self.beta*torch.maximum(torch.tensor([0.], device=x.device), edge_weight.sum() - self.free_edges) \
                - self.entropy*ent

            loss.backward() 

            if self.grad_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(edge_params.grad, self.grad_norm_clip)

            optim.step() 

            with torch.no_grad():
                if out.view(-1).shape[0] == 1:
                    r2 = -666 
                else: 
                    r2 = r2_score(target_preds.detach().cpu().numpy().ravel(), out.detach().cpu().numpy().ravel())

            if self.verbose: 
                print(f'iter: {iter} | loss: {loss.item():.4f} | mse: {mse.item():.4f} | r2: {r2:.3f} | active edges: {(edge_weight > 0.5).sum().item()} / {self.model.edge_index.size(1)} | entropy: {ent.item():.4f}', end='\r')

        # Post-training evaluation with subset edges > 0.5
        if self.verbose:
            print()  # New line after training progress
            with torch.no_grad():
                # Get final edge weights and create binary mask for edges > 0.5
                final_edge_probs, _ = torch.nn.functional.softmax(edge_params.data, dim=0)
                subset_mask = (final_edge_probs > 0.5).float()
                
                # Evaluate performance using only edges > 0.5
                subset_out = self.model(x, edge_mask=subset_mask.view(1, -1))
                if targets is not None:
                    subset_out = subset_out[:, targets]
                
                subset_mse = torch.nn.functional.mse_loss(subset_out, target_preds).item()
                subset_r2 = r2_score(target_preds.detach().cpu().numpy().ravel(), 
                                   subset_out.detach().cpu().numpy().ravel())
                
                # Calculate variance explained (R2 can be negative, so we also show raw correlation)
                target_flat = target_preds.detach().cpu().numpy().ravel()
                pred_flat = subset_out.detach().cpu().numpy().ravel()
                correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
                variance_explained = correlation ** 2 if not np.isnan(correlation) else 0.0
                
                num_selected_edges = (subset_mask > 0.5).sum().item()
                total_edges = len(subset_mask)
                
                print("="*50)
                print("POST-TRAINING EVALUATION (edges > 0.5)")
                print("="*50)
                print(f"Selected edges: {num_selected_edges} / {total_edges} ({100*num_selected_edges/total_edges:.1f}%)")
                print(f"MSE (subset): {subset_mse:.6f}")
                print(f"R² (subset): {subset_r2:.4f}")
                print(f"Variance explained: {variance_explained:.4f}")
                print(f"Correlation: {correlation:.4f}")
                print("="*50)

        edge_scores, _ = torch.nn.functional.softmax(edge_params.data, dim=0).detach().cpu().numpy()
       
        src,dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        edgedf = pd.DataFrame({'source':src, 'target':dst, 'score':edge_scores})

        if return_weights:
            return edgedf, edge_scores
        else:
            return edgedf

    def _explain_nodes(self, x, targets=None, return_weights=False):
        '''
        Compute node-level attributions using gradient descent optimization.
        
        Parameters
        ----------
        x : torch.tensor
            Input features to explain; in shape (B, I).
        targets : list, optional
            Target output indices to explain.
        return_weights : bool, optional (default=False)
            Whether to return raw weights along with the DataFrame.
            
        Returns
        -------
        pd.DataFrame
            Columns ['node', 'score'] for node attributions.
        '''
        
        weights = torch.stack((self.prior*torch.ones(self.model.num_nodes, dtype=torch.float32, device=self.device, requires_grad=True), 
                                -self.prior*torch.ones(self.model.num_nodes, dtype=torch.float32, device=self.device, requires_grad=True)), dim=0)

        node_params = torch.nn.Parameter(weights)
        
        # optimize parameter mask with objective 
        crit = torch.nn.MSELoss()
        optim = self.optimizer([node_params], lr=self.lr, weight_decay=self.weight_decay)

        # calculate tau decay rate
        tau_decay_rate = (self.min_tau / self.tau0) ** (1 / self.iters)

        # get target predictions 
        with torch.no_grad():
            target_preds = self.model(x)
        if targets is not None: 
            target_preds = target_preds[:, targets]

        for iter in range(self.iters):    
            optim.zero_grad()

            tau = max(self.tau0 * tau_decay_rate**iter, self.min_tau)

            node_weight, _ = torch.nn.functional.gumbel_softmax(node_params, dim=0, hard=self.hard, tau=tau)

            out = self.model(x, node_mask=node_weight.view(1, -1))

            if targets is not None: 
                out = out[:, targets]

            mse = crit(out, target_preds)

            node_probs, _ = torch.nn.functional.softmax(node_params, dim=0)
            m = torch.distributions.Bernoulli(probs=node_probs)
            ent = m.entropy().mean()

            loss = mse \
                + self.beta*torch.maximum(torch.tensor([0.], device=x.device), node_weight.sum() - self.free_edges) \
                - self.entropy*ent

            loss.backward() 
            optim.step() 

            with torch.no_grad():
                r2 = r2_score(target_preds.detach().cpu().numpy().ravel(), out.detach().cpu().numpy().ravel())

            if self.verbose: 
                print(f'iter: {iter} | loss: {loss.item():.4f} | mse: {mse.item():.4f} | r2: {r2:.3f} | active nodes: {(node_weight > 0.5).sum().item()} / {self.model.num_nodes} | entropy: {ent.item():.4f}', end='\r')

        # Post-training evaluation with subset nodes > 0.5
        if self.verbose:
            print()  # New line after training progress
            with torch.no_grad():
                # Get final node weights and create binary mask for nodes > 0.5
                final_node_probs, _ = torch.nn.functional.softmax(node_params.data, dim=0)
                subset_mask = (final_node_probs > 0.5).float()
                
                # Evaluate performance using only nodes > 0.5
                subset_out = self.model(x, node_mask=subset_mask.view(1, -1))
                if targets is not None:
                    subset_out = subset_out[:, targets]
                
                subset_mse = torch.nn.functional.mse_loss(subset_out, target_preds).item()
                subset_r2 = r2_score(target_preds.detach().cpu().numpy().ravel(), 
                                   subset_out.detach().cpu().numpy().ravel())
                
                # Calculate variance explained (R2 can be negative, so we also show raw correlation)
                target_flat = target_preds.detach().cpu().numpy().ravel()
                pred_flat = subset_out.detach().cpu().numpy().ravel()
                correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
                variance_explained = correlation ** 2 if not np.isnan(correlation) else 0.0
                
                num_selected_nodes = (subset_mask > 0.5).sum().item()
                total_nodes = len(subset_mask)
                
                print("="*50)
                print("POST-TRAINING EVALUATION (nodes > 0.5)")
                print("="*50)
                print(f"Selected nodes: {num_selected_nodes} / {total_nodes} ({100*num_selected_nodes/total_nodes:.1f}%)")
                print(f"MSE (subset): {subset_mse:.6f}")
                print(f"R² (subset): {subset_r2:.4f}")
                print(f"Variance explained: {variance_explained:.4f}")
                print(f"Correlation: {correlation:.4f}")
                print("="*50)

        node_scores, _ = torch.nn.functional.softmax(node_params.data, dim=0).detach().cpu().numpy()
       
        node_names = np.array(self.model.homo_names)
        nodedf = pd.DataFrame({'node': node_names, 'score': node_scores})

        if return_weights:
            return nodedf, node_scores
        else:
            return nodedf
    
    def tune(self, x, target_ixs=None, min_r2=0.7, beta_step=1.5, max_trials=20, 
             tolerance=1e-3, verbose=True, target='edge', **explain_kwargs):
        """
        Tune beta parameter starting from current value to find maximum sparsity while 
        maintaining minimum performance.
        
        Starts from the user's initial beta and adjusts up/down based on performance:
        - If R² >= min_r2: increase beta (more sparsity) until performance drops
        - If R² < min_r2: decrease beta (less sparsity) until performance recovers
        
        Much more efficient than wide search since user provides good starting point.
        
        Args:
            x : torch.Tensor
                Input data for explanation
            target_ixs : list, optional
                Target output indices to explain
            min_r2 : float, optional (default=0.7)
                Minimum R² threshold to maintain
            beta_step : float, optional (default=1.5)
                Multiplicative step size for beta adjustment (1.5 = 50% increase/decrease)
            max_trials : int, optional (default=20)
                Maximum number of beta adjustments to try
            tolerance : float, optional (default=1e-3)
                Convergence tolerance for fine search
            verbose : bool, optional (default=True)
                Whether to print search progress
            target : str, optional (default='edge')
                Whether to tune for 'edge' or 'node' level attributions
            **explain_kwargs : dict, optional
                Override any explainer parameters during tuning:
                - iters: number of optimization steps
                - lr: learning rate  
                - weight_decay: weight decay
                - free_edges: elements allowed before penalty
                - prior: initial bias for element selection
                - tau0: initial temperature
                - min_tau: minimum temperature
                - hard: use straight-through estimator
                - entropy: entropy bonus strength
                
        Returns:
            dict: Results containing optimal beta, achieved R², number of elements, and final DataFrame
        """
        if target not in ['edge', 'node']:
            raise ValueError(f"target must be 'edge' or 'node', got '{target}'")
            
        if verbose:
            print("="*60)
            print("BETA TUNING - Starting from User's Beta")
            print("="*60)
            print(f"Target: Find max beta with R² >= {min_r2:.3f}")
            print(f"Explanation target: {target}")
            print(f"Starting beta: {self.beta:.4f}")
            print(f"Step size: {beta_step:.2f}x")
            if explain_kwargs:
                print(f"Parameter overrides: {explain_kwargs}")
            print("="*60)
        
        # Store original settings for all tunable parameters
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
            else:
                if verbose:
                    print(f"Warning: Unknown parameter '{param}' ignored")
        
        # Disable verbose during tuning iterations unless specifically requested
        tuning_verbose = self.verbose if 'verbose' in explain_kwargs else False
        
        def evaluate_beta(beta_val):
            """Evaluate performance for a given beta value"""
            if target == 'edge':
                # Initialize edge parameters
                num_elements = self.model.edge_index.size(1)
                weights = torch.stack((self.prior*torch.ones(num_elements, dtype=torch.float32, device=self.device), 
                                        -self.prior*torch.ones(num_elements, dtype=torch.float32, device=self.device)), dim=0)
                params = torch.nn.Parameter(weights)
                
                # Setup training
                crit = torch.nn.MSELoss()
                optim = self.optimizer([params], lr=self.lr, weight_decay=self.weight_decay)
                tau_decay_rate = (self.min_tau / self.tau0) ** (1 / self.iters)
                
                # Get target predictions
                with torch.no_grad():
                    target_preds = self.model(x)
                    if target_ixs is not None: 
                        target_preds = target_preds[:, target_ixs]
                
                # Run training
                for iter in range(self.iters):
                    optim.zero_grad()
                    tau = max(self.tau0 * tau_decay_rate**iter, self.min_tau)
                    weight, _ = torch.nn.functional.gumbel_softmax(params, dim=0, hard=self.hard, tau=tau)
                    out = self.model(x, edge_mask=weight.view(1, -1))
                    if target_ixs is not None:
                        out = out[:, target_ixs]
                    
                    mse = crit(out, target_preds)
                    probs, _ = torch.nn.functional.softmax(params, dim=0)
                    m = torch.distributions.Bernoulli(probs=probs)
                    ent = m.entropy().mean()
                    
                    loss = mse + beta_val*torch.maximum(torch.tensor([0.], device=x.device), weight.sum() - self.free_edges) - self.entropy*ent
                    loss.backward()
                    optim.step()
                    
                    if tuning_verbose and iter % 50 == 0:
                        with torch.no_grad():
                            r2 = r2_score(target_preds.detach().cpu().numpy().ravel(), out.detach().cpu().numpy().ravel())
                        print(f'    iter: {iter} | loss: {loss.item():.4f} | r2: {r2:.3f} | beta: {beta_val:.4f}')
                
                # Evaluate final performance on subset
                with torch.no_grad():
                    final_probs, _ = torch.nn.functional.softmax(params.data, dim=0)
                    subset_mask = (final_probs > 0.5).float()
                    subset_out = self.model(x, edge_mask=subset_mask.view(1, -1))
                    if target_ixs is not None:
                        subset_out = subset_out[:, target_ixs]
                    
                    subset_r2 = r2_score(target_preds.detach().cpu().numpy().ravel(), 
                                       subset_out.detach().cpu().numpy().ravel())
                    num_elements = (subset_mask > 0.5).sum().item()
            
            else:  # target == 'node'
                # Initialize node parameters
                num_elements = self.model.num_nodes
                weights = torch.stack((self.prior*torch.ones(num_elements, dtype=torch.float32, device=self.device), 
                                        -self.prior*torch.ones(num_elements, dtype=torch.float32, device=self.device)), dim=0)
                params = torch.nn.Parameter(weights)
                
                # Setup training
                crit = torch.nn.MSELoss()
                optim = self.optimizer([params], lr=self.lr, weight_decay=self.weight_decay)
                tau_decay_rate = (self.min_tau / self.tau0) ** (1 / self.iters)
                
                # Get target predictions
                with torch.no_grad():
                    target_preds = self.model(x)
                    if target_ixs is not None: 
                        target_preds = target_preds[:, target_ixs]
                
                # Run training
                for iter in range(self.iters):
                    optim.zero_grad()
                    tau = max(self.tau0 * tau_decay_rate**iter, self.min_tau)
                    weight, _ = torch.nn.functional.gumbel_softmax(params, dim=0, hard=self.hard, tau=tau)
                    out = self.model(x, node_mask=weight.view(1, -1))
                    if target_ixs is not None:
                        out = out[:, target_ixs]
                    
                    mse = crit(out, target_preds)
                    probs, _ = torch.nn.functional.softmax(params, dim=0)
                    m = torch.distributions.Bernoulli(probs=probs)
                    ent = m.entropy().mean()
                    
                    loss = mse + beta_val*torch.maximum(torch.tensor([0.], device=x.device), weight.sum() - self.free_edges) - self.entropy*ent
                    loss.backward()
                    optim.step()
                    
                    if tuning_verbose and iter % 50 == 0:
                        with torch.no_grad():
                            r2 = r2_score(target_preds.detach().cpu().numpy().ravel(), out.detach().cpu().numpy().ravel())
                        print(f'    iter: {iter} | loss: {loss.item():.4f} | r2: {r2:.3f} | beta: {beta_val:.4f}')
                
                # Evaluate final performance on subset
                with torch.no_grad():
                    final_probs, _ = torch.nn.functional.softmax(params.data, dim=0)
                    subset_mask = (final_probs > 0.5).float()
                    subset_out = self.model(x, node_mask=subset_mask.view(1, -1))
                    if target_ixs is not None:
                        subset_out = subset_out[:, target_ixs]
                    
                    subset_r2 = r2_score(target_preds.detach().cpu().numpy().ravel(), 
                                       subset_out.detach().cpu().numpy().ravel())
                    num_elements = (subset_mask > 0.5).sum().item()
                
            return subset_r2, num_elements, params
        
        # Adaptive search starting from user's beta
        current_beta = self.beta
        best_beta = current_beta
        best_r2 = 0.0
        total_elements = self.model.edge_index.size(1) if target == 'edge' else self.model.num_nodes
        best_elements = total_elements
        best_params = None
        
        # Step 1: Evaluate starting point
        if verbose:
            print(f"\nStep 1: Evaluating starting beta = {current_beta:.4f}")
        
        try:
            initial_r2, initial_elements, initial_params = evaluate_beta(current_beta)
            
            element_type = "Edges" if target == 'edge' else "Nodes"
            if verbose:
                print(f"  → R² = {initial_r2:.4f}, {element_type} = {initial_elements}")
            
            # Set initial best
            best_beta = current_beta
            best_r2 = initial_r2
            best_elements = initial_elements
            best_params = initial_params
            
            # Step 2: Determine search direction
            if initial_r2 >= min_r2:
                # Performance is good, try increasing beta (more sparsity)
                search_direction = "up"
                if verbose:
                    print(f"  ✓ Good performance! Searching upward for more sparsity...")
            else:
                # Performance is poor, try decreasing beta (less sparsity)
                search_direction = "down"
                if verbose:
                    print(f"  ✗ Poor performance! Searching downward for better performance...")
            
            # Step 3: Search in determined direction
            for trial in range(max_trials):
                if search_direction == "up":
                    test_beta = current_beta * beta_step
                else:
                    test_beta = current_beta / beta_step
                
                if verbose:
                    print(f"\nTrial {trial + 1}: Testing beta = {test_beta:.4f} (direction: {search_direction})")
                
                try:
                    test_r2, test_elements, test_params = evaluate_beta(test_beta)
                    
                    if verbose:
                        print(f"  → R² = {test_r2:.4f}, {element_type} = {test_elements}")
                    
                    if search_direction == "up":
                        if test_r2 >= min_r2:
                            # Still good, keep this as best and continue
                            best_beta = test_beta
                            best_r2 = test_r2
                            best_elements = test_elements
                            best_params = test_params
                            current_beta = test_beta
                            
                            if verbose:
                                print(f"  ✓ Still good! New best: β={best_beta:.4f}")
                        else:
                            # Performance dropped, we've found the boundary
                            if verbose:
                                print(f"  ✗ Performance dropped, boundary found!")
                            break
                    else:  # search_direction == "down"
                        if test_r2 >= min_r2:
                            # Found good performance, this is our answer
                            best_beta = test_beta
                            best_r2 = test_r2
                            best_elements = test_elements
                            best_params = test_params
                            
                            if verbose:
                                print(f"  ✓ Performance recovered! Optimal: β={best_beta:.4f}")
                            break
                        else:
                            # Still poor, keep going down
                            current_beta = test_beta
                            
                            if verbose:
                                print(f"  ✗ Still poor, continuing downward...")
                    
                    # Safety check - don't let beta get too extreme
                    if test_beta > 100 or test_beta < 0.001:
                        if verbose:
                            print(f"  ⚠ Beta limit reached ({test_beta:.4f}), stopping search")
                        break
                        
                except Exception as e:
                    if verbose:
                        print(f"  Error with beta={test_beta:.4f}: {e}")
                    break
                    
        except Exception as e:
            if verbose:
                print(f"Error with initial beta={current_beta:.4f}: {e}")
            # Fall back to original beta if there's an error
            best_beta = self.beta
        
        # Restore all original settings
        for param, value in original_settings.items():
            setattr(self, param, value)
        
        # Set the optimal beta
        self.beta = best_beta
        
        # Create final dataframe with optimal results
        final_df = None
        if best_params is not None:
            scores, _ = torch.nn.functional.softmax(best_params.data, dim=0).detach().cpu().numpy()
            if target == 'edge':
                src, dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
                final_df = pd.DataFrame({'source': src, 'target': dst, 'score': scores})
            else:  # target == 'node'
                node_names = np.array(self.model.homo_names)
                final_df = pd.DataFrame({'node': node_names, 'score': scores})
        
        # Final evaluation with optimal beta
        if verbose:
            print("\n" + "="*60)
            print("TUNING COMPLETE")
            print("="*60)
            print(f"Starting beta: {original_settings['beta']:.4f}")
            print(f"Optimal beta: {best_beta:.4f}")
            print(f"Change: {best_beta/original_settings['beta']:.2f}x")
            print(f"Final R²: {best_r2:.4f}")
            element_type_lower = "edges" if target == 'edge' else "nodes"
            print(f"Selected {element_type_lower}: {best_elements} / {total_elements} ({100*best_elements/total_elements:.1f}%)")
            print("="*60)
        
        results = {
            'starting_beta': original_settings['beta'],
            'optimal_beta': best_beta,
            'beta_change_factor': best_beta / original_settings['beta'],
            'achieved_r2': best_r2,
            'num_elements': best_elements,
            'total_elements': total_elements,
            'sparsity_ratio': best_elements / total_elements,
            'result_df': final_df,
            'target': target
        }
        
        return results