import numpy as np 
import torch 
import pandas as pd 
from sklearn.metrics import r2_score
import copy 
 

class GSNNExplainer:
    r"""Edge-mask optimiser that produces *sparse* explanations.

    The explainer learns a binary mask *m∈\{0,1\}^E* that maximises fidelity
    between the model's prediction on the **masked** graph and the prediction
    on the *full* graph while simultaneously penalising mask size::

        L = MSE\bigl(f(x; m), f(x; 1)\bigr)
            + β \max(0, \|m\|₁ − free_edges)
            − λ H(m)            (optional entropy term)

    Here *m* is obtained via a differentiable Gumbel-Softmax relaxation so the
    optimisation can be performed with vanilla back-prop.  After convergence
    the edge importance score is the softmax probability *p_e = P(m_e=1)*.

    Interpretation
    --------------
    * ``score_e → 1``   edge e is essential for reproducing the original prediction.
    * ``score_e → 0``   edge e can be removed with little impact.

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
        Weight decay applied to the edge logits.
    free_edges : int, optional (default=0)
        Number of edges allowed before the sparsity penalty activates.
    beta : float, optional (default=1.0)
        Coefficient of the sparsity term.
    entropy : float, optional (default=0.0)
        Strength of the entropy bonus (encourages exploration).

    Example
    -------
    >>> explainer = GSNNExplainer(model, data, iters=400, beta=5)
    >>> edge_df = explainer.explain(x, targets=[0])
    >>> edge_df.sort_values('score', ascending=False).head()
    """

    def __init__(self, model, data, ignore_cuda=False, gumbel_softmax=True, hard=False, tau0=3, min_tau=0.5, 
                            prior=1, iters=250, lr=1e-2, weight_decay=1e-5, free_edges=0,
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

    def explain(self, x, targets=None): 
        '''
        initializes and runs gradient descent to select a minimal subset of edges that produce comparable predictions 
        to the full graph. 
        
        Args: 
            x               torch.tensor            inputs to explain; in shape (B, I)

        Returns 
            dataframe                               edgelist with importance scores 
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
            optim.step() 

            with torch.no_grad():
                r2 = r2_score(target_preds.detach().cpu().numpy().ravel(), out.detach().cpu().numpy().ravel())

            if self.verbose: print(f'iter: {iter} | loss: {loss.item():.4f} | mse: {mse.item():.4f} | r2: {r2:.3f} | active edges: {(edge_weight > 0.5).sum().item()} / {self.model.edge_index.size(1)} | entropy: {ent.item():.4f}', end='\r')

        edge_scores, _ = torch.nn.functional.softmax(edge_params.data, dim=0).detach().cpu().numpy()
       
        src,dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        edgedf = pd.DataFrame({'source':src, 'target':dst, 'score':edge_scores})

        return edgedf