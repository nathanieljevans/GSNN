import numpy as np 
import torch 
import pandas as pd 
from sklearn.metrics import r2_score
import copy 


class GSNNExplainer: 

    def __init__(self, model, data, ignore_cuda=False, gumbel_softmax=True, hard=False, tau0=3, min_tau=0.5, 
                            prior=1, iters=250, lr=1e-2, weight_decay=1e-5, free_edges=0,
                                    beta=1, verbose=True, optimizer=torch.optim.Adam): 
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

            loss = mse + self.beta*torch.maximum(torch.tensor([0.], device=x.device), edge_weight.sum() - self.free_edges)

            loss.backward() 
            optim.step() 

            with torch.no_grad():
                r2 = r2_score(target_preds.detach().cpu().numpy().ravel(), out.detach().cpu().numpy().ravel())

            if self.verbose: print(f'iter: {iter} | mse: {mse.item():.8f} | r2: {r2:.3f} | active edges: {(edge_weight > 0.5).sum().item()}', end='\r')

        edge_scores, _ = torch.nn.functional.softmax(edge_params.data, dim=0).detach().cpu().numpy()
       
        src,dst = np.array(self.model.homo_names)[self.model.edge_index.detach().cpu().numpy()]
        edgedf = pd.DataFrame({'source':src, 'target':dst, 'score':edge_scores})

        return edgedf



'''


def get_edge_activations(x, model): 
    model.to(x.device)
    x = node2edge(x, model.edge_index)  # convert x to edge-indexed
    x0 = x
    act = []
    for l in range(model.layers): 
        
        x = model.ResBlocks[0 if model.share_layers else l](x)
        if not model.residual: x += x0
        act.append(x)

    return act


def masked_forward(x, xb, model, mask1, mask2, activations1):
    x = node2edge(x, model.edge_index)  # convert x to edge-indexed
    xb = node2edge(xb, model.edge_index)
    x0 = x

    # mask input features 
    x = mask1*xb + mask2*x

    for l in range(model.layers): 
        x = model.ResBlocks[0 if model.share_layers else l](x)
        if not model.residual: x += x0
        
        x = mask1*activations1[l] + mask2*x

    if model.residual: x /= model.layers

    return edge2node(x, model.edge_index, model.output_node_mask)  # convert x from edge-indexed to node-indexed

'''