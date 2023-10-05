
import torch 
import pandas as pd 
from sklearn.metrics import r2_score
import numpy as np 
import torch.nn.functional as F

from src.models import utils 

def get_edge_activations(x, model): 
    '''get activations of a given observation'''
    model.to(x.device)
    x = utils.node2edge(x, model.edge_index)  # convert x to edge-indexed
    x0 = x
    act = []
    for l in range(model.layers): 
        
        x = model.ResBlocks[0 if model.share_layers else l](x)
        if not model.residual: x += x0
        act.append(x)

    return act


def masked_forward(x, xb, model, mask1, mask2, activations1):
    '''
    Assumes x is `node` indexed 
    ''' 
    x = utils.node2edge(x, model.edge_index)  # convert x to edge-indexed
    xb = utils.node2edge(xb, model.edge_index)
    x0 = x

    # mask input features 
    x = mask1*xb + mask2*x

    for l in range(model.layers): 
        x = model.ResBlocks[0 if model.share_layers else l](x)
        if not model.residual: x += x0
        
        x = mask1*activations1[l] + mask2*x

    if model.residual: x /= model.layers

    return utils.edge2node(x, model.edge_index, model.output_node_mask)  # convert x from edge-indexed to node-indexed


class GSNNExplainer: 

    def __init__(self, model, data, ignore_cuda=False, gumbel_softmax=True, hard=False, tau0=3, min_tau=0.5, 
                            prior=1, targets=None, iters=250, lr=1e-2, weight_decay=1e-5, 
                                    beta=1, verbose=True, optimizer=torch.optim.Adam, free_edges=0): 
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
        self.targets = targets 
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
        
        model = model.eval()
        model = model.to(self.device)
        for p in model.parameters(): 
            p.requires_grad = False 
        self.model = model

    def explain(self, x, baseline=None, return_predictions=False): 
        '''
        initializes and runs gradient descent to select a minimal subset of edges that produce comparable predictions to the full graph. 
        
        Args: 
            x               torch.tensor            inputs to explain; in shape (B, I, 1)

        Returns 
            dataframe                               edgelist with importance scores 
        '''

        if baseline is not None: 
            
            x1 = baseline 
            x2 = x
            
            if (x1.size(0) > 1) | (x2.size(0) > 1): 
                raise NotImplementedError('GSNNExplainer currently only supports single observation explanations')

            x1 = x1.to(self.device).detach()
            x2 = x2.to(self.device).detach()

            yhat1 = self.model(x1).detach()
            yhat2 = self.model(x2).detach()

            _r2 = r2_score(yhat1.cpu().numpy().ravel(), yhat2.cpu().numpy().ravel())
            if _r2 > 0: print('WARNING! r2(yhat_x, yhat_baseline) > 0 - this indicates the model predicts similar outcomes for x and baseline. This may indicate poor observation prediction.')

            target = (yhat2 - yhat1).detach()

            activations1 = [x.detach() for x in get_edge_activations(x1, self.model)]

        else: 
            # if no baseline is provided then "zero" value activations will be used as the baseline. 
            # this is essentially comparing against the "average" activation and is unlikely to represent any one real cell line, drug or observation. 
            x2 = x
            x2 = x2.to(self.device).detach()
            x1 = torch.zeros_like(x2)
            yhat2 = self.model(x2).detach()
            yhat1 = 0. 
            target = yhat2.detach() 
            activations1 = [0.]*self.model.layers 

        
        weights = torch.stack((self.prior*torch.ones(self.model.edge_index.size(1), dtype=torch.float32, device=self.device, requires_grad=True), 
                                -self.prior*torch.ones(self.model.edge_index.size(1), dtype=torch.float32, device=self.device, requires_grad=True)), dim=0)
        edge_params = torch.nn.Parameter(weights)
        
        # optimize parameter mask with objective 
        crit = torch.nn.MSELoss()   # MutualInformationLoss() #PearsonCorrelationLoss() #
        optim = self.optimizer([edge_params], lr=self.lr, weight_decay=self.weight_decay)

        # calculate tau decay rate
        tau_decay_rate = (self.min_tau / self.tau0) ** (1 / self.iters)

        if self.targets is None: 
            targets = torch.arange(len(x.shape[1]))
        else: 
            targets = self.targets

        for iter in range(self.iters):    
            optim.zero_grad()

            tau = max(self.tau0 * tau_decay_rate**iter, self.min_tau)

            if self.gumbel_softmax:
                mask1, mask2 = torch.nn.functional.gumbel_softmax(edge_params, dim=0, hard=self.hard, tau=tau)
            else: 
                mask1, mask2 = torch.Softmax(edge_params, dim=0)

            mask1 = mask1.view(1, -1, 1)
            mask2 = mask2.view(1, -1, 1)

            out = masked_forward(x=x2, xb=x1, model=self.model, mask1=mask1, mask2=mask2, activations1=activations1)
            
            # if mask1 is all ones, then out should be equal to yhat1 
            # if mask2 is all ones, then out should be equal to yhat2
            # so which edges (mask2) are important to predict the difference between yhat2 and yhat1

            out = out - yhat1

            mse = crit(out[:, targets], target[:, targets])

            loss = mse + self.beta*torch.maximum(torch.tensor([0.], device=x1.device), mask2.sum() - self.free_edges)

            loss.backward() 
            optim.step() 

            with torch.no_grad():
                r2 = r2_score(target[:, targets].detach().cpu().numpy().ravel(), out[:, targets].detach().cpu().numpy().ravel())

            print(f'iter: {iter} | mse: {mse.item():.8f} | r2: {r2:.3f} | active edges: {mask2.sum().item()}', end='\r')

        # compute explained variance
        with torch.no_grad():
            mask1, mask2 = torch.nn.functional.softmax(edge_params.data, dim=0)
            mask1 = 1.*(mask1.view(1, -1, 1) > 0.5)
            mask2 = 1.*(mask2.view(1, -1, 1) > 0.5)
            out = masked_forward(x=x2, xb=x1, model=self.model, mask1=mask1, mask2=mask2, activations1=activations1)
            out = out - yhat1
            full_graph_preds = target[:, targets].detach().cpu().numpy().ravel() 
            subgraph_preds = out[:, targets].detach().cpu().numpy().ravel()
            r2 = r2_score(full_graph_preds, subgraph_preds)
            print()
            print('Final r2 (MLE):', r2)

        # get 
        _, edge_scores = torch.nn.functional.softmax(edge_params.data, dim=0).detach().cpu().numpy()
       
        src,dst = self.data.node_names[self.model.edge_index.detach().cpu().numpy()]
        edgedf = pd.DataFrame({'source':src, 'target':dst, 'score':edge_scores})

        if return_predictions: 
            return edgedf, r2, full_graph_preds, subgraph_preds
        else: 
            return edgedf, r2