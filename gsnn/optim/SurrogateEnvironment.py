'''
trains an Heterogenous GSNN ensemble to predict reward 
'''

import torch
import numpy as np 
from gsnn.optim.EarlyStopper import EarlyStopper
from gsnn.models.GSNN import GSNN
from sklearn.metrics import r2_score
import gc
from torch import nn 
from torch.nn import functional as F

from hnet.models.HyperNet import HyperNet
from hnet.train.hnet import init_hnet
from scipy.special import softmax

import warnings
warnings.filterwarnings("ignore")

def ema_probs(N, alpha):
    if N <= 0 or not (0 < alpha < 1):
        raise ValueError("N must be positive and alpha must be between 0 and 1.")
    
    weights = np.zeros(N)
    for i in range(N):
        weights[i] = (1 - alpha) ** i
    
    weights = weights[::-1]  # Reverse the weights to be in ascending order
    weights /= weights.sum()  # Normalize the weights to sum to 1

    return weights

def train(actions, rewards, num_inputs, model_kwargs, model=None,
          batch_size=50, lr=1e-2, wd=0, epochs=100, verbose=False,
          stochastic_channels=10, width=10, samples=100, rewards_mean=None, 
          rewards_std=None, ema_sampling=False, patience=5, min_delta=0.01):
    
    gc.collect() 
    torch.cuda.empty_cache()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

    if (rewards_mean is None) or (rewards_std is None): 
        rewards_mean = rewards.mean()
        rewards_std = rewards.std()

    rewards = (rewards - rewards_mean)/(rewards_std + 1e-8)

    if model is None: 
        gsnn = GSNN(**model_kwargs)
        model = HyperNet(gsnn, stochastic_channels=stochastic_channels, width=width).to(device)
    else: 
        gsnn = model.model
        model = model.to(device)
    
    # BUG: this might be creating large memory usage... ?
    edge_mask = torch.ones((actions.size(0), gsnn.edge_index.size(1),), device=device)
    edge_mask[:, gsnn.function_edge_mask] = actions

    # this should be trainble as well
    x_input = torch.ones((1,num_inputs), dtype=torch.float32, device=device)

    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    crit = torch.nn.GaussianNLLLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


    for epoch in range(epochs):
        model.train()
        losses = []

        if ema_sampling: 
            n_splits = len(actions)//batch_size + 1
            probs = ema_probs(len(actions), alpha=min(2/(len(actions)//4 + 1), 0.5))
            batch_splits = [torch.tensor(np.random.choice(np.arange(len(actions)), p=probs, size=(batch_size)), dtype=torch.long, device=device) for _ in range(n_splits)]
        else: 
            batch_splits = torch.split(torch.randperm(len(actions), device=device), batch_size)


        for idxs in batch_splits:
            optim.zero_grad()
            B_ = len(idxs)
            batch_mask = edge_mask[idxs]
            batch_rewards = rewards[idxs]
            xx = x_input.expand(B_, num_inputs)

            rhat = model(x=xx, samples=samples, kwargs={'edge_mask':batch_mask}) 
            rhat_mu = rhat.mean(0)
            rhat_var = rhat.var(0)
            loss = crit(rhat_mu.view(-1), batch_rewards.view(-1), rhat_var.view(-1))
            losses.append(loss.item())

            loss.backward()
            optim.step()

        if verbose: print(f'optimizing surrogate model... epoch: {epoch}/{epochs} --> loss:{np.mean(losses):.3f}', end='\r')

        if early_stopper.early_stop(loss.item()):
            break
    
    model.eval()
    return model.cpu(), rewards_mean, rewards_std


class SurrogateEnvironment: 
    ''''''

    def __init__(self, data, alpha=0.01, model_kwargs={'channels':5,
                                                        'norm':'layer',
                                                        'layers':10,
                                                        'dropout':0.,
                                                        'dropout_type':'channel',
                                                        'add_function_self_edges':False,
                                                        'bias':False,
                                                        'share_layers':False}, 
 
                                        samples=100, 
                                        stochastic_channels=10,
                                        hnet_width=10): 

        self.data = data
        self.model_kwargs = model_kwargs
        self.model = None
        self.rewards_mean = None
        self.rewards_std = None
        self.alpha = alpha
        self.samples= samples 
        self.stochastic_channels=stochastic_channels
        self.hnet_width=hnet_width

    def optimize(self, recorder, train_kwargs, verbose=False, ema_sampling=False, patience=1000, min_delta=0): 

        gc.collect() 
        torch.cuda.empty_cache()

        actions = torch.stack(recorder.actions, dim=0).to(torch.float32)
        rewards = torch.tensor(recorder.rewards).to(torch.float32)

        self.model, self.rewards_mean, self.rewards_std = train(actions = actions, 
                                                                rewards = rewards, 
                                                                num_inputs = len(self.data.node_names_dict['input']), 
                                                                model_kwargs={'edge_index_dict':self.data.edge_index_dict,
                                                                                'node_names_dict':self.data.node_names_dict,
                                                                                **self.model_kwargs},
                                        **train_kwargs,
                                        stochastic_channels=self.stochastic_channels,
                                        width=self.hnet_width, 
                                        samples=self.samples,
                                        verbose=verbose,
                                        model = self.model, 
                                        rewards_mean=self.rewards_mean,
                                        rewards_std=self.rewards_std,
                                        ema_sampling=ema_sampling,
                                        patience=patience, 
                                        min_delta=min_delta)    

        print()

    def predict(self, actions, samples=None, batch=10, need_grad=False): 

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.model is None: raise Exception('`model` attribute is None - have you run `SurrogateEnvironment.optimize()` yet?')
        
        B = batch

        model = self.model.to(device)
        if not need_grad: model.eval()

        if samples is None: 
            samples = self.samples
        
        edge_mask = torch.ones((actions.size(0), model.model.edge_index.size(1),), device=device)
        edge_mask[:, model.model.function_edge_mask] = actions.to(device)
        
        if not need_grad: 
            with torch.no_grad(): 
                out = []
                for idxs in torch.split(torch.arange(len(actions)), B):
                    xx = torch.ones((len(idxs), len(self.data.node_names_dict['input'])), dtype=torch.float32, device=device, requires_grad=False)
                    out.append(model(x = xx, samples=samples, kwargs={'edge_mask': edge_mask[idxs]}).cpu()) 

                out = torch.cat(out, dim=1)
        else: 
            out = []
            for idxs in torch.split(torch.arange(len(actions)), B):
                xx = torch.ones((len(idxs), len(self.data.node_names_dict['input'])), dtype=torch.float32, device=device, requires_grad=False)
                out.append(model(x = xx, samples=samples, kwargs={'edge_mask': edge_mask[idxs]}).cpu()) 

            out = torch.cat(out, dim=1)

        return out

                    

