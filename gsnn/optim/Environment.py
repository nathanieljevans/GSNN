


import torch 
import torch_geometric as pyg
from sklearn.metrics import r2_score 
import numpy as np
import copy 
from torch.utils.data import DataLoader
import gc 

from gsnn.models.GSNN import GSNN
from gsnn.optim.EarlyStopper import EarlyStopper
from gsnn.models import utils

#import torch._dynamo
#torch._dynamo.config.suppress_errors = True
#torch._dynamo.config.cache_size_limit = 100000


def ema(values, alpha=2/(3+1)):
    """
    Calculate the Exponential Moving Average (EMA)

    Args:
    values (list or array-like): A list of numerical values for which the EMA is to be calculated.
    alpha (float): The smoothing factor, a value between 0 and 1.

    Returns:
    float: The EMA value for the current epoch.
    """
    if not values:
        return None
    
    # Initialize EMA with the first value
    val = values[0]

    # Calculate EMA up to the current epoch (last value in the list)
    for t in range(1, len(values)):
        val = alpha * values[t] + (1 - alpha) * val

    return val

class Environment(): 

    def __init__(self, action_edge_dict, train_dataset, val_dataset, model_kwargs, 
                 training_kwargs, metric='spearman', reward_type='auc', verbose=True,
                 raise_error_on_fail=False): 
        
        self.action_edge_dict = action_edge_dict
        self.train_dataset = train_dataset 
        self.val_dataset = val_dataset 
        self.model_kwargs = model_kwargs 
        self.training_kwargs = training_kwargs
        self.metric = metric
        self.reward_type = reward_type
        self.verbose = verbose
        self.raise_error_on_fail = raise_error_on_fail

        self.edge_index_dict = model_kwargs['edge_index_dict']
        self.model_kwargs.pop('edge_index_dict', None)

        self.N_func = len(model_kwargs['node_names_dict']['function'])
        self.E_func = self.edge_index_dict['function', 'to','function'].size(1) 

    def augment_edge_index(self, action): 

        # add a default fixed edge to the action space
        # all action indexes with value -1 will be set to True in the edge mask 
        action = torch.cat((action.squeeze(), torch.tensor([1], dtype=torch.long, device=action.device)), dim=-1)

        action = action == 1         
        #The action_edge_dict  {key: int (E,)} where the values correspond to the index in `action`
        # therefore a single action can be used to mask multiple edges 
        edge_mask_dict = {key: action[values] for key, values in self.action_edge_dict.items()}

        edge_index_dict_ = {} 
        for key, edge_index in self.edge_index_dict.items():
            if key in edge_mask_dict: 
                edge_index_dict_[key] = edge_index[:, edge_mask_dict[key]]
            else: 
                edge_index_dict_[key] = edge_index

        return edge_index_dict_
    
    def train(self, edge_index_dict): 

        # cuda mem accumulation problem
        gc.collect() 
        torch.cuda.empty_cache()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.training_kwargs['batch'], num_workers=self.training_kwargs['workers'], shuffle=True, persistent_workers=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.training_kwargs['batch'], num_workers=self.training_kwargs['workers'], shuffle=False, persistent_workers=True)
        
        model = GSNN(edge_index_dict=edge_index_dict, **self.model_kwargs).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=self.training_kwargs['lr'])
        crit = torch.nn.MSELoss() 

        best_mean_val = -np.inf
        best_val_score = None
        scores = []
        for epoch in range(self.training_kwargs['max_epochs']): 
            model.train()
            for i, (x,y,*_) in enumerate(train_loader): 
                optim.zero_grad()
                if x.size(0) == 1: continue # BUG workaround: if batch only has 1 obs it fails
                x = x.to(device); y = y.to(device)
                yhat = model(x)
                loss = crit(yhat, y)
                loss.backward()
                optim.step()

                if torch.isnan(loss): 
                    del model; del optim; del crit; del train_loader; del val_loader
                    return -1 
                
                if self.verbose: print(f'[batch: {i+1}/{len(train_loader)}]', end='\r')

            # validation perf 
            y,yhat,_ = utils.predict_gsnn(val_loader, model, device=device, verbose=False)

            # since we have a multioutput prediction problem, we need to return multioutput performances 

            if self.metric == 'mse': 
                val_score = -np.mean((y-yhat)**2, dim=0)
            elif self.metric == 'pearson': 
                val_score = utils.corr_score(y, yhat, method='pearson', multioutput='raw_values')
            elif self.metric == 'spearman': 
                val_score = utils.corr_score(y, yhat, method='spearman', multioutput='raw_values')
            elif self.metric == 'r2': 
                val_score = np.clip(utils.corr_score(y, yhat, method='r2', multioutput='raw_values'),-1,1)
            else:
                raise Exception('unrecognized `metric` type')
            
            scores.append(val_score)
            
            if self.verbose: print(f'\t\trun progress: {epoch}/{self.training_kwargs["max_epochs"]} | train loss: {loss.item():.1f} || mean val perf: {val_score.mean():.3f}', end='\r')

            # use the best val as the reward value 
            # could use running mean too... 
            if (val_score.mean() > best_mean_val): 
                best_val_score = val_score 

        # trying to find cuda mem accumulation 
        del model; del optim; del crit; del train_loader; del val_loader

        if self.reward_type == 'best': 
            reward = best_val_score 
        elif self.reward_type == 'last': 
            reward = val_score
        elif self.reward_type == 'auc': 
            reward = np.sum(np.stack(scores, axis=0), axis=0) # should reward longer training runs as well as high perf
        else: 
            raise NotImplementedError('unrecognized reward type')

        return reward
    
    def validate(self, edge_index_dict): 
        '''
        if critical nodes, such as func->output edges are not included, include them; should speed up convergence.
        '''
        if edge_index_dict['function', 'to', 'output'].size(1) == 0:
            return False
        else: 
            return True

    def run(self, action): 

        # augment edge_index_dict appropriately 
        edge_index_dict = self.augment_edge_index(action=action.cpu())

        if not self.validate(edge_index_dict): return 0

        # train model 
        try: 
            reward = self.train(edge_index_dict)
        except: 
            # failed trials will result in low reward; e.g., nan divergences
            reward = -1
            if self.raise_error_on_fail: raise

        return reward