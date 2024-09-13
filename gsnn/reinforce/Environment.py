


import torch 
import torch_geometric as pyg
from sklearn.metrics import r2_score 
import numpy as np
import copy 
from torch.utils.data import DataLoader
import gc 

from gsnn.models.GSNN import GSNN
from gsnn.reinforce.EarlyStopper import EarlyStopper
from gsnn.reinforce.Actor import Actor
from gsnn.models import utils

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

    def __init__(self, train_dataset, val_dataset, test_dataset, model_kwargs, training_kwargs, device, metric='spearman'): 

        self.train_dataset = train_dataset 
        self.val_dataset = val_dataset 
        self.test_dataset = test_dataset 
        self.model_kwargs = model_kwargs 
        self.training_kwargs = training_kwargs
        self.device = device
        self.metric = metric

        self.edge_index_dict = model_kwargs['edge_index_dict']
        self.model_kwargs.pop('edge_index_dict', None)

        self.N_func = len(model_kwargs['node_names_dict']['function'])
        self.E_func = self.edge_index_dict['function', 'to','function'].size(1) 

    def augment_edge_index(self, action, action_type='node'): 

        edge_index_dict = copy.deepcopy(self.edge_index_dict)
        if action.mean() == 1: return edge_index_dict

        if action_type == 'node': 
            # remove all nodes for which action == 0 
            # assumes action is of length N ('function') nodes
            subset       = action.nonzero(as_tuple=True)[0]

            edge_index_dict['function', 'to','function'] = pyg.utils.subgraph(subset              = subset, 
                                                                              edge_index          = edge_index_dict['function', 'to','function'], 
                                                                              relabel_nodes       = False, 
                                                                              num_nodes           = self.N_func, 
                                                                              return_edge_mask    = False)[0]
            
            # remove any input-> function nodes that aren't in the function subset 
            row,col = edge_index_dict['input', 'to','function']
            edge_index = edge_index_dict['input', 'to','function'][:, torch.isin(col, subset)]
            edge_index_dict['input', 'to','function'] = edge_index 

            # remove any function->output nodes that aren't in the function subset 
            row,col = edge_index_dict['function', 'to','output']
            edge_index = edge_index_dict['function', 'to','output'][:, torch.isin(row, subset)]
            edge_index_dict['function', 'to','output'] = edge_index 

        elif action_type == 'edge': 
            # remove all edges for which action == 0 
            # assumes action is of length E ('function', 'to', 'function') edges
            edge_index_dict['function', 'to','function'] = edge_index_dict['function', 'to','function'][:, action.nonzero(as_tuple=True)[0]]
        else: 
            raise Exception('unrecognized action type')

        return edge_index_dict
    
    def train(self, edge_index_dict, ret_test=False, use_ema=False): 

        # hacky fix for the cuda mem accumulation problem
        # maybe just caused by selecting larger graphs? might need to add penalty, or adjust batch size based on num edges 
        gc.collect() 
        torch.cuda.empty_cache()
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.training_kwargs['batch'], num_workers=self.training_kwargs['workers'], shuffle=True, persistent_workers=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.training_kwargs['batch'], num_workers=self.training_kwargs['workers'], shuffle=False, persistent_workers=True)
        
        if ret_test: test_loader = DataLoader(self.test_dataset, batch_size=self.training_kwargs['batch'], num_workers=self.training_kwargs['workers'], shuffle=False, persistent_workers=True)
        
        # init model 
        model = GSNN(edge_index_dict=edge_index_dict, **self.model_kwargs).to(self.device)

        optim = torch.optim.Adam(model.parameters(), lr=self.training_kwargs['lr'])
        crit = torch.nn.MSELoss() 
        early_stopper = EarlyStopper(patience=self.training_kwargs['patience'], min_delta=self.training_kwargs['min_delta'])

        best_val = -np.inf
        vals = []
        for epoch in range(self.training_kwargs['max_epochs']): 
            model.train()
            for i,(x, y, id) in enumerate(train_loader): 
                optim.zero_grad()
                if x.size(0) == 1: continue # BUG workaround: if batch only has 1 obs it fails
                x = x.to(self.device); y = y.to(self.device)
                yhat = model(x)
                loss = crit(yhat, y)
                loss.backward()
                optim.step()

                if torch.isnan(loss): 
                    del model; del optim; del crit; del early_stopper; del train_loader; del val_loader
                    return -1 
                
                print(f'[batch: {i+1}/{len(train_loader)}]', end='\r')

            # validation perf 
            y,yhat,_ = utils.predict_gsnn(val_loader, model, device=self.device, verbose=False)

            if self.metric == 'mse': 
                val_score = -np.mean((y-yhat)**2)
            elif self.metric == 'pearson': 
                val_score = utils.corr_score(y, yhat, method='pearson')
            elif self.metric == 'spearman': 
                val_score = utils.corr_score(y, yhat, method='spearman')
            elif self.metric == 'r2': 
                # prevent large negative values
                val_score = np.clip(r2_score(y, yhat, multioutput='variance_weighted'), -1, 1) 
            else:
                raise Exception('unrecognized `metric` type')
            
            print(f'\t\trun progress: {epoch}/{self.training_kwargs["max_epochs"]} | train loss: {loss.item():.1f} || val perf: {val_score:.3f}', end='\r')

            # this may help to smooth the variance in val rewards
            if use_ema: 
                vals.append(val_score)
                val_score = ema(vals)

            # use the best val as the reward value 
            # could use running mean too... 
            if val_score > best_val: 
                best_val = val_score 

                if ret_test: 
                    y,yhat,_ = utils.predict_gsnn(test_loader, model, device=self.device, verbose=False)
                    test_dict = {'mse':-np.mean((y-yhat)**2),
                                'pearson':utils.corr_score(y, yhat, method='pearson'),
                                'spearman':utils.corr_score(y, yhat, method='spearman'), 
                                'r2':r2_score(y, yhat, multioutput='variance_weighted')}

            # stop early 
            if early_stopper.early_stop(-val_score): break
        
        # trying to find cuda mem accumulation 
        del model; del optim; del crit; del early_stopper; del train_loader; del val_loader

        if ret_test: 
            return best_val, test_dict
        else: 
            return best_val
    
    def validate(self, edge_index_dict): 
        '''
        if critical nodes, such as func->output edges are not included, include them; should speed up convergence.
        '''
        if edge_index_dict['function', 'to', 'output'].size(1) == 0:
            return False
        else: 
            return True

    def run(self, action, action_type='node'): 

        # augment edge_index_dict appropriately 
        edge_index_dict = self.augment_edge_index(action, action_type)

        if not self.validate(edge_index_dict): return 0

        # train model 
        try: 
            reward = self.train(edge_index_dict)
        except: 
            # failed trials will result in 0 reward; e.g., nan divergences
            reward = -1
            raise

        return reward