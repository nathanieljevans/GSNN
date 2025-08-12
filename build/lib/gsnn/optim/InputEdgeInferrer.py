
import copy 
import torch 
import numpy as np 
from scipy.stats import ttest_1samp
import pandas as pd 
from statsmodels.stats.multitest import multipletests

class InputEdgeInferrer: 
    def __init__(self, model, data, x_train, y_train):
        
        # self.model = copy.deepcopy(model) # BUG: RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.  If you were attempting to deepcopy a module, this may be because of a torch.nn.utils.weight_norm usage, see https://github.com/pytorch/pytorch/pull/103001
        self.model = model
        self.data = data
        self.x_train = x_train
        self.y_train = y_train

    def freeze_(self, model): 
        for param in model.parameters(): 
            param.requires_grad = False

    def infer(self, input_idx, edge_targets=None, iters=1000, lr=1e-2, wd=1e-3, batch_size=512, bootstrap_p=0.75, n_bootstrap=10, dropout=0, device='cpu', verbose=True):
        
        assert len(input_idx) == 1, 'only one input node is supported for now'
        
        if edge_targets is None: 
            edge_targets = torch.arange(self.model.edge_index.size(1), dtype=torch.bool)
            edge_targets[self.model.input_edge_mask] = True

        ws = [] 
        for i in range(n_bootstrap):

            ws.append( self.infer_bs(input_idx=input_idx, 
                                     edge_targets=edge_targets,
                                     iters=iters, 
                                     lr=lr, 
                                     wd=wd, 
                                     batch_size=batch_size, 
                                     bootstrap_p=bootstrap_p, 
                                     dropout=dropout, 
                                     device=device, 
                                     verbose=verbose,
                                     bootstrap_iter=i) )
        w = torch.stack(ws, axis=-1)

        if verbose: print() 
        res = {'input_idx':[], 'func_idx':[], 'src':[], 'dst': [], 'weight': [], 'pval':[], 'prop_gt_zero':[], 'prop_lt_zero':[]}
        i_idxs = edge_targets[self.model.input_edge_mask.cpu()].nonzero(as_tuple=True)[0]
        for i_idx in i_idxs: 
            for x_idx in input_idx:
                if verbose: print(f'computing significances: {i_idx}/{len(i_idxs)} | {x_idx}/{len(input_idx)}', end='\r')

                src_node = int(x_idx)                                                           # x_idx is the src node 
                dst_edge = self.data.edge_index_dict['input', 'to', 'function'][:, int(i_idx)]  # i_idx is the input edge index that x is targeting 
                dst_node = dst_edge[1].item()                                                   # grab the dst of the edge to get the dst node

                weight = w[0, i_idx, :].mean().item()

                # perform two-sided t-test to determine if the weight is significantly different from 0
                pval = ttest_1samp(w[0, i_idx, :].detach().cpu().numpy(), 0).pvalue
                prop_gt_zero = (w[0, i_idx, :].detach().cpu().numpy() > 0).mean()
                prop_lt_zero = (w[0, i_idx, :].detach().cpu().numpy() < 0).mean()
                
                res['input_idx'].append(src_node)
                res['func_idx'].append(dst_node)
                res['src'].append(self.data.node_names_dict['input'][src_node])
                res['dst'].append(self.data.node_names_dict['function'][dst_node])
                res['weight'].append(weight)
                res['pval'].append(pval)
                res['prop_gt_zero'].append(prop_gt_zero)
                res['prop_lt_zero'].append(prop_lt_zero)

        res = pd.DataFrame(res)
        res['pval_adj'] = multipletests(res['pval'], method='fdr_bh')[1]

        # add flag for if edge is in graph 
        src,dst = self.data.edge_index_dict['input', 'to', 'function'].detach().cpu().numpy()
        input_edges = set([(i,j) for i,j in zip(src.tolist(), dst.tolist())])
        in_graph = []
        for i, row in res.iterrows(): 
            in_graph.append((row['input_idx'], row['func_idx']) in input_edges)
        res = res.assign(in_graph=in_graph)

        #TODO: 
        # evaluate performance of w on validation (leftover bootstrap) and compare to w=0 
        # report the MSE improvement 
        
        return res


    def infer_bs(self, input_idx, edge_targets, iters=1000, lr=1e-2, wd=1e-3, batch_size=512, 
                 bootstrap_p=0.75, dropout=0, device='cpu', verbose=True, bootstrap_iter=-1): 
        
        def eval_(model, w, x, y, crit, batch_size, device): 

            losses = []
            for idxs in torch.split(torch.arange(x.size(0)), batch_size): 

                with  torch.no_grad(): 
                    xx_batch = x[idxs].to(device)
                    yy_batch = y[idxs].to(device)
                    
                    i0_ = xx_batch[:, input_idx] @ w
                    e0 = torch.zeros(len(idxs), model.edge_index.size(1), device=device)
                    e0[:, edge_targets] = i0_

                    yhat = model(xx_batch, e0=e0)
                    loss = crit(yy_batch, yhat)

                losses.append(loss.item())
            return np.mean(losses)

        # TODO:  Need to know the MSE of the model with e0=0 and ensure that later MSE is lower than this 

        model = self.model.to(device)
        model.eval()
        self.freeze_(model)
        
        # xavier initialization standard deviation based on the fan-in and fan-out of the w 
        s = 1e-3

        dat = s*torch.randn(1, sum(edge_targets), device=device) 
        w = torch.nn.Parameter(dat)

        optim = torch.optim.Adam([w], lr=lr) # switch to L1...  weight_decay=wd)
        crit = torch.nn.MSELoss()
        dropout = torch.nn.Dropout(dropout)

        subset_idx = torch.randint(0, self.x_train.size(0), (int(bootstrap_p*self.x_train.size(0)),))
        x_train_bs = self.x_train[subset_idx]
        y_train_bs = self.y_train[subset_idx]

        val_mask = torch.ones(self.x_train.size(0), dtype=torch.bool)
        val_mask[torch.unique(subset_idx)] = False
        x_val_bs = self.x_train[~val_mask]
        y_val_bs = self.y_train[~val_mask]

        val0_loss = eval_(model, torch.zeros_like(w), x_val_bs, y_val_bs, crit, batch_size, device)

        for i in range(iters):
            
            train_losses = []
            for idxs in torch.split(torch.randperm(x_train_bs.size(0)), batch_size): 

                optim.zero_grad()

                xx_batch = x_train_bs[idxs].to(device)
                yy_batch = y_train_bs[idxs].to(device)
                
                i0_ = xx_batch[:, input_idx] @ dropout(w)
                e0 = torch.zeros(len(idxs), model.edge_index.size(1), device=device)
                e0[:, edge_targets] = i0_

                yhat = model(xx_batch, e0=e0)
                l1 = torch.norm(w, p=1)
                loss = crit(yy_batch, yhat) + wd*l1
                loss.backward()
                optim.step()

                train_losses.append(loss.item())

            val_loss = eval_(model, w, x_val_bs, y_val_bs, crit, batch_size, device)

            if verbose: print(f'[bootstrap sample: {bootstrap_iter}] --> iter: {i} | train loss: {np.mean(train_losses):.3f} || val0: {val0_loss:.4f}, val loss: {val_loss:.4f}]', end='\r')

        return w.detach()
