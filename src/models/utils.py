import torch
import copy 
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelBinarizer

def _get_regressed_metrics(y, yhat, sig_ids, siginfo): 
    try: 
        r_cell = get_regressed_r(y, yhat, sig_ids, vars=['pert_id', 'pert_dose'], multioutput='uniform_weighted', siginfo=siginfo)
    except: 
        r_cell = -666
    try:
        r_drug = get_regressed_r(y, yhat, sig_ids, vars=['cell_iname', 'pert_dose'], multioutput='uniform_weighted', siginfo=siginfo)
    except: 
        r_drug = -666
    try: 
        r_dose = get_regressed_r(y, yhat, sig_ids, vars=['pert_id', 'cell_iname'], multioutput='uniform_weighted', siginfo=siginfo)
    except: 
        r_dose = -666
    return r_cell, r_drug, r_dose

class TBLogger():
    def __init__(self, root):
        ''''''
        if not os.path.exists(root): os.mkdir(root)
        self.writer = SummaryWriter(log_dir = root)

    def add_hparam_results(self, args, model, data, device, test_loader, val_loader, siginfo, time_elapsed, epoch):

        if args.model == 'nn':
            predict_fn = predict_nn 
        elif args.model == 'gsnn': 
            predict_fn = predict_gsnn
        elif args.model == 'gnn':
            predict_fn = predict_gnn 
        else: 
            raise ValueError(f'unrecognized model type: {args.model}')
        
        y_test, yhat_test, sig_ids_test = predict_fn(test_loader, model, data, device)
        y_val, yhat_val, sig_ids_val = predict_fn(val_loader, model, data, device)

        r_cell_test, r_drug_test, r_dose_test = _get_regressed_metrics(y_test, yhat_test, sig_ids_test, siginfo)
        r_cell_val, r_drug_val, r_dose_val = _get_regressed_metrics(y_val, yhat_val, sig_ids_val, siginfo)

        r2_test = r2_score(y_test, yhat_test, multioutput='variance_weighted')
        r2_val = r2_score(y_val, yhat_val, multioutput='variance_weighted')

        r_flat_test = np.corrcoef(y_test.ravel(), yhat_test.ravel())[0,1]
        r_flat_val = np.corrcoef(y_val.ravel(), yhat_val.ravel())[0,1]

        mse_test = np.mean((y_test - yhat_test)**2)
        mse_val = np.mean((y_val - yhat_val)**2)

        hparam_dict = args.__dict__
        metric_dict = {'r2_test':r2_test, 
                       'r2_val':r2_val, 
                       'r_flat_test':r_flat_test, 
                       'r_flat_val':r_flat_val,
                       'r_cell_test':r_cell_test,
                       'r_cell_val':r_cell_val,
                       'r_drug_test':r_drug_test,
                       'r_drug_val':r_drug_val,
                       'r_dose_test':r_dose_test,
                       'r_dose_val':r_dose_val,
                       'mse_test':mse_test,
                       'mse_val':mse_val,
                       'time_elapsed':time_elapsed,
                       'eval_at_epoch':epoch
                       }
        

        self.writer.add_hparams(hparam_dict, metric_dict)

    def log(self, epoch, train_loss, test_r2, test_r_flat):
        self.writer.add_scalar('train-loss', train_loss, epoch)
        self.writer.add_scalar('test-r2', test_r2, epoch)
        self.writer.add_scalar('test-corr-flat', test_r_flat, epoch)


def get_activation(act): 

    if act == 'relu': 
        return torch.nn.ReLU 
    elif act == 'elu': 
        return torch.nn.ELU 
    elif act == 'tanh': 
        return torch.nn.Tanh
    elif act == 'mish': 
        return torch.nn.Mish 
    elif act == 'selu': 
        return torch.nn.SELU  
    elif act == 'softplus': 
        return torch.nn.Softplus  
    else:
        raise ValueError(f'unrecognized activation function: {act}')

def get_optim(optim): 

    if optim == 'adam': 
        return torch.optim.Adam 
    elif optim == 'sgd': 
        return torch.optim.SGD 
    elif optim == 'rmsprop': 
        return torch.optim.RMSprop
    else:
        raise ValueError(f'unrecognized optim argument: {optim}')
    
def get_crit(crit): 

    if crit == 'mse': 
        return torch.nn.MSELoss
    elif crit == 'huber': 
        return torch.nn.HuberLoss
    else:
        raise ValueError(f'unrecognized optim argument: {crit}')
    
def get_scheduler(optim, args, loader): 

    if args.sched == 'none': 
        return None
    elif args.sched == 'onecycle': 
        return torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr, 
                                                    epochs=args.epochs, 
                                                    steps_per_epoch=len(loader), 
                                                    pct_start=0.3)
    elif args.sched == 'cosine': 
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs*len(loader), eta_min=1e-7)
    else:
        raise ValueError(f'unrecognized lr scheduler: {args.sched}')
    


def get_W1_indices(edge_index, channels): 
    '''
    # how to create input layer , e.g., edge values -> node indices 


    '''
    row = [] 
    col = []
    for edge_id, (src, node_id) in enumerate(edge_index.detach().cpu().numpy().T):
        for k in range(channels): 
            row.append(edge_id)
            col.append(channels*node_id.item() + k)

    row = torch.tensor(row, dtype=torch.float32)
    col = torch.tensor(col, dtype=torch.float32)
    indices = torch.stack((row,col), dim=0)
    return indices


def get_W2_indices(function_nodes, channels): 
    '''
    # how to create node -> node latent weight indices 

    # for node_id in function_nodes 
        # for k in channels: 
            # for k2 in channels: 
                # add weight indice: (node_id + k, node_id + k2)
    '''
    row = []
    col = []
    for node_id in function_nodes: 
        for k in range(channels): 
            for k2 in range(channels): 
                row.append(channels*node_id.item() + k)
                col.append(channels*node_id.item() + k2)

    row = torch.tensor(row, dtype=torch.float32)
    col = torch.tensor(col, dtype=torch.float32)
    indices = torch.stack((row,col), dim=0)
    return indices

def get_W3_indices(edge_index, function_nodes, channels): 
    '''
    # how to create node -> edge indices 

    # for node_id in function_nodes 

        # filter to edges from node_id 
        # src, dst = edge_index
        # out_edges = (src == node_id).nonzero()
        # for k in channels: 
            # for out_edge_idx in out_edges: 
                # add weight indice:   (node_id + k, out_edge_idx)
    '''
    row = [] 
    col = []
    for node_id in function_nodes: 
        
        src,dst = edge_index 
        out_edges = (src == node_id).nonzero(as_tuple=True)[0]

        for k in range(channels):
            
            for edge_id in out_edges: 

                row.append(channels*node_id.item() + k)
                col.append(edge_id.item())

    row = torch.tensor(row, dtype=torch.float32)
    col = torch.tensor(col, dtype=torch.float32)
    indices = torch.stack((row,col), dim=0)
    return indices

def node2edge(x, edge_index): 
    '''
    convert from node indexed attributes to edge indexed attributes
    '''
    src,dst = edge_index 
    return x[:, src] 

def edge2node(x, edge_index, output_node_mask): 
    ''' 
    convert from edge indexed attributes `x` to node indexed attributes
    NOTE: only maps to output nodes (eg., in-degree = 1) to avoid collisions; all other nodes (input nodes + function nodes) will have value of 0. 
    '''
    output_nodes = output_node_mask.nonzero(as_tuple=True)[0]
    src, dst = edge_index 
    output_edge_mask = torch.isin(dst, output_nodes)

    B = x.size(0)
    out = torch.zeros(B, output_node_mask.size(0), dtype=torch.float32, device=x.device)
    out[:, dst[output_edge_mask].view(-1)] = x[:, output_edge_mask].view(B, -1)

    return out

def predict_gsnn(loader, model, data, device): 

    model = model.eval()

    ys = [] 
    yhats = [] 
    sig_ids = []
    
    with torch.no_grad(): 
        for i,(x, y, sig_id) in enumerate(loader): 
            print(f'progress: {i}/{len(loader)}', end='\r')

            yhat = model(x.to(device))[:, data.output_node_mask]
            y = y.to(device).squeeze(-1)[:, data.output_node_mask]

            yhat = yhat.detach().cpu() 
            y = y.detach().cpu()

            ys.append(y)
            yhats.append(yhat)
            sig_ids += sig_id

    y = torch.cat(ys, dim=0).detach().cpu().numpy()
    yhat = torch.cat(yhats, dim=0).detach().cpu().numpy()

    return y, yhat, sig_ids

def predict_nn(loader, model, data, device): 

    model = model.eval()

    ys = [] 
    yhats = [] 
    sig_ids = []
    
    with torch.no_grad(): 
        for i,(x, y, sig_id) in enumerate(loader): 
            print(f'progress: {i}/{len(loader)}', end='\r')

            x = x[:, data.input_node_mask].to(device).squeeze(-1)
            yhat = model(x)
            y = y.to(device).squeeze(-1)[:, data.output_node_mask]

            yhat = yhat.detach().cpu() 
            y = y.detach().cpu()

            ys.append(y)
            yhats.append(yhat)
            sig_ids += sig_id

    y = torch.cat(ys, dim=0).detach().cpu().numpy()
    yhat = torch.cat(yhats, dim=0).detach().cpu().numpy()

    return y, yhat, sig_ids

def predict_gnn(loader, model, data, device): 

    model = model.eval()

    ys = [] 
    yhats = [] 
    sig_ids = []
    
    with torch.no_grad(): 
        for i,(batch) in enumerate(loader): 

            yhat = model(edge_index=batch.edge_index.to(device), x=batch.x.to(device))
            
            #  select output nodes
            yhat = yhat[batch.output_node_mask]
            y = batch.y.to(device)[batch.output_node_mask]

            B = len(batch.sig_id)

            yhat = yhat.view(B, -1).detach().cpu()
            y = y.view(B, -1).detach().cpu()

            ys.append(y)
            yhats.append(yhat)
            sig_ids += batch.sig_id

    y = torch.cat(ys, dim=0).detach().cpu().numpy()
    yhat = torch.cat(yhats, dim=0).detach().cpu().numpy()

    return y, yhat, sig_ids

def randomize(data): 
    print('NOTE: RANDOMIZING EDGE INDEX')
    # permute edge index 
    edge_index = copy.deepcopy(data.edge_index)
    func_nodes  = (~(data.input_node_mask | data.output_node_mask)).nonzero(as_tuple=True)[0].detach().cpu().numpy()

    src,dst = edge_index[:, data.input_edge_mask]
    dst = torch.tensor(np.random.choice(func_nodes, size=(len(dst))), dtype=torch.long)
    edge_index[:, data.input_edge_mask] = torch.stack((src, dst), dim=0)


    func_edge_mask = ~(data.input_edge_mask | data.output_edge_mask)
    src,dst = edge_index[:, func_edge_mask]
    src = torch.tensor(np.random.choice(func_nodes, size=(len(dst))), dtype=torch.long)
    dst = torch.tensor(np.random.choice(func_nodes, size=(len(dst))), dtype=torch.long)
    edge_index[:, func_edge_mask] = torch.stack((src, dst), dim=0)


    src,dst = edge_index[:, data.output_edge_mask]
    src = torch.tensor(np.random.choice(func_nodes, size=(len(dst))), dtype=torch.long)
    edge_index[:, data.output_edge_mask] = torch.stack((src, dst), dim=0)

    return edge_index



def corr_score(y, yhat, multioutput='uniform_weighted'): 
    '''
    calculate the average pearson correlation score

    y (n_samples, n_outputs): 
    yhat (n_samples, n_outputs):
    '''
    if len(y.shape) == 1: 
        y = y.reshape(-1,1)
        yhat = yhat.reshape(-1,1)

    corrs = []
    for i in range(y.shape[1]): 
        if (np.std(y[:, i]) == 0) : 
            # this occurs when a landmark gene zscore will be all zeros for a batch. 
            p = 0
        
        elif (np.std(yhat[:, i]) == 0): 
            # this occurs if an entire batch is made up of the same replicate; can occur in test sets 
            p = 0

        else: 
            p = np.corrcoef(y[:, i], yhat[:, i])[0,1]
            
        corrs.append( p ) 

    if multioutput == 'uniform_weighted': 
        return np.nanmean(corrs)
    elif multioutput == 'raw_values': 
        return corrs
    else:
        raise ValueError('unrecognized multioutput value, expected one of "uniform_weighted", "raw_values"')

def regress_out(y, df, vars): 
    '''
    regress out variance from certain variables 

    inputs 
        y       numpy array     signal to modify 
        df      dataframe       co-variates options 
        vars    list<str>       variables to regress out; must be columns in dataframe 

    outputs 
        numpy array     augmented y signal 
    ''' 
    str_vars = df[vars].astype(str).agg('__'.join, axis=1)

    lb = LabelBinarizer() 
    one_hot_vars = lb.fit_transform(str_vars)

    reg = LinearRegression() 
    reg.fit(one_hot_vars, y)

    y_vars = reg.predict(one_hot_vars)
    y_res = y - y_vars

    return y_res 

def get_regressed_r(y, yhat, sig_ids, vars, data='../../data/', multioutput='uniform_weighted', siginfo=None): 

    
    if siginfo is None: siginfo = pd.read_csv(f'{data}/siginfo_beta.txt', sep='\t', low_memory=False)[['sig_id', 'pert_id', 'cell_iname', 'pert_dose']]

    df = pd.DataFrame({'sig_id':sig_ids}).merge(siginfo, on='sig_id', how='left')

    y_res = regress_out(y, df, vars=vars)
    yhat_res = regress_out(yhat, df, vars=vars)

    return corr_score(y_res, yhat_res, multioutput=multioutput)
