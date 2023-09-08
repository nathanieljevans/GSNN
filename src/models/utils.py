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
import torch_geometric as pyg
from sklearn.preprocessing import minmax_scale

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

    def log(self, epoch, train_loss, val_r2, val_r_flat):
        self.writer.add_scalar('train-loss', train_loss, epoch)
        self.writer.add_scalar('val-r2', val_r2, epoch)
        self.writer.add_scalar('val-corr-flat', val_r_flat, epoch)


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
    

def _degree_to_channels(edge_index, min_size=2, max_size=25, transform=np.sqrt, verbose=True, scale_by='in_degree', clip_degree=250): 
    '''
    utility function to create variable number of channels per function node, dependent on the degree of each node. 

    # channels = minmax_scale(transform(degree), range=(min_size, max_size))

    Args: 
        edge_index          torch.tensor            COO format graph edge index 
        min_size            int                     minimum number of channels 
        max_size            int                     maximum number of channels 
        transform           function                transformation to be applied to degree prior to min-max scaling between range( )
        verbose             bool                    whether to print summary statistics to console 
        scale_by            str                     the choice of scaling metric, options: 'in_degree', 'out_degree', 'degree' 
        clip_degree         int;None                whether to clip the maximum degree value; useful if there are outliers with large degree

    Returns:
        scaled_channels 
    '''
    num_nodes = torch.unique(edge_index.view(-1)).size(0)
    row, col = edge_index 
    out_degree = pyg.utils.degree(row, num_nodes).detach().cpu().numpy() 
    in_degree = pyg.utils.degree(col, num_nodes).detach().cpu().numpy() 

    if scale_by == 'in_degree': 
        degree = in_degree 
    elif scale_by == 'out_degree': 
        degree = out_degree 
    elif scale_by == 'degree': 
        degree = in_degree + out_degree 
    else: 
        raise ValueError(f'`_degree_to_channels` got unexpected `scale_by` argument, expected one of: in_degree, out_degree, degree but got: {scale_by}')
    
    if clip_degree is not None: 
        degree = np.clip(degree, 0, clip_degree)

    scaled_channels = transform(degree)                                                                                     # apply degree transformation 
    func_node_mask = (in_degree > 0) * (out_degree > 0)
    scaled_channels[func_node_mask] = minmax_scale(scaled_channels[func_node_mask], feature_range=(min_size, max_size))     # scale between `min_size` and `max_size`
    scaled_channels = np.array([int(np.round(x, decimals=0)) for x in scaled_channels])                                     # ensure integers 
    scaled_channels[~func_node_mask] = 0                                                                                    # only function nodes need hidden channels; input/output nodes have no function. To ensure proper indexing, we will have a surrogate index for input/output nodes. Note: this does not impact the number of parameters. 
    if verbose: print('mean # of function node channels (scaled)', np.mean(scaled_channels[func_node_mask]))
    return scaled_channels


def get_W1_indices(edge_index, channels, function_nodes, scale_by_degree=True): 
    '''
    how to create input layer , e.g., edge values -> node indices 

    Args: 
        edge_index          torch tensor            COO edge index describing the structural graph 
        channels            int                     if `scale_by_degree` is false, then this will be the number of hidden channels in each function 
                                                    node; otherwise this will be the maximum number of channels in the function node.  
        function_nodes
        scale_by_degree     bool                    whether to scale the number of hidden channels based on the function node degree; input and output 
                                                    nodes will have 0 channels
    
    Returns: 
        indices             torch tensor            COO format for the W1 indices
        _channels           numpy array             the number of hidden channels for each node, indexed by node id; e.g., node 10 will have _channels[10] hidden channels
    '''

    if scale_by_degree:
        _channels = _degree_to_channels(edge_index, max_size=channels)
    else: 
        _channels = np.zeros(edge_index.size(1), dtype=int) 
        _channels[function_nodes] = channels

    row = []
    col = []
    for edge_id, (_, node_id) in enumerate(edge_index.detach().cpu().numpy().T):
        if node_id not in function_nodes: continue # skip the output nodes 
        c = _channels[node_id] # number of func. node channels 
        node_id_idx0 = np.sum(_channels[:node_id.item()])       # node indexing: index of the first hidden channel for a given function node 
        for k in range(c): 
            row.append(edge_id)
            col.append(node_id_idx0 + k)

    row = torch.tensor(row, dtype=torch.float32)
    col = torch.tensor(col, dtype=torch.float32)
    indices = torch.stack((row,col), dim=0)
    return indices, _channels


def get_W2_indices(function_nodes, channels): 
    '''
    how to create node -> node latent weight indices for W2 

    Args: 
        function_nodes      torch tensor        the node index for function nodes (e.g., in_degree > 0 and out_degree > 0)
        channels            numpy array         the number of hidden channels for each function node, indexed by node 

    Returns: 
        indices             torch tensor        COO format edge indices for W2 
    '''
    row = []
    col = []
    for node_id in function_nodes: 
        c = channels[node_id]                                  # number of func. node channels 
        node_id_idx0 = np.sum(channels[:node_id.item()])       # node indexing: index of the first hidden channel for a given function node 
        for k in range(c): 
            for k2 in range(c): 
                row.append(node_id_idx0 + k)
                col.append(node_id_idx0 + k2)

    row = torch.tensor(row, dtype=torch.float32)
    col = torch.tensor(col, dtype=torch.float32)
    indices = torch.stack((row,col), dim=0)
    return indices

def get_W3_indices(edge_index, function_nodes, channels): 
    '''
    how to create node -> edge indices for W3

    Args: 
        edge_index          torch tensor        COO edge index describing the structural graph 
        function_nodes      torch tensor        the node index for function nodes (e.g., in_degree > 0 and out_degree > 0)
        channels            numpy array         the number of hidden channels for each function node, indexed by node 

    Returns: 
        indices             torch tensor        COO format edge indices for W2 
    '''
    row = [] 
    col = []
    for node_id in function_nodes: 
        
        # get the edge ids of the function node 
        src,_ = edge_index 
        out_edges = (src == node_id).nonzero(as_tuple=True)[0]

        c = channels[node_id]                                  # number of func. node channels 
        node_id_idx0 = np.sum(channels[:node_id.item()])       # node indexing: index of the first hidden channel for a given function node 

        for k in range(c):
            for edge_id in out_edges: 
                row.append(node_id_idx0 + k)
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

    # randomize the input edges (e.g., drug targets and omics)
    src,dst = edge_index[:, data.input_edge_mask]
    dst = torch.tensor(np.random.choice(func_nodes, size=(len(dst))), dtype=torch.long)
    edge_index[:, data.input_edge_mask] = torch.stack((src, dst), dim=0)

    # randomize the function node connections
    func_edge_mask = ~(data.input_edge_mask | data.output_edge_mask)
    src,dst = edge_index[:, func_edge_mask]
    src = torch.tensor(np.random.choice(func_nodes, size=(len(dst))), dtype=torch.long)
    dst = torch.tensor(np.random.choice(func_nodes, size=(len(dst))), dtype=torch.long)
    edge_index[:, func_edge_mask] = torch.stack((src, dst), dim=0)

    # randomize the output edge mask (e.g., endogenous feature connections)
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


def next_divisor(N, X):
    '''
    returns the smallest divisor of N which is larger than or equal to X
    '''
    i = X
    while N % i != 0:
        i += 1

    return i