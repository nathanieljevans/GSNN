import torch
import copy 
import numpy as np
from collections import Counter
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelBinarizer
import torch_geometric as pyg
from sklearn.preprocessing import minmax_scale
from scipy.stats import spearmanr
    

def compute_sample_weights(sig_ids, max_prob_fold_diff=100):
    """
    Calculate the sample weights based on the joint frequency of cell lines and perturbation IDs.
    
    Args:
        cell_lines (numpy.ndarray): A numpy array containing the cell line for each example.
        pert_ids (numpy.ndarray): A numpy array containing the perturbation ID for each example.
    
    Returns:
        torch.Tensor: A PyTorch tensor containing the sample weights for each example.
    """

    cell_lines, pert_ids = get_sigid_attrs(sig_ids)

    # Combine cell_lines and pert_ids into pairs
    cell_line_pert_pairs = list(zip(cell_lines, pert_ids))

    # Calculate the joint frequency of each unique cell line and pert_id pair
    pair_counts = Counter(cell_line_pert_pairs)
    total_count = len(cell_lines)

    # Calculate the inverse joint frequency and assign it as weight for each example
    weights = np.array([total_count / pair_counts[pair] for pair in cell_line_pert_pairs], dtype=np.float32)

    # Convert the weights to a PyTorch tensor
    sample_weights = torch.from_numpy(weights)

    sample_prob = sample_weights / sample_weights.sum()

    clip_low = sample_prob.min()
    clip_high = clip_low*max_prob_fold_diff

    sample_prob = np.clip(sample_prob, clip_low, clip_high)

    print()
    print('Balancing training obs. sampling probabilities...')
    print('max prob. fold change (min-max)', max_prob_fold_diff)
    print('\tmin sample prob:', sample_prob.min())
    print('\tmax sample prob', sample_prob.max())
    print('\taverage sample prob', sample_prob.mean())
    print()

    return sample_prob


def get_sigid_attrs(sig_ids):
    """
    Extract cell line names (cell_inames) and perturbation IDs (pert_id) from the given signature IDs (sig_ids).
    
    Args:
        sig_ids (list or array-like): A list or array of signature IDs to be parsed.
    
    Returns:
        tuple: A tuple containing two lists:
            - cell_inames (list): Cell line names corresponding to the input signature IDs.
            - pert_ids (list): Perturbation IDs corresponding to the input signature IDs.
    """
    cell_inames = []
    pert_ids = []

    for sig_id in sig_ids:
        try: 
            # MET001_N8_XH:BRD-U44432129:100:336

            # Split the sig_id using '_' and ':'
            parts = sig_id.split('_')
            cell_iname = parts[1]

            # Extract the pert_id from the second part
            pert_id = parts[2].split(':')[1]

            cell_inames.append(cell_iname)
            pert_ids.append(pert_id)
        except: 
            raise ValueError(f'failed `sig_id` parse: {sig_id}')

    return cell_inames, pert_ids


def _get_regressed_metrics(y, yhat, sig_ids, siginfo, ignore_errors=True): 
    try: 
        r_cell = get_regressed_r(y, yhat, sig_ids, vars=['pert_id', 'pert_dose'], multioutput='uniform_weighted', siginfo=siginfo)
    except: 
        r_cell = -666
        if not ignore_errors: raise
    try:
        r_drug = get_regressed_r(y, yhat, sig_ids, vars=['cell_iname', 'pert_dose'], multioutput='uniform_weighted', siginfo=siginfo)
    except: 
        r_drug = -666
        if not ignore_errors: raise
    try: 
        r_dose = get_regressed_r(y, yhat, sig_ids, vars=['pert_id', 'cell_iname'], multioutput='uniform_weighted', siginfo=siginfo)
    except: 
        r_dose = -666
        if not ignore_errors: raise
    return r_cell, r_drug, r_dose


class TBLogger:
    def __init__(self, root):
        if not os.path.exists(root):
            os.mkdir(root)
        self.writer = SummaryWriter(log_dir=root)

    def add_hparam_results(self, args, model, data, device, test_loader, val_loader, siginfo, time_elapsed, epoch):
        if args.model == 'nn':
            predict_fn = predict_nn 
        elif args.model == 'gsnn':
            predict_fn = predict_gsnn
        elif args.model == 'gnn':
            predict_fn = predict_gnn
        else:
            raise ValueError(f'unrecognized model type: {args.model}')
        
        y_test, yhat_test, sig_ids_test = predict_fn(test_loader, model, device)
        y_val, yhat_val, sig_ids_val = predict_fn(val_loader, model, device)

        #r_cell_test, r_drug_test, r_dose_test = _get_regressed_metrics(y_test, yhat_test, sig_ids_test, siginfo)
        #r_cell_val, r_drug_val, r_dose_val = _get_regressed_metrics(y_val, yhat_val, sig_ids_val, siginfo)

        r2_test = r2_score(y_test, yhat_test, multioutput='variance_weighted')
        r2_val = r2_score(y_val, yhat_val, multioutput='variance_weighted')

        r_flat_test = np.corrcoef(y_test.ravel(), yhat_test.ravel())[0, 1]
        r_flat_val = np.corrcoef(y_val.ravel(), yhat_val.ravel())[0, 1]

        median_r_val = corr_score(y_val, yhat_val, multioutput='uniform_median')
        median_r_test = corr_score(y_test, yhat_test, multioutput='uniform_median')

        mean_r_val = corr_score(y_val, yhat_val, multioutput='uniform_weighted')
        mean_r_test = corr_score(y_test, yhat_test, multioutput='uniform_weighted')

        mse_test = np.mean((y_test - yhat_test)**2)
        mse_val = np.mean((y_val - yhat_val)**2)

        hparam_dict = args.__dict__
        metric_dict = {
            'median_r_val': median_r_val,
            'median_r_test': median_r_test,
            'mean_r_val': mean_r_val,
            'mean_r_test': mean_r_test,
            'r2_test': r2_test,
            'r2_val': r2_val,
            'r_flat_test': r_flat_test,
            'r_flat_val': r_flat_val,
            #'r_cell_test': r_cell_test,
            #'r_cell_val': r_cell_val,
            #'r_drug_test': r_drug_test,
            #'r_drug_val': r_drug_val,
            #'r_dose_test': r_dose_test,
            #'r_dose_val': r_dose_val,
            'mse_test': mse_test,
            'mse_val': mse_val,
            'time_elapsed': time_elapsed,
            'eval_at_epoch': epoch
        }

        self.writer.add_hparams(hparam_dict, metric_dict)

        return metric_dict, yhat_test, sig_ids_test

    def log(self, epoch, train_metrics, val_metrics):
        # Expecting train_metrics and val_metrics to be dictionaries,
        # something like: {'loss': ..., 'r2': ..., 'r_flat': ...}
        train_loss = train_metrics.get('loss', None)
        val_r2 = val_metrics.get('r2', None)
        val_r_flat = val_metrics.get('r_flat', None)

        if train_loss is not None:
            self.writer.add_scalar('train-loss', train_loss, epoch)
        if val_r2 is not None:
            self.writer.add_scalar('val-r2', val_r2, epoch)
        if val_r_flat is not None:
            self.writer.add_scalar('val-corr-flat', val_r_flat, epoch)


def get_activation(act): 

    if act == 'relu': 
        return torch.nn.ReLU 
    elif act == 'leakyrelu':
        return torch.nn.LeakyReLU
    elif act == 'prelu': 
        return torch.nn.PReLU
    elif act == 'elu': 
        return torch.nn.ELU 
    elif act == 'gelu': 
        return torch.nn.GELU 
    elif act == 'tanh': 
        return torch.nn.Tanh
    elif act == 'mish': 
        return torch.nn.Mish 
    elif act == 'selu': 
        return torch.nn.SELU  
    elif act == 'softplus': 
        return torch.nn.Softplus  
    elif act == 'linear': 
        return torch.nn.Identity
    else:
        raise ValueError(f'unrecognized activation function: {act}')

def get_optim(optim): 

    if optim == 'adam': 
        return torch.optim.Adam 
    elif optim == 'adan': 
        try: 
            from adan import Adan
        except: 
            raise ImportError('adan not installed. Please install adan (see: https://github.com/sail-sg/Adan)')
        return Adan
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
    

def _degree_to_channels(edge_index, min_size=3, max_size=25, transform=np.sqrt, verbose=False, scale_by='degree', clip_degree=250): 
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

def predict_gsnn(loader, model, device, verbose=True): 

    model = model.eval()

    ys = [] 
    yhats = [] 
    sig_ids = []
    
    with torch.no_grad(): 
        for i,(x, y, *sig_id) in enumerate(loader): 
            if verbose: print(f'progress: {i}/{len(loader)}', end='\r')

            yhat = model(x.to(device))

            y = y.to(device)

            yhat = yhat.detach().cpu() 
            y = y.detach().cpu()

            ys.append(y)
            yhats.append(yhat)
            sig_ids += sig_id

    y = torch.cat(ys, dim=0).detach().cpu().numpy()
    yhat = torch.cat(yhats, dim=0).detach().cpu().numpy()

    return y, yhat, sig_ids

def predict_nn(loader, model, device, verbose=True): 

    model = model.eval()

    ys = [] 
    yhats = [] 
    sig_ids = []
    
    with torch.no_grad(): 
        for i,(x, y, sig_id) in enumerate(loader): 
            if verbose: print(f'progress: {i}/{len(loader)}', end='\r')

            x = x.to(device).squeeze(-1)
            yhat = model(x)
            y = y.to(device).squeeze(-1)

            yhat = yhat.detach().cpu() 
            y = y.detach().cpu()

            ys.append(y)
            yhats.append(yhat)
            sig_ids += np.array(sig_id).ravel().tolist()

    y = torch.cat(ys, dim=0).detach().cpu().numpy()
    yhat = torch.cat(yhats, dim=0).detach().cpu().numpy()

    return y, yhat, sig_ids

def predict_gnn(loader, model, device, verbose=True): 

    model = model.eval()

    ys = [] 
    yhats = [] 
    sig_ids = []
    
    with torch.no_grad(): 
        for i,(batch) in enumerate(loader): 
            if verbose: print(f'progress: {i}/{len(loader)}', end='\r')
            
            yhat_dict = model({k:v.to(device) for k,v in batch.x_dict.items()}, 
                              {k:v.to(device) for k,v in batch.edge_index_dict.items()})
            
            #  select output nodes
            yhat = yhat_dict['output']
            y = batch.y_dict['output'].to(device)

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
    '''
    
    '''
    print('NOTE: RANDOMIZING EDGE INDEX')
    # permute edge index 
    edge_index_dict = copy.deepcopy(data.edge_index_dict)
    N_funcs = len(data.node_names_dict['function'])

    # randomize the input edges (e.g., drug targets and omics)
    # randomly select drug targets from all possible proteins 
    src,dst = edge_index_dict['input', 'to', 'function']
    dst = torch.tensor(np.random.choice(np.arange(N_funcs), size=(len(dst))), dtype=torch.long)
    edge_index_dict['input', 'to', 'function'] = torch.stack((src, dst), dim=0)

    # randomize the function node connections
    src,dst = edge_index_dict['function', 'to', 'function']
    src = torch.tensor(np.random.choice(np.arange(N_funcs), size=(len(dst))), dtype=torch.long)
    dst = torch.tensor(np.random.choice(np.arange(N_funcs), size=(len(dst))), dtype=torch.long)
    edge_index_dict['function', 'to', 'function'] = torch.stack((src, dst), dim=0)

    # randomize the output edge mask (e.g., endogenous feature connections)
    src,dst = edge_index_dict['function', 'to', 'output']
    src = torch.tensor(np.random.choice(np.arange(N_funcs), size=(len(dst))), dtype=torch.long)
    edge_index_dict['function', 'to', 'output'] = torch.stack((src, dst), dim=0)

    return edge_index_dict



def corr_score(y, yhat, multioutput='uniform_weighted', method='pearson', eps=1e-6): 
    '''
    calculate the average pearson correlation score

    y (n_samples, n_outputs): 
    yhat (n_samples, n_outputs):
    '''
    if len(y.shape) == 1: 
        y = y.reshape(-1,1)
        yhat = yhat.reshape(-1,1)

    if method == 'pearson': 
        metric = lambda x,y: np.corrcoef(x, y)[0,1]
    elif method == 'spearman': 
        metric = lambda x,y: spearmanr(x,y)[0]
    elif method == 'r2': 
        #NOTE: hacky since r2 is not a corr. 
        metric = lambda x,y: r2_score(x,y)
    else:
        raise ValueError('unrecognized metric')

    corrs = []
    for i in range(y.shape[1]): 
        if (np.std(y[:, i]) < eps) | (np.std(yhat[:, i]) < eps): 
            p = 0
        else: 
            p = metric(y[:, i], yhat[:, i])
            
        corrs.append( p ) 

    if multioutput == 'uniform_weighted': 
        return np.nanmean(corrs)
    elif multioutput == 'uniform_median': 
        return np.nanmedian(corrs)
    elif multioutput == 'raw_values': 
        return np.array(corrs)
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
    if y.shape[1] == 1: y = y.ravel()

    str_vars = df[vars].astype(str).agg('__'.join, axis=1)

    lb = LabelBinarizer() 
    one_hot_vars = lb.fit_transform(str_vars)

    #reg = MultiOutputRegressor(SGDRegressor())
    reg = LinearRegression()
    reg.fit(one_hot_vars, y)

    y_vars = reg.predict(one_hot_vars)
    y_res = y - y_vars

    return y_res 

def bootstrap_r(y, yhat, multioutput='uniform_weighted', n=100, q_lower=0.025, q_upper=0.975): 
    '''
    To get a better estimate of validation performance, we compute the validation 95% confidence interval of average pearson correlation. 

    Args: 
        y               np.array            true values 
        yhat            np.array            predicted values 
        multioutput     str                 method to handle multioutput prediction [uniform_weighted, raw_values]
        n               int                 number of bootstrapped samples to compute 
        q_lower         float               lower bound quantile 
        q_upper         float               upper bound quantile 

    Returns:
        r_low, r_up                         the lower and upper quantile of the (average) pearson correlation of y,yhat
    '''
    
    r = []
    for i in range(n): 
        idxs = np.random.choice(np.arange(0, y.shape[0]), size=y.shape[0], replace=True)
        r.append(corr_score(y[idxs], yhat[idxs], multioutput=multioutput))

    r_low = np.quantile(np.array(r), q=q_lower)
    r_up = np.quantile(np.array(r), q=q_upper)
    return r_low, r_up

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