import torch
import copy 
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score

class TBLogger():
    def __init__(self, root):
        ''''''
        if not os.path.exists(root): os.mkdir(root)
        self.writer = SummaryWriter(log_dir = root)

    def add_hparam_results(self, args, y, yhat):

        hparam_dict = args.__dict__
        metric_dict = {'R2':r2_score(y, yhat, multioutput='variance_weighted'), 
                       'r_flat': np.corrcoef(y.ravel(), yhat.ravel())[0,1],
                       'MSE': np.mean((y - yhat)**2)}
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

    y = torch.cat(ys, dim=0)
    yhat = torch.cat(yhats, dim=0)

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

    y = torch.cat(ys, dim=0)
    yhat = torch.cat(yhats, dim=0)

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