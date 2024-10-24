

import torch
import numpy as np 
from geomloss import SamplesLoss
from scipy.stats import wasserstein_distance_nd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import umap
from gsnn.ot.mmd import compute_scalar_mmd
from sklearn.metrics import r2_score

def freeze_(model):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

def unfreeze_(model):
    for param in model.parameters():
        param.requires_grad_(True)
    model.train()

def eval(T, sampler, batch_size=64, partition='val', agg='mean', max_n=None): 
    '''estimate of val performance'''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shd = SamplesLoss('sinkhorn', p=2, blur=0.05)

    freeze_(T) 
    mmds_ = []
    shds_ = []
    wass_ = [] 
    mu_r2 = [] 
    for i in range(len(sampler)): 
        print(f'[evaluating condition {i}/{len(sampler)}]', end='\r')

        with torch.no_grad(): 
            X,y, x_cell, x_drug, y0 = sampler.sample_(i, batch_size=max_n, partition=partition)
            X = X.to(device); y=y.to(device); y0 = y0.to(device)

            yhat = [] 
            for idx in torch.split(torch.arange(X.shape[0]), batch_size): 
                yhat.append( T(X[idx]) + y0[idx] )
            yhat = torch.cat(yhat, dim=0)

            mu_delta_hat = yhat.mean(dim=0) - y0.mean(dim=0)
            mu_delta = y.mean(dim=0) - y0.mean(dim=0)

            mu_r2.append(r2_score(mu_delta.detach().cpu().numpy(), mu_delta_hat.detach().cpu().numpy()))

            mmds_.append(compute_scalar_mmd(y.detach().cpu().numpy(), yhat.detach().cpu().numpy()))
            shds_.append(shd(yhat, y).item())
            wass_.append(wasserstein_distance_nd(yhat.detach().cpu().numpy(), y.detach().cpu().numpy()))

    if agg == 'mean':
        return np.mean(mmds_), np.mean(shds_), np.mean(wass_), np.mean(mu_r2)
    elif agg == 'none':
        return mmds_, shds_, wass_, mu_r2
    else:
        raise ValueError('agg must be one of mean, none')


def plot_f_score(reducer, f, x_drug, x_cell, ax, x_min, x_max, y_min, y_max, n=15):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Transform the grid back to the original space
    grid_transformed = reducer.inverse_transform(grid)
    
    # Predict using the model f
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_transformed, dtype=torch.float32).to(device)
        yf = torch.cat((x_cell[0].unsqueeze(0).expand(grid_tensor.size(0), -1), 
                        x_drug[0].unsqueeze(0).expand(grid_tensor.size(0), -1), 
                        grid_tensor), dim=-1)
        z = f(yf).cpu().numpy().reshape(xx.shape)
    
    # Plot the contour
    c = ax.contourf(xx, yy, z, levels=15, cmap="RdBu_r", alpha=0.6)
    ax.contour(xx, yy, z, levels=15, colors="k", linewidths=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.colorbar(c, ax=ax)


def plot_transport_plan(sampler, T, F=None, conditions=None, dim_red='pca', max_n=500, plot_y0=True):

    if conditions is None: 
        conditions = range(len(sampler))   

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    for i in conditions:

        cond = sampler.conditions.iloc[i]
        drg_ = cond.drug 
        cell_ = cond.cell_line 
        dose_ = cond.dose

        with torch.no_grad():
            X,y, x_cell, x_drug, y0 = sampler.sample_(idx=i, batch_size=max_n)
            X = X.to(device); y=y.to(device); x_cell = x_cell.to(device); x_drug = x_drug.to(device); y0 = y0.to(device)
            yf = torch.cat((x_cell, x_drug, y0), dim=-1)

            delta = [] 
            for idx in torch.split(torch.arange(X.shape[0]), 25): 
                delta.append(T(X[idx].to(device)).detach().cpu())
            delta = torch.cat(delta, dim=0)

        y0 = y0.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        yhat = delta.detach().cpu().numpy() + y0

        # dim red on each condition to better separate perturbation effects 
        if dim_red == 'pca': 
            reducer = PCA(2)
            reducer.fit(np.concatenate([y0, y, yhat], axis=0))
        elif dim_red == 'umap': 
            reducer = umap.UMAP(2, metric='cosine', min_dist=0.5)
            targets = None # np.concatenate([np.zeros(y0.shape[0]), np.ones(y.shape[0])])
            reducer.fit(np.concatenate([y0, y, yhat], axis=0), y=targets)
        else:
            raise ValueError('dim_red must be one of pca, umap')

        pc_y0 = reducer.transform(y0)
        pc_y = reducer.transform(y)
        pc_yhat = reducer.transform(yhat)

        x_min = np.min([pc_y0[:,0].min(), pc_y[:,0].min(), pc_yhat[:,0].min()])
        x_max = np.max([pc_y0[:,0].max(), pc_y[:,0].max(), pc_yhat[:,0].max()])
        y_min = np.min([pc_y0[:,1].min(), pc_y[:,1].min(), pc_yhat[:,1].min()])
        y_max = np.max([pc_y0[:,1].max(), pc_y[:,1].max()])

        x_margin = 0.1*(x_max - x_min)
        y_margin = 0.1*(y_max - y_min)
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        f, ax = plt.subplots(1,1, figsize=(6,6))
        if F is not None: plot_f_score(reducer, F, x_drug, x_cell, ax, x_min, x_max, y_min, y_max, n=25)
        if plot_y0: 
            for ii in range(pc_y0.shape[0]): 
                plt.plot((pc_y0[ii,0], pc_yhat[ii, 0]),(pc_y0[ii,1], pc_yhat[ii, 1]),  'r-', alpha=0.33)
        if plot_y0: plt.plot(*pc_y0.T, 'ko', label='y0', markersize=5)
        if plot_y0: plt.plot(*pc_y0.T, 'ro', label='y0', markersize=3)
        plt.plot(*pc_yhat.T, 'ks', label='yhat', alpha=1., markersize=5)
        plt.plot(*pc_yhat.T, 'rs', label='yhat', alpha=1., markersize=3)
        plt.plot(*pc_y.T, 'k.', label='y')
        plt.xlabel('u1')
        plt.ylabel('u2')
        plt.title(f'Condition: {drg_}, {cell_}, {dose_}')
        plt.legend()
        plt.show()