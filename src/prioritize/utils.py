
import argparse 
import pandas as pd 
import torch
import numpy as np
import os 
import shutil 
import sys 
sys.path.append('../')
from src.models.GSNN import GSNN
from src.models.NN import NN

def set_drug_concs(xx, drugs, concs, data, verbose=True): 
    '''

    args: 
        xx      torch.tesnsor       the input vector for a GSNN model - Assumes all drug concentrations are zero. 
        drugs   list                list of pert_id strings to set respective conc value to 
        concs   list                list of floats to set respective drug conc value 
        data    pyg.Data            data object to use 
        verbose str                 verbosity 

    returns: 
        torch.tensor                the xx value with the drugs set to concs values. 
    '''


    for drug, conc in zip(drugs, concs): 
        if verbose: print('setting drug conc: ', drug, conc)

        drug_idx = data.node_names.tolist().index('DRUG__' + drug)
        xx[:, drug_idx] = dose2scale(conc)

    return xx.detach().clone()


def get_base_X(cell_inames, proc, data, instinfo): 
    '''
    find an observation of the given `cell_inames` and load to memory, then set all drugs to zero. To be used as base X values. Will be returned with shape (B, N) 
    where B is length of `cell_inames` and N is the number of nodes. 

    Args: 
        cell_inames         list-like           cell_inames base X inputs to load 
        proc                str                 path to /proc/ directory 
        data                pyg.Data            gsnn graph object 
        instinfo            dataframe           

    Returns 
        X                   torch.tensor        The no-drug X inputs for a given experiment, shape B,N
    ''' 
    sigids = np.load(f'{proc}/sig_ids.npy', allow_pickle=True).tolist()
    instinfo = instinfo[lambda x: x.sig_id.isin(sigids)]
    cell2sigid = instinfo[['sig_id', 'cell_iname']].set_index('cell_iname').to_dict()['sig_id']
    X = [] 
    for cell_iname in cell_inames: 
        sigid = cell2sigid[cell_iname]
        obs = torch.load(f'{proc}/obs/{sigid}.pt')
        X.append(obs['x'].view(1, -1))

    X = torch.cat(X, dim=0)

    drug_nodes = [x for x in data.node_names if 'DRUG__' in x] 
    drug_node_idxs = np.isin(data.node_names, drug_nodes).nonzero()[0]
    
    # set drug values to zero 
    X[:, drug_node_idxs] = 0. 

    return X

def dose2scale(x, eps=1e-6): 
    return (np.log10(x + eps) - np.log10(eps))/-np.log10(eps)


def make_drug_inputs(args, data, drugs, cells, doses, siginfo): 
    '''
    
    Args: 
        args        namespace   commandline args 
        data        pyg.Data    GSNN graph data object 
        drugs       list        drugs to screen 
        cells       list        cell lines to screen 
        doses       list        doses to screen 

    '''

    res = {'cell_iname':[], 'pert_id_1':[],'pert_id_2':[], 'dose_um_1':[], 'dose_um_2':[]}
    X = []
    cell_x_dict = {cell:get_base_X([cell], args.proc, data, siginfo) for cell in cells}


    # ensure all drugs are in namespace 
    for drug in drugs: 
        drug_ = 'DRUG__' + drug 
        if drug_ not in data.node_names: raise ValueError(f'unrecognized drug: {drug}')

    drugspace_1 = drugs
    drugspace_2 = drugs + ['none'] if args.combo else ['none']

    print(f'screening {len(drugspace_1)} drugs over {len(doses)} doses')
    print(f'screening 2-drug combination agents:', args.combo)

    # list of 2-drug sets specifying the drug combinations that have been tested already; NOTE: we assume drug1,drug2 = drug2,drug1
    drug_combo_list = []

    for i,cell_iname in enumerate(cells): 

        for j1, pert1 in enumerate(drugspace_1):

            for j2, pert2 in enumerate(drugspace_2): 

                for k1, dose1 in enumerate(doses): 

                    for k2, dose2 in enumerate(doses): 

                        # to avoid multible obs per "none"
                        if pert2 == 'none': dose2 = 0.

                        print(f'progress: \t{cell_iname} ({i}/{len(cells)}) \t| {pert1}={dose1:.4f} \t| {pert2}={dose2:.4f})', end='\r')

                        drug_comb = set([f'{pert1}::{dose1}', f'{pert2}::{dose2}', cell_iname])
                        if drug_comb in drug_combo_list: continue  
                        drug_combo_list.append(drug_comb)

                        # encode drug 1 and cell context features 
                        xx = cell_x_dict[cell_iname].clone().detach().squeeze(0)
                        drug1_idx = data.node_names.tolist().index('DRUG__' + pert1)
                        xx[drug1_idx] = dose2scale(dose1)

                        if pert2 != 'none': 
                            # second drug feature 
                            drug2_idx = data.node_names.tolist().index('DRUG__' + pert2)
                            xx[drug2_idx] = dose2scale(dose2)
                        
                        X.append(xx)
                        res['cell_iname'].append(cell_iname)
                        res['pert_id_1'].append(pert1)
                        res['pert_id_2'].append(pert2)
                        res['dose_um_1'].append(dose1)
                        res['dose_um_2'].append(dose2)

    if args.verbose: print()

    X = torch.stack(X, dim=0)
    res = pd.DataFrame(res)

    return X, res 

def predict_expr(model, X, data, device='cuda', batch=250, verbose=True): 
    
    if verbose: print(f'predicting perturbed expression [batch={batch}, device={device}]...')

    model = model.to(device)
    model.eval()

    if isinstance(model, GSNN): 
        print('expr model type: GSNN')
        forward = lambda x: model(x)[:, data.output_node_mask]
    elif isinstance(model, NN): 
        print('expr model type: NN')
        forward = lambda x: model(x.squeeze(-1)[:, data.input_node_mask])
    else: 
        raise NotImplementedError('unrecognized expr model type')
    
    pred_expr = []
    with torch.no_grad(): 
        for batch_idxs in torch.split(torch.arange(0, len(X)), split_size_or_sections=batch): 
            if verbose: print(f'\tprogress: {batch_idxs[-1]}/{len(X)}', end='\r')
            xx = X[batch_idxs].unsqueeze(-1).to(device)
            out = forward(xx)
            pred_expr.append(out.detach())

    pred_expr = torch.cat(pred_expr, dim=0)
    if verbose: print()
    return pred_expr


def evaluate_drugs(viab_model, pred_expr, meta, res_lines, sens_lines, N=1000, verbose=True): 
    '''
    
    Args: 
        viab_model          torch.nn.Module             the cell viability predictor 
        pred_expr           torch.tensor                predicted perturbed expression values 
        meta                pd.DataFrame                metadata for the `pred_expr` data. 
        res_lines           list-like                   the "background" cell lines; those intended to have a less desirable response (e.g., the more resistant lines)
        sens_lines          list-like                   the "target" cell lines; those intended to have a desirable response (e.g., the more sensitive lines)
        N                   int                         number of monte carlo simulations to run (default=1000)
        verbose             bool                        whether to print progress reports to console 

    Returns: 
        pd.DataFrame                                    the drug results with have columns: pert_id, dose_um, diff_mean, diff_var, p_sens 
    '''
    if verbose: print(f'Calculating drug priortization metrics...')
    if verbose: print(f'\t# of "target" lines: {len(sens_lines)}')
    if verbose: print(f'\t# of "background" lines: {len(res_lines)}')
    if verbose: print(f'\t# of MC simulations: {N}')

    res = {'pert_id_1':[], 'dose_um_1':[], 'pert_id_2':[], 'dose_um_2':[], 'diff_mean':[], 'diff_var':[], 'p_sens':[]}

    drug_df = meta[['pert_id_1', 'dose_um_1', 'pert_id_2', 'dose_um_2']].drop_duplicates()
    
    jj = 0
    for i,row in drug_df.iterrows():  
        if verbose: print(f'\tprogress: {jj+1}/{len(drug_df)}', end='\r'); jj+=1
        
        # get the obs. idxs for the "target" and "background"
        drug_idxs_sens = meta[lambda x: (x.pert_id_1 == row.pert_id_1) & 
                              (x.dose_um_1 == row.dose_um_1) & 
                              (x.pert_id_2 == row.pert_id_2) & 
                              (x.dose_um_2 == row.dose_um_2) & 
                              (x.cell_iname.isin(sens_lines))].index.values 
        drug_idxs_res = meta[lambda x: (x.pert_id_1 == row.pert_id_1) & 
                              (x.dose_um_1 == row.dose_um_1) & 
                              (x.pert_id_2 == row.pert_id_2) & 
                              (x.dose_um_2 == row.dose_um_2) & 
                              (x.cell_iname.isin(res_lines))].index.values 

        with torch.no_grad(): 
            # predict cell viability within "target" lines + sample
            x_sens = pred_expr[drug_idxs_sens]
            yhat_sens_dist = viab_model(x_sens.cpu())
            yhat_sens_rvs = yhat_sens_dist.sample((N,)) # (N, B)

            # predict cell viability within "background" lines + sample 
            x_res = pred_expr[drug_idxs_res]
            yhat_res_dist = viab_model(x_res.cpu())
            yhat_res_rvs = yhat_res_dist.sample((N,)) # (N, B)

        #               V--Target Mixture--V      V--Background Mixture--V
        diffs       = yhat_sens_rvs.mean(dim=1) - yhat_res_rvs.mean(dim=1)  # (N,)

        diff_mean   = diffs.mean()
        diff_var    = diffs.var()
        p_sens      = (diffs < 0).type(torch.float32).mean()

        res['pert_id_1'].append(row.pert_id_1)
        res['dose_um_1'].append(row.dose_um_1)
        res['pert_id_2'].append(row.pert_id_2)
        res['dose_um_2'].append(row.dose_um_2)
        res['diff_mean'].append(diff_mean.item())
        res['diff_var'].append(diff_var.item()) 
        res['p_sens'].append(p_sens.item())

    res = pd.DataFrame(res)

    if verbose: print()
    return res