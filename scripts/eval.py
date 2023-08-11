'''
loads a model and predicts data split results, computes the correlation within gene, drug, cell and r-cell within gene. outputs are saved in uid folder as: 
r_gene_dict.pkl 
r_cell_gene_dict.pkl 
r_cell_dict.pkl 
r_drug_dict.pkl 

example usage: 
    $ python eval.py --data ../output/exp1/proc/ --uid ../output/nn/4a98c844-1be8-48c8-9ec8-03081cb7d391/ --model_name model-6.pt 
    $ python eval.py --data ../output/exp1/proc/ --uid ../output/gsnn/09461877-9965-4dd8-a668-97f77ef73717/ --model_name model-6.pt 
'''


import torch 
import pandas as pd 
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as DataLoader2
from sklearn.metrics import r2_score 
from matplotlib import pyplot as plt 
import argparse 
import pickle as pkl

import sys
sys.path.append('../')
from src.data.LincsDataset import LincsDataset
from src.data.pygLincsDataset import pygLincsDataset
from src.models.utils import predict_nn, predict_gsnn, predict_gnn, get_regressed_r, corr_score


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../processed_data/',
                        help="path to data directory")
    
    parser.add_argument("--siginfo", type=str, default='../../data/siginfo_beta.txt',
                        help="path to data directory")
    
    parser.add_argument("--uid", type=str,
                        help="path to uid directory")
    
    parser.add_argument("--model_name", type=str, default='model-100.pt',
                        help="model name")
    
    parser.add_argument("--workers", type=int, default=1,
                        help="num dataloader workers")
    
    parser.add_argument("--batch", type=int, default=50,
                        help="batch size")
    
    parser.add_argument("--ids", type=str, default='val_obs.npy',
                        help="the data split to evaluate [val_obs.npy, test_obs.npy, train_obs.npy]")
    
    parser.add_argument("--verbose", action='store_true',
                        help="print status to console")
    
    return parser.parse_args()


if __name__ == '__main__': 

    args = get_args()
    if args.verbose: print(args)

    # load model 
    model = torch.load(f'{args.uid}/{args.model_name}')
    typ = str(type(model)).split('.')[-1].strip("'>")

    if typ == 'NN':
        predict_fn = predict_nn 
        dataset = LincsDataset
        dataloader = DataLoader
    elif typ == 'GSNN': 
        predict_fn = predict_gsnn
        dataset = LincsDataset
        dataloader = DataLoader
    elif typ in ['GAT', 'GIN', 'GCN']:
        predict_fn = predict_gnn 
        dataset = pygLincsDataset
        dataloader = DataLoader2
    else: 
        raise ValueError(f'unrecognized model type: {args.model}')
    
    # load val data 
    val_ids = np.load(args.data + '/' + args.ids, allow_pickle=True)
    data = torch.load(args.uid + '/Data.pt')

    if typ in ['GAT', 'GIN', 'GCN']:
        val_dataset = dataset(root=args.data, sig_ids=val_ids, data=data)
    else: 
        val_dataset = dataset(root=args.data, sig_ids=val_ids)

    val_loader = dataloader(val_dataset, num_workers=args.workers, batch_size=args.batch) 

    if torch.cuda.is_available(): 
        device = 'cuda'
    else: 
        device = 'cpu'
    if args.verbose: print('device:', device)

    y_val, yhat_val, sig_ids_val = predict_fn(val_loader, model, data, device)

    if args.verbose: print('loading siginfo...')
    siginfo = pd.read_csv(args.siginfo, low_memory=False, sep='\t')

    if args.verbose: print('computing metrics...')
    # gene r2s 
    r_gene = corr_score(y_val, yhat_val, multioutput='raw_values')
    out_names = [x.split('__')[1] for x in data.node_names[data.output_node_mask]]
    r_gene_dict = {n:v for n,v in zip(out_names, r_gene)}

    r_cell_gene = get_regressed_r(y_val, yhat_val, sig_ids_val, vars=['pert_id', 'pert_dose'], data='../../data/', multioutput='raw_values', siginfo=siginfo)
    r_cell_gene_dict = {n:v for n,v in zip(out_names, r_cell_gene)}

    # drug perf 
    id_info = pd.DataFrame({'sig_id':sig_ids_val}).merge(siginfo, how='left', on='sig_id')

    r_drug_dict = {}
    for pid in id_info.pert_id.unique():
        mask = id_info.pert_id.values == pid
        yy = y_val[mask, :].ravel()
        yyhat = yhat_val[mask, :].ravel()

        r_drug_dict[pid] = corr_score(yy, yyhat)

    # drug perf 
    id_info = pd.DataFrame({'sig_id':sig_ids_val}).merge(siginfo, how='left', on='sig_id')

    r_cell_dict = {}
    for cid in id_info.cell_iname.unique():
        mask = id_info.cell_iname.values == cid
        yy = y_val[mask, :].ravel()
        yyhat = yhat_val[mask, :].ravel()

        r_cell_dict[cid] = corr_score(yy, yyhat)

    if args.verbose: print('saving to disk...')
    with open(f'{args.uid}/r_gene_dict.pkl', 'wb') as f: 
        pkl.dump(r_gene_dict, f)

    with open(f'{args.uid}/r_cell_gene_dict.pkl', 'wb') as f: 
        pkl.dump(r_cell_gene_dict, f)

    with open(f'{args.uid}/r_cell_dict.pkl', 'wb') as f: 
        pkl.dump(r_cell_dict, f)

    with open(f'{args.uid}/r_drug_dict.pkl', 'wb') as f: 
        pkl.dump(r_drug_dict, f)


    

    