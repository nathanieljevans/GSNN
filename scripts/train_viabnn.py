'''
Train a cell viability predictor (nn)

NOTE: The drugs that are not well predicted by the GSNN, similarly have poor prediction of cell viability; however, training the cell viab. predictor on all drugs tends to 
lead to better performing models. 

example usage: 

```
(gsnn) $ python train_viabnn.py --exp ../output/exp1-na/ --uid GSNN1/93e89891-a363-44fa-9bd9-8ea067b04130 --model model-45.pt
```

The resulting `ViabPredictor` model takes pertrubed expression as input (outputs of the GSNN model) and predicts cell viability or log-fold change (depending on user choice). 
'''

import torch 
import argparse
import numpy as np 
import pickle as pkl 
from torch.utils.data import DataLoader
import pandas as pd 
from sklearn.metrics import r2_score 
from matplotlib import pyplot as plt 

import sys 
sys.path.append('../.')
from src.data.PrismDataset import PrismDataset
from src.models.utils import predict_gsnn
from src.models.NN import NN

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../../data/',
                        help="path to data directory")
    
    parser.add_argument("--exp", type=str, default=None,
                        help="path to experiment directory; should have sub-directory `/proc/`")
    
    parser.add_argument("--uid", type=str, default=None,
                        help="GSNN model `uid` string; NOTE: must include any subdirs from exp->uid")
    
    parser.add_argument("--model", type=str, default='model-100.pt',
                        help="model name")
    
    parser.add_argument("--target", type=str, default='log_fold_change',
                        help="cell viability target: 'log_fold_change' or 'cell_viability'")
    
    parser.add_argument("--gsnn_batch", type=int, default=512,
                        help="GSNN batch size to use")
    
    parser.add_argument("--batch", type=int, default=1024,
                        help="batch size to use while training cell viab predictor")
    
    parser.add_argument("--min_gsnn_gene_corr", type=float, default=0.25,
                        help="minimum GSNN test set performance (pearson corr.) to use as input to cell viab predictor")
    
    parser.add_argument("--epochs", type=int, default=250,
                        help="number of epochs to train")
    
    parser.add_argument("--channels", type=int, default=100,
                        help="cell viab predictor nn hidden channels")
    
    parser.add_argument("--layers", type=int, default=2,
                        help="cell viab predictor nn layers")
    
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout rate to use while training cell viab predictor")
    
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate to use for cell viab predictor")
    
    return parser.parse_args()


def get_viab_predictor_drug_perf(pred, train_sigids, test_sigids, val_sigids): 
    '''
    '''
    pred.eval()

    train_drugs = np.array([x.split('::')[1] for x in np.array(train_sigids)]) #)[best_inferred_drugs_train_idxs]])
    test_drugs = np.array([x.split('::')[1] for x in np.array(test_sigids)]) #)[best_inferred_drugs_test_idxs]])
    val_drugs = np.array([x.split('::')[1] for x in np.array(val_sigids)]) #)[best_inferred_drugs_val_idxs]])

    drug_perf = {}
    for drug in np.unique(train_drugs): 
        drug_idxs = (train_drugs == drug).nonzero()[0]
        _yhat = pred(x_train2[drug_idxs]).squeeze()
        _y = y_train[drug_idxs]
        drug_perf[drug] = r2_score(_y.detach().cpu().numpy(), _yhat.detach().cpu().numpy())

    drug_perf = pd.DataFrame(drug_perf, index=[0]).T.reset_index().rename({'index': 'pert_id', 0:'r2_train'}, axis=1)

    drug_perf2 = {}
    for drug in np.unique(test_drugs): 
        drug_idxs = (test_drugs == drug).nonzero()[0]
        _yhat = pred(x_test2[drug_idxs]).squeeze()
        _y = y_test[drug_idxs]
        drug_perf2[drug] = r2_score(_y.detach().cpu().numpy(), _yhat.detach().cpu().numpy())

    drug_perf2 = pd.DataFrame(drug_perf2, index=[0]).T.reset_index().rename({'index': 'pert_id', 0:'r2_test'}, axis=1)

    drug_perf3 = {}
    for drug in np.unique(val_drugs): 
        drug_idxs = (val_drugs == drug).nonzero()[0]
        _yhat = pred(x_val2[drug_idxs]).squeeze()
        _y = y_val[drug_idxs]
        drug_perf3[drug] = r2_score(_y.detach().cpu().numpy(), _yhat.detach().cpu().numpy())

    drug_perf3 = pd.DataFrame(drug_perf3, index=[0]).T.reset_index().rename({'index': 'pert_id', 0:'r2_val'}, axis=1)

    drug_perf = drug_perf.merge(drug_perf2, on='pert_id', how='left').merge(drug_perf3, on='pert_id', how='left').sort_values(by='r2_test', ascending=False)
    
    return drug_perf


if __name__ == '__main__': 

    # get args 
    args = get_args()

    print()
    print(args)
    print()

    with open(f'{args.exp}/{args.uid}/cellviab_args.log', 'w') as f: 
        f.write(str(args))

    data = torch.load(f'{args.exp}/proc/Data.pt')
    gsnn = torch.load(f'{args.exp}/{args.uid}/{args.model}').eval()

    # predict yhat 
    prism_train_ids = np.load(f'{args.exp}/proc/prism_train_obs.npy', allow_pickle=True)
    prism_test_ids = np.load(f'{args.exp}/proc/prism_test_obs.npy', allow_pickle=True)
    prism_val_ids = np.load(f'{args.exp}/proc/prism_val_obs.npy', allow_pickle=True)

    # prism datasets 
    train_dataset = PrismDataset(root=f'{args.exp}/proc/', data=data, sig_ids=prism_train_ids, target=args.target)
    test_dataset = PrismDataset(root=f'{args.exp}/proc/', data=data, sig_ids=prism_test_ids, target=args.target)
    val_dataset = PrismDataset(root=f'{args.exp}/proc/', data=data, sig_ids=prism_val_ids, target=args.target)

    train_loader = DataLoader(train_dataset, batch_size=args.gsnn_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.gsnn_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.gsnn_batch)

    print('predicting GSNN(X_train) ...')
    y_train, x_pert_train, train_sigids = predict_gsnn(train_loader, gsnn, data, device='cuda')
    x_train = torch.FloatTensor(x_pert_train); y_train = torch.FloatTensor(y_train)
    print()
    print('predicting GSNN(X_val) ...')
    y_val, x_pert_val, val_sigids = predict_gsnn(val_loader, gsnn, data, device='cuda')
    x_val = torch.FloatTensor(x_pert_val); y_val = torch.FloatTensor(y_val)
    print()
    print('predicting GSNN(X_test) ...')
    y_test, x_pert_test, test_sigids = predict_gsnn(test_loader, gsnn, data, device='cuda')
    x_test = torch.FloatTensor(x_pert_test); y_test = torch.FloatTensor(y_test)
    print()
    print()

    r_gene_dict = pkl.load(open(f'{args.exp}/{args.uid}/r_gene_dict.pkl', 'rb'))
    r_gene = pd.DataFrame(r_gene_dict, index=[0]).T.reset_index().rename({'index':'gene', 0:'r'}, axis=1).sort_values('r', ascending=False)

    # LINCS (pertrubed expression) genes that are well predicted by the GSNN model 
    # note: must use `eval.py` to create these dictionaries 
    best_inferred_genes = r_gene[lambda x: x.r > args.min_gsnn_gene_corr].gene.values

    print('# of GSNN "predictable" genes:', len(best_inferred_genes))

    best_inferred_genes_idx = np.isin(data.node_names[data.output_node_mask], np.array(['LINCS__' + x for x in best_inferred_genes])).nonzero()[0]

    x_train2 = x_train[:, best_inferred_genes_idx]
    x_test2 = x_test[:, best_inferred_genes_idx]
    x_val2 = x_val[:, best_inferred_genes_idx]

    mu = x_train2.mean(dim=0).unsqueeze(0)
    sigma = x_train2.std(dim=0).unsqueeze(0)

    x_train2 = (x_train2 - mu)/(sigma + 1e-12)
    x_test2 = (x_test2 - mu)/(sigma + 1e-12)
    x_val2 = (x_val2 - mu)/(sigma + 1e-12)

    print('train set:', x_train2.size(0))
    print('val set:', x_val2.size(0))
    print('test set:', x_test2.size(0))

    pred = NN(in_channels=x_train2.shape[1], hidden_channels=args.channels, layers=args.layers, out_channels=1, dropout=args.dropout, nonlin=torch.nn.ELU, out=None, norm=True)
    
    print('ViabPredictor Architecture:')
    print()
    print(pred)
    print()
    print() 

    optim = torch.optim.Adam(pred.parameters(), lr=args.lr) 
    crit = torch.nn.MSELoss() 

    for epoch in range(args.epochs): 

        for idx in torch.split(torch.arange(x_train2.size(0)), args.batch):
            optim.zero_grad()
            pred.train()
            yhat_train = pred(x_train2[idx]).squeeze()
            loss = crit(yhat_train, y_train[idx] )
            loss.backward()
            optim.step()

        pred.eval() 
        yhat_train = pred(x_train2).squeeze()
        r_train = np.corrcoef(yhat_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())[0,1]
        yhat_val = pred(x_val2).squeeze()
        r2_val = r2_score(y_val.detach().cpu().numpy(), yhat_val.detach().cpu().numpy())
        r_val = np.corrcoef(y_val.detach().cpu().numpy(), yhat_val.detach().cpu().numpy())[0,1]

        print(f'epoch: {epoch} | train r: {r_train:.2f} | val r2: {r2_val:.2f} | val r: {r_val:.2f}', end='\r')

    torch.save(pred, f'{args.exp}/{args.uid}/ViabPredictor.pt')

    # save the meta data 
    meta_dict = {'best_inferred_genes':best_inferred_genes, 
                 'best_inferred_genes_idxs':best_inferred_genes_idx,
                 'mu':mu,
                 'sigma':sigma}
    
    pkl.dump(meta_dict, open(f'{args.exp}/{args.uid}/ViabPredictor_metadict.pkl', 'wb'))

    print('evaluating ViabPredictor performance...')

    drug_perf = get_viab_predictor_drug_perf(pred, train_sigids, test_sigids, val_sigids)

    # load GSNN drug performances for comparison
    r_drug_dict = pkl.load(open('../output/exp1-na/GSNN1/93e89891-a363-44fa-9bd9-8ea067b04130/r_drug_dict.pkl', 'rb'))
    r_drug = pd.DataFrame(r_drug_dict, index=[0]).T.reset_index().rename({'index':'pert_id', 0:'gsnn_r'}, axis=1).sort_values('gsnn_r', ascending=False)
    df = r_drug.merge(drug_perf, on='pert_id', how='left')

    df.to_csv(f'{args.exp}/{args.uid}/ViabPredictorDrugPerf.csv')

    plt.figure()
    plt.plot(df.gsnn_r, df.r2_train, 'r.', label=f'PRISM-TRAIN corr: {df[lambda x: (x.gsnn_r > 0) & (x.r2_train > 0)][["gsnn_r", "r2_train"]].corr(method="spearman").values[0,1]:.2f}')
    plt.plot(df.gsnn_r, df.r2_test, 'b.', label=f'PRISM-TEST corr: {df[lambda x: (x.gsnn_r > 0) & (x.r2_test > 0)][["gsnn_r", "r2_test"]].corr(method="spearman").values[0,1]:.2f}')
    plt.plot(df.gsnn_r, df.r2_val, 'g.', label=f'PRISM-VAL corr: {df[lambda x: (x.gsnn_r > 0) & (x.r2_val > 0)][["gsnn_r", "r2_val"]].corr(method="spearman").values[0,1]:.2f}')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot((0,1), (0,1), 'k--')
    plt.xlabel('GSNN Perf. (r) (across all LINCS)')
    plt.ylabel('Viab. Pred. Perf. (r2)')
    plt.title('GSNN vs ViabPred perf. (by drug)')
    plt.legend()
    plt.savefig(f'{args.exp}/{args.uid}/ViabPredPerf_vs_GSNNPerf_by_drug.png')

    print('Top 25 best-performing drugs: ')
    print(df.head(25))
    print()

    # since train/test/val is split by cell line it doesn't make sense to evaluate within cell line and data split, so we will combine them all together and then eval by cell line. 
    x = torch.cat((x_train2, x_val2, x_test2), dim=0)
    y = torch.cat((y_train, y_val, y_test), dim=-1)
    sigids = train_sigids + val_sigids + test_sigids 
    inames = np.array([sid.split('::')[0] for sid in sigids])
    cell_perf = {} 
    for iname in inames: 
        idxs = (iname == inames).nonzero()[0]
        cell_perf[iname] = r2_score(y[idxs].detach().cpu().numpy(), pred(x[idxs]).squeeze().detach().cpu().numpy())

    print(cell_perf)

    pkl.dump(cell_perf, open(f'{args.exp}/{args.uid}/viab_cell_perf_dict.pkl', 'wb'))

    print('complete.')
