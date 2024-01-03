'''
Train a cell viability predictor (nn)

NOTE: The drugs that are not well predicted by the GSNN, similarly have poor prediction of cell viability; however, training the cell viab. predictor on all drugs tends to 
lead to better performing models. 

example usage: 

```
(gsnn) $ python train_viabnn.py --data ../../output/ --proc ../output/SignalTransduction/proc/ --fold ../output/SignalTransduction/FOLD-1/ --model_dir ../output/SignalTransduction/FOLD-1/GSNN/c82456f4-6c84-4100-a83a-5d6d6d80b94d --model_name model-100.pt
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
import os 

import sys 
sys.path.append('../.')
from src.data.PrismDataset import PrismDataset
from src.models.utils import predict_gsnn
from src.models.GSNN import GSNN 
from src.models.NN import NN 
from src.uncertainty.NNEnsemble import NNEnsemble

from src.uncertainty.DistNN import DistNN
from src.uncertainty.utils import root_mean_squared_picp_error
torch.multiprocessing.set_sharing_strategy('file_system')


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../../data/',
                        help="path to data directory")
    
    parser.add_argument("--proc", type=str, default=None,
                        help="path to processed data dir")
    
    parser.add_argument("--fold", type=str, default=None,
                        help="path to fold directory; must contain 'prism_XXX_ids.npy")
    
    parser.add_argument("--model_dir", type=str, default=None,
                        help="directory containing the model")
    
    parser.add_argument("--model_name", type=str, default='model-100.pt',
                        help="model name")
    
    parser.add_argument("--target", type=str, default='cell_viability',
                        help="cell viability target: 'log_fold_change' or 'cell_viability'")
    
    parser.add_argument("--gsnn_batch", type=int, default=256,
                        help="GSNN batch size to use")
    
    parser.add_argument("--batch", type=int, default=512,
                        help="batch size to use while training cell viab predictor")
    
    parser.add_argument("--epochs", type=int, default=250,
                        help="number of epochs to train")
    
    parser.add_argument("--channels", type=int, default=100,
                        help="cell viab predictor nn hidden channels")
    
    parser.add_argument("--layers", type=int, default=1,
                        help="cell viab predictor nn layers")
    
    parser.add_argument("--workers", type=int, default=12,
                        help="number of workers to use for the data loader when predicting transcriptional activations")
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout rate to use while training cell viab predictor")
    
    parser.add_argument("--min_feature_variance", type=float, default=0.,
                        help="input features with variance less than this value will be excluded")
    
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate to use for cell viab predictor")
    
    parser.add_argument('--recompute_activations', action='store_true', 
                        help='whether to recompute and overwrite the transcriptional model activations even if they are available in `model_dir`')
    
    parser.add_argument('--use_last_activation', action='store_true', 
                        help='wheter to use the last activation of the transcriptional model or the predicted perturbed expression')
    
    parser.add_argument('--fgsm', action='store_true', 
                        help='whether to use the `fast gradient sign method` for advesarial data augmentation during training.')
    
    parser.add_argument("--N", type=int, default=1,
                        help="number of models to include in the ensemble")
    
    parser.add_argument("--target_distribution", type=str, default='beta',
                        help="options: ['gaussian', 'beta', 'gamma']")
    
    parser.add_argument("--use_ensemble_mixture", action='store_true', 
                        help='wheter to use the a mixture model of all ensemble predictions or to use an aggregate single distribution.')
    
    return parser.parse_args()


def get_last_edge_activations(loader, model, data, device='cuda', return_last_activation=False): 

    if return_last_activation:
        print('using the last activation of the transcriptional model as input to the viability predictor.')
    else: 
        print('using the predicted perturbed expression as input to the viability model.')

    if isinstance(model, GSNN): 
        print('transcriptional model is type: GSNN')
        if return_last_activation: 
            new_forward = lambda x: model(x.to(device), return_last_activation=return_last_activation)
        else: 
            new_forward = lambda x: model(x.to(device), return_last_activation=return_last_activation)[:, data.output_node_mask]
    elif isinstance(model, NN): 
        print('transcriptional model is type: NN')
        if return_last_activation: 
            raise NotImplementedError()
        else: 
            new_forward = lambda x: model(x[:, data.input_node_mask].squeeze(-1).to(device))

    else: 
        raise ValueError('unrecognized model type! should be either a GSNN or NN. ')
    
    out = [] 
    ys = [] 
    sig_ids = []
    with torch.no_grad(): 
        model.eval() 
        model = model.to(device)
        for ii, (x, y, sig_ids) in enumerate(loader): 
            print(f'\tprogress: {ii}/{len(loader)}', end='\r')
            xx = new_forward(x).detach().cpu()
            out.append(xx)
            ys.append(y)
            sig_ids+=sig_ids
    print()

    out = torch.cat(out, dim=0)
    y = torch.cat(ys, dim=0).view(-1)
    return y.type(torch.float32), out.type(torch.float32), sig_ids



def train_viabnn(args, x_train, y_train, x_val, y_val, device='cuda'): 


    # normalize viab inputs 
    mu = x_train.mean(dim=0, keepdim=True)
    sigma = x_train.std(dim=0, keepdim=True)

    print('input feature variance quantiles (every 0.1):', np.quantile(sigma.detach().cpu().numpy(), q=np.linspace(0,1,10)))

    # additional transforms? 
    # input selections ? variance filter? 
    input_idxs = (sigma.view(-1) > args.min_feature_variance).nonzero(as_tuple=True)[0]

    pred = DistNN(in_channels       = len(input_idxs), 
                      hidden_channels   = args.channels, 
                      layers            = args.layers, 
                      dropout           = args.dropout, 
                      nonlin            = torch.nn.ELU, 
                      norm              = torch.nn.LayerNorm,
                      transform         = (mu, sigma),
                      input_idxs        = input_idxs,
                      dist              = args.target_distribution).to(device)
    
    print()
    print('ViabPredictor Architecture:')
    print()
    print(pred)
    print()
    print() 

    n_params = sum([p.numel() for p in pred.parameters()])
    print('# params:', n_params)

    optim = torch.optim.Adam(pred.parameters(), lr=args.lr) 

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    epsilon = 0.01 # data is normal 
    if args.fgsm: print('FGSM eps:', epsilon)

    for epoch in range(args.epochs): 

        for idx in torch.split(torch.arange(x_train.size(0)), args.batch):
            optim.zero_grad()
            pred.train()
            xx = x_train[idx]
            if args.fgsm: xx.requires_grad = True
            pred_dist = pred(xx)
            y_true = y_train[idx]
            loss = -pred_dist.log_prob(y_true).mean()
            loss.backward()

            # advesarial data augmentation 
            # fast gradient sign method
            if args.fgsm: 
                xx2 = (xx + epsilon * xx.grad.data.sign()).detach() 
                pred_dist = pred(xx2)
                loss = -pred_dist.log_prob(y_true).mean()
                loss.backward()

            optim.step()

        with torch.no_grad(): 
            pred.eval() 
            pred_dist_train = pred(x_train)
            NLL_train = -pred_dist_train.log_prob(y_train).mean()

            pred_dist_val = pred(x_val)
            NLL_val = -pred_dist_val.log_prob(y_val).mean()
            r2_val = r2_score(y_val.detach().cpu().numpy(), pred_dist_val.mean.detach().cpu().numpy())
            rmspicpe = -1# root_mean_squared_picp_error(pred_dist_val, y_val)

        print(f'epoch: {epoch} | train NLL: {NLL_train:.3f} | val NLL: {NLL_val:.3f} | val. r2: {r2_val:.2f} | val. rmspicpe: {rmspicpe :.3f}', end='\r')

    return pred


if __name__ == '__main__': 

    # get args 
    args = get_args()

    print()
    print(args)
    print()

    with open(f'{args.model_dir}/cellviab_args.log', 'w') as f: 
        f.write(str(args))

    data = torch.load(f'{args.model_dir}/Data.pt')
    model = torch.load(f'{args.model_dir}/{args.model_name}').eval()

    # predict yhat 
    prism_train_ids = np.load(f'{args.fold}/prism_train_obs.npy', allow_pickle=True)
    prism_test_ids  = np.load(f'{args.fold}/prism_test_obs.npy', allow_pickle=True)
    prism_val_ids   = np.load(f'{args.fold}/prism_val_obs.npy', allow_pickle=True)

    # prism datasets 
    train_dataset   = PrismDataset(root=args.proc, data=data, sig_ids=prism_train_ids, target=args.target)
    test_dataset    = PrismDataset(root=args.proc, data=data, sig_ids=prism_test_ids, target=args.target)
    val_dataset     = PrismDataset(root=args.proc, data=data, sig_ids=prism_val_ids, target=args.target)

    train_loader    = DataLoader(train_dataset, batch_size=args.gsnn_batch, num_workers=args.workers)
    test_loader     = DataLoader(test_dataset, batch_size=args.gsnn_batch, num_workers=args.workers)
    val_loader      = DataLoader(val_dataset, batch_size=args.gsnn_batch, num_workers=args.workers)

    if os.path.exists(f'{args.model_dir}/activations_dict.pkl') and (not args.recompute_activations):
        print('activations already exist! loading from disk.')
        with open(f'{args.model_dir}/activations_dict.pkl', 'rb') as f: 
            activations_dict = pkl.load(f)

        if activations_dict['target'] != args.target: 
            raise ValueError('Pre-computed activations have wrong target. Use `--recompute_activations` flag with new target. ')

        y_train = activations_dict['train']['y_viab_target']
        x_train = activations_dict['train']['activations']
        train_sigids = activations_dict['train']['sig_ids']

        y_val = activations_dict['val']['y_viab_target']
        x_val = activations_dict['val']['activations']
        val_sigids = activations_dict['val']['sig_ids']

        y_test = activations_dict['test']['y_viab_target']
        x_test = activations_dict['test']['activations']
        test_sigids = activations_dict['test']['sig_ids']
    else:  
        print('predicting model(X_train) ...')
        y_train, x_train, train_sigids = get_last_edge_activations(train_loader, model, data, return_last_activation=args.use_last_activation)
        print()
        print('predicting model(X_val) ...')
        y_val, x_val, val_sigids = get_last_edge_activations(val_loader, model, data, return_last_activation=args.use_last_activation)
        print()
        print('predicting model(X_test) ...')
        y_test, x_test, test_sigids = get_last_edge_activations(test_loader, model, data, return_last_activation=args.use_last_activation)
        print()
        print()

        activations_dict = {'train':{'activations': x_train, 
                                     'y_viab_target':y_train,
                                     'sig_ids':train_sigids},
                            'val':{'activations': x_val, 
                                   'y_viab_target':y_val,
                                   'sig_ids':val_sigids},
                            'test':{'activations': x_test, 
                                    'y_viab_target':y_test,
                                    'sig_ids':test_sigids}}
        
        activations_dict['target'] = args.target
        
        with open(f'{args.model_dir}/activations_dict.pkl', 'wb') as f: 
            pkl.dump(activations_dict, f)

    x_train = x_train.squeeze(-1)
    x_val = x_val.squeeze(-1)
    x_test = x_test.squeeze(-1)

    if args.target == 'cell_viability':
        # beta target fails otherwise
        y_train = torch.clamp(y_train, 1e-6, 1-1e-6) # can't be 0 or 1
        y_val = torch.clamp(y_val, 1e-6, 1-1e-6) # can't be 0 or 1
        y_test = torch.clamp(y_test, 1e-6, 1-1e-6) # can't be 0 or 1

    print('train set:', x_train.size(0))
    print('val set:', x_val.size(0))
    print('test set:', x_test.size(0))

    preds = []
    for i in range(args.N):
        print()
        print()
        print(f'training ensemble model {i + 1}')
        preds.append(train_viabnn(args, x_train, y_train, x_val, y_val).cpu())
    
    
    ensemble = NNEnsemble(preds, use_mixture=args.use_ensemble_mixture)

    pred_mixture = ensemble(x_val)

    NLL_val = -pred_mixture.log_prob(y_val).mean().detach().cpu().item()
    #rmspicpe_val = root_mean_squared_picp_error(pred_mixture, y_val) # mixtures don't currently support icdf
    r2_val = r2_score(y_val.detach().cpu().numpy(), pred_mixture.mean.detach().cpu().numpy())

    print()
    print()
    print('Ensemble performance (validation set):')
    print('\tNLL: ', NLL_val)
    #print('\tRMSPICPE: ', rmspicpe_val)
    print('\tR2: ', r2_val)
    print()

    ensemble.args = args
    torch.save(ensemble, f'{args.model_dir}/ViabNNEnsemble.pt')

'''
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

def get_viab_predictor_drug_perf(pred, train_sigids, test_sigids, val_sigids): 

    pred.eval()

    train_drugs = np.array([x.split('::')[1] for x in np.array(train_sigids)])
    test_drugs = np.array([x.split('::')[1] for x in np.array(test_sigids)])
    val_drugs = np.array([x.split('::')[1] for x in np.array(val_sigids)])

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
'''