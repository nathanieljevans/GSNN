import torch 
import argparse 
import uuid
import os 
import time
import numpy as np 
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import pandas as pd 

import sys 
sys.path.append('../.')
from src.models.GSNN import GSNN
from src.data.LincsDataset import LincsDataset
from src.models import utils 

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../processed_data/',
                        help="path to data directory")
    
    parser.add_argument("--siginfo", type=str, default='../../data/',
                        help="path to siginfo directory")
    
    parser.add_argument("--out", type=str, default='../output/',
                        help="path to output directory")
    
    parser.add_argument("--batch", type=int, default=50,
                        help="training batch size")
    
    parser.add_argument("--workers", type=int, default=12,
                        help="number of workers for dataloaders")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    
    parser.add_argument("--randomize", action='store_true',
                        help="whether to randomize the structural graph")
    
    parser.add_argument("--no_residual", action='store_true',
                        help="disable residual connections")
    
    parser.add_argument("--ignore_cuda", action='store_true',
                        help="whether to ignore available cuda GPU")
    
    parser.add_argument("--cell_agnostic", action='store_true',
                        help="if true, will set all non-drug input nodes to zero, removing all contextual information")
    
    parser.add_argument("--channels", type=int, default=3,
                        help="number of channels for each function node")
    
    parser.add_argument("--layers", type=int, default=10,
                        help="number of layers of message passing")
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="")
    
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    
    parser.add_argument("--wd", type=float, default=0.,
                        help="weight decay")
    
    parser.add_argument("--nonlin", type=str, default='elu',
                        help="non-linearity function to use [relu, elu, mish, softplus, tanh]")
    
    parser.add_argument("--optim", type=str, default='adam',
                        help="optimization algorithm to use [adam, sgd, rmsprop]")
    
    parser.add_argument("--crit", type=str, default='mse',
                        help="loss function (criteria) to use [mse, huber]")
    
    parser.add_argument("--sched", type=str, default='none',
                        help="lr scheduler [onecycle, cosine, none]")
    
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping by norm")
    
    parser.add_argument("--save_every", type=int, default=20,
                        help="saves model results and weights every X epochs")

    parser.add_argument("--dropout_type", type=str, default='edgewise',
                        help="type of dropout to perform: 'edgewise' will drop edges across all layers (e.g., same edge removed for all layers). \
                            'layerwise' will dropout unique edges from each layer. 'nodewise' will dropout all edges from a given node.")                   
    
    parser.add_argument("--norm", type=str, default='layer',
                        help="norm type [edge-batch, layer-batch, layer, group]")
    
    parser.add_argument("--no_bias", action='store_true',
                        help="whether to include a bias term in the function node neural networks.")
    
    parser.add_argument("--stochastic_depth", action='store_true',
                        help="whether to use the `stochastic depth` technique during training (https://arxiv.org/abs/1603.09382)")
    
    parser.add_argument("--share_layers", action='store_true',
                        help="whether to use share function node parameters across layers.")
    
    parser.add_argument("--fix_hidden_channels", action='store_true',
                        help="if true, all function nodes will have `channels` hidden units, otherwise, the number of hidden channels of each function node will depend on the function node degree (e.g., nodes with more inputs/output will have more hidden channels.).")
    
    parser.add_argument("--null_inflation", type=float, default=0.05,
                        help="proportion of training dataset that should be inflated with 'null' obs, e.g., zero-drug, zero-output")
    
    args = parser.parse_args()

    # checks 
    assert args.dropout_type in ['edgewise', 'layerwise', 'nodewise'], f'unexpected argument `dropout_type` : {args.dropout_type}'
    assert args.norm in ['layer', 'edge-batch', 'layer-batch', 'group'], f'unexpected argument `norm` : {args.norm}'

    return args
    


if __name__ == '__main__': 

    time0 = time.time() 

    # get args 
    args = get_args()
    args.model = 'gsnn'

    print()
    print(args)
    print()

    # create uuid 
    uid = str(uuid.uuid4())
    args.uid = uid
    print('UID:', uid)
    out_dir = f'{args.out}/{uid}'
    if not os.path.exists(args.out): 
        os.mkdir(args.out)
    os.mkdir(out_dir)

    with open(f'{out_dir}/args.log', 'w') as f: 
        f.write(str(args))

    if torch.cuda.is_available() and not args.ignore_cuda: 
        device = 'cuda'
    else: 
        device = 'cpu'
    args.device = device
    print('using device:', device)

    data = torch.load(f'{args.data}/Data.pt')

    train_ids = np.load(f'{args.data}/train_obs.npy', allow_pickle=True)
    train_dataset = LincsDataset(root=f'{args.data}', sig_ids=train_ids, data=data, null_inflation=args.null_inflation)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True)

    test_ids = np.load(f'{args.data}/test_obs.npy', allow_pickle=True)
    test_dataset = LincsDataset(root=f'{args.data}', sig_ids=test_ids, data=data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)

    val_ids = np.load(f'{args.data}/val_obs.npy', allow_pickle=True)
    val_dataset = LincsDataset(root=f'{args.data}', sig_ids=val_ids, data=data)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)

    if args.randomize: data.edge_index = utils.randomize(data)

    torch.save(data, out_dir + '/Data.pt')

    model = GSNN(edge_index=data.edge_index, 
             channels=args.channels, 
             input_node_mask=data.input_node_mask, 
             output_node_mask=data.output_node_mask, 
             layers=args.layers, 
             dropout=args.dropout,
             residual=not args.no_residual,
             nonlin=utils.get_activation(args.nonlin),
             dropout_type=args.dropout_type,
             norm=args.norm,
             bias=~args.no_bias,
             stochastic_depth=args.stochastic_depth,
             share_layers=args.share_layers,
             fix_hidden_channels=args.fix_hidden_channels).to(device)
    
    n_params = sum([p.numel() for p in model.parameters()])
    args.n_params = n_params
    print('# params', n_params)

    optim = utils.get_optim(args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = utils.get_crit(args.crit)()
    scheduler = utils.get_scheduler(optim, args, train_loader)
    logger = utils.TBLogger(out_dir + '/tb/')

    if args.cell_agnostic: 
        # get indices of all non-drug input nodes 
        omic_input_idxs = (data.input_node_mask & torch.tensor(['DRUG_' not in x for x in data.node_names], dtype=torch.bool)).nonzero(as_tuple=True)[0]
        
    siginfo = pd.read_csv(f'{args.siginfo}/siginfo_beta.txt', sep='\t', low_memory=False)

    for epoch in range(1, args.epochs+1):
        big_tic = time.time()
        model = model.train()
        losses = []
        for i,(x, y, sig_id) in enumerate(train_loader): 

            if len(sig_id) == 1: continue # BUG workaround: if batch only has 1 obs it fails

            if args.cell_agnostic: x[:, omic_input_idxs] = 0.

            tic = time.time()
            optim.zero_grad() 

            yhat = model(x.to(device))[:, data.output_node_mask]
            y = y.to(device).squeeze(-1)[:, data.output_node_mask]

            loss = crit(yhat, y)

            loss.backward()
            if args.clip_grad is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optim.step()
            if scheduler is not None: scheduler.step()

            with torch.no_grad(): 

                yhat = yhat.detach().cpu().numpy() 
                y = y.detach().cpu().numpy() 
                
                r2 = r2_score(y, yhat, multioutput='variance_weighted')
                r_flat = np.corrcoef(y.ravel(), yhat.ravel())[0,1]
                losses.append(loss.item())

                print(f'epoch: {epoch} || batch: {i}/{len(train_loader)} || loss: {loss.item():.3f} || r2: {r2:.3f} || r (flat): {r_flat:.2f} || elapsed: {(time.time() - tic):.2f} s' , end='\r')
        
        loss_train = np.mean(losses)

        y,yhat,sig_ids = utils.predict_gsnn(val_loader, model, data, device)
        r_cell, r_drug, r_dose = utils._get_regressed_metrics(y, yhat, sig_ids, siginfo)
        r2_val = r2_score(y, yhat, multioutput='variance_weighted')
        r_flat_val = np.corrcoef(y.ravel(), yhat.ravel())[0,1]

        logger.log(epoch, loss_train, r2_val, r_flat_val)

        if (epoch % args.save_every == 0): 

            time_elapsed = time.time() - time0
            # add test results + hparams
            logger.add_hparam_results(args=args, 
                                    model=model, 
                                    data=data, 
                                    device=device, 
                                    test_loader=test_loader, 
                                    val_loader=val_loader, 
                                    siginfo=siginfo,
                                    time_elapsed=time_elapsed,
                                    epoch=epoch)

            torch.save(model, out_dir + f'/model-{epoch}.pt')

        print(f'Epoch: {epoch} || loss (train): {loss_train:.3f} || r2 (val): {r2_val:.2f} || r flat (val): {r_flat_val:.2f} || r cell: {r_cell:.2f} || r drug: {r_drug:.2f} || elapsed: {(time.time() - big_tic)/60:.2f} min')
