import torch 
import argparse 
import uuid
import os 
import time
import numpy as np 
from sklearn.metrics import r2_score
import torch_geometric as pyg 

import warnings
warnings.filterwarnings("ignore")
'''
UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
'''

import sys 
sys.path.append('../.')
from src.data.pygLincsDataset import pygLincsDataset
from src.models import utils 

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../processed_data/',
                        help="path to data directory")
    
    parser.add_argument("--out", type=str, default='../output/',
                        help="path to output directory")
    
    parser.add_argument("--gnn", type=str, default='GCN',
                        help="GNN archtecture to use, supports: GCN, GAT, GIN")
    
    parser.add_argument("--batch", type=int, default=100,
                        help="training batch size")
    
    parser.add_argument("--workers", type=int, default=12,
                        help="number of workers for dataloaders")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs")
    
    parser.add_argument("--randomize", action='store_true',
                        help="whether to randomize the structural graph")
    
    parser.add_argument("--ignore_cuda", action='store_true',
                        help="whether to ignore available cuda GPU")
    
    parser.add_argument("--channels", type=int, default=32,
                        help="number of channels for each function node")
    
    parser.add_argument("--layers", type=int, default=10,
                        help="number of layers of message passing")
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="")
    
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    
    parser.add_argument("--wd", type=float, default=0.,
                        help="weight decay")
    
    return parser.parse_args()
    


if __name__ == '__main__': 

    # get args 
    args = get_args()

    print()
    print(args)
    print()

    # create uuid 
    uid = uuid.uuid4() 
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

    print('using device:', device)

    data = torch.load(f'{args.data}/Data.pt')

    train_ids = np.load(f'{args.data}/train_obs.npy', allow_pickle=True)
    train_dataset = pygLincsDataset(root=f'{args.data}', sig_ids=train_ids, data=data)
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True)

    test_ids = np.load(f'{args.data}/test_obs.npy', allow_pickle=True)
    test_dataset = pygLincsDataset(root=f'{args.data}', sig_ids=test_ids, data=data)
    test_loader = pyg.loader.DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True)

    if args.randomize: 
        data.edge_index = utils.randomize(data)

    torch.save(data, out_dir + '/Data.pt')

    if args.gnn == 'GAT': 
        GNN = pyg.nn.models.GAT
    elif args.gnn == 'GCN': 
        GNN = pyg.nn.models.GAT
    elif args.gnn == 'GIN': 
        GNN = pyg.nn.models.GIN
    else: 
        raise ValueError(f'unrecognized `gnn` value. Expected one of ["GAT" "GCN" "GIN"] but got: {args.gnn}')

    model = GNN(in_channels=1, 
                hidden_channels=args.channels, 
                num_layers=args.layers,
                out_channels=1, 
                dropout=args.dropout, 
                act='elu',
                act_first=False,
                act_kwargs=None,
                norm='batch',
                norm_kwargs=None,
                jk='cat').to(device)
    
    n_params = sum([p.numel() for p in model.parameters()])
    print('# params', n_params)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        big_tic = time.time()
        model = model.train()
        losses = []
        for i, batch in enumerate(train_loader): 
            tic = time.time()
            optim.zero_grad() 

            yhat = model(edge_index=batch.edge_index.to(device), x=batch.x.to(device))

            #  select output nodes
            yhat = yhat[batch.output_node_mask]
            y = batch.y.to(device)[batch.output_node_mask]

            loss = crit(yhat, y)
            loss.backward()
            optim.step()

            with torch.no_grad(): 

                B = len(batch.sig_id)

                yhat = yhat.view(B, -1).detach().cpu().numpy() 
                y = y.view(B, -1).detach().cpu().numpy() 
                
                r2 = r2_score(y, yhat, multioutput='variance_weighted')
                r_flat = np.corrcoef(y.ravel(), yhat.ravel())[0,1]
                losses.append(loss.item())

                print(f'epoch: {epoch} || batch: {i}/{len(train_loader)} || loss: {loss.item():.3f} || r2: {r2:.3f} || r (flat): {r_flat:.2f} || elapsed: {(time.time() - tic):.2f} s' , end='\r')
        
        loss_train = np.mean(losses)

        y,yhat,sig_ids = utils.predict_gnn(test_loader, model, data, device)
        r2_test = r2_score(y, yhat, multioutput='variance_weighted')
        r_flat_test = np.corrcoef(y.ravel(), yhat.ravel())[0,1]

        torch.save(model, out_dir + '/model.pt')

        print(f'Epoch: {epoch} || loss (train): {loss_train:.3f} || r2 (test): {r2_test:.2f} || r flat (test): {r_flat_test:.2f} || elapsed: {(time.time() - big_tic)/60:.2f} min')