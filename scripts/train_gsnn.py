import torch 
import argparse 
import uuid
import os 
import time
import numpy as np 
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

import sys 
sys.path.append('../.')
from src.models.GSNN import GSNN
from src.data.LincsDataset import LincsDataset
from src.models import utils 

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../processed_data/',
                        help="path to data directory")
    
    parser.add_argument("--out", type=str, default='../output/',
                        help="path to output directory")
    
    parser.add_argument("--batch", type=int, default=100,
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
    
    parser.add_argument("--channels", type=int, default=5,
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
    train_dataset = LincsDataset(root=f'{args.data}', sig_ids=train_ids)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True)

    test_ids = np.load(f'{args.data}/test_obs.npy', allow_pickle=True)
    test_dataset = LincsDataset(root=f'{args.data}', sig_ids=test_ids)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True)

    if args.randomize: 
        data.edge_index = utils.randomize(data)

    torch.save(data, out_dir + '/Data.pt')

    model = GSNN(edge_index=data.edge_index, 
             channels=args.channels, 
             input_node_mask=data.input_node_mask, 
             output_node_mask=data.output_node_mask, 
             layers=args.layers, 
             dropout=args.dropout,
             residual=not args.no_residual,
             nonlin=torch.nn.ELU).to(device)
    
    n_params = sum([p.numel() for p in model.parameters()])
    print('# params', n_params)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        big_tic = time.time()
        model = model.train()
        losses = []
        for i,(x, y, sig_id) in enumerate(train_loader): 
            tic = time.time()
            optim.zero_grad() 

            yhat = model(x.to(device))[:, data.output_node_mask]
            y = y.to(device).squeeze(-1)[:, data.output_node_mask]

            loss = crit(yhat, y)
            loss.backward()
            optim.step()

            with torch.no_grad(): 

                yhat = yhat.detach().cpu().numpy() 
                y = y.detach().cpu().numpy() 
                
                r2 = r2_score(y, yhat, multioutput='variance_weighted')
                r_flat = np.corrcoef(y.ravel(), yhat.ravel())[0,1]
                losses.append(loss.item())

                print(f'epoch: {epoch} || batch: {i}/{len(train_loader)} || loss: {loss.item():.3f} || r2: {r2:.3f} || r (flat): {r_flat:.2f} || elapsed: {(time.time() - tic):.2f} s' , end='\r')
        
        loss_train = np.mean(losses)

        y,yhat,sig_ids = utils.predict_gsnn(test_loader, model, data, device)
        r2_test = r2_score(y, yhat, multioutput='variance_weighted')
        r_flat_test = np.corrcoef(y.ravel(), yhat.ravel())[0,1]

        torch.save(model, out_dir + '/model.pt')

        print(f'Epoch: {epoch} || loss (train): {loss_train:.3f} || r2 (test): {r2_test:.2f} || r flat (test): {r_flat_test:.2f} || elapsed: {(time.time() - big_tic)/60:.2f} min')