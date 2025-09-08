
import torch
from gsnn.models.NN import NN
from gsnn.ot.utils import freeze_, unfreeze_
from gsnn.models.GSNN import GSNN
import numpy as np
from geomloss import SamplesLoss    
from gsnn.models.AE import AE

class SHD(): 
    '''optimal transport with wasserstein approximation via sinkhorn distance'''

    def __init__(self, args, data): 

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.args = args
        self.data = data

        if args.arch == 'gsnn': 
            self.model    = GSNN(edge_index_dict                 = data.edge_index_dict, 
                            node_names_dict                 = data.node_names_dict,
                            channels                        = args.channels, 
                            layers                          = args.layers, 
                            dropout                         = args.dropout,
                            nonlin                          = torch.nn.ELU,
                            bias                            = True,
                            share_layers                    = False,
                            add_function_self_edges         = True,
                            norm                            = 'layer',
                            checkpoint                      = args.checkpoint).to(self.device)
            
        elif args.arch == 'nn':
            self.model = NN(in_channels          = len(data.node_names_dict['input']),
                        out_channels         = len(data.node_names_dict['output']),
                        hidden_channels      = args.channels, 
                        layers               = args.layers,
                        dropout              = args.dropout,
                        norm                 = torch.nn.LayerNorm).to(self.device)
            
        elif args.arch == 'ae':
            self.model = AE(data, hidden_channels=args.channels, latent_dim=args.latent_dim, out_channels=len(data.node_names_dict['output']),
                        layers=args.layers, dropout=args.dropout, nonlin=torch.nn.ELU, out=None, norm=torch.nn.LayerNorm).to(self.device)
        else:
            raise ValueError('args.T_arch must be one of gsnn, nn')
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

        self.crit = SamplesLoss('sinkhorn', p=args.p, blur=args.blur, reach=args.reach, debias=args.debias, scaling=args.scaling)

    def step(self, sampler): 
        
        device = self.device
        args = self.args 
        self.model.train()
        self.model.to(device)
        
        ii = 0
        losses = []
        for i in torch.randperm(len(sampler)).detach().cpu().numpy().tolist(): 
            self.optim.zero_grad()
            X,y, x_cell, x_drug, y0 = sampler.sample_(i, batch_size=args.batch_size, ret_all_y=True)
            X = X.to(device); y=y.to(device); x_cell = x_cell.to(device); x_drug = x_drug.to(device); y0 = y0.to(device)

            yhat = self.model(X) + y0.detach()
            loss = self.crit(yhat, y)
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
            print(f'optimizing T...{ii}/{len(sampler)} [loss:{loss.item():.3f}]', end='\r'); ii+=1

        return np.mean(losses)

    def state_dict(self): 
        return {'T':self.model}
    
    def get_T(self): 
        return self.model   
    