
import torch
from gsnn.models.NN import NN
from gsnn.ot.utils import freeze_, unfreeze_
from gsnn.models.GSNN import GSNN
import numpy as np
from geomloss import SamplesLoss    

class SHD(): 
    '''optimal transport with wasserstein approximation via sinkhorn distance'''

    def __init__(self, args, data, sampler): 

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.args = args
        self.data = data

        if args.T_arch == 'gsnn': 
            self.T    = GSNN(edge_index_dict                 = data.edge_index_dict, 
                            node_names_dict                 = data.node_names_dict,
                            channels                        = args.channels, 
                            layers                          = args.layers, 
                            dropout                         = args.dropout,
                            residual                        = True,
                            nonlin                          = torch.nn.GELU,
                            bias                            = True,
                            share_layers                    = False,
                            add_function_self_edges         = True,
                            norm                            = 'layer',
                            checkpoint                      = args.checkpoint).to(self.device)
        elif args.T_arch == 'nn':
            self.T = NN(in_channels          = data.node_names_dict['input'].shape[0],
                        out_channels         = data.node_names_dict['output'].shape[0],
                        hidden_channels      = args.channels, 
                        layers               = args.layers,
                        dropout              = args.dropout,
                        norm                 = torch.nn.LayerNorm).to(self.device)
        else:
            raise ValueError('args.T_arch must be one of gsnn, nn')
        
        self.t_optim = torch.optim.Adam(self.T.parameters(), lr=args.T_lr, weight_decay=args.wd)

        self.crit = SamplesLoss('sinkhorn', p=2, blur=args.blur, reach=args.reach, debias=True, scaling=args.scaling)

    def step(self, sampler): 
        
        t_loss = self.optimize_T(sampler)

        return t_loss
   
    def optimize_T(self, sampler):

        device = self.device
        args = self.args 
        self.T.train()
        unfreeze_(self.T)
        
        ii = 0
        losses = []
        for i in torch.randperm(len(sampler)).detach().cpu().numpy().tolist(): 
            print(f'optimizing T... [{ii}/{len(sampler)}]', end='\r'); ii+=1
            X,y, x_cell, x_drug, y0 = sampler.sample_(i, batch_size=args.batch_size, ret_all_y=True)
            X = X.to(device); y=y.to(device); x_cell = x_cell.to(device); x_drug = x_drug.to(device); y0 = y0.to(device)

            self.t_optim.zero_grad()
            yhat = self.T(X) + y0 
            loss = self.crit(yhat, y)
            loss.backward()
            self.t_optim.step()
            losses.append(loss.item())

        return np.mean(losses)


    def state_dict(self): 
        return {'T':self.T}
    
    def get_T(self): 
        return self.T   
    