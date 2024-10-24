
import torch
from gsnn.models.NN import NN
from gsnn.ot.utils import freeze_, unfreeze_
from gsnn.models.GSNN import GSNN

class NOT(): 
    '''neural optimal transport'''

    def __init__(self, args, data, sampler): 

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.args = args
        self.data = data

        n_inputs = sampler.n_cell_lines + sampler.n_drugs + len(data.X2Y0_idxs)
        self.f = NN(in_channels          = n_inputs,
                    hidden_channels     = args.f_channels, 
                    out_channels        = 1, 
                    layers              = args.f_layers, 
                    norm                = torch.nn.LayerNorm).to(self.device)

        self.f_optim = torch.optim.Adam(self.f.parameters(), lr=args.f_lr, weight_decay=args.wd)

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

    def step(self, sampler): 
        
        f_loss = self.optimize_f(sampler)
        t_loss = self.optimize_T(sampler)

        return t_loss + f_loss
    
    def optimize_f(self, sampler): 

        freeze_(self.T); unfreeze_(self.f)
        device = self.device
        args = self.args

        X,y, x_cell, x_drug, y0 = sampler.sample(batch_size=args.batch_size)
        X = X.to(device); y=y.to(device); x_cell = x_cell.to(device); x_drug = x_drug.to(device); y0 = y0.to(device)
        yf = torch.cat((x_cell, x_drug, y), dim=-1)
        
        ii = 0
        for i in torch.randperm(len(sampler)).detach().cpu().numpy().tolist(): 
            print(f'optimizing f... [{ii}/{len(sampler)}]', end='\r'); ii+=1
            X,y, x_cell, x_drug, y0 = sampler.sample_(i, batch_size=args.batch_size)
            X = X.to(device); y=y.to(device); x_cell = x_cell.to(device); x_drug = x_drug.to(device); y0 = y0.to(device)
            yf = torch.cat((x_cell, x_drug, y), dim=-1)
            for j in range(args.f_iters):

                with torch.no_grad():
                    yhat = self.T(X) + y0
                    yhatf = torch.cat((x_cell, x_drug, yhat), dim=-1)

                self.f_optim.zero_grad() 
                yhat_score = self.f(yhatf).view(-1)
                y_score = self.f(yf).view(-1)
                f_loss = yhat_score.mean() - y_score.mean()

                f_loss.backward()
                self.f_optim.step()

        return f_loss.item()


    def optimize_T(self, sampler):

        unfreeze_(self.T); freeze_(self.f)
        device = self.device
        args = self.args 
        
        # optimize T 
        ii = 0
        for i in torch.randperm(len(sampler)).detach().cpu().numpy().tolist(): 
            print(f'optimizing T... [{ii}/{len(sampler)}]', end='\r'); ii+=1
            X,y, x_cell, x_drug, y0 = sampler.sample_(i, batch_size=args.batch_size)
            X = X.to(device); y=y.to(device); x_cell = x_cell.to(device); x_drug = x_drug.to(device); y0 = y0.to(device)
            
            for k in range(args.T_iters):
                self.t_optim.zero_grad()
                yhat = self.T(X) + y0 
                yhatf = torch.cat((x_cell, x_drug, yhat), dim=-1)

                cost_loss = torch.nn.functional.mse_loss(yhat, y0) 
                yhat_score = self.f(yhatf).view(-1)
                t_loss = cost_loss - yhat_score.mean()

                t_loss.backward() 

                self.t_optim.step() 

        return t_loss.item()
    

    def state_dict(self): 
        return {'f':self.f, 'T':self.T}
    
    def get_T(self): 
        return self.T   
    
    def get_f(self):
            return self.f