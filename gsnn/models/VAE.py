'''
assumes input are binary variables
'''

import numpy as np
from gsnn.models.NN import NN
from gsnn.optim.EarlyStopper import EarlyStopper
from torch import nn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, num_layers=2, hidden_channels=256, dropout=0., norm=torch.nn.BatchNorm1d):
        '''
        '''
        super().__init__()
        self.latent_dim      = latent_dim

        self.encoder         = NN(in_channels=input_dim, 
                                  hidden_channels=hidden_channels, 
                                  out_channels=latent_dim*2, 
                                  layers=num_layers, 
                                  dropout=dropout, 
                                  nonlin=torch.nn.Mish, 
                                  out=None, 
                                  norm=norm)
        
        self.decoder         = NN(in_channels=latent_dim, 
                                  hidden_channels=hidden_channels, 
                                  out_channels=input_dim, 
                                  layers=num_layers, 
                                  dropout=dropout, 
                                  nonlin=torch.nn.Mish, 
                                  out=None, 
                                  norm=norm)
        
    def forward(self, x):
        
        x       = x.view(x.size(0), -1)
        h       = self.encoder(x)
        mu      = h[:, :self.latent_dim]
        logvar  = h[:, self.latent_dim:]
        
        std             = torch.exp( 0.5*logvar )
        err             = torch.randn(std.size(), device=x.device) 

        z =  mu + std*err

        x_hat   = self.decoder(z)
        return x_hat, mu, logvar

    def get_loss(self, x, beta=1): 
        x_hat, mu, logvar = self.forward(x)

        recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')  # Normalize by batch size

        # KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # Normalize by batch siz

        # Total loss
        total_loss = recon_loss + beta*kl_divergence

        return total_loss, recon_loss, kl_divergence
    
    def optimize(self, x, device='cpu', lr=1e-3, epochs=100, batch_size=1024, verbose=True, beta=1, patience=-1, train_p=0.9): 

        # split train/test 
        idx = torch.randperm(x.size(0))
        n_train = int(len(idx)*train_p)
        x_train = x[idx[:n_train]].to(device)
        x_test = x[idx[n_train:]].to(device)

        self.to(device)

        optim = torch.optim.Adam(self.parameters(), lr=lr)

        patience = np.inf if patience == -1 else patience
        early_stopper = EarlyStopper(patience=patience)

        best_state = None 
        best_rec = np.inf

        for epoch in range(epochs): 
            self.train()

            # 1 cycle beta anealing
            #_beta = (torch.cos( torch.tensor(2*torch.pi*(epoch/epochs)-torch.pi) ) + 1)/2* beta 
            _beta = beta

            for idx in torch.randperm(x_train.size(0)).split(batch_size):

                optim.zero_grad()
                loss, mse, kl = self.get_loss(x_train[idx].to(device), beta=_beta)
                loss.backward()
                optim.step()

            with torch.no_grad(): 
                recs = [] 
                kls = []
                for idx in torch.randperm(x_test.size(0)).split(batch_size):
                    loss, rec, kl = self.get_loss(x_test[idx].to(device), beta=_beta)
                    recs.append(rec.item())
                    kls.append(kl.item())
                recs = np.mean(recs)
                kls = np.mean(kls)

                if rec < best_rec:
                    best_rec = rec
                    best_state = self.state_dict()

                if early_stopper.early_stop(rec): break 

            if verbose: print(f'Epoch {epoch}--> test mse: {recs:.4f}, test kl: {kls:.4f}, beta: {_beta:.4f}', end='\r')

        self.load_state_dict(best_state)

        if verbose: 
            print() 
            # eval 
            self.eval()
            xhats = []
            for idx in torch.arange(x_test.size(0)).split(batch_size): 
                with torch.no_grad(): xhat = self.decode(self.encode(x_test[idx].to(device)))
                xhats.append(xhat)
            xhats = torch.cat(xhats, dim=0)
            r2 = r2_score(x_test.cpu().numpy(), xhats.detach().cpu().numpy(), multioutput='variance_weighted')
            print('VAE test R^2:', r2)
            print()

    def encode(self, x): 
        self.eval()
        with torch.no_grad(): 
            return self.encoder(x)[:, :self.latent_dim]
    
    def decode(self, z): 
        self.eval()
        with torch.no_grad(): 
            return self.decoder(z)