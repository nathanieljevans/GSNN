'''
assumes input are binary variables
'''

import numpy as np
from gsnn.models.NN import NN
from gsnn.optim.EarlyStopper import EarlyStopper
from torch import nn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

class VAE(nn.Module):
    def __init__(self, x, latent_dim=32, num_layers=2, hidden_channels=256, dropout=0.):
        '''
        '''
        super().__init__()
        self.latent_dim      = latent_dim
        self.x               = x
        self.encoder         = NN(in_channels=x.size(1), hidden_channels=hidden_channels, out_channels=latent_dim*2, layers=num_layers, dropout=dropout, nonlin=torch.nn.ELU, out=None, norm=torch.nn.BatchNorm1d)
        self.decoder         = NN(in_channels=latent_dim, hidden_channels=hidden_channels, out_channels=x.size(1), layers=num_layers, dropout=dropout, nonlin=torch.nn.ELU, out=None, norm=torch.nn.BatchNorm1d)
        self.trained         = False
        
    def forward(self, x):
        
        x       = x.view(x.size(0), -1)
        h       = self.encoder(x)
        mu      = h[:, :self.latent_dim]
        logvar  = h[:, self.latent_dim:]
        
        std             = torch.exp( 0.5*logvar )
        err             = torch.randn(std.size(), device=x.device) 

        z =  mu + std*err

        x_hat   = self.decoder(z).sigmoid()
        return x_hat, mu, logvar

    def get_loss(self, x, beta=1): 
        x_hat, mu, logvar = self.forward(x)

        # Reconstruction loss (binary cross entropy for binary variables)
        recon_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)  # Normalize by batch size

        # KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # Normalize by batch size

        # Total loss
        total_loss = recon_loss + beta*kl_divergence

        return total_loss, recon_loss, kl_divergence
    
    def train(self, device='cpu', lr=1e-3, epochs=100, verbose=True, beta=1, patience=-1): 

        # split train/test 
        idx = torch.randperm(self.x.size(0))
        n_train = int(len(idx)*0.75)
        x_train = self.x[idx[:n_train]].to(device)
        x_test = self.x[idx[n_train:]].to(device)

        self.to(device)

        optim = torch.optim.Adam(self.parameters(), lr=lr)

        patience = np.inf if patience == -1 else patience
        early_stopper = EarlyStopper(patience=patience)

        for epoch in range(epochs): 
            self.training = True

            optim.zero_grad()
            loss, bce, kl = self.get_loss(x_train, beta=beta)

            loss.backward()
            optim.step()

            with torch.no_grad(): 
                loss, rec, kl = self.get_loss(x_test, beta=beta)
                if early_stopper.early_stop(loss): break

            if verbose: print(f'Epoch {epoch}: loss: {loss:.4f} | bce: {bce:.3f} || kl: {kl:.3f}', end='\r')

        self.trained = True

        print()

        if verbose: 
            # eval aupr 
            x_hat, mu, logvar = self.forward(x_test)
            aupr = []; auroc = []
            for i in range(x_hat.shape[1]): 
                
                y_true = x_test[:, i].detach().cpu().numpy()
                y_pred = x_hat[:, i].detach().cpu().numpy() 
                if y_true.sum() == 0: continue
                aupr.append( average_precision_score( y_true, y_pred ))
                auroc.append( roc_auc_score( y_true, y_pred ))

            print('test aupr:', np.mean(aupr))
            print('test auroc:', np.mean(auroc))

            return aupr, auroc

    def embed(self, device='cpu'): 
        self.training=False
        encoder = self.encoder.to(device)
        x = self.x.to(device)
        return encoder(x).detach().cpu().numpy()[:, :self.latent_dim]