# some elements are adapted from : https://github.com/bunnech/cellot/blob/main/cellot/train/train.py#L44
# See gsnn/external/cellot.py for license and citation information 

import torch
from gsnn.models.NN import NN
from gsnn.ot.utils import freeze_, unfreeze_
from gsnn.models.GSNN import GSNN
import numpy as np
from gsnn.external.cellot import ICNN, compute_loss_f, compute_loss_g, compute_w2_distance
import copy
from gsnn.models.VAE import VAE
import os

class OTICNN(): 
    '''optimal transport via input convex neural networks'''

    def __init__(self, args, data, sampler, latent_dim=50): 

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.args = args
        self.data = data

        self.f_dict = {i:ICNN(input_dim         = latent_dim,
                                            hidden_units        = [args.channels]*args.layers, 
                                            fnorm_penalty       = args.icnn_reg).to(device) for i in range(len(sampler))}
        
        self.g_dict = {i:ICNN(input_dim         = latent_dim,
                                            hidden_units        = [args.channels]*args.layers, 
                                            fnorm_penalty       = args.icnn_reg).to(device) for i in range(len(sampler))}
        
        self.f_optim_dict = {i:torch.optim.Adam(self.f_dict[i].parameters(), lr=args.T_lr, betas=(0.5, 0.9)) for i in range(len(sampler))}
        self.g_optim_dict = {i:torch.optim.Adam(self.g_dict[i].parameters(), lr=args.T_lr, betas=(0.5, 0.9)) for i in range(len(sampler))}

        # train encoder/decoder 
        vae_ = VAE(input_dim=len(data.node_names_dict['output']), 
                  latent_dim=latent_dim, 
                  num_layers=2, 
                  hidden_channels=1024, 
                  dropout=0).to(device)
        
        X = sampler.sample_from_all_profiles(partition='train', N=None)
        vae_.optimize(x=X, 
                      device='cuda', 
                      lr=1e-4, 
                      epochs=500,
                      batch_size=1024, 
                      verbose=True, 
                      beta=1e-4, 
                      patience=200, 
                      train_p=0.9)
        freeze_(vae_)
        vae_.eval()
        self.vae = vae_.to(device)

    def step(self, sampler): 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  

        losses = []
        for i in range(len(sampler)):
            
            # CellOT embeds into a latent space using SCGEN, currently working in native space

            # CellOT has source/target flipped from my expectation; they are treating "source" as the perturbed data 
            # at least compared to this implementation: https://github.com/atong01/ot-icnn-minimal/blob/main/icnn/icnn.py
            #  `ot-icnn-minimal` vs `cellot` -->  x = target, y = source,  grad f(x) = yhat vs grad g(source) = targethat

            # so if source = perts ~ y , and target = controls ~ x
            # and grad f(controls) = pert-hat
            # then grad g(perts) = controls-hat

            # optimize g
            freeze_(self.f_dict[i]); unfreeze_(self.g_dict[i])
            for j in range(self.args.icnn_g_iters): 
                #y = sampler.sample_targets(i, partition='train', batch_size=self.args.batch_size).to(device)
                source = sampler.sample_controls(i, partition='train', batch_size=self.args.batch_size).to(device)
                with torch.no_grad(): source = self.vae.encode(source).detach()
                source.requires_grad_(True)

                self.g_optim_dict[i].zero_grad()
                gl = compute_loss_g(self.f_dict[i], self.g_dict[i], source).mean()
                if not self.g_dict[i].softplus_W_kernels and self.g_dict[i].fnorm_penalty > 0:
                    gl = gl + self.g_dict[i].penalize_w()

                gl.backward()
                self.g_optim_dict[i].step()

            # optimize f 
            freeze_(self.g_dict[i]); unfreeze_(self.f_dict[i])
            #y = sampler.sample_targets(i, partition='train', batch_size=self.args.batch_size).to(device)
            #y = sampler.sample_controls(i, partition='train', batch_size=self.args.batch_size).to(device)
            #with torch.no_grad(): y = self.vae.encode(y).detach()
            #x = sampler.sample_controls(i, partition='train', batch_size=y.size(0)).to(device)
            
            targets = sampler.sample_targets(i, partition='train', batch_size=self.args.batch_size).to(device)
            with torch.no_grad(): targets = self.vae.encode(targets).detach()

            source = sampler.sample_controls(i, partition='train', batch_size=targets.size(0)).to(device)
            with torch.no_grad(): source = self.vae.encode(source).detach()
            
            targets.requires_grad_(True); source.requires_grad_(True)
            transport = self.g_dict[i].transport(source).detach()

            self.f_optim_dict[i].zero_grad()
            fl = compute_loss_f(self.f_dict[i], self.g_dict[i], source, targets, transport=transport).mean()
            fl.backward()
            self.f_optim_dict[i].step()
            self.f_dict[i].clamp_w()

            transport = self.g_dict[i].transport(source).detach()
            dist = compute_w2_distance(self.f_dict[i], self.g_dict[i], source, targets, transport=transport)

            losses.append(dist.item())
            print(f'optimizing f and g... [{i}/{len(sampler)}] -> f loss: {fl.item():.2f}, g loss: {gl.item():.2f}', end='\r')

        return np.mean(losses)

    def state_dict(self): 
        return {'T':self.get_T(),
                'f_dict':copy.deepcopy(self.f_dict),
                'g_dict':copy.deepcopy(self.g_dict)}
    
    def get_T(self): 
        # the CellOT uses g to transport, but I thought f was usually used to transport
        # when we use g, the computed W2 distance and our eval metrics are inversely correlated ... 
        # QUESTION: are they calculating transportation from perturbed -> controls? rather than my expectation of controls -> pertrubed ?
        return Transporter(self.g_dict, self.vae)#, self.scaler)

class Transporter(torch.nn.Module): 
    def __init__(self, model_dict, vae): 
        super().__init__()
        self.model_dict = copy.deepcopy(model_dict)
        self.vae = vae

    def forward(self, x, cond_idx): 
        x = self.vae.encode(x).detach()
        x.requires_grad_(True)
        m = self.model_dict[cond_idx]
        m.eval()
        t = m.transport(x).detach()
        t = self.vae.decode(t).detach()
        return t 
    


'''

def check_loss(*args):
    # https://github.com/bunnech/cellot/blob/main/cellot/train/train.py#L44
    for arg in args:
        if torch.isnan(arg):
            raise ValueError
        

class RuntimeScaler(): 

    def __init__(self, sampler, subset=4096, eps=1e-2): 

        self.eps = eps
        self.param_dict = {}
        for i in range(len(sampler)):
            print(f'fitting scaler... [{i}/{len(sampler)}]', end='\r')
            source = sampler.sample_controls(i, partition='train', batch_size=subset)
            target = sampler.sample_targets(i, partition='train', batch_size=subset)
            st = torch.cat((source, target), dim=0)

            self.param_dict[i] = (st.mean(0), st.std(0))
    
    def scale(self, i, x):
        mean, std = self.param_dict[i]
        return (x - mean.to(x.device)) / (std.to(x.device) + self.eps)
    
    def inv_scale(self, i, x):
        mean, std = self.param_dict[i]
        return x * (std.to(x.device) + self.eps) + mean.to(x.device)





def optimize_f(f, g, optim, x, y, reg=0):

    optim.zero_grad()
    grad_gy = transport(g,y)
    f_loss = torch.mean(f(x) - f(grad_gy)) # minimizing f(x) and maximizing f(grad(g(y)))

    if reg > 0:
        f_loss += reg * torch.sum(torch.stack([torch.sum(torch.nn.functional.relu(-w.weight) ** 2) / 2 for w in f.Wzs]))

    f_loss.backward()
    optim.step()

    return f,optim,f_loss.item()

def optimize_g(g, f, optim, x, y, reg=0):

    optim.zero_grad()
    grad_gy = transport(g,y)
    f_grad_gy = f(grad_gy)
    y_dot_grad_gy = torch.sum(torch.mul(y, grad_gy), axis=1, keepdim=True)
    g_loss = torch.mean(f_grad_gy - y_dot_grad_gy) # minimizing f(grad(g(y))) and maximizing y^T grad(g(y))
    if reg > 0:
        g_loss += reg * torch.sum(torch.stack([torch.sum(torch.nn.functional.relu(-w.weight) ** 2) / 2 for w in g.Wzs]))
    g_loss.backward()
    optim.step()

    return g, optim, g_loss.item()
'''