from torch_geometric.nn import GAE, SAGEConv
import umap 
from matplotlib import pyplot as plt 
import seaborn as sbn
import torch
import numpy as np
import torch_geometric as pyg

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, conv=pyg.nn.GATConv, norm=pyg.nn.PairNorm, nonlin=torch.nn.GELU, layers=10):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            conv(in_channels, hidden_channels)] + 
            [conv(hidden_channels, hidden_channels) for i in range(layers-1)])
        
        self.norms = torch.nn.ModuleList(
            [norm(hidden_channels) for i in range(layers)])

        self.activations = torch.nn.ModuleList()
        self.activations.extend([nonlin()]*layers)

        self.jk = pyg.nn.JumpingKnowledge('cat') 

        self.lin = torch.nn.Linear(layers*hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        out = [] 
        for conv, norm, act in zip(self.convs, self.norms, self.activations):
            x = conv(x, edge_index)
            x = norm(x)
            x = act(x)
            out.append(x)

        out = self.jk(out) 
        out = self.lin(out)
        return out

class GraphAutoencoder:
    def __init__(self, data, embedding_dim): 

        self.data = data 
        self.embedding_dim = embedding_dim 
        self.model = None 

        self.model = GAE(Encoder(self.data.x.size(1), self.embedding_dim))
        self.trained = False

    def train(self, device, lr=1e-4, epochs=50, verbose=True):

        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs): 
            model.train()

            optimizer.zero_grad()
            z = model.encode(self.data.x.to(device), self.data.edge_index.to(device))
            loss = model.recon_loss(z, self.data.edge_index.to(device))
            loss.backward()
            optimizer.step()

            if verbose: print(f'Epoch {epoch}: loss: {loss.item():.4f}', end='\r')

        self.trained = True
        self.model = model.cpu()

    def embed(self, device='cpu'): 
        model = self.model.to(device)
        x = self.data.x
        edge_index = self.data.edge_index
        return model.encode(x.to(device), edge_index.to(device)).detach().cpu().numpy()
    
    def plot(self, colors=None, figsize=(8, 8), verbose=True):
        if not self.trained: print('Warning: the Graph Autoencoder has not yet been trained.') 

        z = self.embed()
        z = (z - z.mean())/(z.std(0) + 1e-8)
        reducer = umap.UMAP() 
        if verbose: print('running umap embedding...')
        z = reducer.fit_transform(z)

        plt.figure(figsize=figsize)
        if colors is None: 
            plt.plot(z[:,0], z[:,1], 'k.', alpha=0.5)
        else: 
            cs = np.unique(colors)
            for c in cs: 
                idx = (np.array(colors) == c).nonzero()[0]
                alpha = np.clip(1 - (np.array(colors) == c).mean(), 0.25, 1)
                plt.scatter(z[idx,0], z[idx,1], c=c, alpha=alpha, marker='.')
        plt.show()
