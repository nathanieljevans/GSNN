
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import DeepGraphInfomax, SAGEConv

import umap 
from matplotlib import pyplot as plt 
import seaborn as sbn
import networkx as nx
import torch
import numpy as np
import torch_geometric as pyg

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, conv='gin', norm=pyg.nn.PairNorm, nonlin=torch.nn.GELU, layers=10, dropout=0.):
        super().__init__()

        if conv == 'gin': 
            conv = lambda i,o: pyg.nn.GINConv(nn=torch.nn.Sequential(torch.nn.Linear(i, o), 
                                               nonlin(), 
                                               torch.nn.Linear(o, o)))
        elif conv == 'gat': 
            conv = lambda i,o: pyg.nn.GATConv(i, o, heads=1, concat=False, add_self_loops=True)
        elif conv == 'gcn': 
            conv = lambda i,o: pyg.nn.GCNConv(i, o)
        elif conv == 'sage': 
            conv = lambda i,o: pyg.nn.SAGEConv(i, o)
        else: 
            raise Exception()


        self.convs = torch.nn.ModuleList([
            conv(in_channels, hidden_channels)] + 
            [conv(hidden_channels, hidden_channels) for i in range(layers-1)])
        
        self.norms = torch.nn.ModuleList(
            [norm(hidden_channels) for i in range(layers)])

        self.activations = torch.nn.ModuleList()
        self.activations.extend([nonlin()]*layers)

        self.jk = pyg.nn.JumpingKnowledge('cat') 

        self.lin = torch.nn.Linear(layers*hidden_channels, hidden_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        out = [] 
        for conv, norm, act in zip(self.convs, self.norms, self.activations):
            x = conv(x, edge_index)
            x = torch.nn.functional.dropout(x, p=self.dropout)
            x = norm(x)
            x = act(x)
            out.append(x)

        out = self.jk(out) 
        out = self.lin(out)
        return out.tanh()
    

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

class DGI(): 
    def __init__(self, data, embedding_dim, dropout=0., layers=10, conv='gin'): 

        self.data = data 
        self.embedding_dim = embedding_dim 
        self.model = None 

        self.model =  DeepGraphInfomax(
            hidden_channels=self.embedding_dim, encoder=Encoder(self.data.x.size(1), self.embedding_dim, dropout=dropout, layers=layers, conv=conv),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption)
        
        self.trained = False

    def train(self, device, lr=1e-4, epochs=50, verbose=True):

        #train_loader = NeighborLoader(self.data, num_neighbors=num_neighbors, batch_size=batch_size,
        #                      shuffle=True, num_workers=workers, input_nodes=None)

        model = self.model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs): 
            model.train()

            total_loss = total_examples = 0
            
            optimizer.zero_grad()
            pos_z, neg_z, summary = model(self.data.x.to(device), self.data.edge_index.to(device))
            loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pos_z.size(0)
            total_examples += pos_z.size(0)

            if verbose: print(f'Epoch {epoch}: loss: {total_loss / total_examples:.4f}', end='\r')

        self.trained = True
        self.model = model.cpu()

    def embed(self, device='cpu'): 
        model = self.model.to(device)
        model.eval()
        x = self.data.x
        edge_index = self.data.edge_index
        return model.encoder(x.to(device), edge_index.to(device)).detach().cpu().numpy()
    
    def plot(self, colors=None, figsize=(8, 8), verbose=True):
        if not self.trained: print('Warning: the Node2Vec model has not yet been trained.') 

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
