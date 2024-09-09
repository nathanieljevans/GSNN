
from torch_geometric.nn import Node2Vec as n2v
import torch 
import matplotlib.pyplot as plt
import umap
import numpy as np

class Node2Vec(): 

    def __init__(self, edge_index, embedding_dim=128, walk_length=20, context_size=10, 
                                    walks_per_node=10, num_negative_samples=1, p=1., q=1., sparse=True): 

        self.edge_index = edge_index 

        self.model = n2v(edge_index,
                                embedding_dim=embedding_dim,
                                walk_length=walk_length,
                                context_size=context_size,
                                walks_per_node=walks_per_node,
                                num_negative_samples=num_negative_samples,
                                p=p,
                                q=q,
                                sparse=sparse) 
    
        self.trained = False
        
    def train(self, lr=1e-3, num_workers=5, epochs=100, batch_size=128, verbose=True): 
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.model.to(device)

        model.train()

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
        optim = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

        for epoch in range(epochs): 
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optim.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optim.step()
                total_loss += loss.item()
            
            if verbose: print(f'epoch: {epoch} || loss: {total_loss / len(loader):.3f}', end='\r')

        self.model = model.cpu()
        self.trained = True 

    def embed(self): 
        model = self.model
        model.eval()
        with torch.no_grad(): 
            return model().detach().cpu().numpy()

    def plot(self, colors=None, figsize=(8, 8), verbose=True):
        if not self.trained: print('Warning: the Node2Vec model has not yet been trained.') 
        z = self.embed()
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