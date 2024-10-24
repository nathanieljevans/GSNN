
import os 
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph, bipartite_subgraph, to_undirected

class ExperienceRecord:
    def __init__(self, root, name='ExperienceRecord.pt'):
        """
        """
        self.root = root
        self.name = name
        self.actions = []       # mask of nodes selected 
        self.rewards = []       # the performance returned 
        self.check_and_load()

    def add(self, action, reward):
        """
        """
        self.actions.append(action.cpu())
        self.rewards.append(reward)
        self.save()

    def __len__(self):
        return len(self.rewards)
    
    def check_and_load(self):

        path = self.root + '/' + self.name 

        if os.path.exists(path): 
            print('loading experiences from disk...', end='')
            exp_dict = torch.load(path)
            self.actions = [x.cpu() for x in exp_dict['actions']]
            self.rewards = exp_dict['rewards']
            print(f'{len(self.rewards)} experiences loaded.')
        else:
            print('no past experiences to load.')

    def save(self): 
        # TODO: saving and loading should be args dependent and be unique files so that we can parrallelize this to many processes (slurm nodes)
        path = self.root + '/' + self.name 

        torch.save({'actions':self.actions, 
                    'rewards':self.rewards}, path)
        

    def get(self, idxs, edge_index_dict, x_dict): 
        
        data_ls = []
        for idx in idxs: 
            r = torch.tensor(self.rewards[idx], dtype=torch.float32).view(1,1)
            a = self.actions[idx]
            data = HeteroData()
            
            n_input = torch.max(edge_index_dict['input', 'to', 'function'][0]) + 1
            n_func = torch.max(edge_index_dict['function', 'to', 'function']) + 1
            n_output = torch.max(edge_index_dict['function', 'to', 'output'][1]) + 1

            data['input', 'to', 'function'].edge_index = bipartite_subgraph(subset=(torch.arange(n_input), a.nonzero(as_tuple=True)[0]), 
                                                                            edge_index=edge_index_dict['input', 'to', 'function'],
                                                                            size=(n_input, n_func))[0]
            data['function', 'to', 'function'].edge_index = subgraph(subset=a.nonzero(as_tuple=True)[0], 
                                                                     edge_index=edge_index_dict['function', 'to', 'function'],
                                                                     num_nodes=(n_func))[0]
            data['function', 'to', 'output'].edge_index = bipartite_subgraph(subset=(a.nonzero(as_tuple=True)[0], torch.arange(n_output)), 
                                                                            edge_index=edge_index_dict['function', 'to', 'output'],
                                                                            size=(n_func, n_output))[0]
            
            data['function', 'to', 'function'].edge_index = to_undirected(data['function', 'to', 'function'].edge_index)

            data['function', 'to', 'input'].edge_index = torch.stack((data['input', 'to', 'function'].edge_index[1], 
                                                                      data['input', 'to', 'function'].edge_index[0]), dim=0)
            
            data['output', 'to', 'function'].edge_index = torch.stack((data['function', 'to', 'output'].edge_index[1], 
                                                                      data['function', 'to', 'output'].edge_index[0]), dim=0)

            data['input'].num_nodes = n_input
            data['function'].num_nodes = n_func
            data['output'].num_nodes = n_output
            data.reward = r 
            data.action = a 

            for k,v in x_dict.items(): 
                data[k].x = v

            data_ls.append(data)

        return data_ls
    

'''

    def batch(self, idxs, edge_index_dict, x_dict): 
        
        data_ls = []
        for idx in idxs: 
            r = torch.tensor(self.rewards[idx], dtype=torch.float32).view(1,1)
            a = self.actions[idx].view(1,-1 )
            data = HeteroData()
            for k,v in x_dict.items(): 
                data[k].x = v
            for k,v in edge_index_dict.items(): 
                data[k].edge_index = v
            data.reward = r 
            data.action = a 
            data_ls.append(data)

        batched_data = Batch.from_data_list(data_ls)
        return batched_data
    
'''
