
import os 
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph, bipartite_subgraph, to_undirected
import uuid 

class ExperienceRecord:
    def __init__(self, root):
        """
        """
        self.root = root
        self.actions = []       # mask of nodes selected 
        self.rewards = []       # the performance returned 
        self.uids = []
        self.check_and_load()

    def add(self, action, reward):
        """
        """
        self.uids.append(str(uuid.uuid4()))
        self.actions.append(action.cpu().view(-1))
        self.rewards.append(reward)
        self.save()

    def __len__(self):
        return len(self.rewards)
    
    def load(self, uids): 

        for ii,uid in enumerate(uids): 
            print(f'loading experience from disk {ii}/{len(uids)}', end='\r')
            (action, reward) = torch.load(self.root + '/' + uid)
            self.actions.append(action.detach().cpu().type(torch.bool))
            self.rewards.append(reward)
            self.uids.append(uid.split('.')[0])
    
    def check_and_load(self):

        path = self.root + '/'

        if os.path.exists(path): 
            #print('loading experiences from disk...', end='')
            
            new_uid_fnames = [x for x in os.listdir(path) if x.split('.')[0] not in self.uids]
            self.load(new_uid_fnames)
            
            print(f'{len(new_uid_fnames)} experiences loaded.')
        else:
            os.makedirs(self.root, mode=0o777, exist_ok=True)
            print('no past experiences to load.')

    def save(self): 
        
        disk_uids = [x.split('.')[0] for x in os.listdir(self.root)]
        uids_to_save = [x for x in self.uids if x not in disk_uids]
        
        for uid in uids_to_save: 

            idx = self.uids.index(uid)
            action = self.actions[idx]
            reward = self.rewards[idx]

            torch.save((action.detach().cpu().type(torch.bool), reward), self.root + '/' + uid + '.pt')
        

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
