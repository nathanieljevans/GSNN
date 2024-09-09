
from torch.utils.data import Dataset
import torch_geometric as pyg
import numpy as np
import h5py
import torch
import time
import copy 



class pygLincsDataset(Dataset):
    def __init__(self, root, sig_ids, data):
        '''
        '''
        super().__init__()

        self.sig_ids            = np.array(sig_ids)
        self.root               = root 
        self.data               = data


    def __len__(self):
        return len(self.sig_ids)

    def __getitem__(self, idx):

        # create data object 

        data = pyg.data.Data()

        sig_id      = self.sig_ids[idx]

        obs         = torch.load(f'{self.root}/obs/{sig_id}.pt')

        x = obs['x']
        y = obs['y']

        data.edge_index = self.data.edge_index
        data.x = x 
        data.y = y 
        data.output_node_mask = self.data.output_node_mask
        data.sig_id = sig_id

        return data
