
from torch.utils.data import Dataset

import numpy as np
import h5py
import torch
import time
import copy 



class LincsDataset(Dataset):
    def __init__(self, root, sig_ids, data, null_inflation=0.):
        '''

        null_inflation          amount to inflate dataset with null observations (e.g., zero drug, zero output)
        '''
        super().__init__()

        self.sig_ids  = np.array(sig_ids)
        self.root     = root 
        self.data     = data

        self.num_null_obs = int(len(self.sig_ids)*null_inflation)
        self.drug_mask = torch.tensor(['DRUG__' in x for x in self.data.node_names])


    def __len__(self):
        return len(self.sig_ids) + self.num_null_obs

    def __getitem__(self, idx):

        if idx >= len(self.sig_ids): # NOTE: Null Obs. Inflation
            idx                 = torch.randint(0, len(self.sig_ids), size=(1,)).item() 
            sig_id              = self.sig_ids[idx]
            obs                 = torch.load(f'{self.root}/obs/{sig_id}.pt')
            x                   = obs['x']
            x[self.drug_mask]   = 0.                                                    # set all drugs equal to zero 
            y                   = torch.zeros_like(obs['y'])                            # set all outputs equal to zero 
            return x, y, 'NULL_OBS__' + sig_id

        else: 
            sig_id      = self.sig_ids[idx]
            obs         = torch.load(f'{self.root}/obs/{sig_id}.pt')
            x           = obs['x']
            y           = obs['y']
            return x, y, sig_id
