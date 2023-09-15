
from torch.utils.data import Dataset

import numpy as np
import h5py
import torch
import time
import copy 



class PrismDataset(Dataset):
    def __init__(self, root, sig_ids, data, target='log_fold_change'):
        '''
        '''
        super().__init__()

        self.sig_ids  = np.array(sig_ids)
        self.root     = root 
        self.data     = data
        self.target   = target

    def __len__(self):
        return len(self.sig_ids)

    def __getitem__(self, idx):

        sig_id      = self.sig_ids[idx]
        obs         = torch.load(f'{self.root}/obs_prism/{sig_id}.pt')
        x           = obs['x']
        y           = obs[self.target]
        return x, y, sig_id
