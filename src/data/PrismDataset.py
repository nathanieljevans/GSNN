
from torch.utils.data import Dataset

import numpy as np
import h5py
import torch
import time
import copy 



class PrismDataset(Dataset):
    def __init__(self, root, sig_ids, data, target='log_fold_change', eps=1e-8):
        '''
        '''
        super().__init__()

        self.sig_ids  = np.array([x for x in sig_ids if x not in ['nan']])
        self.root     = root 
        self.data     = data
        self.target   = target
        self.eps      = eps

    def __len__(self):
        return len(self.sig_ids)

    def __getitem__(self, idx):

        sig_id      = self.sig_ids[idx]
        obs         = torch.load(f'{self.root}/obs_prism/{sig_id}.pt')
        x           = obs['x']
        y           = obs[self.target]

        if self.target == 'cell_viability': 
            # clip to 0,1 - noninclusive
            y = np.clip(y, self.eps, 1-self.eps)

        return x, y, sig_id
