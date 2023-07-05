
from torch.utils.data import Dataset

import numpy as np
import h5py
import torch
import time
import copy 



class LincsDataset(Dataset):
    def __init__(self, root, sig_ids):
        '''
        '''
        super().__init__()

        self.sig_ids            = np.array(sig_ids)
        self.root               = root 


    def __len__(self):
        return len(self.sig_ids)

    def __getitem__(self, idx):

        # create data object 

        sig_id      = self.sig_ids[idx]

        obs         = torch.load(f'{self.root}/obs/{sig_id}.pt')

        x = obs['x']
        y = obs['y']

        return x, y, sig_id
