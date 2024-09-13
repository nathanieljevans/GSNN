
from torch.utils.data import Dataset

import numpy as np
import h5py
import torch
import time
import copy 



class LincsDataset(Dataset):
    def __init__(self, root, sig_ids, data, siginfo):
        '''

        '''
        super().__init__()

        self.sig_ids  = np.array(sig_ids)
        self.root     = root 
        self.data     = data
        
        siginfo  = siginfo[lambda x: x.sig_id.isin(self.sig_ids)]
        siginfo  = siginfo.set_index('sig_id')[['pert_id', 'pert_dose', 'cell_iname']]
        self.siginfo = siginfo

    def __len__(self):
        return len(self.sig_ids)

    def __getitem__(self, idx):

        sig_id      = self.sig_ids[idx]

        info        = self.siginfo.loc[sig_id]
        pert_id     = info.pert_id
        conc_um     = info.pert_dose
        cell_iname  = info.cell_iname
        x_drug      = self.data.x_dict['drug_dict'][pert_id](conc_um)
        x_cell      = self.data.x_dict['cell_dict'][cell_iname]
        x           = x_drug + x_cell 

        y           = torch.load(f'{self.root}/obs/{sig_id}.pt')

        subgraph_var = 'DRUG__' + pert_id

        return x.to_dense().detach(), y.to_dense().detach(), sig_id, subgraph_var
