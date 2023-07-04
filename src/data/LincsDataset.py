
from torch.utils.data import Dataset

import numpy as np
import h5py
import torch
import time
import copy 



def get_x_pert(pert_idx, conc_um, num_inputs, logscale=True):
    '''
    '''
    x_pert = torch.zeros(num_inputs, dtype=torch.float32)
    x_pert[pert_idx] = conc_um
    if logscale: x_pert = torch.log10(x_pert + 1)
    return x_pert.view(-1)


class LincsDataset(Dataset):
    def __init__(self, root, graph, sig_ids):
        '''
        '''
        super().__init__()

        self.sig_ids            = np.array(sig_ids)
        self.root               = root 

        self.data               = graph
        self.num_perts          = len(self.data.drug_nodes)

        '''
        This is a bit nuanced, what we're setting up is the "input" *edge* for a given drug.
        '''
        src,dst = self.data.edge_index 
        edge_inputs = [x.split('__')[1] for x in self.data.node_names[src]]
        self.node2_edgeidx = {input_node:i for i,input_node in enumerate(edge_inputs)}

    def __len__(self):
        return len(self.sig_ids)

    def __getitem__(self, idx):

        # create data object 

        sig_id      = self.sig_ids[idx]

        obs         = torch.load(f'{self.root}/obs/{sig_id}.pt')

        pert_id     = obs['pert_id']
        cell_iname  = obs['cell_iname']
        cell_idx    = obs['cell_idx']
        conc_um     = obs['conc_um']
        time        = obs['time_hr'] 

        attr = torch.tensor(self.data.omics_dict[cell_iname], dtype=torch.float32)
        x = get_x_pert(self.node2_edgeidx[pert_id], conc_um, num_inputs=self.data.num_edges, logscale=True)
        y = obs['y'].detach().type(torch.float32)

        return x, attr, y, sig_id, pert_id
