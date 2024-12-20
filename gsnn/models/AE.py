
import torch 
from .NN import NN

class AE(torch.nn.Module): 

    def __init__(self, data, hidden_channels, latent_dim, out_channels, layers=2, dropout=0, 
                        nonlin=torch.nn.ELU, out=None, norm=torch.nn.BatchNorm1d): 
        '''
        
        Args: 
            in_channels             int                 number of input channels 
            hidden_channels         int                 number of hidden channels per layer 
            out_channels            int                 number of output channels 
            layers                  int                 number of hidden layers 
            dropout                 float               dropout regularization probability 
            nonlin                  pytorch.module      non-linear activation function 
            out                     pytorch.module      output transformation to be applied 
            norm                    pytorch.module      normalization method to use 
        '''
        super().__init__()
        
        n_drug_features = len([x for x in data.node_names_dict['input'] if 'DRUG__' in x])
        self.register_buffer('drug_input_ixs', torch.tensor([i for i,x in enumerate(data.node_names_dict['input']) if 'DRUG__' in x], dtype=torch.long)) 
        self.drug_enc = NN(in_channels=n_drug_features, hidden_channels=hidden_channels, out_channels=latent_dim,
                                layers=layers, dropout=dropout, nonlin=nonlin, out=out, norm=norm)
        
        n_cell_features = len([x for x in data.node_names_dict['input'] if 'DRUG__' not in x])
        self.register_buffer('cell_input_ixs', torch.tensor([i for i,x in enumerate(data.node_names_dict['input']) if 'DRUG__' not in x], dtype=torch.long))
        self.cell_enc = NN(in_channels=n_cell_features, hidden_channels=hidden_channels, out_channels=latent_dim,
                                layers=layers, dropout=dropout, nonlin=nonlin, out=out, norm=norm)  
        
        self.dec = NN(in_channels=latent_dim, hidden_channels=hidden_channels, out_channels=out_channels,
                        layers=layers, dropout=dropout, nonlin=nonlin, out=out, norm=norm) 
        

    def forward(self, x): 

        x_drug = x[:, self.drug_input_ixs]
        x_cell = x[:, self.cell_input_ixs]

        z_drug = self.drug_enc(x_drug)
        z_cell = self.cell_enc(x_cell)

        z = z_drug + z_cell

        xhat = self.dec(z)

        return xhat