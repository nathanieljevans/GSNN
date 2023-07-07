import torch 




class NN(torch.nn.Module): 

    def __init__(self, in_channels, hidden_channels, out_channels, layers, dropout, nonlin=torch.nn.ELU): 
        super().__init__()
        
        seq = [torch.nn.Linear(in_channels, hidden_channels), torch.nn.BatchNorm1d(hidden_channels), nonlin(), torch.nn.Dropout(dropout)] 
        for i in range(layers - 1): 
            seq += [torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.BatchNorm1d(hidden_channels), nonlin(), torch.nn.Dropout(dropout)]
        seq += [torch.nn.Linear(hidden_channels, out_channels)]

        self.nn = torch.nn.Sequential(*seq)

    def forward(self, x): 

        return self.nn(x)

        