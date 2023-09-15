import torch 




class NN(torch.nn.Module): 

    def __init__(self, in_channels, hidden_channels, out_channels, layers, dropout, nonlin=torch.nn.ELU, out=None, norm=True): 
        super().__init__()
        
        seq = [torch.nn.Linear(in_channels, hidden_channels)]
        if norm: seq.append(torch.nn.BatchNorm1d(hidden_channels))
        seq += [nonlin(), torch.nn.Dropout(dropout)] 
        for i in range(layers - 1): 
            seq += [torch.nn.Linear(hidden_channels, hidden_channels)]
            if norm: seq.append(torch.nn.BatchNorm1d(hidden_channels))
            seq += [nonlin(), torch.nn.Dropout(dropout)]
        seq += [torch.nn.Linear(hidden_channels, out_channels)]
        if out is not None: seq += [out()]

        self.nn = torch.nn.Sequential(*seq)

    def forward(self, x): 

        return self.nn(x)

        