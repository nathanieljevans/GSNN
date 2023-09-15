import torch 




class Logistic(torch.nn.Module): 

    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)
        

    def forward(self, x): 
        return torch.sigmoid(self.lin(x))

        