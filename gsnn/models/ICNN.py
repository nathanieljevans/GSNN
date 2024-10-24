import torch 

class ICNN(torch.nn.Module):
    """Input Convex Neural Network"""
    #https://github.com/atong01/ot-icnn-minimal/blob/main/icnn/icnn.py

    def __init__(self, in_channels=2, hidden_channels=64, layers=4):
        super().__init__()

        Wzs = []
        Wzs.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(layers - 1):
            Wzs.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=False))
        Wzs.append(torch.nn.Linear(hidden_channels, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = []
        for _ in range(layers - 1):
            Wxs.append(torch.nn.Linear(in_channels, hidden_channels))
        Wxs.append(torch.nn.Linear(in_channels, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)
        self.act = torch.nn.Softplus()

    def forward(self, x):
        z = self.act(self.Wzs[0](x))
        for Wz, Wx in zip(self.Wzs[1:-1], self.Wxs[:-1]):
            z = self.act(Wz(z) + Wx(x))
        return self.Wzs[-1](z) + self.Wxs[-1](x)