import torch

class NN(torch.nn.Module):
    """Fully-connected baseline: Linear blocks with optional norm, activation, dropout."""

    def __init__(self, in_channels, hidden_channels, out_channels, layers=2, dropout=0,
                        nonlin=torch.nn.ELU, out=None, norm=torch.nn.LayerNorm):
        """Build stack of linear layers; ``out`` is an optional module class after the last linear."""
        super().__init__()
        
        seq = [torch.nn.Linear(in_channels, hidden_channels)]
        if norm is not None: seq.append(norm(hidden_channels))
        seq += [nonlin(), torch.nn.Dropout(dropout)] 
        for _ in range(layers - 1): 
            seq += [torch.nn.Linear(hidden_channels, hidden_channels)]
            if norm is not None: seq.append(norm(hidden_channels))
            seq += [nonlin(), torch.nn.Dropout(dropout)]
        seq += [torch.nn.Linear(hidden_channels, out_channels)]
        if out is not None: seq += [out()]

        self.nn = torch.nn.Sequential(*seq)

    def forward(self, x):
        """Flat features in, predictions out (shape depends on ``out_channels``)."""
        return self.nn(x)

        