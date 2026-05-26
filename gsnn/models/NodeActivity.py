import torch
import torch.nn as nn
from gsnn.models.SignedMessagePassing import SignedMessagePassing

class NodeActivity(torch.nn.Module):
    r"""Node-wise channel attention, mediated by external features. This is performed once per forward pass, 
    and the node attention are applied at every layer. 

    This requires an FUNCTION_NODE feature matrix to be provided, and parameters are shared across all nodes.
    e.g., mutation or expression features for function nodes. Note, missingness is not supported. 

    Parameters
    ----------
    channel_groups : Sequence[int] or Tensor
        A 1-D list/array mapping *global channel index* → *node index*.
        Length equals the total number of hidden channels across all nodes.
    activity_dim : int,
        The dimension of the function node activity features.
    dropout : float, optional (default=0.0)
        Dropout probability applied inside the MLP.
    temperature : float, optional (default=1.0)
        Sigmoid temperature applied to the logits before gating.
        Lower values produce sharper (closer to 0/1) gates; higher values
        produce softer (closer to 0.5) gates.

    """

    def __init__(self, 
                 channel_groups, 
                 activity_dim=1, 
                 dropout: float = 0.0, 
                 temperature: float = 1.0, 
                 channels=16, 
                 mode='per-node'):
        super().__init__()

        self.mode = mode

        # Convert and store channel → node mapping
        self.register_buffer('channel_groups', torch.as_tensor(channel_groups, dtype=torch.long))
        self.Ncg = len(channel_groups)

        self.n_nodes: int = int(self.channel_groups.max().item() + 1)
        self.temperature: float = float(temperature)
        
        # Compute how many channels belong to each node (assume uniform)
        self.channels_per_node: int = int(self.channel_groups.numel() // self.n_nodes)
        self.activity_dim: int = int(activity_dim)

        self.dropout = nn.Dropout(dropout)

        if mode == 'per-node':
            out_dim = 1 
        elif mode == 'per-channel':
            out_dim = self.channels_per_node
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'per-node' or 'per-channel'.")

        # Shared MLP that maps a vector of node-channels → scalar gate
        self.mlp = nn.Sequential(
            nn.Linear(self.activity_dim, channels),
            nn.GELU(),
            nn.LayerNorm(channels),
            nn.Linear(channels, out_dim),
        )

    def get_alpha_mean(self):
        if hasattr(self, 'store_alpha_mean'):
            return self.store_alpha_mean
        else:
            raise ValueError("Alpha mean not stored. Please run forward pass first.")

    def forward(self, x: torch.Tensor):
        """Infer the node attention/activity.

        Parameters
        ----------
        x : Tensor of shape ``(B, Nf, activity_dim)`` or ``(B, Nf)``
            Function node features. If ``activity_dim == 1`` a 2-D tensor of
            shape ``(B, Nf)`` is also accepted and will be unsqueezed
            internally. ``Nf`` must equal ``self.n_nodes``.

        Returns
        -------
        Tensor : (B, Nf * C_pn)
            Node activity/attention applied per channel.
        """

        if x.dim() == 2:
            if self.activity_dim != 1:
                raise ValueError(
                    f"Got 2-D x of shape {tuple(x.shape)} but activity_dim="
                    f"{self.activity_dim}; expected 3-D (B, Nf, activity_dim)."
                )
            x = x.unsqueeze(-1)  # (B, Nf, 1)

        B, Nf, F = x.shape
        if Nf != self.n_nodes:
            raise ValueError(
                f"Expected x with Nf={self.n_nodes} function nodes, got Nf={Nf}."
            )
        if F != self.activity_dim:
            raise ValueError(
                f"Expected x with activity_dim={self.activity_dim}, got {F}."
            )

        logits = self.mlp(x)                                     # (B, Nf, 1) or (B, Nf, C_pn)
        alpha = (logits / self.temperature).sigmoid()            # (B, Nf, 1) or (B, Nf, C_pn)
        self.store_alpha_mean = alpha.mean(dim=0)

        # dropout nodes to zero 
        alpha = self.dropout(alpha)

        if self.mode == 'per-node':
            alpha_per_channel = alpha.expand(-1, -1, self.channels_per_node)  # (B, Nf, C_pn)
            alpha_per_channel = alpha_per_channel.reshape(B, self.Ncg)        # (B, Nf * C_pn)
        elif self.mode == 'per-channel':
            alpha_per_channel = alpha.reshape(B, self.Ncg)            # (B, Nf * C_pn)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'per-node' or 'per-channel'.")

        return alpha_per_channel