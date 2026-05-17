import torch
import torch.nn as nn
from gsnn.models.NodeAttention import NodeAttention
from gsnn.models.NodeMLP import NodeMLP
from gsnn.models.utils import apply_norm_and_nonlin
from gsnn.models.GroupLayerNorm import GroupLayerNorm
from gsnn.models.SoftmaxGroupNorm import SoftmaxGroupNorm
from gsnn.models.GroupBatchNorm import GroupBatchNorm
from gsnn.models.GroupRMSNorm import GroupRMSNorm
from gsnn.models.GroupEMANorm import GroupEMANorm
from gsnn.models.ChannelEMANorm import ChannelEMANorm
from gsnn.models.SparseLinear import SparseLinear
import warnings


class ResBlock(torch.nn.Module): 

    def __init__(self, bias, nonlin, indices_params, dropout=0., norm='layer', init='xavier', 
                 lin_in=None, lin_out=None, residual=True, norm_first=True, node_attn=False, attn_mlp_hidden=32,
                 learn_residual=True, affine=True, node_mlp=True, node_mlp_hidden=32, 
                 edge_index=None, edge_weight=None): 
        r"""A residual block for GSNN that applies sparse linear transformations with optional normalization.

        Each ResBlock consists of:
            1. Input sparse linear transformation (W_in)
            2. Normalization (optional)
            3. Nonlinearity
            4. Output sparse linear transformation (W_out)
            5. Residual connection (optional)

        The block operates on edge features and uses sparse linear layers to maintain the graph structure constraints.

        Args:
            bias (bool): If set to :obj:`False`, the layers will not learn an additive bias.
            nonlin (torch.nn.Module): Activation function class (e.g., :obj:`torch.nn.ELU`).
            indices_params (tuple): A tuple containing:
                - w_in_indices (Tensor): Indices for input sparse linear layer
                - w_out_indices (Tensor): Indices for output sparse linear layer
                - w_in_size (tuple): Size specification for input layer
                - w_out_size (tuple): Size specification for output layer
                - channel_groups (list): Mapping of channels to their respective nodes
            dropout (float, optional): Dropout probability. (default: :obj:`0.`)
            norm (str, optional): Normalization type (:obj:`'layer'`, :obj:`'batch'`, :obj:`'softmax'`, 
                :obj:`'groupbatch'`, :obj:`'edgebatch'`, :obj:`'rms'`, :obj:`'ema'`, :obj:`'channelema'` or :obj:`'none'`). (default: :obj:`'layer'`)
            init (str, optional): Weight initialization strategy (:obj:`'xavier'` or :obj:`'kaiming'`).
                (default: :obj:`'xavier'`)
            lin_in (SparseLinear, optional): Predefined input linear layer. If :obj:`None`, constructed 
                from indices_params. (default: :obj:`None`)
            lin_out (SparseLinear, optional): Predefined output linear layer. If :obj:`None`, constructed 
                from indices_params. (default: :obj:`None`)
            residual (bool, optional): If set to :obj:`True`, adds residual connections. (default: :obj:`True`)
            norm_first (bool, optional): If set to :obj:`True`, apply normalization before nonlinearity. (default: :obj:`True`) 
            learn_residual (bool, optional): If set to :obj:`True`, learn the residual connection. (default: :obj:`True`)
            affine (bool, optional): If set to :obj:`True`, the normalization layers will learn an additive bias and scale. (default: :obj:`True`)
            node_mlp (bool, optional): If set to :obj:`True`, apply additional MLP processing per node to enhance 
                representational capacity while maintaining graph structure constraints. (default: :obj:`True`)
            node_mlp_hidden (int, optional): Hidden dimension size for the node MLP when enabled. 
                (default: :obj:`128`)

        Example:
            >>> # Create indices for a simple graph with 2 edges and 1 function node with 3 channels
            >>> w_in_indices = torch.tensor([[0, 1], [0, 1]])  # 2 edges, 2 channels
            >>> w_out_indices = torch.tensor([[0, 1], [0, 1]])
            >>> w_in_size = (2, 3)  # (num_edges, num_channels)
            >>> w_out_size = (3, 2)  # (num_channels, num_edges)
            >>> channel_groups = [0, 0, 0]  # All channels belong to node 0
            >>> indices_params = (w_in_indices, w_out_indices, w_in_size, w_out_size, channel_groups)
            >>> # Create ResBlock
            >>> block = ResBlock(
            ...     bias=True,
            ...     nonlin=torch.nn.ELU,
            ...     indices_params=indices_params
            ... )
            >>> # Forward pass
            >>> x = torch.randn(32, 2)  # [batch_size, num_edges]
            >>> batch_params = (None, None)  # Normally computed by GSNN
            >>> out = block(x, batch_params)
            >>> print(out.shape)  # [32, 2]
        """

        super().__init__()
        
        w_in_indices, w_out_indices, w_in_size, w_out_size, channel_groups = indices_params
        self.residual = residual
        self.norm_first = norm_first
        self.norm = norm
        self.dropout = dropout
        self.node_attn = NodeAttention(channel_groups, channels=attn_mlp_hidden, edge_index=edge_index, edge_weight=edge_weight) if node_attn else None 
        self.register_buffer('channel_groups', torch.tensor(channel_groups, dtype=torch.long))
        self.n_nodes = int(self.channel_groups.max().item() + 1)
        self.channels_per_node = int(self.channel_groups.numel() // self.n_nodes)

        if norm == 'layer': 
            _norm = lambda: GroupLayerNorm(channel_groups)
        elif norm == 'batch': 
            _norm = lambda: torch.nn.BatchNorm1d(len(channel_groups), eps=1e-3, affine=affine)
        elif norm == 'batch++':
            self._edge_norm = torch.nn.BatchNorm1d(w_in_size[0], eps=1e-3, affine=affine) 
            _norm = lambda: torch.nn.BatchNorm1d(len(channel_groups), eps=1e-3, affine=affine)
        elif norm == 'groupbatch': 
            _norm = lambda: GroupBatchNorm(channel_groups, affine=affine)
        elif norm == 'edgebatch':
            _norm = lambda: torch.nn.Identity()
            self._edge_norm = torch.nn.BatchNorm1d(w_in_size[0], eps=1e-3, affine=affine) 
        elif norm == 'softmax': 
            _norm = lambda: SoftmaxGroupNorm(channel_groups)
            if norm_first: warnings.warn('It is recommended to apply softmax normalization after the nonlinearity (set norm_first=False)')
        elif norm == 'rms':
            _norm = lambda: GroupRMSNorm(channel_groups, affine=affine)
        elif norm == 'ema':
            _norm = lambda: GroupEMANorm(channel_groups, affine=affine)
        elif norm == 'channelema':
            self._edge_norm = ChannelEMANorm(w_in_size[0], affine=affine)
            _norm = lambda: ChannelEMANorm(len(channel_groups), affine=affine)
        elif norm == 'none': 
            _norm = lambda: torch.nn.Identity()
        else:
            raise ValueError(f'unrecognized norm type: {norm}. Available options: layer, batch, groupbatch, edgebatch, softmax, rms, ema, channelema, none')

        if lin_in is not None:
            self.lin_in = lin_in 
        else:
            self.lin_in = SparseLinear(indices=w_in_indices, size=w_in_size, bias=bias, init=init)
        
        if lin_out is not None:
            self.lin_out = lin_out
        else:
            self.lin_out = SparseLinear(indices=w_out_indices, size=w_out_size, bias=bias, init=init)

        self.norm = _norm()
        self.nonlin = nonlin()
        self.mask = None 
        self._store_activations = False  
        self.learn_residual = learn_residual
        self.node_mlp = node_mlp
        self.node_mlp_hidden = node_mlp_hidden

        if self.learn_residual: 
            self.residual_weight = nn.Parameter(torch.tensor(1.0))
        else: 
            self.residual_weight = 1.0

        # Optional node MLP for enhanced representational capacity per node
        if self.node_mlp:
            self.mlp = NodeMLP(self.channels_per_node, self.node_mlp_hidden, nonlin, dropout)
        else:
            self.mlp = None

    def set_node_mask(self, mask): 
        """
        Set a mask to restrict which channels or nodes are active in the computation.

        Args:
            mask (torch.Tensor): A boolean mask indicating which positions remain active.
        """
        self.node_mask = mask

    def forward(self, x, batch_params, node_err=None, fn_activity=None): 
        r"""Implements the forward pass of the residual block.

        The forward pass consists of:
            1. Edge batch normalization (if configured)
            2. Input sparse linear transformation
            3. Optional node error addition
            4. Normalization and nonlinearity
            5. Node masking (if configured)
            6. Output sparse linear transformation
            7. Dropout
            8. Residual connection (if enabled)

        Args:
            x (Tensor): Edge features of shape :obj:`[batch_size, num_edges]` or 
                :obj:`[batch_size, num_edges, 1]`.
            batch_params (tuple): A tuple containing:
                - batched_indices_in (Tensor): Batched indices for input sparse linear layer
                - batched_indices_out (Tensor): Batched indices for output sparse linear layer
            node_err (Tensor, optional): Node-level error terms to be added after input transformation.
                Shape :obj:`[batch_size, num_nodes]`. (default: :obj:`None`)
            fn_activity (Tensor, optional): Function node activity of shape :obj:`[batch_size, num_function_nodes]`. Used for 
                computing node activity. (default: :obj:`None`)

        Returns:
            Tensor: Transformed edge features of shape :obj:`[batch_size, num_edges]`.

        Example:
            >>> # Using the block from the class example
            >>> x = torch.randn(32, 2)  # [batch_size, num_edges]
            >>> # Create batched indices (normally done by GSNN)
            >>> batch_in = torch.tensor([[0, 0], [0, 1]])
            >>> batch_out = torch.tensor([[0, 1], [0, 0]])
            >>> batch_params = (batch_in, batch_out)
            >>> # Forward pass
            >>> out = block(x, batch_params)
            >>> print(out.shape)  # [32, 2]
            >>> # With node errors
            >>> node_err = torch.randn(32, 1)  # [batch_size, num_nodes]
            >>> out = block(x, batch_params, node_err=node_err)
            >>> print(out.shape)  # [32, 2]
        """

        if hasattr(self, '_edge_norm'): 
            out = self._edge_norm(x)
        else: 
            out = x

        # EDGE CHANEL INDEXED --- ^^^^ HERE ^^^^
        out = self.lin_in(out, batched_indices=batch_params[0])    
        # NODE CHANNEL INDEXED --- vvvv HERE vvvv

        ###### in development ######
        if node_err is not None:
            out = out + node_err.unsqueeze(-1)  
        ############################
        
        out = apply_norm_and_nonlin(self.norm, self.nonlin, out, self.norm_first)

        # drops out node channels (not edge channels)
        out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)

        # Optional node MLP processing ##########################################
        # Applies additional MLP to each node's representation independently
        # This enhances representational capacity while maintaining graph constraints
        if self.node_mlp:
            # Reshape: [batch_size, num_nodes * channels_per_node] -> [batch_size, num_nodes, channels_per_node]
            #out = out.squeeze(-1).view(-1, self.channels_per_node)
            out = out.squeeze(-1).view(-1, self.n_nodes, self.channels_per_node)
            out = self.mlp(out)
            # Reshape back: [batch_size, num_nodes, channels_per_node] -> [batch_size, num_nodes * channels_per_node]
            out = out.view(-1, self.n_nodes*self.channels_per_node)
        #####################################################################

        if self.node_attn is not None:
            out = self.node_attn(out)

        if self._store_activations: self._last_activation = out

        if self.node_mask is not None: 
            out = out.squeeze(-1) * self.node_mask.squeeze(-1)

        if fn_activity is not None:
            # squeeze trailing singleton so 3-D (B, NC, 1) doesn't broadcast
            # against 2-D fn_activity (B, NC) into (B, NC, NC).
            out = out.squeeze(-1) * fn_activity

        # NODE CHANEL INDEXED --- ^^^^ HERE ^^^^
        out = self.lin_out(out, batched_indices=batch_params[1]) 
        # EDGE CHANEL INDEXED --- vvvv HERE vvvv

        if self.residual: 
            out = out.squeeze(-1) + self.residual_weight.relu() * x 

        return out


