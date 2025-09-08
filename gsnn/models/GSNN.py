from tkinter.constants import FALSE
import torch
import torch.nn as nn
import torch.nn.functional 
from torch.utils.checkpoint import checkpoint #_sequential
from gsnn.models.SparseLinear import batch_graphs
import numpy as np
from gsnn.models.SparseLinear import SparseLinear
from gsnn.models.GroupLayerNorm import GroupLayerNorm
from gsnn.models.SoftmaxGroupNorm import SoftmaxGroupNorm
from gsnn.models.GroupBatchNorm import GroupBatchNorm
from gsnn.models.GroupRMSNorm import GroupRMSNorm
from gsnn.models.GroupEMANorm import GroupEMANorm
from gsnn.models.ChannelEMANorm import ChannelEMANorm
import warnings
import pandas as pd
import torch_geometric as pyg


def hetero2homo(edge_index_dict, node_names_dict, edge_weight_dict=None): 
    r"""Convert a heterogeneous GSNN graph into a homogeneous graph representation.

    The GSNN pipeline distinguishes three edge types:
        1. ('input', 'to', 'function')
        2. ('function', 'to', 'function')
        3. ('function', 'to', 'output')

    This function stacks these edge sets into one homogeneous graph and returns
    boolean masks that let you recover the original node semantics.

    Args:
        edge_index_dict (Dict[Tuple[str, str, str], Tensor]): Edge-type mapping where each
            value is a :obj:`LongTensor` with shape :obj:`[2, num_edges_of_type]`.
        node_names_dict (Dict[str, List[str]]): Mapping of node types ('input', 'function', 'output')
            to their respective node names.
        edge_weight_dict (Dict[Tuple[str, str, str], Tensor]): Dictionary mapping edge types to edge weights.
            Expected keys are ('input', 'to', 'function'), ('function', 'to', 'function'), and 
            ('function', 'to', 'output'). Values should be tensors of shape :obj:`[num_edges]`.
            (default: :obj:`None`)

    Returns:
        tuple: A tuple containing:
            - edge_index (Tensor): Homogeneous edge indices of shape :obj:`[2, num_edges]`
            - input_mask (Tensor): Boolean mask for input nodes of shape :obj:`[num_nodes]`
            - output_mask (Tensor): Boolean mask for output nodes of shape :obj:`[num_nodes]`
            - num_nodes (int): Total number of nodes in the homogeneous graph
            - homo_names (List[str]): Node names in homogeneous ordering
            - edge_weight (Optional[Tensor]): Homogeneous edge weights of shape :obj:`[num_edges]`,
              or :obj:`None` if :obj:`edge_weight_dict` was :obj:`None`.

    Example:
        >>> edge_index_dict = {
        ...     ('input', 'to', 'function'): torch.tensor([[0, 1], [0, 0]]),
        ...     ('function', 'to', 'function'): torch.tensor([[0], [0]]),
        ...     ('function', 'to', 'output'): torch.tensor([[0], [0]])
        ... }
        >>> node_names_dict = {
        ...     'input': ['in1', 'in2'],
        ...     'function': ['func1'],
        ...     'output': ['out1']
        ... }
        >>> edge_index, in_mask, out_mask, n_nodes, names = hetero2homo(
        ...     edge_index_dict, node_names_dict
        ... )
        >>> print(edge_index.shape)  # [2, 4]
        >>> print(in_mask.sum())  # 2 (number of input nodes)
        >>> print(out_mask.sum())  # 1 (number of output nodes)
    """

    # convert edge_index_dict to edge_index (homogenous)
    input_edge_index = edge_index_dict['input', 'to', 'function'].clone()
    function_edge_index = edge_index_dict['function', 'to', 'function'].clone()
    output_edge_index = edge_index_dict['function', 'to', 'output'].clone()

    N_input = len(node_names_dict['input'])
    N_function = len(node_names_dict['function'])
    N_output = len(node_names_dict['output'])

    # add offsets to treat as unique nodes 
    input_edge_index[0, :] = input_edge_index[0,:] + N_function  # increment input nodes only 

    output_edge_index[1, :] = output_edge_index[1, :] + N_function + N_input # increment output nodes only 

    edge_index = torch.cat((function_edge_index, input_edge_index, output_edge_index), dim=1)

    if edge_weight_dict is not None:
        edge_weight = torch.cat((edge_weight_dict['function', 'to', 'function'], 
                                 edge_weight_dict['input', 'to', 'function'], 
                                 edge_weight_dict['function', 'to', 'output']), dim=0)
    else:
        edge_weight = None
    
    input_node_mask = torch.zeros((N_input + N_function + N_output), dtype=torch.bool)
    input_nodes = torch.arange(N_input) + N_function
    input_node_mask[input_nodes] = True

    output_node_mask = torch.zeros((N_input + N_function + N_output), dtype=torch.bool)
    output_nodes = torch.arange(N_output) + N_function + N_input
    output_node_mask[output_nodes] = True

    num_nodes = N_input + N_function + N_output

    homo_names = node_names_dict['function'] + node_names_dict['input'] + node_names_dict['output']

    return edge_index, input_node_mask, output_node_mask, num_nodes, homo_names, edge_weight


def get_Win_indices(edge_index, channels, function_nodes): 
    r"""Build sparse COO indices for the input weight matrix :math:`W_{in}`.

    Args:
        edge_index (Tensor): Homogeneous edge index of shape :obj:`[2, num_edges]`.
        channels (int or Tensor): If int, every function node gets the same number of hidden channels.
            If 1-D tensor/array, it must contain the per-node channel count of length :obj:`num_nodes`.
        function_nodes (Tensor): Index list of nodes that represent functions.

    Returns:
        tuple: A tuple containing:
            - indices (Tensor): COO indices of shape :obj:`[2, nnz]` for sparse tensor construction
            - channel_count (numpy.ndarray): Per-node channel counts for later reuse

    Example:
        >>> edge_index = torch.tensor([[0, 1], [1, 0]])  # 2 edges
        >>> channels = 3  # 3 channels per function node
        >>> function_nodes = torch.tensor([0])  # Node 0 is a function node
        >>> indices, counts = get_Win_indices(edge_index, channels, function_nodes)
        >>> print(indices.shape)  # [2, 6] (2 edges * 3 channels)
        >>> print(counts)  # [3, 0] (3 channels for node 0, 0 for node 1)
    """

    # channels should be of size (Num_Nodes)
    num_nodes = torch.unique(edge_index.view(-1)).size(0)
    _channels = np.zeros(num_nodes, dtype=int)

    # Convert function node indices to numpy for numpy array indexing
    func_nodes_np = function_nodes.detach().cpu().numpy()

    # Populate per-node channel counts
    if isinstance(channels, (int, np.integer)):
        _channels[func_nodes_np] = int(channels)
    else:
        ch_arr = np.asarray(channels, dtype=int)
        if ch_arr.shape[0] != int(num_nodes):
            raise ValueError(
                f"channels must be an int or a length-{int(num_nodes)} array; got shape {ch_arr.shape}"
            )
        _channels = ch_arr.copy()

    row = []
    col = []
    edge_np = edge_index.detach().cpu().numpy()
    func_nodes_set = set(func_nodes_np.tolist())
    for edge_id, (_, node_id) in enumerate(edge_np.T):
        # skip edges whose destination is not a function node
        if int(node_id) not in func_nodes_set:
            continue
        c = int(_channels[int(node_id)])  # number of func. node channels
        node_id_idx0 = int(np.sum(_channels[: int(node_id) ]))  # index of first hidden channel for this node
        for k in range(c):
            row.append(edge_id)
            col.append(node_id_idx0 + k)

    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    indices = torch.stack((row,col), dim=0)
    return indices, _channels



def get_Wout_indices(edge_index, function_nodes, channels): 
    r"""Build sparse COO indices for the output weight matrix :math:`W_{out}`.

    Args:
        edge_index (Tensor): Homogeneous edge index of shape :obj:`[2, num_edges]`.
        function_nodes (Tensor): Index list of nodes that represent functions.
        channels (numpy.ndarray): Array indicating the number of channels for each node.

    Returns:
        Tensor: COO indices of shape :obj:`[2, nnz]` for sparse tensor construction.

    Example:
        >>> edge_index = torch.tensor([[0, 1], [1, 0]])  # 2 edges
        >>> function_nodes = torch.tensor([0])  # Node 0 is a function node
        >>> channels = np.array([3, 0])  # 3 channels for node 0, 0 for node 1
        >>> indices = get_Wout_indices(edge_index, function_nodes, channels)
        >>> print(indices.shape)  # [2, 6] (3 channels * 2 edges)
    """

    row = [] 
    col = []
    for node_id in function_nodes: 
        
        # get the edge ids of the function node 
        src,_ = edge_index 
        out_edges = (src == node_id).nonzero(as_tuple=True)[0]

        c = channels[int(node_id)]                                  # number of func. node channels 
        node_id_idx0 = np.sum(channels[:node_id.item()])       # node indexing: index of the first hidden channel for a given function node 

        for k in range(c):
            for edge_id in out_edges: 
                row.append(node_id_idx0 + k)
                col.append(edge_id.item())

    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    indices = torch.stack((row,col), dim=0)
    return indices







def node2edge(x, edge_index): 
    r"""Convert node-level features to edge-level features. Every out-going edge receives the feature of the source node.

    Args:
        x (Tensor): Node features of shape :obj:`[batch_size, num_nodes]`.
        edge_index (Tensor): Edge indices of shape :obj:`[2, num_edges]`.

    Returns:
        Tensor: Edge features of shape :obj:`[batch_size, num_edges]`.

    Example:
        >>> x = torch.randn(32, 4)  # [batch_size, num_nodes]
        >>> edge_index = torch.tensor([[0, 1], [1, 2]])  # 2 edges
        >>> edge_features = node2edge(x, edge_index)
        >>> print(edge_features.shape)  # [32, 2]
    """
    src,dst = edge_index 
    return x[:, src] 





def edge2node(x, edge_index, output_node_mask): 
    r"""Convert edge-level features back to node-level features, focusing on output nodes.

    Typically, output nodes should be designed to have an in-degree of 1, however, in the case of multiple edges per output node, 
    the output features are summed and normalized by the square root of the in-degree.

    Args:
        x (Tensor): Edge features of shape :obj:`[batch_size, num_edges]`.
        edge_index (Tensor): Edge indices of shape :obj:`[2, num_edges]`.
        output_node_mask (Tensor): Boolean mask of shape :obj:`[num_nodes]` indicating output nodes.

    Returns:
        Tensor: Node features of shape :obj:`[batch_size, num_output_nodes]`.

    Example:
        >>> x = torch.randn(32, 3)  # [batch_size, num_edges]
        >>> edge_index = torch.tensor([[0, 1, 1], [2, 2, 3]])  # 3 edges
        >>> output_mask = torch.tensor([0, 0, 1, 1])  # Nodes 2,3 are outputs
        >>> node_features = edge2node(x, edge_index, output_mask)
        >>> print(node_features.shape)  # [32, 2]
    """

    output_node_ixs = output_node_mask.nonzero(as_tuple=True)[0]
    src, dst = edge_index 
    output_edge_mask = torch.isin(dst, output_node_ixs)

    B = x.size(0)
    out = torch.zeros(B, output_node_mask.size(0), dtype=torch.float32, device=x.device)

    #out[:, dst[output_edge_mask].view(-1)] = x[:, output_edge_mask].view(B, -1)
    idx = dst[output_edge_mask].view(1, -1).expand(B, -1)
    src = x[:, output_edge_mask].view(B, -1)
    out = out.scatter_add(1, idx, src)

    # this is only applicable if there are many edges per output node 
    # user can define the graph structure to avoid this but jic... 
    deg_in = torch.bincount(dst, minlength=out.size(1)).clamp_min(1)
    out = out / deg_in.sqrt()

    return out







def get_conv_indices(edge_index, channels, function_nodes): 
    r"""Compute indexing structures for convolutional (sparse linear) layers.

    Args:
        edge_index (Tensor): Homogeneous edge indices of shape :obj:`[2, num_edges]`.
        channels (int): Number of channels per function node.
        function_nodes (Tensor): Indices of function nodes.

    Returns:
        tuple: A tuple containing:
            - w_in_indices (Tensor): Indexing for :math:`W_{in}`
            - w_out_indices (Tensor): Indexing for :math:`W_{out}`
            - w_in_size (tuple): Size specification for :math:`W_{in}`
            - w_out_size (tuple): Size specification for :math:`W_{out}`
            - channel_groups (List[int]): List mapping each channel to its node

    Example:
        >>> edge_index = torch.tensor([[0, 1], [1, 0]])  # 2 edges
        >>> channels = 3  # 3 channels per function node
        >>> function_nodes = torch.tensor([0])  # Node 0 is a function node
        >>> indices = get_conv_indices(edge_index, channels, function_nodes)
        >>> print(len(indices))  # 5 (w_in_indices, w_out_indices, sizes, groups)
    """

    E = edge_index.size(1)  
    w_in_indices, node_hidden_channels = get_Win_indices(edge_index, channels, function_nodes)
    w_out_indices = get_Wout_indices(edge_index, function_nodes, node_hidden_channels)
    w_in_size = (E, np.sum(node_hidden_channels))
    w_out_size = (np.sum(node_hidden_channels), E)

    channel_groups = [] 
    for node_id, c in enumerate(node_hidden_channels): 
        for i in range(c): 
            channel_groups.append(node_id)

    return (w_in_indices, w_out_indices, w_in_size, w_out_size, channel_groups)






def apply_norm_and_nonlin(norm, nonlin, out, norm_first): 
    r"""Apply normalization and nonlinearity to the input tensor.

    Args:
        norm (callable): Normalization layer or operation.
        nonlin (callable): Nonlinear activation function.
        out (Tensor): Input tensor to be normalized and activated.
        norm_first (bool): If :obj:`True`, apply normalization before nonlinearity.

    Returns:
        Tensor: The transformed tensor.

    Example:
        >>> norm = torch.nn.BatchNorm1d(32)
        >>> nonlin = torch.nn.ReLU()
        >>> x = torch.randn(16, 32)  # [batch_size, num_features]
        >>> # Apply normalization first
        >>> out = apply_norm_and_nonlin(norm, nonlin, x, norm_first=True)
        >>> print(out.shape)  # [16, 32]
    """
    if norm_first: 
        out = norm(out)
        out = nonlin(out)  
    else: 
        out = nonlin(out)  
        out = norm(out)

    return out

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
class SignedMessagePassing(pyg.nn.MessagePassing):
    def __init__(self, edge_weight, edge_index):
        super().__init__(aggr='add')
        self.register_buffer('edge_weight', edge_weight)
        self.register_buffer('edge_index', edge_index)

    def forward(self, x):
        # x is (B, N) where N is the number of function nodes
        B, N = x.shape
        
        # Create function node mask - function nodes are at indices 0 to N-1 in the homogeneous graph
        src_func = self.edge_index[0, :] < N
        dst_func = self.edge_index[1, :] < N 
        func_mask = src_func & dst_func 
        
        # Get function-only edges
        func_edge_index = self.edge_index[:, func_mask]
        func_edge_weight = self.edge_weight[func_mask]
        
        # Create batched edge indices for all batch items
        E = func_edge_index.size(1)
        batched_edge_indices = func_edge_index.repeat(1, B).contiguous()
        batch_idx = torch.repeat_interleave(torch.arange(B, dtype=torch.long, device=x.device), E).contiguous()
        
        # Add offsets for each batch item
        src_incr = batch_idx * N
        dst_incr = batch_idx * N
        incr = torch.stack((src_incr, dst_incr), dim=0)
        batched_edge_indices += incr
        
        # Create batched edge weights
        batched_edge_weights = func_edge_weight.repeat(B)
        
        # Reshape x to (B*N, 1) for batched processing
        x_flat = x.view(-1, 1)
        
        # Process all batch items at once
        out_flat = self.propagate(
            batched_edge_indices,
            x=x_flat,
            edge_weight=batched_edge_weights
        )
        
        # Reshape back to (B, N) - output should have same shape as input
        out = out_flat.view(B, N)
        
        return out

    def message(self, x_i, x_j, edge_weight):
        # Use source node features with edge weights
        return x_j.view(-1, 1) * edge_weight.view(-1, 1)



class NodeAttention(torch.nn.Module):
    r"""Node-wise channel attention.

    The layer learns a single scalar attention coefficient \(\alpha_{b,n}\) per **node** *n*
    for every sample in the batch *b*.  The coefficient is obtained by first
    aggregating the (optionally weighted) hidden channels that belong to the
    node and then normalising the aggregated scores across all nodes with a
    sigmoid gates per node (no cross-node normalization). The resulting
    attention weights can be:

    1. **Interpreted** - \(\alpha_{b,n}\) tells how important node *n* was for
       the current forward pass.
    2. **Applied** - the coefficients are broadcast back to the individual
       channels that originated from the node and multiplied with the original
       activations, producing an attention-modulated output.

    Parameters
    ----------
    channel_groups : Sequence[int] or Tensor
        A 1-D list/array mapping *global channel index* → *node index*.
        Length equals the total number of hidden channels across all nodes.
    dropout : float, optional (default=0.0)
        Dropout probability applied to the node-level attention weights.
    temperature : float, optional (default=1.0)
        Softmax temperature.  Lower values produce sharper distributions.

    Examples
    --------
    >>> # Suppose we have 2 nodes with 3 channels each (total 6 channels)
    >>> ch_groups = [0, 0, 0, 1, 1, 1]
    >>> attn = NodeAttention(ch_groups, dropout=0.1)
    >>> x = torch.randn(8, 6)  # (batch=8, channels=6)
    >>> out, alpha = attn(x, return_alpha=True)
    >>> out.shape          # same shape as input
    torch.Size([8, 6])
    >>> alpha.shape        # one scalar per node
    torch.Size([8, 2])
    """

    def __init__(self, channel_groups, 
                 dropout: float = 0.0, 
                 temperature: float = 1.0, 
                 channels=128,
                 edge_index=None,
                 edge_weight=None):
        super().__init__()

        # Convert and store channel → node mapping
        self.register_buffer('channel_groups', torch.as_tensor(channel_groups, dtype=torch.long))

        self.n_nodes: int = int(self.channel_groups.max().item() + 1)
        self.dropout: float = float(dropout)
        self.temperature: float = float(temperature)
        
        if (edge_weight is not None) and (edge_index is not None):
            self.signed_message_passing = SignedMessagePassing(edge_weight, edge_index)
        else:
            self.signed_message_passing = torch.nn.Identity()

        # Compute how many channels belong to each node (assume uniform)
        self.channels_per_node: int = int(self.channel_groups.numel() // self.n_nodes)

        # Shared two-layer MLP that maps a vector of node-channels → scalar gate
        self.mlp = nn.Sequential(
            nn.Linear(self.channels_per_node, channels),
            nn.ELU(),
            nn.LayerNorm(channels),
            nn.Linear(channels, 1),
        )

        # will hold the last computed attention for inspection
        self._last_alpha = None

    def forward(self, x: torch.Tensor, *, return_alpha: bool = False):
        """Apply node attention.

        Parameters
        ----------
        x : Tensor of shape (B, C)
            Input activations ordered so that channels belonging to the same
            node are indexed according to `channel_groups`.
        return_alpha : bool, optional (default=False)
            If *True*, the method returns a tuple ``(out, alpha)`` where
            ``alpha`` is the attention matrix of shape *(B, n_nodes)*.

        Returns
        -------
        Tensor or Tuple[Tensor, Tensor]
            The attention-modulated activations (and, optionally, the node
            coefficients).
        """
        x = x.squeeze(-1)

        B, C = x.shape
        if C != self.channel_groups.numel():
            raise ValueError(
                f"Expected input with {self.channel_groups.numel()} channels, got {C}.")

        x_nodes = x.view(B, self.n_nodes, self.channels_per_node)             # (B, N, C_pn)

        alpha = self.mlp(x_nodes).squeeze(-1)                                 # (B, N); logits 

        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)

        # signed message passing (sum of neighbors * edge sign)
        alpha = self.signed_message_passing(alpha)

        alpha_per_channel = alpha.sigmoid().unsqueeze(-1).expand(-1, -1, self.channels_per_node)   # (B,N,C_pn)
        out = (x_nodes * alpha_per_channel).reshape(B, C)                                # (B, C)

        self._last_alpha = alpha

        return (out, alpha) if return_alpha else out
        

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

class NodeMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, nonlin, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nonlin(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, C)
        x = self.mlp(x)
        return x


class ResBlock(torch.nn.Module): 

    def __init__(self, bias, nonlin, indices_params, dropout=0., norm='layer', init='xavier', 
                 lin_in=None, lin_out=None, residual=True, norm_first=True, node_attn=False, 
                 learn_residual=True, affine=True, node_mlp=True, node_mlp_hidden=256, 
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
        self.node_attn = NodeAttention(channel_groups, edge_index=edge_index, edge_weight=edge_weight) if node_attn else None 
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

    def forward(self, x, batch_params, node_err=None): 
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

        out = self.lin_in(out, batched_indices=batch_params[0])    

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

        out = self.lin_out(out, batched_indices=batch_params[1]) 


        if self.residual: 
            out = out.squeeze(-1) + self.residual_weight.relu() * x 
        
        return out
    

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

class GSNN(torch.nn.Module): 

    def __init__(self, edge_index_dict, node_names_dict, channels, layers, dropout=0., nonlin=torch.nn.ELU, bias=True, 
                 share_layers=True, add_function_self_edges=True, norm='layer', init='degree_normalized', verbose=False, 
                 edge_channels=1, checkpoint=False, residual=True, norm_first=True, node_attn=False, 
                 node_mlp=True, node_mlp_hidden=128, edge_weight_dict=None):
        r"""Graph Structured Neural Network (GSNN) that constrains neural network architecture using a predefined graph structure.
        Unlike traditional GNNs that learn from graph structure, GSNN uses the graph to constrain which variables can directly 
        influence each other. The model operates on edge features rather than node features and supports cyclic graphs.

        The architecture uses three types of nodes:
            1. Input nodes: Represent observed variables
            2. Function nodes: Represent latent variables parameterized by neural networks
            3. Output nodes: Represent target variables

        Only function nodes are trainable; input and output nodes pass/receive information unchanged.

        Args:
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): Dictionary mapping edge types to edge indices.
                Expected keys are ('input', 'to', 'function'), ('function', 'to', 'function'), and 
                ('function', 'to', 'output'). Values should be tensors of shape :obj:`[2, num_edges]`.
            node_names_dict (Dict[str, List[str]]): Dictionary mapping node types ('input', 'function', 'output') 
                to their respective node names.
            channels (int): Number of hidden channels per function node.
            layers (int): Number of sequential sparse linear layers to propagate information across the graph.
            dropout (float, optional): Dropout probability. (default: :obj:`0.`)
            nonlin (torch.nn.Module, optional): Activation function. (default: :obj:`torch.nn.ELU`)
            bias (bool, optional): If set to :obj:`False`, the layer will not learn an additive bias. 
                (default: :obj:`True`)
            share_layers (bool, optional): If set to :obj:`True`, reuse layer parameters across all layers.
                (default: :obj:`True`)
            add_function_self_edges (bool, optional): If set to :obj:`True`, add self-connections to function nodes.
                (default: :obj:`True`)
            norm (str, optional): Normalization type (:obj:`'layer'`, :obj:`'batch'`, :obj:`'softmax'`, 
                :obj:`'groupbatch'`, :obj:`'edgebatch'`, :obj:`'rms'`, :obj:`'ema'`, :obj:`'channelema'` or :obj:`'none'`). (default: :obj:`'groupbatch'`)
            init (str, optional): Weight initialization strategy (:obj:`'xavier'` or :obj:`'kaiming'`).
                (default: :obj:`'xavier'`)
            verbose (bool, optional): If set to :obj:`True`, print debugging information. (default: :obj:`False`)
            edge_channels (int, optional): Number of latent edge feature channels to replicate.
                (default: :obj:`1`)
            checkpoint (bool, optional): If set to :obj:`True`, use gradient checkpointing to reduce memory usage.
                (default: :obj:`False`)
            residual (bool, optional): If set to :obj:`True`, add residual connections. (default: :obj:`True`)
            norm_first (bool, optional): If set to :obj:`True`, apply normalization before nonlinearity. (default: :obj:`True`)
            node_attn (bool, optional): If set to :obj:`True`, apply node attention. (default: :obj:`False`)
            node_mlp (bool, optional): If set to :obj:`True`, apply additional MLP processing per node to enhance 
                representational capacity while maintaining graph structure constraints. (default: :obj:`True`)
            node_mlp_hidden (int, optional): Hidden dimension size for the node MLP when enabled. 
                (default: :obj:`128`)
            edge_weight_dict (Dict[Tuple[str, str, str], Tensor], optional): Dictionary mapping edge types to edge weights.
                Expected keys are ('input', 'to', 'function'), ('function', 'to', 'function'), and 
                ('function', 'to', 'output'). Values should be tensors of shape :obj:`[num_edges]`.
                (default: :obj:`None`)

        Example:
            >>> # Define a simple graph with 2 input nodes, 1 function node, and 1 output node
            >>> edge_index_dict = {
            ...     ('input', 'to', 'function'): torch.tensor([[0, 1], [0, 0]]),  # 2 input edges
            ...     ('function', 'to', 'function'): torch.tensor([[0], [0]]),     # 1 self edge
            ...     ('function', 'to', 'output'): torch.tensor([[0], [0]])        # 1 output edge
            ... }
            >>> node_names_dict = {
            ...     'input': ['in1', 'in2'],
            ...     'function': ['func1'],
            ...     'output': ['out1']
            ... }
            >>> model = GSNN(
            ...     edge_index_dict=edge_index_dict,
            ...     node_names_dict=node_names_dict,
            ...     channels=16,
            ...     layers=3
            ... )
            >>> x = torch.randn(32, 2)  # batch_size=32, num_input_nodes=2
            >>> out = model(x)
            >>> print(out.shape)  # [32, 1] (batch_size, num_output_nodes)
        """
        super().__init__()

        # Optional: add multiple latent edge features per edge
        # NOTE: this will scale the total number of channels (be careful)
        if edge_channels > 1:
            edge_index_dict['function', 'to', 'function'] = edge_index_dict['function', 'to', 'function'].repeat(1, edge_channels)

        edge_index, input_node_mask, output_node_mask, \
            self.num_nodes, self.homo_names, self.edge_weights = hetero2homo(edge_index_dict, 
                                                                             node_names_dict, 
                                                                             edge_weight_dict)


        self.nonlin                     = nonlin
        self.bias                       = bias
        self.share_layers               = share_layers
        self.layers                     = layers 
        self.channels                   = channels
        self.add_function_self_edges    = add_function_self_edges
        self.verbose                    = verbose
        self.edge_channels              = edge_channels
        self.checkpoint                 = checkpoint
        self.norm                       = norm
        self.dropout                    = dropout
        self.residual                   = residual
        self.node_attn                  = node_attn
        self.norm_first                  = norm_first
        self.node_mlp                   = node_mlp
        self.node_mlp_hidden            = node_mlp_hidden

        if self.checkpoint: 
            # BUG:  checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
            #       with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
            #       /home/teddy/miniconda3/envs/gsnn-lib/lib/python3.12/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
            warnings.filterwarnings("ignore", category=FutureWarning)

        self.register_buffer('output_node_mask', output_node_mask)
        self.register_buffer('input_node_mask', input_node_mask)

        function_nodes = (~(input_node_mask | output_node_mask)).nonzero(as_tuple=True)[0]
        if add_function_self_edges: 
            if verbose: print('Augmenting `edge index` with function node self-edges.')
            edge_index = torch.cat((edge_index, torch.stack((function_nodes, function_nodes), dim=0)), dim=1)

        self.register_buffer('edge_index', edge_index)
        self.E = self.edge_index.size(1)
        self.N = len(node_names_dict['input']) + len(node_names_dict['function']) + len(node_names_dict['output'])

        func_edge_mask = torch.isin(edge_index[0], function_nodes) & torch.isin(edge_index[1], function_nodes) # edges from function -> function / e.g., not an input or output edge 
        inp_edge_mask = torch.isin(edge_index[0], input_node_mask.nonzero(as_tuple=True)[0]) & torch.isin(edge_index[1], function_nodes) # edges from input -> function
        output_edge_mask = ~(func_edge_mask | inp_edge_mask)
        self.register_buffer('function_edge_mask', func_edge_mask) 
        self.register_buffer('input_edge_mask', inp_edge_mask)
        self.register_buffer('output_edge_mask', output_edge_mask)

        self.indices_params = get_conv_indices(edge_index, channels, function_nodes)

        if self.share_layers: 
            w_in_indices, w_out_indices, w_in_size, w_out_size, channel_groups = self.indices_params
            lin_in = SparseLinear(indices=w_in_indices, size=w_in_size, bias=bias, init=init)
            lin_out = SparseLinear(indices=w_out_indices, size=w_out_size, bias=bias, init=init)
        else: 
            lin_in = None
            lin_out = None

        self.ResBlocks = torch.nn.ModuleList([ResBlock(bias             = self.bias,    
                                                       nonlin           = self.nonlin, 
                                                       dropout          = dropout, 
                                                       norm             = norm, 
                                                       init             = init, 
                                                       indices_params   = self.indices_params,
                                                       lin_in           = lin_in,
                                                       lin_out          = lin_out,
                                                       node_attn        = self.node_attn,
                                                       norm_first       = self.norm_first,
                                                       residual         = self.residual,
                                                       node_mlp         = self.node_mlp,
                                                       node_mlp_hidden  = self.node_mlp_hidden, 
                                                       edge_index       = self.edge_index,
                                                       edge_weight      = self.edge_weights) for i in range(self.layers)])
        
        self._B             = None
        self._batch_params  = None

        self.scale = torch.tensor(self.layers**(0.5), dtype=torch.float32)


    def get_batch_params(self, B, device): 
        r"""Retrieves or computes batch-specific indexing parameters for sparse linear layers.

        This method caches the batch parameters to avoid recomputing them for the same batch size.
        The parameters are used to efficiently perform batched sparse matrix operations.

        Args:
            B (int): Batch size.
            device (torch.device): Device on which to place the computed parameters.

        Returns:
            tuple: A tuple containing:
                - batched_indices_in (Tensor): Batched indices for input sparse linear layer
                - batched_indices_out (Tensor): Batched indices for output sparse linear layer

        Example:
            >>> model = GSNN(edge_index_dict, node_names_dict, channels=16, layers=3)
            >>> # Get batch parameters for batch size 32
            >>> batch_params = model.get_batch_params(32, torch.device('cuda'))
            >>> # Parameters are cached for subsequent calls
            >>> same_params = model.get_batch_params(32, torch.device('cuda'))
            >>> # Different batch size triggers recomputation
            >>> new_params = model.get_batch_params(64, torch.device('cuda'))
        """

        if (self._B == B) and (self._batch_params is not None): 
            # caching batch params
            return self._batch_params
        else: 
            self._B = B
            # precompute edge batching so it doesn't have to be done in every resblock 
            batched_edge_indices_in = batch_graphs(N=self.ResBlocks[0].lin_in.N,
                                                            M=self.ResBlocks[0].lin_in.M, 
                                                            edge_index = self.ResBlocks[0].lin_in.indices, 
                                                            B=B, 
                                                            device=device)
            
            batched_edge_indices_out = batch_graphs(N=self.ResBlocks[0].lin_out.N,
                                                            M=self.ResBlocks[0].lin_out.M, 
                                                            edge_index = self.ResBlocks[0].lin_out.indices, 
                                                            B=B, 
                                                            device=device)
            
            self._batch_params = (batched_edge_indices_in, batched_edge_indices_out)
        
        return self._batch_params
    
    def _preprocess(self, x, node_mask=None): 
        r"""Preprocesses input features before applying residual blocks.

        This method:
            1. Converts input node features to a full node feature tensor
            2. Converts node features to edge features
            3. Applies node masking if provided
            4. Computes batch parameters for sparse operations

        Args:
            x (Tensor): Input features of shape :obj:`[batch_size, num_input_nodes]`.
            node_mask (Tensor, optional): node weights for function nodes of shape 
                :obj:`[B, num_nodes]`. (default: :obj:`None`)

        Returns:
            tuple: A tuple containing:
                - x (Tensor): Edge features of shape :obj:`[batch_size, num_edges]`
                - batch_params (tuple): Batch-specific parameters for sparse operations
                - modules (List[ResBlock]): List of residual blocks to apply

        Example:
            >>> model = GSNN(edge_index_dict, node_names_dict, channels=16, layers=3)
            >>> x = torch.randn(32, 2)  # [batch_size, num_input_nodes]
            >>> # Basic preprocessing
            >>> edge_feats, params, blocks = model._preprocess(x, None)
            >>> print(edge_feats.shape)  # [32, num_edges]
            >>> # With node masking
            >>> node_mask = torch.ones(32, 4)  # [B, num_nodes]
            >>> edge_feats, params, blocks = model._preprocess(x, node_mask)
        """

        B = x.size(0)
        x_node = torch.zeros((B, self.num_nodes), device=x.device, dtype=torch.float32)
        idx = self.input_node_mask.nonzero(as_tuple=True)[0].unsqueeze(0).expand(B, -1)  # Shape: (B, num_input_nodes)
        x_node = x_node.scatter_add(1, idx, x)

        x = node2edge(x_node, self.edge_index)  # convert x to edge-indexed
        
        modules = [blk for blk in self.ResBlocks]

        # faster if we do this up front 
        if node_mask is not None: 
            #node_mask = torch.stack([torch.isin(modules[0].channel_groups, node_mask[i].nonzero(as_tuple=True)[0]) for i in range(node_mask.size(0))], dim=0)
            node_mask = node_mask[:, modules[0].channel_groups]
        for mod in modules: mod.set_node_mask(node_mask)

        batch_params = self.get_batch_params(B, x.device)

        return x, batch_params, modules
    
    def prune(self, threshold=1e-2, verbose=False): 
        r"""Prunes the model by removing channels with small weights.

        This method removes channels whose maximum absolute weight value across all layers is below
        the specified threshold. This can significantly reduce model size while maintaining performance.
        Remember to reinitialize the optimizer after pruning if using during training.

        Args:
            threshold (float, optional): The threshold below which weights are considered insignificant.
                (default: :obj:`1e-2`)
            verbose (bool, optional): If set to :obj:`True`, print pruning statistics.
                (default: :obj:`False`)

        Returns:
            int: Number of parameters removed by pruning.

        Example:
            >>> # Create a model with 16 channels per function node
            >>> model = GSNN(edge_index_dict, node_names_dict, channels=16, layers=3)
            >>> # Train the model...
            >>> # Prune channels with small weights
            >>> removed_params = model.prune(threshold=1e-2, verbose=True)
            >>> print(f'Removed {removed_params} parameters')
        """

        w1 = [] ; w2 = []
        for mod in self.ResBlocks: 
            w1.append(mod.lin_in.values)
            w2.append(mod.lin_out.values)
        w1 = torch.stack(w1, dim=0)
        w2 = torch.stack(w2, dim=0)

        w1_abs_max = torch.max(torch.abs(w1), dim=0).values
        w2_abs_max = torch.max(torch.abs(w2), dim=0).values

        keep_idxs1 = (w1_abs_max >= threshold).nonzero(as_tuple=True)[0]
        keep_idxs2 = (w2_abs_max >= threshold).nonzero(as_tuple=True)[0]

        for mod in self.ResBlocks:
            mod.lin_in.prune(keep_idxs1)
            mod.lin_out.prune(keep_idxs2)

            # reset cached batch params 
            self._B = None; self._batch_params = None

        if verbose: 
            print(f'Pruned in/out: {w1.size(1) - len(keep_idxs1)}, {w2.size(1) - len(keep_idxs2)} -> remaining in/out: {len(keep_idxs1)}, {len(keep_idxs2)}')

        # return the number of parameters removed 
        return w1.size(1) - len(keep_idxs1) + w2.size(1) - len(keep_idxs2)

        

    def forward(self, x, node_mask=None, edge_mask=None, ret_edge_out=False, e0=None, node_errs=None):
        r"""Implements the forward pass of the GSNN model.

        The model first converts node features to edge features, then applies a sequence of sparse linear 
        transformations constrained by the graph structure. Each layer consists of:
            1. Input transformation (W_in)
            2. Normalization (optional)
            3. Nonlinearity
            4. Output transformation (W_out)
            5. Residual connection (optional)

        Args:
            x (Tensor): Input node features of shape :obj:`[batch_size, num_input_nodes]`.
            node_mask (Tensor, optional): Boolean mask for function nodes of shape :obj:`[batch_size, num_nodes]`.
                If provided, masks out specific function nodes during computation. (default: :obj:`None`)
            edge_mask (Tensor, optional): Boolean mask for edges of shape :obj:`[batch_size, num_edges]`.
                If provided, masks out specific edges during computation. (default: :obj:`None`)
            ret_edge_out (bool, optional): If set to :obj:`True`, return edge-level features instead of 
                node-level features. (default: :obj:`False`)
            e0 (Tensor, optional): Initial edge features of shape :obj:`[batch_size, num_edges]`. Used for 
                inferring input errors. (default: :obj:`None`)
            node_errs (List[Tensor], optional): List of node errors per layer, each of shape 
                :obj:`[batch_size, num_nodes]`. Length must match number of layers. (default: :obj:`None`)

        Returns:
            Tensor: If :obj:`ret_edge_out=False`, returns node-level output features of shape 
            :obj:`[batch_size, num_output_nodes]`. Otherwise, returns edge-level features of shape 
            :obj:`[batch_size, num_edges]`.

        Example:
            >>> # Using the model from the class example
            >>> x = torch.randn(32, 2)  # batch_size=32, num_input_nodes=2
            >>> # Basic forward pass
            >>> out = model(x)
            >>> print(out.shape)  # [32, 1]
            >>> # Get edge-level features
            >>> edge_out = model(x, ret_edge_out=True)
            >>> print(edge_out.shape)  # [32, 4] (batch_size, num_edges)
            >>> # Using masks
            >>> node_mask = torch.ones(32, 4)  # [batch_size, num_nodes]
            >>> edge_mask = torch.ones(32, 4)  # [batch_size, num_edges]
            >>> out = model(x, node_mask=node_mask, edge_mask=edge_mask)
            >>> print(out.shape)  # [32, 1]
        """

        ############ in dev ################
        if node_errs is None:
            node_errs = [None]*self.layers
        else: 
            if len(node_errs) != self.layers: 
                raise ValueError('node_errs must be the same length as the number of layers')
        ###################################

        x, batch_params, modules = self._preprocess(x, node_mask)

        if e0 is not None:
            x = x + e0
        
        # mask input edges (otherwise input edges get missed)
        if edge_mask is not None: x = x*edge_mask

        if self.checkpoint and self.training: x.requires_grad_(True)
        for i, (mod,nerr) in enumerate(zip(modules, node_errs)): 

            if self.checkpoint and self.training: 
                x = checkpoint(mod, x, batch_params, node_err=nerr, use_reentrant=False).squeeze(-1)
            else: 
                x = mod(x, batch_params, node_err=nerr).squeeze(-1)

            if edge_mask is not None: x = x*edge_mask

        # under assumption that each layer output is iid unit normal (weak assumption since layer outputs will be correlated)
        # then x = N(0,1) + N(0,1) + ... + N(0,1) = N(0, sqrt(layers))
        if self.residual: x = x / self.scale

        if ret_edge_out: 
            return x
        else: 
            out = edge2node(x, self.edge_index, self.output_node_mask)[:, self.output_node_mask]
            return out

    def get_node_activations(self, x, agg='sum', inference=True):

        with torch.inference_mode(mode=inference):  
            for mod in self.ResBlocks: mod._store_activations = True
            preds = self.forward(x)

            activations = []
            for mod in self.ResBlocks:  
                activations.append(mod._last_activation.squeeze(-1)) 
                del mod._last_activation

            activations = torch.stack(activations, dim=0) 

            if agg == 'sum': 
                activations = activations.sum(dim=0) 
            elif agg == 'mean': 
                activations = activations.mean(dim=0)
            elif agg == 'max': 
                activations = activations.max(dim=0).values
            elif agg == 'last': 
                activations = activations[-1, :, :] # L,B,C*N
            elif agg == 'all': 
                activations = activations.permute(1, 0, 2) # (L, B, CN) -> (B, L, CN)
            else: 
                raise ValueError(f'Invalid aggregation method: {agg}')

            for mod in self.ResBlocks: mod._store_activations = False 

            node_activation_dict = {}
            for i in range(int(self.ResBlocks[0].channel_groups.max().item()) + 1): 
                ixs = (self.ResBlocks[0].channel_groups == i).to(activations.device)
                
                if len(activations.shape) == 2: 
                    node_acts = activations[:, ixs] 
                elif len(activations.shape) == 3: 
                    B, L, CN = activations.shape
                    node_acts = activations[:, :, ixs].reshape(B, -1)

                node_name = self.homo_names[i] 
                node_activation_dict[node_name] = node_acts 

            return node_activation_dict

    def get_node_attention(self, x):
        """Return per-layer node-level attention weights.

        Parameters
        ----------
        x : Tensor (B, num_input_nodes)
            Input features; typically supply a single sample (B=1).

        Returns
        -------
        Dict[str, Tensor]
            Mapping from node name to a tensor of shape (L, B) with attention
            weights per layer (L) and batch element (B).
        """
        assert self.node_attn is not None, "Node attention is not enabled in this model."

        # Run a forward pass to cache _last_alpha inside each NodeAttention
        _ = self.forward(x)

        layer_attns: list[torch.Tensor] = []  # each entry: (B, N_fn)
        B = x.size(0)

        # mask and size for function nodes (those with attention)
        function_mask = ~(self.input_node_mask | self.output_node_mask)
        N_fn = int(function_mask.sum().item())
        device = self.edge_index.device

        for blk in self.ResBlocks:
            if blk.node_attn is not None and hasattr(blk.node_attn, "_last_alpha"):
                alpha = blk.node_attn._last_alpha  # (B, N_fn)
                if alpha is None:
                    raise RuntimeError("Attention not stored; ensure NodeAttention forward was called.")
                # Ensure expected 2D shape (B, N_fn)
                if alpha.dim() == 1:
                    alpha = alpha.unsqueeze(0)
                layer_attns.append(alpha)
            else:
                # If no attention in this block, fill with zeros (B, N_fn)
                layer_attns.append(torch.zeros(B, N_fn, device=device))

        # Stack into (L, B, N_fn)
        layer_attns = torch.stack(layer_attns, dim=0)

        # Build dict: node name -> (L, B)
        function_idxs = function_mask.nonzero(as_tuple=True)[0].tolist()
        node_attn_dict = {}
        for j, node_idx in enumerate(function_idxs):
            node_name = str(self.homo_names[int(node_idx)])
            node_attn_dict[node_name] = layer_attns[..., j]

        return node_attn_dict

        # Build DataFrame: each column is a layer
        #data = {f"layer{idx}": vals for idx, vals in enumerate(layer_attns)}
        #data["node"] = np.array(self.homo_names)[(~(self.input_node_mask | self.output_node_mask )).detach().cpu().numpy()]
        #df = pd.DataFrame(data)
        #df = df.set_index("node")
        #return df
