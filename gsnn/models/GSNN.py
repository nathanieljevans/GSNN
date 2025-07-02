import torch
import torch.nn.functional 
from torch.utils.checkpoint import checkpoint #_sequential
from torch.autograd import Variable
from gsnn.models.SparseLinear import batch_graphs
import numpy as np
from gsnn.models.SparseLinear import SparseLinear
from gsnn.models.GroupLayerNorm import GroupLayerNorm
from gsnn.models.SoftmaxGroupNorm import SoftmaxGroupNorm
from gsnn.models.GroupBatchNorm import GroupBatchNorm
import warnings 


def hetero2homo(edge_index_dict, node_names_dict): 
    r"""Convert a heterogeneous GSNN graph into a single COO *edge_index*.

    The GSNN pipeline distinguishes three edge types::

        ('input', 'to', 'function')
        ('function', 'to', 'function')
        ('function', 'to', 'output')

    This function stacks those edge sets into one homogeneous graph and returns
    boolean masks that let you recover the original node semantics.

    Args:
        edge_index_dict (dict[tuple, Tensor]): Edge-type mapping where each
            value is a :obj:`LongTensor` with shape :obj:`[2, E_t]`.
        node_names_dict (dict[str, list[str]]): Mapping ``'input'`` /
            ``'function'`` / ``'output'`` → list of node names.

    Shapes:
        - **edge_index** – :math:`(2, E)`
        - **input_node_mask / output_node_mask** – :math:`(N,)`

    Returns:
        Tuple[Tensor, Tensor, Tensor, int, list[str]]:
        ``(edge_index, input_mask, output_mask, num_nodes, homo_names)``.
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
    
    input_node_mask = torch.zeros((N_input + N_function + N_output), dtype=torch.bool)
    input_nodes = torch.arange(N_input) + N_function
    input_node_mask[input_nodes] = True

    output_node_mask = torch.zeros((N_input + N_function + N_output), dtype=torch.bool)
    output_nodes = torch.arange(N_output) + N_function + N_input
    output_node_mask[output_nodes] = True

    num_nodes = N_input + N_function + N_output

    homo_names = node_names_dict['function'] + node_names_dict['input'] + node_names_dict['output']

    return edge_index, input_node_mask, output_node_mask, num_nodes, homo_names


def get_Win_indices(edge_index, channels, function_nodes): 
    r"""Build sparse COO indices for the *input* weight matrix :math:`W_{in}`.

    Args:
        edge_index (Tensor): Homogeneous edge index with shape :obj:`[2, E]`.
        channels (int | Tensor): If *int*, every function node gets the same
            number of hidden channels.  If 1-D tensor/array, it must contain
            the per-node channel count of length :math:`N`.
        function_nodes (Tensor): Index list of nodes that represent *functions*.

    Shapes:
        - **edge_index** – :math:`(2, E)`
        - **indices** – :math:`(2, \text{nnz})`

    Returns:
        tuple[Tensor, np.ndarray]: *(indices, channel_count)* where
        **indices** are the coordinates required by
        :class:`torch.sparse_coo_tensor` to build :math:`W_{in}` and
        **channel_count** is a NumPy array of per-node channels for later
        reuse.
    """

    # channels should be of size (Num_Nodes)
    num_nodes = torch.unique(edge_index.view(-1)).size(0)
    _channels = np.zeros(num_nodes, dtype=int) 
    _channels[function_nodes] = channels

    row = []
    col = []
    for edge_id, (_, node_id) in enumerate(edge_index.detach().cpu().numpy().T):
        if node_id not in function_nodes: continue # skip the output nodes 
        c = _channels[node_id] # number of func. node channels 
        node_id_idx0 = np.sum(_channels[:node_id.item()])       # node indexing: index of the first hidden channel for a given function node 
        for k in range(c): 
            row.append(edge_id)
            col.append(node_id_idx0 + k)

    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    indices = torch.stack((row,col), dim=0)
    return indices, _channels



def get_Wout_indices(edge_index, function_nodes, channels): 
    """
    Compute the indices used to construct the output linear transformation matrix (W_out).

    Args:
        edge_index (torch.Tensor): A 2D tensor (2, E) representing homogeneous edge indices.
        function_nodes (torch.Tensor): A 1D tensor of function node indices.
        channels (numpy.ndarray): An array indicating the number of channels for each function node.

    Returns:
        indices (torch.Tensor): A 2D tensor of shape (2, ???) for sparse indexing of W_out.

    Notes:
        ??? Confirm if ordering of edges and nodes in channels array matches the node indexing.
    """

    row = [] 
    col = []
    for node_id in function_nodes: 
        
        # get the edge ids of the function node 
        src,_ = edge_index 
        out_edges = (src == node_id).nonzero(as_tuple=True)[0]

        c = channels[node_id]                                  # number of func. node channels 
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
    """
    Convert node-level features to edge-level features.

    Args:
        x (torch.Tensor): Node features of shape (B, N), where B is batch size and N is number of nodes.
        edge_index (torch.Tensor): A 2D tensor (2, E) of edge indices.

    Returns:
        (torch.Tensor): Edge features of shape (B, E).

    Notes:
        ??? Ensure that the indexing aligns so that x[:, src] gives correct features for each edge.
    """
    src,dst = edge_index 
    return x[:, src] 

def edge2node(x, edge_index, output_node_mask): 
    """
    Convert edge-level features back to node-level features, focusing on output nodes.

    Args:
        x (torch.Tensor): Edge features of shape (B, E).
        edge_index (torch.Tensor): A 2D tensor (2, E) of edge indices.
        output_node_mask (torch.Tensor): Boolean mask indicating which nodes are output nodes.

    Returns:
        (torch.Tensor): Output node features of shape (B, number_of_output_nodes).

    Notes:
        Only output nodes are nonzero to avoid potential collisions if in-degree > 1
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

    return out







def get_conv_indices(edge_index, channels, function_nodes): 
    """
    Compute the indexing structures for convolutional (sparse linear) layers.

    Args:
        edge_index (torch.Tensor): A 2D tensor (2, E) representing homogeneous edges.
        channels (int): Number of channels per function node or ??? if variable.
        function_nodes (torch.Tensor): Indices of function nodes.

    Returns:
        (tuple): A tuple containing:
            - w_in_indices (torch.Tensor): Indexing for W_in.
            - w_out_indices (torch.Tensor): Indexing for W_out.
            - w_in_size (tuple): Size specification for W_in.
            - w_out_size (tuple): Size specification for W_out.
            - channel_groups (list): A list indicating which node each channel belongs to.

    Notes:
        ??? Clarify how channel_groups maps back to nodes.
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
    """
    Apply normalization and nonlinearity to the input tensor.

    Args:
        norm (callable): A normalization layer or operation.
        nonlin (callable): A nonlinear activation function.
        out (torch.Tensor): The input tensor to be normalized and activated.
        norm_first (bool): If True, apply normalization before nonlinearity; otherwise reverse.

    Returns:
        (torch.Tensor): The transformed tensor.

    Notes:
        ??? Confirm that out shape matches the expected shape for norm and nonlin.
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



class ResBlock(torch.nn.Module): 

    def __init__(self, bias, nonlin, indices_params, dropout=0., norm='layer', init='xavier', lin_in=None, lin_out=None, residual=True): 
        """
        A residual block that applies sparse linear transformations, normalization, and nonlinearities.

        Args:
            bias (bool): Whether to use a bias term in the linear transformations.
            nonlin (callable): The nonlinear activation function class (e.g., torch.nn.ELU).
            indices_params (tuple): Indexing parameters for constructing sparse linear layers.
            dropout (float, optional): Dropout probability. Default is 0.
            norm (str, optional): Normalization type ('layer', 'batch', 'softmax', 'groupbatch', 'edgebatch' or 'none'). Default is 'layer'.
            init (str, optional): Initialization strategy for weights ('xavier', 'kaiming', ???). Default is 'xavier'.
            lin_in (SparseLinear, optional): Predefined input linear layer. If None, constructed from indices_params.
            lin_out (SparseLinear, optional): Predefined output linear layer. If None, constructed from indices_params.
            residual (bool, optional): If True, adds the input to the output (residual connection). Default is True. 

        Notes:
            ??? Clarify expected input shapes for forward method and how node_mask interacts.
        """

        super().__init__()
        
        w_in_indices, w_out_indices, w_in_size, w_out_size, channel_groups = indices_params
        self.residual = residual
        
        self.norm = norm
        self.dropout = dropout
        self.register_buffer('channel_groups', torch.tensor(channel_groups, dtype=torch.long))

        if norm == 'layer': 
            _norm = lambda: GroupLayerNorm(channel_groups)
            self.norm_first = True 
        elif norm == 'batch': 
            _norm = lambda: torch.nn.BatchNorm1d(len(channel_groups), eps=1e-3, affine=False)
            self.norm_first = True
        elif norm == 'groupbatch': 
            _norm = lambda: GroupBatchNorm(channel_groups)
            self.norm_first = True
        elif norm == 'edgebatch':
            _norm = lambda: torch.nn.Identity()
            self._edge_norm = torch.nn.BatchNorm1d(w_in_size[0], eps=1e-2)
            self.norm_first = True
        elif norm == 'softmax': 
            _norm = lambda: SoftmaxGroupNorm(channel_groups)
            self.norm_first = False
        elif norm == 'none': 
            _norm = lambda: torch.nn.Identity()
            self.norm_first = True
        else:
            raise ValueError('unrecognized norm type')

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

    def set_node_mask(self, mask): 
        """
        Set a mask to restrict which channels or nodes are active in the computation.

        Args:
            mask (torch.Tensor): A boolean mask indicating which positions remain active.

        Notes:
            ??? Confirm shape and type of mask, and its expected dimensions relative to output features.
        """

        self.node_mask = mask

    def forward(self, x, batch_params, node_err=None): 
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input edge-level features, shape (B, C) or (B, C, 1) ???.
            batch_params (tuple): Precomputed batch-specific parameters for sparse indexing.

        Returns:
            (torch.Tensor): Output features with the same shape as x.

        Notes:
            ??? Confirm that input x matches the indexing defined in indices_params.
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

        if self.node_mask is not None: 
            out = out.squeeze(-1) * self.node_mask.squeeze(-1)

        out = self.lin_out(out, batched_indices=batch_params[1]) 

        # drops out edge features, not node hidden channels 
        out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)

        if self.residual: 
            out = out.squeeze(-1) + x 
        
        return out
    


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

class GSNN(torch.nn.Module): 

    def __init__(self, edge_index_dict, node_names_dict, channels, layers, dropout=0., nonlin=torch.nn.ELU, bias=True, 
                 share_layers=True, add_function_self_edges=True, norm='layer', init='xavier', verbose=False, 
                 edge_channels=1, checkpoint=False, residual=True):
        """
        A Graph-based neural network model that operates on heterogeneous inputs converted to a homogeneous format.
        It uses residual blocks to process node-to-edge converted features and return outputs on specified nodes.

        Args:
            edge_index_dict (dict): A dictionary mapping edge types to edge index tensors.
            node_names_dict (dict): A dictionary mapping node types ('input', 'function', 'output') to node names.
            channels (int): Number of channels allocated per function node. ???
            layers (int): Number of residual layers in the network.
            dropout (float, optional): Dropout probability. Default is 0.
            nonlin (callable, optional): Nonlinear activation (e.g., torch.nn.ELU). Default is torch.nn.ELU.
            bias (bool, optional): Whether to include bias terms in linear layers. Default is True.
            share_layers (bool, optional): If True, reuse the same layer parameters for all layers. Default is True.
            add_function_self_edges (bool, optional): If True, add self-edges for function nodes. Default is True.
            norm (str, optional): Normalization type ('layer', 'batch', 'softmax', 'none'). Default is 'layer'.
            init (str, optional): Weight initialization ('kaiming', 'xavier', ???). Default is 'kaiming'.
            verbose (bool, optional): If True, print debugging information. Default is False.
            edge_channels (int, optional): Number of latent edge feature channels to replicate. Default is 1.
            checkpoint (bool, optional): If True, use gradient checkpointing. Default is False.

        Notes:
            ??? Confirm if channels are uniform across all function nodes or variable.
        """
        super().__init__()

        # Optional: add multiple latent edge features per edge
        # NOTE: this will scale the total number of channels (be careful)
        if edge_channels > 1:
            edge_index_dict['function', 'to', 'function'] = edge_index_dict['function', 'to', 'function'].repeat(1, edge_channels)

        edge_index, input_node_mask, output_node_mask, self.num_nodes, self.homo_names = hetero2homo(edge_index_dict, node_names_dict)

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
                                                       residual         = self.residual) for i in range(self.layers)])
        
        self._B             = None
        self._batch_params  = None

        self.scale = torch.tensor(self.layers**(0.5), dtype=torch.float32)


    def get_batch_params(self, B, device): 
        """
        Retrieve or compute the batch-specific indexing parameters for sparse linear layers.

        Args:
            B (int): Batch size.
            device (torch.device): The device on which the computation will occur.

        Returns:
            (tuple): A tuple of batched indices for the input and output sparse linear transforms.

        Notes:
            ??? Confirm if this caching works correctly for variable batch sizes.
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
    
    def _preprocess(self, x, node_mask): 
        """
        Preprocess the input features before applying the residual blocks.

        Args:
            x (torch.Tensor): Input features of shape (B, ???), corresponding to input nodes only.
            node_mask (torch.Tensor or None): A mask indicating which nodes should remain active.

        Returns:
            x (torch.Tensor): Edge-level features after node-to-edge conversion.
            batch_params (tuple): Batch-specific indexing parameters.
            modules (list): A list of the ResBlocks to be applied.

        Notes:
            ??? Confirm shape transformations and the purpose of node_mask here.
        """

        B = x.size(0)
        x_node = torch.zeros((B, self.num_nodes), device=x.device, dtype=torch.float32)
        idx = self.input_node_mask.nonzero(as_tuple=True)[0].unsqueeze(0).expand(B, -1)  # Shape: (B, num_input_nodes)
        x_node = x_node.scatter_add(1, idx, x)

        x = node2edge(x_node, self.edge_index)  # convert x to edge-indexed
        
        modules = [blk for blk in self.ResBlocks]

        # faster if we do this up front 
        if node_mask is not None: node_mask = torch.stack([torch.isin(modules[0].channel_groups, node_mask[i].nonzero(as_tuple=True)[0]) for i in range(node_mask.size(0))], dim=0)
        for mod in modules: mod.set_node_mask(node_mask)

        batch_params = self.get_batch_params(B, x.device)

        return x, batch_params, modules
    
    def prune(self, threshold=1e-2, verbose=False): 
        """
        Prune the model by removing channels with small weights. Remember to reinitialize the optimizer if using during training. 

        Args:
            threshold (float, optional): The threshold below which weights are considered insignificant. Default is 1e-2.

        Notes:
            ??? Confirm the exact pruning strategy and how it interacts with the model.
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
        """
        Forward pass of the GSNN model.

        Args:
            x (torch.Tensor): Input features of shape (B, ???), representing input nodes only.
            node_mask (torch.Tensor, optional): A boolean mask for nodes, shape (B, N). Default is None.
            edge_mask (torch.Tensor, optional): A mask applied to edge features, shape (B, E). Default is None.
            ret_edge_out (bool, optional): If True, return edge-level outputs instead of node-level. Default is False.
            e0 (torch.Tensor, optional): Initial edge-level features. Default is None. Used for inferring input errors. 

        Returns:
            (torch.Tensor): If ret_edge_out is False, node-level output features (B, number_of_output_nodes).
                            Otherwise, edge-level features (B, E).

        Notes:
            ??? Confirm what happens if node_mask or edge_mask are partially active or mismatched in shape.
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

        if self.checkpoint and self.training: x.requires_grad_(True)
        for i, (mod,nerr) in enumerate(zip(modules, node_errs)): 

            if self.checkpoint and self.training: 
                x = checkpoint(mod, x, batch_params, node_err=nerr, use_reentrant=False)
            else: 
                x = mod(x, batch_params, node_err=nerr)

            if edge_mask is not None: x = x*edge_mask

        # under assumption that each layer output is iid unit normal (weak assumption since layer outputs will be correlated)
        # then x = N(0,1) + N(0,1) + ... + N(0,1) = N(0, sqrt(layers))
        if not self.residual: x = x / self.scale

        if ret_edge_out: 
            return x
        else: 
            out = edge2node(x, self.edge_index, self.output_node_mask)[:, self.output_node_mask]
            return out