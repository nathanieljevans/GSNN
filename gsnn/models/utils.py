import torch
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


#########################################################################################################################
######################################### GSNN utils ################################################################
#########################################################################################################################


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

#########################################################################################################################
######################################### ResBlock utils ################################################################
#########################################################################################################################

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
######################################### Prediction utils ################################################################
#########################################################################################################################

def predict_gsnn(loader, model, device, verbose=True):
    """Run ``model`` on ``loader``; return stacked numpy ``y``, ``yhat``, and sig ids from batches."""

    model = model.eval()

    ys = [] 
    yhats = [] 
    sig_ids = []
    
    with torch.no_grad(): 
        for i,(x, y, *sig_id) in enumerate(loader): 
            if verbose: print(f'progress: {i}/{len(loader)}', end='\r')

            yhat = model(x.to(device))

            y = y.to(device)

            yhat = yhat.detach().cpu() 
            y = y.detach().cpu()

            ys.append(y)
            yhats.append(yhat)
            sig_ids += sig_id

    y = torch.cat(ys, dim=0).detach().cpu().numpy()
    yhat = torch.cat(yhats, dim=0).detach().cpu().numpy()

    return y, yhat, sig_ids




def corr_score(y, yhat, multioutput='uniform_weighted', method='pearson', eps=1e-6): 
    '''
    calculate the average pearson correlation score

    y (n_samples, n_outputs): 
    yhat (n_samples, n_outputs):
    '''
    if len(y.shape) == 1: 
        y = y.reshape(-1,1)
        yhat = yhat.reshape(-1,1)

    if method == 'pearson': 
        metric = lambda x,y: np.corrcoef(x, y)[0,1]
    elif method == 'spearman': 
        metric = lambda x,y: spearmanr(x,y)[0]
    elif method == 'r2': 
        #NOTE: hacky since r2 is not a corr. 
        metric = lambda x,y: r2_score(x,y)
    else:
        raise ValueError('unrecognized metric')

    corrs = []
    for i in range(y.shape[1]): 
        if (np.std(y[:, i]) < eps) | (np.std(yhat[:, i]) < eps): 
            p = 0
        else: 
            p = metric(y[:, i], yhat[:, i])
            
        corrs.append( p ) 

    if multioutput == 'uniform_weighted': 
        return np.nanmean(corrs)
    elif multioutput == 'uniform_median': 
        return np.nanmedian(corrs)
    elif multioutput == 'raw_values': 
        return np.array(corrs)
    else:
        raise ValueError('unrecognized multioutput value, expected one of "uniform_weighted", "raw_values"')