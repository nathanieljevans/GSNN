import torch 
import numpy as np 
import torch_geometric as pyg 
import scipy 
from gsnn.models.GSNN import get_conv_indices

class dense_func_node(torch.nn.Module): 
    def __init__(self, lin_in, lin_out, nonlin, norm, node_mlp=None): 
        super().__init__()
        self.lin_in = lin_in
        self.lin_out = lin_out
        # `nonlin` may be an instantiated module or a class – handle both
        self.nonlin = nonlin if isinstance(nonlin, torch.nn.Module) else nonlin()
        self.node_mlp = node_mlp  # Optional node MLP
        
        channels = lin_in.out_features  # hidden channels produced by `lin_in`

        # ------------------------------------------------------------------
        # Normalisation layers (mirrors gsnn.models.GSNN.ResBlock logic)
        # ------------------------------------------------------------------
        if norm == 'layer':
            assert False, 'Layer norm not implemented for extracted single-node functions'
            self.norm = torch.nn.LayerNorm(channels, elementwise_affine=False)
            self.norm_first = True
            # TODO: copy params from gsnn norm 
        elif norm == 'batch':
            assert False, 'Batch norm not implemented for extracted single-node functions'
            self.norm = torch.nn.BatchNorm1d(channels, eps=1e-3, affine=False)
            self.norm_first = True
            # TODO: copy params from gsnn norm 
        elif norm in ('groupbatch', 'edgebatch'):
            raise NotImplementedError('Group/edge batch norm not implemented for extracted single-node functions')
        elif norm == 'softmax':
            # Approximate SoftmaxGroupNorm with per-feature softmax.
            self.norm = torch.nn.Softmax(dim=1)
            self.norm_first = False  # match ResBlock behaviour
        elif norm == 'none':
            self.norm = torch.nn.Identity()
            self.norm_first = True
        else:
            raise ValueError(f"Unrecognized norm type '{norm}'")

    def forward(self, x):
        """Forward pass reproducing ResBlock ordering of norm / nonlin."""
        x = self.lin_in(x)
        if self.norm_first:
            x = self.norm(x)
            x = self.nonlin(x)
        else:
            x = self.nonlin(x)
            x = self.norm(x)
        
        # Apply optional node MLP if present
        if self.node_mlp is not None:
            # Reshape for node MLP: (batch_size, channels) -> (batch_size, 1, channels)
            batch_size = x.size(0)
            channels = x.size(1)
            x = x.view(batch_size, 1, channels)
            x = self.node_mlp(x)
            # Reshape back: (batch_size, 1, channels) -> (batch_size, channels)
            x = x.view(batch_size, channels)
            
        x = self.lin_out(x) 
        return x




def extract_entity_function(node, model, data, layer=0): 
    r"""Extract the *stand-alone* MLP that implements a single GSNN function node.

    Given a trained :class:`~gsnn.models.GSNN.GSNN` model and the graph that
    was used to train it, this helper rebuilds the exact linear-nonlinear
    sequence that corresponds to a single *function* node at a particular
    layer.  The returned module consumes the latent representations of its
    input edges and produces the hidden activations that are sent to its
    outgoing edges, replicating the behaviour inside the parent GSNN.

    Parameters
    ----------
    node : str
        Name of the *function* node to extract (must exist in
        ``data.node_names_dict['function']``).
    model : gsnn.models.GSNN.GSNN
        Reference GSNN model (weights are copied; the original model remains
        unchanged).
    data : torch_geometric.data.HeteroData
        Heterogeneous graph object used for training.
    layer : int, optional (default=0)
        Index of the GSNN layer (``ResBlocks[layer]``) from which to extract
        the node-specific sub-network.

    Returns
    -------
    func : torch.nn.Module
        A dense two-layer network ``func(x_in) -> x_out`` that is numerically
        equivalent to the chosen node inside the GSNN.
    meta : dict
        Dictionary with

        * ``'input_edge_names'``  – list[str] of incoming edge names
        * ``'output_edge_names'`` – list[str] of outgoing edge names

    Example
    -------
    >>> func_node, meta = extract_entity_function('func3', model, data, layer=1)
    >>> y = func_node(torch.randn(len(meta['input_edge_names'])))
    >>> print(meta['output_edge_names'])
    """

    model = model.cpu()

    # total number of nodes 
    N = len(data.node_names_dict['input']) + len(data.node_names_dict['function']) + len(data.node_names_dict['output'])

    # get homogenous network index; see hetero2homo (GSNN) for reference
    node_idx = data.node_names_dict['function'].index(node)
    #node_idx = torch.tensor([node_idx], dtype=torch.long)
    # convert to edge indexing
    #node_idx = (utils.node2edge(torch.arange(N).unsqueeze(0), model.edge_index) == node_idx).nonzero(as_tuple=True)[0]
    #print(node_idx)

    # NOTE THESE ARE EDGE INDICES (NOT NODE INDICES)
    row,col = model.edge_index.detach().cpu()
    input_edges = (col == node_idx).nonzero(as_tuple=True)[0]
    output_edges = (row == node_idx).nonzero(as_tuple=True)[0]

    node_names = np.array(data.node_names_dict['function'] + data.node_names_dict['input'] + data.node_names_dict['output'])
    inp_edge_names = node_names[row[input_edges]]
    out_edge_names = node_names[col[output_edges]]

    # the hidden channel indices relevant to `node`
    function_nodes = torch.arange(len(data.node_names_dict['function']))
    
    w1_indices, w_out_indices, w_in_size, w_out_size, channel_groups = get_conv_indices(model.edge_index, model.channels, function_nodes)
    
    assert (w1_indices == model.ResBlocks[layer].lin_in.indices).all(), 'W1 indices do not match model W1 indices'

    channel_groups = np.array(channel_groups)
    hidden_idxs = torch.tensor((channel_groups == node_idx).nonzero()[0], dtype=torch.long)
    N_channels = len(hidden_idxs)

    # we have a bipartite network from edge_idx -> function node hidden layers
    indices, values = pyg.utils.bipartite_subgraph(subset           = (input_edges, hidden_idxs), 
                                                   edge_index       = model.ResBlocks[layer].lin_in.indices, 
                                                   edge_attr        = model.ResBlocks[layer].lin_in.values.data, 
                                                   relabel_nodes    = True, 
                                                   return_edge_mask = False,
                                                   size             = (model.edge_index.size(1), len(channel_groups)))
    
    w1_smol = scipy.sparse.coo_array((values.detach(), (indices[0,:].detach(), indices[1,:].detach())), shape=(len(input_edges), N_channels)).todense()
    
    if hasattr(model.ResBlocks[layer].lin_in, 'bias'):
        # Bias vector is defined per hidden channel (out_features). Select
        # the channels that belong to the current node (hidden_idxs).
        w1_bias = model.ResBlocks[layer].lin_in.bias[hidden_idxs].detach().numpy()
    else: 
        w1_bias = None

    lin1_smol = torch.nn.Linear(*w1_smol.shape)
    lin1_smol.weight = torch.nn.Parameter(torch.tensor(w1_smol.T, dtype=torch.float32))
    if w1_bias is not None: lin1_smol.bias = torch.nn.Parameter(torch.tensor(w1_bias.squeeze(), dtype=torch.float32))

    indices, values = pyg.utils.bipartite_subgraph(subset=(hidden_idxs, output_edges), 
                                                   edge_index=model.ResBlocks[layer].lin_out.indices, 
                                                   edge_attr=model.ResBlocks[layer].lin_out.values, 
                                                   relabel_nodes=True, 
                                                   return_edge_mask=False)

    w3_smol = scipy.sparse.coo_array((values.detach(), (indices[0,:].detach(), indices[1,:].detach())), shape=(N_channels, len(output_edges.view(-1)))).todense()
    
    if hasattr(model.ResBlocks[layer].lin_out, 'bias'): 
        w3_bias = model.ResBlocks[layer].lin_out.bias[output_edges].detach().numpy()
    else: 
        w3_bias = None

    lin3_smol = torch.nn.Linear(*w3_smol.shape)
    lin3_smol.weight = torch.nn.Parameter(torch.tensor(w3_smol.T, dtype=torch.float32))
    if w3_bias is not None: lin3_smol.bias = torch.nn.Parameter(torch.tensor(w3_bias.squeeze(), dtype=torch.float32))

    norm = getattr(model, 'norm', 'none')
    
    # Extract node MLP if present
    node_mlp = None
    if hasattr(model.ResBlocks[layer], 'node_mlp') and model.ResBlocks[layer].node_mlp and model.ResBlocks[layer].mlp is not None:
        # Extract the specific node batch norm params (one param per node)
        # Create a new MLP with node-specific batch norm running stats
        original_mlp = model.ResBlocks[layer].mlp
        
        # Create a new sequential module with extracted components
        new_layers = []
        for i, layer_module in enumerate(original_mlp):
            if isinstance(layer_module, torch.nn.BatchNorm1d):
                # Extract running stats for this specific node
                new_bn = torch.nn.BatchNorm1d(1, eps=layer_module.eps, momentum=layer_module.momentum, 
                                            affine=layer_module.affine, track_running_stats=layer_module.track_running_stats)
                
                # Copy the running stats for the specific node
                if layer_module.track_running_stats:
                    new_bn.running_mean.data = layer_module.running_mean[node_idx:node_idx+1].clone()
                    new_bn.running_var.data = layer_module.running_var[node_idx:node_idx+1].clone()
                    new_bn.num_batches_tracked.data = layer_module.num_batches_tracked.clone()
                
                # Copy affine parameters if present
                if layer_module.affine:
                    new_bn.weight.data = layer_module.weight[node_idx:node_idx+1].clone()
                    new_bn.bias.data = layer_module.bias[node_idx:node_idx+1].clone()
                
                new_layers.append(new_bn)
            else:
                # For non-BatchNorm layers, copy as-is
                new_layers.append(layer_module)
        
        node_mlp = torch.nn.Sequential(*new_layers)
        
    
    func = dense_func_node(lin_in=lin1_smol, lin_out=lin3_smol, nonlin=model.ResBlocks[layer].nonlin, norm=norm, node_mlp=node_mlp)
    func = func.eval()

    meta = {'input_edge_names':inp_edge_names, 'output_edge_names':out_edge_names}

    return func, meta