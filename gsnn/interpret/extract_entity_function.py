import torch 
import numpy as np 
import torch_geometric as pyg 
import scipy 
import gsnn.models.utils as utils
from gsnn.models.GSNN import get_conv_indices

class dense_func_node(torch.nn.Module): 
    def __init__(self, lin_in, lin_out, nonlin, norm): 
        super().__init__()
        self.lin_in = lin_in
        self.lin_out = lin_out
        # `nonlin` may be an instantiated module or a class – handle both
        self.nonlin = nonlin if isinstance(nonlin, torch.nn.Module) else nonlin()

        channels = lin_in.out_features  # hidden channels produced by `lin_in`

        # ------------------------------------------------------------------
        # Normalisation layers (mirrors gsnn.models.GSNN.ResBlock logic)
        # ------------------------------------------------------------------
        if norm == 'layer':
            self.norm = torch.nn.LayerNorm(channels, elementwise_affine=False)
            self.norm_first = True
        elif norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(channels, eps=1e-3, affine=False)
            self.norm_first = True
        elif norm in ('groupbatch', 'edgebatch'):
            # For extracted single-node functions there is no meaningful group/edge
            # structure anymore – fall back to Identity so the operation is a no-op.
            self.norm = torch.nn.Identity()
            self.norm_first = True
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
        x = self.lin_out(x) 
        return x




def extract_entity_function(node, model, data, layer=0): 
    '''
    
    Args: 
        node        (str)                   node name to extract
        model       (GSNN)                  reference model 
        data        (HeteroData)            graph data 
        layer       (int)                   layer to extract 

    Returns: 
        func        (torch.nn.Module)       extracted entity function 
        meta        (dict)                  input edge names, output edge names 
    '''

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
    func = dense_func_node(lin_in=lin1_smol, lin_out=lin3_smol, nonlin=model.ResBlocks[0].nonlin, norm=norm)

    meta = {'input_edge_names':inp_edge_names, 'output_edge_names':out_edge_names}

    return func, meta