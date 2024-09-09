import torch 
import numpy as np 
import torch_geometric as pyg 
import scipy 
import gsnn.models.utils as utils
from gsnn.models.GSNN import get_conv_indices

class dense_func_node(torch.nn.Module): 
    def __init__(self, lin1, lin3, nonlin, norm, lin2=None): 
        super().__init__()
        self.lin1 = lin1 
        self.lin2 = lin2 
        self.lin3 = lin3 
        self.nonlin = nonlin 
        
        if norm == 'layer': 
            channels = lin1.size(1)
            self.norm = torch.nn.LayerNorm(channels, elementwise_affine=False)

    def forward(self, x):
        x = self.lin1(x)
        if hasattr(self, 'norm'): x = self.norm(x)
        x = self.nonlin(x)
        if self.lin2 is not None: 
            x = self.lin2(x)
            if hasattr(self, 'norm'): x = self.norm(x)
            x = self.nonlin(x)
        x = self.lin3(x) 

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
    assert model.fix_hidden_channels == True, 'degree scaled function nodes are not supported for entity extraction'
    assert model.two_layer_conv == False, 'two layer function nodes are not supported for entity extraction'

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
    
    w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups = get_conv_indices(model.edge_index, model.channels, function_nodes, fix_hidden_channels=True)
    
    assert (w1_indices == model.ResBlocks[layer].lin1.indices).all(), 'W1 indices do not match model W1 indices'

    channel_groups = np.array(channel_groups)
    hidden_idxs = torch.tensor((channel_groups == node_idx).nonzero()[0], dtype=torch.long)
    N_channels = len(hidden_idxs)

    # we have a bipartite network from edge_idx -> function node hidden layers
    indices, values = pyg.utils.bipartite_subgraph(subset           = (input_edges, hidden_idxs), 
                                                   edge_index       = model.ResBlocks[layer].lin1.indices, 
                                                   edge_attr        = model.ResBlocks[layer].lin1.values.data, 
                                                   relabel_nodes    = True, 
                                                   return_edge_mask = False,
                                                   size             = (model.edge_index.size(1), len(channel_groups)))
    
    w1_smol = scipy.sparse.coo_array((values.detach(), (indices[0,:].detach(), indices[1,:].detach())), shape=(len(input_edges), N_channels)).todense()
    
    if hasattr(model.ResBlocks[layer].lin1, 'bias'): 
        w1_bias = model.ResBlocks[layer].lin1.bias[output_edges].detach().numpy()
    else: 
        w1_bias = None

    lin1_smol = torch.nn.Linear(*w1_smol.shape)
    lin1_smol.weight = torch.nn.Parameter(torch.tensor(w1_smol.T, dtype=torch.float32))
    if w1_bias is not None: lin1_smol.bias = torch.nn.Parameter(torch.tensor(w1_bias.squeeze(), dtype=torch.float32))

    indices, values = pyg.utils.bipartite_subgraph(subset=(hidden_idxs, output_edges), 
                                                   edge_index=model.ResBlocks[layer].lin3.indices, 
                                                   edge_attr=model.ResBlocks[layer].lin3.values, 
                                                   relabel_nodes=True, 
                                                   return_edge_mask=False)

    w3_smol = scipy.sparse.coo_array((values.detach(), (indices[0,:].detach(), indices[1,:].detach())), shape=(N_channels, len(output_edges.view(-1)))).todense()
    
    if hasattr(model.ResBlocks[layer].lin3, 'bias'): 
        w3_bias = model.ResBlocks[layer].lin3.bias[output_edges].detach().numpy()
    else: 
        w3_bias = None

    lin3_smol = torch.nn.Linear(*w3_smol.shape)
    lin3_smol.weight = torch.nn.Parameter(torch.tensor(w3_smol.T, dtype=torch.float32))
    if w3_bias is not None: lin3_smol.bias = torch.nn.Parameter(torch.tensor(w3_bias.squeeze(), dtype=torch.float32))

    norm = 'layer' if hasattr(model, 'norm') else 'none'
    func = dense_func_node(lin1=lin1_smol, lin2=None, lin3=lin3_smol, nonlin=model.ResBlocks[0].nonlin, norm=norm)

    meta = {'input_edge_names':inp_edge_names, 'output_edge_names':out_edge_names}

    return func, meta