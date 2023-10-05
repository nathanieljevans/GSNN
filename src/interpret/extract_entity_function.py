import torch 
import numpy as np 
import torch_geometric as pyg 
import scipy 
import src.models.utils as utils
from src.models.GSNN import get_conv_indices

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




def extract_entity_function(node, model, data, X, layer=0): 

    node_idx = data.node_names.tolist().index(node)

    #(data.edge_index == model.edge_index).all())

    # NOTE THESE ARE EDGE INDICES (NOT NODE INDICES)
    row,col = model.edge_index
    input_edges = (col == node_idx).nonzero().squeeze()
    output_edges = (row == node_idx).nonzero().squeeze()

    inp_edge_names = data.node_names[row[input_edges]]
    out_edge_names = data.node_names[col[output_edges]]

    # the hidden channel indices relevant to `node`
    # NOTE: this only works for scale_channels_by_degree = False
    function_nodes = (~(model.input_node_mask | model.output_node_mask)).nonzero(as_tuple=True)[0]
    
    w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups = get_conv_indices(model.edge_index, model.channels, function_nodes, False)

    if not (w1_indices.size() == model.ResBlocks[layer].lin1.indices.size()): 
        w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups = get_conv_indices(model.edge_index, model.channels, function_nodes, True)
    else: 
        print('NOTE: number of channels are scaled by degree.')
    
    assert (w1_indices == model.ResBlocks[layer].lin1.indices).all(), 'W1 indices do not match model W1 indices'

    channel_groups = np.array(channel_groups)

    hidden_idxs = torch.tensor((channel_groups == node_idx).nonzero()[0], dtype=torch.long)

    src,dst = model.ResBlocks[layer].lin1.indices 

    indices, values = pyg.utils.bipartite_subgraph(subset           = (input_edges, hidden_idxs), 
                                                   edge_index       = model.ResBlocks[layer].lin1.indices, 
                                                   edge_attr        = model.ResBlocks[layer].lin1.values.data, 
                                                   relabel_nodes    = True, 
                                                   return_edge_mask = False)
    
    N_channels = len(hidden_idxs)

    w1_smol = scipy.sparse.coo_array((values.detach(), (indices[0,:].detach(), indices[1,:].detach())), shape=(len(input_edges), N_channels)).todense()
    w1_bias = model.ResBlocks[layer].lin1.bias[hidden_idxs].detach().numpy()

    lin1_smol = torch.nn.Linear(*w1_smol.shape)
    lin1_smol.weight = torch.nn.Parameter(torch.tensor(w1_smol.T, dtype=torch.float32))
    lin1_smol.bias = torch.nn.Parameter(torch.tensor(w1_bias.squeeze(), dtype=torch.float32))

    if hasattr(model.ResBlocks[layer], 'lin2'): 
        indices, values = pyg.utils.bipartite_subgraph(subset=(hidden_idxs, hidden_idxs), edge_index=model.ResBlocks[layer].lin2.indices, edge_attr=model.ResBlocks[layer].lin2.values, relabel_nodes=True, return_edge_mask=False)

        w2_smol = scipy.sparse.coo_array((values.detach(), (indices[0,:].detach(), indices[1,:].detach())), shape=(model.channels, model.channels)).todense()
        w2_bias = model.ResBlocks[layer].lin2.bias[hidden_idxs].detach().numpy()

        lin2_smol = torch.nn.Linear(*w2_smol.shape)
        lin2_smol.weight = torch.nn.Parameter(torch.tensor(w2_smol.T, dtype=torch.float32))
        lin2_smol.bias = torch.nn.Parameter(torch.tensor(w2_bias.squeeze(), dtype=torch.float32))
    else: 
        lin2_smol = None

    indices, values = pyg.utils.bipartite_subgraph(subset=(hidden_idxs, output_edges), 
                                                   edge_index=model.ResBlocks[layer].lin3.indices, 
                                                   edge_attr=model.ResBlocks[layer].lin3.values, 
                                                   relabel_nodes=True, 
                                                   return_edge_mask=False)

    w3_smol = scipy.sparse.coo_array((values.detach(), (indices[0,:].detach(), indices[1,:].detach())), shape=(N_channels, len(output_edges.view(-1)))).todense()
    w3_bias = model.ResBlocks[layer].lin3.bias[output_edges].detach().numpy()

    lin3_smol = torch.nn.Linear(*w3_smol.shape)
    lin3_smol.weight = torch.nn.Parameter(torch.tensor(w3_smol.T, dtype=torch.float32))
    lin3_smol.bias = torch.nn.Parameter(torch.tensor(w3_bias.squeeze(), dtype=torch.float32))

    func = dense_func_node(lin1=lin1_smol, lin2=lin2_smol, lin3=lin3_smol, nonlin=model.ResBlocks[0].nonlin, norm='layer' if hasattr(model, 'norm') else 'none')

    meta = {'input_edge_names':inp_edge_names, 'output_edge_names':out_edge_names}

    eX = utils.node2edge(X, model.edge_index)
    ex_smol = eX[:, input_edges]

    return func, ex_smol, meta