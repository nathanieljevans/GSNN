import torch 
import numpy as np 
import torch_geometric as pyg 
import scipy 
import src.models.utils as utils

class dense_func_node(torch.nn.Module): 
    def __init__(self, lin1, lin2, lin3, nonlin): 
        super().__init__()
        self.lin1 = lin1 
        self.lin2 = lin2 
        self.lin3 = lin3 
        self.nonlin = nonlin 

    def forward(self, x):
        x = self.lin1(x)
        x = self.nonlin(x)
        x = self.lin2(x)
        x = self.nonlin(x)
        x = self.lin3(x) 

        return x


def extract_entity_function(node, model, data, X): 

    eX = utils.node2edge(X, model.edge_index)

    node_idx = data.node_names.tolist().index(node)

    # NOTE THESE ARE EDGE INDICES (NOT NODE INDICES)
    row,col = data.edge_index
    input_edges = (col == node_idx).nonzero().squeeze()
    output_edges = (row == node_idx).nonzero().squeeze()

    inp_edge_names = data.node_names[row[input_edges]]
    out_edge_names = data.node_names[col[output_edges]]

    # the hidden channel indices relevant to `node`
    hidden_idxs = np.arange(node_idx*model.channels, node_idx*model.channels + model.channels) 

    ex_smol = eX[:, input_edges]

    indices, values = pyg.utils.bipartite_subgraph(subset=(input_edges, hidden_idxs), edge_index=model.lin1.indices, edge_attr=model.lin1.values, relabel_nodes=True, return_edge_mask=False)

    w1_smol = scipy.sparse.coo_array((values.detach(), (indices[0,:].detach(), indices[1,:].detach())), shape=(len(input_edges), model.channels)).todense()
    w1_bias = model.lin1.bias[hidden_idxs].detach().numpy()

    lin1_smol = torch.nn.Linear(*w1_smol.shape)
    lin1_smol.weight = torch.nn.Parameter(torch.tensor(w1_smol.T, dtype=torch.float32))
    lin1_smol.bias = torch.nn.Parameter(torch.tensor(w1_bias.squeeze(), dtype=torch.float32))

    indices, values = pyg.utils.bipartite_subgraph(subset=(hidden_idxs, hidden_idxs), edge_index=model.lin2.indices, edge_attr=model.lin2.values, relabel_nodes=True, return_edge_mask=False)

    w2_smol = scipy.sparse.coo_array((values.detach(), (indices[0,:].detach(), indices[1,:].detach())), shape=(model.channels, model.channels)).todense()
    w2_bias = model.lin2.bias[hidden_idxs].detach().numpy()

    lin2_smol = torch.nn.Linear(*w2_smol.shape)
    lin2_smol.weight = torch.nn.Parameter(torch.tensor(w2_smol.T, dtype=torch.float32))
    lin2_smol.bias = torch.nn.Parameter(torch.tensor(w2_bias.squeeze(), dtype=torch.float32))

    indices, values = pyg.utils.bipartite_subgraph(subset=(hidden_idxs, output_edges), edge_index=model.lin3.indices, edge_attr=model.lin3.values, relabel_nodes=True, return_edge_mask=False)

    w3_smol = scipy.sparse.coo_array((values.detach(), (indices[0,:].detach(), indices[1,:].detach())), shape=(model.channels, len(output_edges.view(-1)))).todense()
    w3_bias = model.lin3.bias[output_edges].detach().numpy()

    lin3_smol = torch.nn.Linear(*w3_smol.shape)
    lin3_smol.weight = torch.nn.Parameter(torch.tensor(w3_smol.T, dtype=torch.float32))
    lin3_smol.bias = torch.nn.Parameter(torch.tensor(w3_bias.squeeze(), dtype=torch.float32))

    func = dense_func_node(lin1_smol, lin2_smol, lin3_smol, nonlin=model.nonlin)

    meta = {'input_edge_names':inp_edge_names, 'output_edge_names':out_edge_names}

    return func, ex_smol, meta