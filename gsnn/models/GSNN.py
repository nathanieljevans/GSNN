import torch 
from gsnn.models import utils
from torch.utils.checkpoint import checkpoint_sequential
from torch.autograd import Variable
import torch_geometric as pyg
import numpy as np
from gsnn.models.ResBlock import ResBlock
import networkx as nx
from gsnn.proc.subset import subset_graph


def get_conv_indices(edge_index, channels, function_nodes, fix_hidden_channels): 
    E = edge_index.size(1)  
    w1_indices, node_hidden_channels = utils.get_W1_indices(edge_index, channels, function_nodes, scale_by_degree=not fix_hidden_channels)
    w2_indices = utils.get_W2_indices(function_nodes, node_hidden_channels)
    w3_indices = utils.get_W3_indices(edge_index, function_nodes, node_hidden_channels)
    w1_size = (E, np.sum(node_hidden_channels))
    w2_size = (np.sum(node_hidden_channels), np.sum(node_hidden_channels))
    w3_size = (np.sum(node_hidden_channels), E)

    channel_groups = [] 
    for node_id, c in enumerate(node_hidden_channels): 
        for i in range(c): 
            channel_groups.append(node_id)

    return (w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups)
    
def hetero2homo(edge_index_dict, node_names_dict): 

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

    return edge_index, input_node_mask, output_node_mask, num_nodes 

def batch_graphs(N, M, edge_index, B, device, edge_mask=None):
    '''
    Create batched edge_index/edge_weight tensors for bipartite graphs.

    Args:
        N (int): Size of the first set of nodes in each bipartite graph.
        M (int): Size of the second set of nodes in each bipartite graph.
        edge_index (tensor): edge index tensor to batch.
        B (int) batch size
        device (str)

    Returns:
        torch.Tensor: Batched edge index.
    '''
    E = edge_index.size(1)
    batched_edge_indices = edge_index.repeat(1, B).contiguous()
    batch_idx = torch.repeat_interleave(torch.arange(B, dtype=torch.long, device=device), E).contiguous()

    if edge_mask is not None: 
        ## this batching process takes much longer
        edge_subset = torch.cat([mask + E*i for i,mask in enumerate(edge_mask)], dim=-1) # this takes most of the time
        batched_edge_indices = batched_edge_indices[:, edge_subset]
        batch_idx = batch_idx[edge_subset]
    else: 
        edge_subset=None

    src_incr = batch_idx*N
    dst_incr = batch_idx*M
    incr = torch.stack((src_incr, dst_incr), dim=0)
    batched_edge_indices += incr

    return batched_edge_indices, edge_subset

def get_edge_masks(subgraph_dict, node_names_dict, edge_index, two_layers, indices_params): 
        
    N = len(node_names_dict['input']) + len(node_names_dict['function']) + len(node_names_dict['output'])  
    w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups = indices_params

    subgraph_dict_ = {}
    i = 0
    for var, (inp, func, out) in subgraph_dict.items(): 
        i+=1
        print(f'initializing subgraph indexing [part 1/2]: {i}/{len(subgraph_dict)}', end='\r')
        # convert to homo indexing 
        subset = torch.cat((inp + len(node_names_dict['function']),
                                func, 
                                out + len(node_names_dict['function']) + len(node_names_dict['input'])), dim=-1) 
        _,_,edge_mask = pyg.utils.subgraph(subset=subset, 
                                        edge_index=edge_index, 
                                        relabel_nodes=False, 
                                        num_nodes=N, 
                                        return_edge_mask=True)
        subgraph_dict_[var] = edge_mask
    print()

    subgraph_dict = subgraph_dict_

    # convert subgraph dict (var->subedges) to (var->edge_mask); one for each lin layer 
    # input edges are edge indexed (not node indexed)

    # build graph
    G = nx.DiGraph()
    # add 
    G.add_edges_from(['E__'+str(int(i.item())), 'C1__' + str(int(j.item()))] for i,j in zip(*w1_indices))
    if two_layers: 
        G.add_edges_from(['C1__'+str(int(i.item())), 'C2__' + str(int(j.item()))] for i,j in zip(*w2_indices))
        G.add_edges_from(['C2__'+str(int(i.item())), 'E__' + str(int(j.item()))] for i,j in zip(*w3_indices))
    else: 
        G.add_edges_from(['C1__'+str(int(i.item())), 'E__' + str(int(j.item()))] for i,j in zip(*w3_indices))

    depth = 3 if two_layers else 2

    # lin1 
    lin1_edge_mask_dict={}
    lin2_edge_mask_dict={}
    lin3_edge_mask_dict={}
    i=0
    distance_dicts=({},{})
    for var, edges in subgraph_dict.items(): 
        i+=1
        print(f'initializing subgraph indexing [part 2/2]: {i}/{len(subgraph_dict)}', end='\r')
        edges = edges.nonzero(as_tuple=True)[0].detach().cpu().numpy().tolist() # convert from mask to edge idxs 
        roots = leafs = ['E__' + str(x) for x in edges]
        sG, distance_dicts = subset_graph(G, depth=depth, roots=roots, leafs=leafs, verbose=True, distance_dicts=distance_dicts, return_dicts=True)

        lin1_edge_mask_dict[var] = torch.tensor([sG.has_edge('E__'+str(int(i)), 'C1__' + str(int(j))) for (i,j) in zip(*w1_indices)], dtype=torch.bool).nonzero(as_tuple=True)[0]
        if two_layers: 
            lin2_edge_mask_dict[var] = torch.tensor([sG.has_edge('C1__'+str(int(i)), 'C2__' + str(int(j))) for (i,j) in zip(*w2_indices)], dtype=torch.bool).nonzero(as_tuple=True)[0]
            lin3_edge_mask_dict[var] = torch.tensor([sG.has_edge('C2__'+str(int(i)), 'E__' + str(int(j))) for (i,j) in zip(*w3_indices)], dtype=torch.bool).nonzero(as_tuple=True)[0]
        else: 
            lin3_edge_mask_dict[var] = torch.tensor([sG.has_edge('C1__'+str(int(i)), 'E__' + str(int(j))) for (i,j) in zip(*w3_indices)], dtype=torch.bool).nonzero(as_tuple=True)[0]
    print()
    return lin1_edge_mask_dict, lin2_edge_mask_dict, lin3_edge_mask_dict

class GSNN(torch.nn.Module): 

    def __init__(self, edge_index_dict, node_names_dict, channels, layers, residual=True, dropout=0., 
                            nonlin=torch.nn.GELU, bias=True, share_layers=True, fix_hidden_channels=True, two_layer_conv=False, 
                                add_function_self_edges=False, norm='layer', init='kaiming', verbose=False, edge_channels=1, checkpoint=False,
                                  fix_inputs=True, dropout_type='node', subgraph_dict=None):
        super().__init__()

        # add multiple latent edge features per edge
        if edge_channels > 1:
            edge_index_dict['function', 'to', 'function'] = edge_index_dict['function', 'to', 'function'].repeat(1, edge_channels)

        # convert edge_index_dict to edge_index (homogenous)
        edge_index, input_node_mask, output_node_mask, self.num_nodes = hetero2homo(edge_index_dict, node_names_dict)
        self.share_layers = share_layers            # whether to share function node parameters across layers
        self.register_buffer('output_node_mask', output_node_mask)
        self.register_buffer('input_node_mask', input_node_mask)
        self.layers = layers 
        self.residual = residual
        self.channels = channels
        self.add_function_self_edges = add_function_self_edges
        self.fix_hidden_channels = fix_hidden_channels 
        self.two_layer_conv = two_layer_conv
        self.verbose = verbose
        self.edge_channels = edge_channels
        self.checkpoint = checkpoint
        self.fix_inputs = fix_inputs
        self.dropout_type = dropout_type

        function_nodes = (~(input_node_mask | output_node_mask)).nonzero(as_tuple=True)[0]
        if add_function_self_edges: 
            if verbose: print('Augmenting `edge index` with function node self-edges.')
            edge_index = torch.cat((edge_index, torch.stack((function_nodes, function_nodes), dim=0)), dim=1)
        self.register_buffer('edge_index', edge_index)
        self.E = self.edge_index.size(1)                             # number of edges 
        self.N = len(node_names_dict['input']) + len(node_names_dict['function']) + len(node_names_dict['output'])      # number of nodes

        self.register_buffer('function_edge_mask', torch.isin(edge_index[0], function_nodes)) # edges from a function node / e.g., not an input or output edge 
        self.register_buffer('input_edge_mask', self.input_node_mask[self.edge_index[0]].type(torch.float32))

        self.dropout = dropout

        self.indices_params = get_conv_indices(edge_index, channels, function_nodes, fix_hidden_channels)

        if subgraph_dict is not None: 
            self.lin1_edge_mask_dict, self.lin2_edge_mask_dict, self.lin3_edge_mask_dict = get_edge_masks(subgraph_dict, 
                                                                                            node_names_dict, 
                                                                                            edge_index=self.edge_index, 
                                                                                            two_layers=self.two_layer_conv, 
                                                                                            indices_params=self.indices_params)

        _n = 1 if self.share_layers else self.layers
        self.ResBlocks = torch.nn.ModuleList([ResBlock(self.edge_index, channels, function_nodes, fix_hidden_channels, 
                                                       bias, nonlin, residual=residual, two_layers=two_layer_conv, 
                                                       dropout=dropout, norm=norm, init=init, fix_inputs=fix_inputs, 
                                                       dropout_type=dropout_type, indices_params=self.indices_params) for i in range(_n)])

    def get_batch_params(self, B, subgraph, device): 

         # get edge masks
        if subgraph is not None: 
            if self.two_layer_conv: 
                edge_mask_params = ([self.lin1_edge_mask_dict[v] for v in subgraph],
                                    [self.lin2_edge_mask_dict[v] for v in subgraph],
                                    [self.lin3_edge_mask_dict[v] for v in subgraph])
            else: 
                edge_mask_params = ([self.lin1_edge_mask_dict[v] for v in subgraph],
                                    None,
                                    [self.lin3_edge_mask_dict[v] for v in subgraph])
        else: 
            edge_mask_params=[None, None, None]



        # precompute edge batching so it doesn't have to be done in every resblock 
        batched_edge_indices1, edge_subset1 = batch_graphs(N=self.ResBlocks[0].lin1.N,
                                                         M=self.ResBlocks[0].lin1.M, 
                                                         edge_index = self.ResBlocks[0].lin1.indices, 
                                                         B=B, 
                                                         device=device, 
                                                         edge_mask=edge_mask_params[0])
        
        if self.two_layer_conv: 
            batched_edge_indices2, edge_subset2 = batch_graphs(N=self.ResBlocks[0].lin2.N,
                                                            M=self.ResBlocks[0].lin2.M, 
                                                            edge_index = self.ResBlocks[0].lin2.indices, 
                                                            B=B, 
                                                            device=device, 
                                                            edge_mask=edge_mask_params[1])
        else: 
            batched_edge_indices2=edge_subset2=None
        
        batched_edge_indices3, edge_subset3 = batch_graphs(N=self.ResBlocks[0].lin3.N,
                                                         M=self.ResBlocks[0].lin3.M, 
                                                         edge_index = self.ResBlocks[0].lin3.indices, 
                                                         B=B, 
                                                         device=device, 
                                                         edge_mask=edge_mask_params[2])
        
        return (batched_edge_indices1, batched_edge_indices2, batched_edge_indices3, edge_subset1, edge_subset2, edge_subset3)

    def forward(self, x, mask=None, subgraph=None):
        '''
        Assumes x is the values of the "input" nodes ONLY
        ''' 
        B = x.size(0)
        x_node = torch.zeros((B, self.num_nodes), device=x.device, dtype=torch.float32)
        x_node[:, self.input_node_mask] = x

        x = utils.node2edge(x_node, self.edge_index)  # convert x to edge-indexed
        
        x0 = x.clone()
        if self.share_layers: 
            modules = [self.ResBlocks[0] for i in range(self.layers)]
        else: 
            modules = [self.ResBlocks[i] for i in range(self.layers)]

        if mask is not None: 
            for mod in modules: mod.set_mask(mask)
        if not self.residual: 
            for mod in modules: mod.set_x0(x0)

        batch_params = self.get_batch_params(B, subgraph, x.device)
        
        if self.checkpoint and self.training: 
            x = Variable(x, requires_grad=True)
            x = checkpoint_sequential(functions=modules, segments=self.layers, input=x, use_reentrant=True)
        else:
            for mod in modules: x = mod(x, batch_params)

        if self.residual: x /= self.layers

        out = utils.edge2node(x, self.edge_index, self.output_node_mask)

        # NOTE: returns only the "output" nodes 
        return out[:, self.output_node_mask]

