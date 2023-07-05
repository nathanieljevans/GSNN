import torch

def get_W1_indices(edge_index, channels): 
    '''
    # how to create input layer , e.g., edge values -> node indices 


    '''
    row = [] 
    col = []
    for edge_id, (src, node_id) in enumerate(edge_index.detach().cpu().numpy().T):
        for k in range(channels): 
            row.append(edge_id)
            col.append(channels*node_id.item() + k)

    row = torch.tensor(row, dtype=torch.float32)
    col = torch.tensor(col, dtype=torch.float32)
    indices = torch.stack((row,col), dim=0)
    return indices


def get_W2_indices(function_nodes, channels): 
    '''
    # how to create node -> node latent weight indices 

    # for node_id in function_nodes 
        # for k in channels: 
            # for k2 in channels: 
                # add weight indice: (node_id + k, node_id + k2)
    '''
    row = []
    col = []
    for node_id in function_nodes: 
        for k in range(channels): 
            for k2 in range(channels): 
                row.append(channels*node_id.item() + k)
                col.append(channels*node_id.item() + k2)

    row = torch.tensor(row, dtype=torch.float32)
    col = torch.tensor(col, dtype=torch.float32)
    indices = torch.stack((row,col), dim=0)
    return indices

def get_W3_indices(edge_index, function_nodes, channels): 
    '''
    # how to create node -> edge indices 

    # for node_id in function_nodes 

        # filter to edges from node_id 
        # src, dst = edge_index
        # out_edges = (src == node_id).nonzero()
        # for k in channels: 
            # for out_edge_idx in out_edges: 
                # add weight indice:   (node_id + k, out_edge_idx)
    '''
    row = [] 
    col = []
    for node_id in function_nodes: 
        
        src,dst = edge_index 
        out_edges = (src == node_id).nonzero(as_tuple=True)[0]

        for k in range(channels):
            
            for edge_id in out_edges: 

                row.append(channels*node_id.item() + k)
                col.append(edge_id.item())

    row = torch.tensor(row, dtype=torch.float32)
    col = torch.tensor(col, dtype=torch.float32)
    indices = torch.stack((row,col), dim=0)
    return indices

def node2edge(x, edge_index): 
    '''
    convert from node indexed attributes to edge indexed attributes
    '''
    src,dst = edge_index 
    return x[:, src] 

def edge2node(x, edge_index, output_node_mask): 
    ''' 
    convert from edge indexed attributes `x` to node indexed attributes
    NOTE: only maps to output nodes (eg., in-degree = 1) to avoid collisions; all other nodes (input nodes + function nodes) will have value of 0. 
    '''
    output_nodes = output_node_mask.nonzero(as_tuple=True)[0]
    src, dst = edge_index 
    output_edge_mask = torch.isin(dst, output_nodes)

    B = x.size(0)
    out = torch.zeros(B, output_node_mask.size(0), dtype=torch.float32, device=x.device)
    out[:, dst[output_edge_mask].view(-1)] = x[:, output_edge_mask].view(B, -1)

    return out

