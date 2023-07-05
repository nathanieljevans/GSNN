def get_degree(edge_index, idx): 
    '''
    return the in/out degree of a given node
    '''
    src,dst = edge_index 
    
    out_degree = (src == idx).sum()
    in_degree = (dst == idx).sum()

    return in_degree, out_degree


def get_in_out_ixs(edge_index, idx): 
    '''
    Return the edge id (e.g., index of each edge in `edge_index` that corresponds to a given node `idx` inputs/outputs)

    example: 
        edge_index = [[0, 1, 2, 3],
                      [1, 2, 3, 4]]
                       |  |  |  | 
        id/index       0  1  2  3

        if idx = 1, return the edge ids to/from node 1

        in_ixs = [0], out_ixs = [1]
    '''
    src,dst = edge_index 
    
    in_ixs = (dst == idx).nonzero().view(-1)
    out_ixs= (src == idx).nonzero().view(-1)

    return in_ixs, out_ixs


def get_children(edge_index, idx): 
    '''
    return the descendents of a given node. 
    '''
    src,dst = edge_index 
    return dst[src == idx]