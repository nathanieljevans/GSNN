import numpy as np
from collections import deque
import networkx as nx


def bfs_distance(G, start_node, depth, node_names):
    r"""Perform breadth-first search from a start node and return shortest path distances.

    This function computes the shortest path distances from a starting node to all reachable
    nodes within a specified maximum depth. The distances are returned as a numpy array
    where unreachable nodes are marked with infinity.

    Args:
        G (networkx.DiGraph): A directed graph to perform BFS on.
        start_node: The starting node for the breadth-first search.
        depth (int): The maximum depth to explore from the start node.
        node_names (list): List of all node names in the graph, used for indexing.

    Returns:
        numpy.ndarray: An array of shape :obj:`[len(node_names)]` where each element
            represents the shortest path distance from the start node to the corresponding
            node. Unreachable nodes are marked with :obj:`np.inf`.

    Example:
        >>> import networkx as nx
        >>> G = nx.DiGraph()
        >>> G.add_edges_from([(0, 1), (1, 2), (0, 3), (3, 2)])
        >>> node_names = [0, 1, 2, 3]
        >>> distances = bfs_distance(G, 0, 2, node_names)
        >>> print(distances)  # [0, 1, 2, 1]
    """
    distances = {start_node: 0}
    queue = deque([(start_node, 0)])  # Store tuples (node, depth)

    while queue:
        current_node, current_depth = queue.popleft()

        # If we have reached the maximum depth, skip further exploration
        if current_depth > depth:
            continue

        for neighbor in G.successors(current_node):
            if neighbor not in distances:
                distances[neighbor] = current_depth + 1
                queue.append((neighbor, current_depth + 1))

    # convert to array for ease of selection later
    node2idx = {name:i for i,name in enumerate(node_names)}
    out = np.inf*np.ones((len(node_names),))
    for node, dist in distances.items(): 
        out[node2idx[node]] = dist
    return out

def get_all_possible_paths_set(G, rG, root, leaf, depth, root_distance_dict, leaf_distance_dict, node_names):
    r"""Compute shortest path lengths from root to leaf that pass through each node.

    This function calculates the shortest path length from a root node to a leaf node
    that goes through each node in the graph. For each node n, it computes:
    min_path_length(root → n → leaf) = distance(root → n) + distance(n → leaf)
    
    The algorithm uses forward BFS from the root and reverse BFS from the leaf,
    then combines the distances to find the shortest path through each node.
    Caching dictionaries are used to avoid recomputing distances for the same nodes.

    Args:
        G (networkx.DiGraph): The original directed graph.
        rG (networkx.DiGraph): The reverse of the original graph (all edges flipped).
        root: The starting node (root node).
        leaf: The target node (leaf node).
        depth (int): The maximum depth to explore in BFS calculations.
        root_distance_dict (dict): Cache dictionary for root node distances.
        leaf_distance_dict (dict): Cache dictionary for leaf node distances.
        node_names (list): List of all node names in the graph.

    Returns:
        tuple: A tuple containing:
            - spl (numpy.ndarray): Shortest path lengths from root to leaf through each node.
                Shape :obj:`[len(node_names)]` where each element represents the minimum
                path length from root to leaf that passes through the corresponding node.
                Nodes that cannot be reached from root or cannot reach leaf are marked with :obj:`np.inf`.
            - root_distance_dict (dict): Updated cache of root distances.
            - leaf_distance_dict (dict): Updated cache of leaf distances.

    Example:
        >>> import networkx as nx
        >>> G = nx.DiGraph()
        >>> G.add_edges_from([(0, 1), (1, 2), (0, 3), (3, 2)])
        >>> rG = G.reverse()
        >>> node_names = [0, 1, 2, 3]
        >>> root_dist_dict = {}
        >>> leaf_dist_dict = {}
        >>> spl, root_dict, leaf_dict = get_all_possible_paths_set(
        ...     G, rG, 0, 2, 3, root_dist_dict, leaf_dist_dict, node_names
        ... )
        >>> print(spl)  # [inf, 2, 0, 2] - shortest path lengths through each node
        >>> # Node 0: inf (cannot reach leaf 2)
        >>> # Node 1: 2 (path 0→1→2)
        >>> # Node 2: 0 (is the leaf itself)
        >>> # Node 3: 2 (path 0→3→2)
    """

    # Step 1: Perform BFS from the root
    if root in root_distance_dict: 
        root_distances = root_distance_dict[root]
    else: 
        root_distances = bfs_distance(G, root, depth=depth, node_names=node_names)
        root_distance_dict[root] = root_distances

    # Step 2: Perform reverse BFS from the leaf
    if leaf in leaf_distance_dict: 
        leaf_distances = leaf_distance_dict[leaf]
    else: 
        leaf_distances = bfs_distance(rG, leaf, depth=depth, node_names=node_names)
        leaf_distance_dict[leaf] = leaf_distances

    spl = root_distances + leaf_distances 
    return spl, root_distance_dict, leaf_distance_dict 



def subset_graph(G, depth, roots, leafs, verbose=True, distance_dicts=None, return_dicts=False): 
    r"""Subset a graph to include only nodes that lie on paths from roots to leaves.

    This function creates a subgraph by identifying nodes that have at least one path
    from any root node to any leaf node within the specified depth. The algorithm
    computes shortest path distances from all roots to all leaves and includes nodes
    that lie on paths of length less than or equal to the specified depth.

    Args:
        G (networkx.DiGraph): The original directed graph to subset.
        depth (int): The maximum path length to consider when determining node inclusion.
        roots (list): List of root node identifiers.
        leafs (list): List of leaf node identifiers.
        verbose (bool, optional): If :obj:`True`, print progress information. (default: :obj:`True`)
        distance_dicts (tuple, optional): Pre-computed distance dictionaries for caching.
            Should be a tuple of (root_distance_dict, leaf_distance_dict). (default: :obj:`None`)
        return_dicts (bool, optional): If :obj:`True`, return the distance dictionaries
            along with the subgraph for potential reuse. (default: :obj:`False`)

    Returns:
        networkx.DiGraph or tuple: If :obj:`return_dicts=False`, returns the subgraph.
            If :obj:`return_dicts=True`, returns a tuple containing:
            - subgraph (networkx.DiGraph): The subsetted graph
            - distance_dicts (tuple): Cached distance dictionaries for reuse

    Example:
        >>> import networkx as nx
        >>> G = nx.DiGraph()
        >>> G.add_edges_from([(0, 1), (1, 2), (0, 3), (3, 2), (2, 4), (4, 5)])
        >>> roots = [0]
        >>> leafs = [2, 5]
        >>> subgraph = subset_graph(G, depth=3, roots=roots, leafs=leafs)
        >>> print(list(subgraph.nodes()))  # [0, 1, 2, 3] - nodes on paths to targets
    """
    rG = G.reverse() # reverse all edges 
    node_names = sorted(list(G.nodes()))

    node_mask = np.zeros((len(node_names,)))
    if distance_dicts is not None: 
        root_distance_dict, leaf_distance_dict = distance_dicts
    else: 
        root_distance_dict = {}
        leaf_distance_dict = {}
    for i,root in enumerate(roots): 
        for j,leaf in enumerate(leafs): 
            if verbose: print(f'subgraph progress: {i+1}/{len(roots)} [{j+1}/{len(leafs)}]', end='\r')
            spl, root_distance_dict, leaf_distance_dict = get_all_possible_paths_set(G, 
                                                                                          rG,
                                                                                          root=root, 
                                                                                          leaf = leaf, 
                                                                                          depth=depth,
                                                                                          root_distance_dict = root_distance_dict,
                                                                                          leaf_distance_dict = leaf_distance_dict,
                                                                                          node_names = node_names)
            
            node_mask += 1.*(spl <= depth) 

    node_mask = node_mask > 0 # could threshold on a value here; interpretation: minimum number of paths from drugs->outputs that go through a given node
    nodes = set(np.array(node_names)[node_mask].tolist())
    subgraph = G.subgraph(nodes)

    # unfreeze the digraph 
    subgraph = subgraph.copy()

    if return_dicts:
        return subgraph, (root_distance_dict, leaf_distance_dict)
    else: 
        return subgraph
    


def build_nx(func_df, targets, outputs): 
    r"""Build a NetworkX directed graph from function interactions, drug targets, and outputs.

    This function constructs a heterogeneous directed graph with three types of nodes:
    - Drug nodes (prefixed with 'DRUG__')
    - Function/Protein nodes (prefixed with 'PROTEIN__') 
    - RNA/Output nodes (prefixed with 'RNA__' and 'LINCS__')

    The graph represents a biological signaling network where drugs target proteins,
    proteins interact with each other, and proteins regulate RNA outputs.

    Args:
        func_df (pandas.DataFrame): DataFrame containing protein-protein interactions.
            Must have columns 'source' and 'target' representing interacting proteins.
        targets (pandas.DataFrame): DataFrame containing drug-target interactions.
            Must have columns 'pert_id' (drug identifier) and 'target' (protein target).
        outputs (list): List of RNA/gene identifiers that represent the outputs.

    Returns:
        networkx.DiGraph: A directed graph representing the biological network with
            drug-protein, protein-protein, and protein-RNA interactions.

    Example:
        >>> import pandas as pd
        >>> func_df = pd.DataFrame({
        ...     'source': ['PROTEIN__A', 'PROTEIN__B'],
        ...     'target': ['PROTEIN__B', 'PROTEIN__C']
        ... })
        >>> targets = pd.DataFrame({
        ...     'pert_id': ['drug1', 'drug2'],
        ...     'target': ['A', 'B']
        ... })
        >>> outputs = ['gene1', 'gene2']
        >>> G = build_nx(func_df, targets, outputs)
        >>> print(list(G.nodes()))  # ['DRUG__drug1', 'DRUG__drug2', 'PROTEIN__A', ...]
    """
    G = nx.DiGraph()

    # function -> function 
    for i,edge in func_df.iterrows(): 
        G.add_edge(edge.source, edge.target)

    # drug -> function
    for i,edge in targets.iterrows(): 
        if ('PROTEIN__' + edge.target) in G: 
            G.add_edge('DRUG__' + edge.pert_id, 'PROTEIN__' + edge.target)
        else: 
            print(f'warning: {edge.target} is not present in graph, this DTI will not be added.')

    # function -> output edges
    for out in outputs: 
        # add the edge even if the RNA doesn't exist; will get filtered in next step
        G.add_edge('RNA__' + out, 'LINCS__' + out)

    return G
