'''
Implement simply coarsening of a network. 

should have various algorithms implemented and organized. 

Importantly, the coarsening methods must not change the connectivity of the network, for instance, 
the list of descendents/ancestors of a grouped node should be equivalent to any of the individual nodes that were grouped. 

valid node aggregations would be:

original:   A (input) -> B -> C -> D (output)
coarsened:  A (input) -> BC (function) -> D (output)


# Now I'm wondering if this example is really valid... 
# arguably, if there are no other constraints, then we can't specify B,C by strucutre since they have identical descendents and ancestors. 
                  /-> B --\
original:(input) A         D (output) 
                  \-> C --/

coarsened:  A (input) -> BC (function) -> D (output)



It's also possible that there are "off shoots" that are valid paths, but could be largely simplified. 

A 
|
V     
B <-> C <-> D <-> E 
|
V
C

In this case, C,D,E are valid paths, but don't are cycles that could be aggregated into B. 


To generalize these cases, let's try to define what a valid aggregation is. 

Let I be all input nodes in the network and O be all output nodes. 
It is a valid aggregation of a set of nodes (N) if for every node i in N: 

ancestors(i) \intersection I = ancestors(N) \intersection I 
and 
descedants(i) \intersection I = descedants(N) \intersection I



Call this IO equivalence: 
- Input ancestor equivalence 
- Output descedent equivalence 


Arguably, if any two nodes have IO equivalence, then we can't define them only by their structure. 
Although, there might be an argument that path lengths will impact the IO equivalence, e.g., 
A,B are IO equivalent but A has bath 1 to I0 and B has path 20 to I0. 

We could use a diffusion process to capture path lengths and structure.

This would need to be done with one diffusion channel per input/output node. 

'''

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import degree, from_networkx, scatter



def io_equivalence(G, input_nodes, function_nodes, output_nodes): 

    res = {} 
    for f in function_nodes: 

        input_ancestors = set(list(nx.ancestors(G, f)))
        output_descendants = set(list(nx.descendants(G, f)))

        res[f] = {'input_ancestors': input_ancestors, 'output_descendants': output_descendants}

    # now compute IO equivalence between nodes 
    input_equivalence = np.zeros((len(function_nodes), len(function_nodes)))
    output_equivalence = np.zeros((len(function_nodes), len(function_nodes)))

    for i, f1 in enumerate(function_nodes): 
        for j, f2 in enumerate(function_nodes): 
            if f1 == f2: continue 

            input_equivalence[i, j] = len(res[f1]['input_ancestors'].intersection(res[f2]['input_ancestors']))
            output_equivalence[i, j] = len(res[f1]['output_descendants'].intersection(res[f2]['output_descendants']))

    return input_equivalence, output_equivalence




def diff_equivalence(G, sources, nodes, iters: int = 10):
    """Compute diffusion-based relational equivalence scores.

    A *channel* is defined for every node contained in *sources*.  Diffusion is
    performed independently inside each channel, meaning that information from
    one source never influences another source.  Concretely, we initialise a
    one–hot vector for every source (shape :math:`[N, |sources|]`, where *N* is
    the number of nodes in *G*).  At every iteration the feature matrix is
    propagated along *out-going* edges using a row-normalised transition
    matrix.  After *iters* iterations we return the diffusion values for the
    queried *nodes*.

    Parameters
    ----------
    G : networkx.DiGraph
        The graph on which diffusion is performed.
    sources : list
        A list of node identifiers that act as sources/channels.
    nodes : list
        Nodes (subset of *G*) for which diffusion scores are returned.
    iters : int, optional (default=100)
        Number of synchronous diffusion steps.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(nodes), len(sources))`` containing diffusion
        scores.  ``scores[i, j]`` corresponds to the amount of signal that
        reached ``nodes[i]`` from ``sources[j]`` after ``iters`` steps.
    """

    # Convert *G* into a PyG *Data* object and extract edge index.
    data = from_networkx(G)
    edge_index = data.edge_index  # [2, E]
    num_nodes = data.num_nodes

    # Build mapping from node identifier to positional index used by PyG.
    nodelist = list(G.nodes())
    node2idx = {n: i for i, n in enumerate(nodelist)}

    # Sanity checks to ensure all requested nodes exist.
    for n in sources + nodes:
        if n not in node2idx:
            raise ValueError(f"Node '{n}' is not present in graph.")

    source_indices = [node2idx[s] for s in sources]
    target_indices = [node2idx[n] for n in nodes]

    # Initialise feature matrix ``x`` with one-hot encoding for each source.
    x = torch.zeros((num_nodes, len(source_indices)), dtype=torch.float)
    for col, idx in enumerate(source_indices):
        x[idx, col] = 1.0

    # Pre-compute out-degrees for normalisation (avoid division by zero).
    out_deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    out_deg[out_deg == 0.0] = 1.0  # prevents NaNs for sink nodes

    # Perform synchronous diffusion for the specified number of iterations.
    src, dst = edge_index  # unpack for readability
    for _ in range(iters):
        print(f'diffusion iteration {_} of {iters}', end='\r')
        # Distribute signal from *src* to *dst* (row-normalised by out-degree).
        messages = x[src] / out_deg[src].unsqueeze(1)
        x = scatter(messages, dst, dim=0, dim_size=num_nodes, reduce='sum')

    # Extract and return diffusion values for the requested nodes.
    scores = x[target_indices].detach().cpu().numpy()
    return scores



def diff_io_equivalence(G, input_nodes, function_nodes, output_nodes, iters: int = 10):
    """Placeholder for future implementation of combined diffusion-based
    input/output equivalence.  Currently unimplemented.

    This function remains as a stub so that *coarsen.py* can be imported
    without raising syntax errors.  It will be completed in a future commit.
    """
    input_scores = diff_equivalence(G, input_nodes, function_nodes, iters)
    output_scores = diff_equivalence(G.reverse(), output_nodes, function_nodes, iters)

    io_scores = np.concatenate((input_scores, output_scores), axis=-1)

    return io_scores





