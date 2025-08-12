"""
construct.py - Build pruned constraint networks for the GSNN model
=================================================================

This helper stitches together several edge lists into a directed
heterogeneous graph, removes nodes that do **not** lie on any path from
an *input* to an *output*, and returns the result as a PyTorch-Geometric
`HeteroData` object ready for use in GSNN.

Accepted edge tables (all `pd.DataFrame` with columns `src`, `dst`)
------------------------------------------------------------------
1. **input_edges**    - critical inputs  →  function nodes
2. **mediator_edges** - (optional) mediator inputs → function nodes.
   Mediators are *only* retained if they target a function node that
   survives the pruning step.
3. **function_edges** - function  →  function (latent logic)
4. **output_edges**   - function  →  output

ASCII toy graph
---------------
```
input_A      mediator_M
   │             │
   ▼             │
 func_X ◀────────┘
   │
   ▼
 out_Y
```
`mediator_M` is kept **only** if `func_X` remains on a viable
`input_A → … → out_Y` path.

Quick usage example
-------------------
```python
from gsnn.proc.construct import construct_network

data = construct_network(input_edges, function_edges,
                         output_edges, mediator_edges)
print(data.graph_summary)
```

Returned attributes
-------------------
* `data.node_names_dict` - mapping of node types to names
* `data.edge_index_dict` - edge indices (PyTorch tensors)
* `data.graph_summary`   - pruning statistics & graph metrics
* `data.build_args`      - parameters used to build the graph
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from .subset import subset_graph


def construct_network(input_edges, function_edges, output_edges, mediator_edges=None, 
                     depth=10, verbose=True):
    """Construct and filter constraint network for GSNN model.
    
    Args:
        input_edges: DataFrame with columns 'src', 'dst' for input→function edges
        function_edges: DataFrame with columns 'src', 'dst' for function→function edges  
        output_edges: DataFrame with columns 'src', 'dst' for function→output edges
        mediator_edges: Optional DataFrame with columns 'src', 'dst' for mediator→function edges (inputs that are only included if they target a function node retained in the final graph)
        depth: Maximum path length to consider (default: 10)
        verbose: If True, print progress and summary (default: True)
    
    Returns:
        HeteroData object with filtered network structure and metadata
    """
    if verbose: print('building candidate network...')
    # Build complete network
    G = nx.DiGraph()
    
    # Track node types and initial counts
    inputs, mediators, functions, outputs = set(), set(), set(), set()
    
    # Add edges and collect node types
    for _, row in input_edges.iterrows():
        G.add_edge(f"input_{row['src']}", f"func_{row['dst']}")
        inputs.add(row['src'])
        functions.add(row['dst'])
    
    if mediator_edges is not None:
        mediators.update(mediator_edges['src'].tolist())
        functions.update(mediator_edges['dst'].tolist())
    
    for _, row in function_edges.iterrows():
        G.add_edge(f"func_{row['src']}", f"func_{row['dst']}")
        functions.add(row['src'])
        functions.add(row['dst'])
    
    for _, row in output_edges.iterrows():
        G.add_edge(f"func_{row['src']}", f"out_{row['dst']}")
        functions.add(row['src'])
        outputs.add(row['dst'])
    
    initial_counts = {'input': len(inputs), 'mediator': len(mediators), 
                     'function': len(functions), 'output': len(outputs)}
    
    # Filter: keep only nodes on paths from inputs to outputs
    roots = [f"input_{i}" for i in inputs]
    leaves = [f"out_{o}" for o in outputs]
    G_filtered = subset_graph(G, depth, roots, leaves, verbose=verbose)

        # Add mediator nodes and edges after filtering, but only if they target a retained function node
    if mediator_edges is not None:
        if verbose: print('adding mediator edges...')
        for _, row in mediator_edges.iterrows():
            func_node = f"func_{row['dst']}"
            if func_node in G_filtered:
                med_node = f"med_{row['src']}"
                G_filtered.add_edge(med_node, func_node)
    
    # Extract filtered node sets
    node_names = {'input': [], 'function': [], 'output': []}
    node_to_idx = {'input': {}, 'function': {}, 'output': {}}
    
    for node in G_filtered.nodes():
        if node.startswith('input_') or node.startswith('med_'):
            name = node.split('_', 1)[1]
            if name not in node_to_idx['input']:
                node_to_idx['input'][name] = len(node_names['input'])
                node_names['input'].append(name)
        elif node.startswith('func_'):
            name = node.split('_', 1)[1]
            if name not in node_to_idx['function']:
                node_to_idx['function'][name] = len(node_names['function'])
                node_names['function'].append(name)
        elif node.startswith('out_'):
            name = node.split('_', 1)[1]
            if name not in node_to_idx['output']:
                node_to_idx['output'][name] = len(node_names['output'])
                node_names['output'].append(name)
    
    # Build edge tensors
    edges = {('input', 'to', 'function'): [], 
             ('function', 'to', 'function'): [],
             ('function', 'to', 'output'): []}
    
    for src, dst in G_filtered.edges():
        if (src.startswith('input_') or src.startswith('med_')) and dst.startswith('func_'):
            src_idx = node_to_idx['input'][src.split('_', 1)[1]]
            dst_idx = node_to_idx['function'][dst.split('_', 1)[1]]
            edges[('input', 'to', 'function')].append([src_idx, dst_idx])
        elif src.startswith('func_') and dst.startswith('func_'):
            src_idx = node_to_idx['function'][src.split('_', 1)[1]]
            dst_idx = node_to_idx['function'][dst.split('_', 1)[1]]
            edges[('function', 'to', 'function')].append([src_idx, dst_idx])
        elif src.startswith('func_') and dst.startswith('out_'):
            src_idx = node_to_idx['function'][src.split('_', 1)[1]]
            dst_idx = node_to_idx['output'][dst.split('_', 1)[1]]
            edges[('function', 'to', 'output')].append([src_idx, dst_idx])
    
    # Create HeteroData
    data = HeteroData()
    data.node_names_dict = node_names
    data.edge_index_dict = {k: torch.tensor(v, dtype=torch.long).T if v else torch.empty((2, 0), dtype=torch.long)
                            for k, v in edges.items()}
    
    # Calculate summary statistics
    n_nodes = sum(len(v) for v in node_names.values())
    n_edges = sum(e.shape[1] for e in data.edge_index_dict.values())
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
    clustering = nx.average_clustering(G_filtered.to_undirected()) if G_filtered.nodes() else 0
    
    # Count removed nodes
    final_inputs = len([n for n in G_filtered.nodes() if n.startswith('input_')])
    final_mediators = len([n for n in G_filtered.nodes() if n.startswith('med_')])
    final_functions = len([n for n in G_filtered.nodes() if n.startswith('func_')])
    final_outputs = len([n for n in G_filtered.nodes() if n.startswith('out_')])
    
    summary = {
        'inputs_included': final_inputs,
        'inputs_removed': initial_counts['input'] - final_inputs,
        'mediators_included': final_mediators,
        'mediators_removed': initial_counts['mediator'] - final_mediators,
        'functions_included': final_functions,
        'functions_removed': initial_counts['function'] - final_functions,
        'outputs_included': final_outputs,
        'outputs_removed': initial_counts['output'] - final_outputs,
        'total_nodes': n_nodes,
        'total_edges': n_edges,
        'density': density,
        'avg_degree': avg_degree,
        'avg_clustering': clustering
    }
    
    data.graph_summary = summary
    data.build_args = {'depth': depth, 'has_mediators': mediator_edges is not None}
    
    if verbose:
        print(f"\n{'='*50}")
        print("Network Construction Summary:")
        print(f"{'='*50}")
        print(f"Inputs: {final_inputs} included, {summary['inputs_removed']} removed")
        print(f"Mediators: {final_mediators} included, {summary['mediators_removed']} removed")
        print(f"Functions: {final_functions} included, {summary['functions_removed']} removed")
        print(f"Outputs: {final_outputs} included, {summary['outputs_removed']} removed")
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {n_nodes}")
        print(f"  Edges: {n_edges}")
        print(f"  Density: {density:.4f}")
        print(f"  Avg Degree: {avg_degree:.2f}")
        print(f"  Avg Clustering: {clustering:.4f}")
        print(f"{'='*50}\n")
    
    return data