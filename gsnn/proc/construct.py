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
from gsnn.proc.construct import GSNNNetworkConstructor

builder = GSNNNetworkConstructor(depth=10, verbose=True)
data = builder.build(input_edges, function_edges,
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




class GSNNNetworkConstructor:

    def __init__(self, depth=10, 
                       verbose=True):

        self.depth = depth 
        self.verbose = verbose
        

    def build(self, input_edges, 
                    function_edges, 
                    output_edges, 
                    mediator_edges=None,
                    input_names=None,
                    output_names=None,
                    function_names=None,
                    mediator_names=None):
        """Construct and filter constraint network for GSNN model.
        
        Behavior of fixed node name arguments:
        - input_names, mediator_names, function_names, output_names specify nodes that MUST
          be included in the final node lists, even if they have no incident edges.
        - Additional nodes of each type that are inferred from the provided edge tables are
          also included. In other words, the final node sets are supersets of the fixed names.
        - Inputs and mediators are combined into a single 'input' type in the order
          input_names + mediator_names, followed by any additional discovered inputs.
        
        Args:
            input_edges: DataFrame with columns 'src', 'dst' for input→function edges
            function_edges: DataFrame with columns 'src', 'dst' for function→function edges  
            output_edges: DataFrame with columns 'src', 'dst' for function→output edges
            mediator_edges: Optional DataFrame with columns 'src', 'dst' for mediator→function edges (only retained if they target a function node retained in the pruned graph)
            input_names: Optional list of input node names to force-include
            mediator_names: Optional list of mediator node names to force-include (combined with inputs)
            function_names: Optional list of function node names to force-include
            output_names: Optional list of output node names to force-include
        
        Returns:
            HeteroData object with filtered network structure and metadata
        """
        if self.verbose: print('building candidate network...')
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
        
        # If fixed name lists are provided, ensure they contribute to initial counts
        if input_names is not None:
            inputs.update(list(input_names))
        if mediator_names is not None:
            mediators.update(list(mediator_names))
        if function_names is not None:
            functions.update(list(function_names))
        if output_names is not None:
            outputs.update(list(output_names))

        initial_counts = {'input': len(inputs), 'mediator': len(mediators), 
                          'function': len(functions), 'output': len(outputs)}
        
        # Filter: keep only nodes on paths from inputs to outputs
        # Only include roots/leaves that actually exist in G to avoid BFS on missing nodes
        roots = [f"input_{i}" for i in inputs if f"input_{i}" in G]
        leaves = [f"out_{o}" for o in outputs if f"out_{o}" in G]
        G_filtered = subset_graph(G, self.depth, roots, leaves, verbose=self.verbose)
        if self.verbose: print() 

        # Add mediator nodes and edges after filtering, but only if they target a retained function node
        if mediator_edges is not None:
            if self.verbose: print('adding mediator edges...')
            for _, row in mediator_edges.iterrows():
                func_node = f"func_{row['dst']}"
                if func_node in G_filtered:
                    med_node = f"med_{row['src']}"
                    G_filtered.add_edge(med_node, func_node)

        # Helper to deduplicate while preserving order
        def _dedupe_preserve_order(sequence):
            seen = set()
            result = []
            for item in sequence:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result

        # Build node name lists honoring any fixed name inputs
        node_names = {'input': [], 'function': [], 'output': []}

        # Inputs (combine inputs + mediators if provided; maintain order input_names + mediator_names)
        discovered_inputs = []
        for node in G_filtered.nodes():
            if node.startswith('input_') or node.startswith('med_'):
                discovered_inputs.append(node.split('_', 1)[1])
        discovered_inputs = _dedupe_preserve_order(discovered_inputs)

        fixed_inputs = []
        if input_names is not None or mediator_names is not None:
            fixed_inputs = (list(input_names) if input_names is not None else []) + \
                           (list(mediator_names) if mediator_names is not None else [])
            fixed_inputs = _dedupe_preserve_order(fixed_inputs)
            node_names['input'] = _dedupe_preserve_order(list(fixed_inputs) + [n for n in discovered_inputs if n not in fixed_inputs])
        else:
            node_names['input'] = discovered_inputs

        # Functions
        discovered_functions = []
        for node in G_filtered.nodes():
            if node.startswith('func_'):
                discovered_functions.append(node.split('_', 1)[1])
        discovered_functions = _dedupe_preserve_order(discovered_functions)

        if function_names is not None:
            fn_fixed = _dedupe_preserve_order(list(function_names))
            node_names['function'] = _dedupe_preserve_order(list(fn_fixed) + [n for n in discovered_functions if n not in fn_fixed])
        else:
            node_names['function'] = discovered_functions

        # Outputs
        discovered_outputs = []
        for node in G_filtered.nodes():
            if node.startswith('out_'):
                discovered_outputs.append(node.split('_', 1)[1])
        discovered_outputs = _dedupe_preserve_order(discovered_outputs)

        if output_names is not None:
            out_fixed = _dedupe_preserve_order(list(output_names))
            node_names['output'] = _dedupe_preserve_order(list(out_fixed) + [n for n in discovered_outputs if n not in out_fixed])
        else:
            node_names['output'] = discovered_outputs

        # Build index maps from finalized node lists
        node_to_idx = {
            'input': {name: idx for idx, name in enumerate(node_names['input'])},
            'function': {name: idx for idx, name in enumerate(node_names['function'])},
            'output': {name: idx for idx, name in enumerate(node_names['output'])},
        }
        
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
        
        # Create HeteroData container and attach metadata
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
        
        # Count removed nodes (use finalized node lists for inputs/functions/outputs)
        final_inputs = len(node_names['input'])
        # Mediators are combined with inputs; estimate mediator inclusion from provided names and discovered 'med_' nodes
        if mediator_names is not None:
            final_mediators = len(_dedupe_preserve_order(list(mediator_names)))
        else:
            final_mediators = len([n for n in G_filtered.nodes() if n.startswith('med_')])
        final_functions = len(node_names['function'])
        final_outputs = len(node_names['output'])
        
        # Calculate isolated nodes (nodes with no incident edges)
        isolated_inputs = 0
        isolated_outputs = 0
        
        # Count isolated inputs (inputs with no outgoing edges)
        input_edges = edges[('input', 'to', 'function')]
        connected_inputs = set()
        for edge in input_edges:
            connected_inputs.add(edge[0])  # edge[0] is the input node index
        isolated_inputs = final_inputs - len(connected_inputs)
        
        # Count isolated outputs (outputs with no incoming edges)
        output_edges = edges[('function', 'to', 'output')]
        connected_outputs = set()
        for edge in output_edges:
            connected_outputs.add(edge[1])  # edge[1] is the output node index
        isolated_outputs = final_outputs - len(connected_outputs)
        
        summary = {
            'inputs_included': final_inputs,
            'inputs_removed': max(initial_counts['input'] - final_inputs, 0),
            'mediators_included': final_mediators,
            'mediators_removed': max(initial_counts['mediator'] - final_mediators, 0),
            'functions_included': final_functions,
            'functions_removed': max(initial_counts['function'] - final_functions, 0),
            'outputs_included': final_outputs,
            'outputs_removed': max(initial_counts['output'] - final_outputs, 0),
            'isolated_inputs': isolated_inputs,
            'isolated_outputs': isolated_outputs,
            'total_nodes': n_nodes,
            'total_edges': n_edges,
            'density': density,
            'avg_degree': avg_degree,
            'avg_clustering': clustering
        }
        
        data.graph_summary = summary
        data.build_args = {'depth': self.depth, 'has_mediators': (mediator_edges is not None) or (mediator_names is not None and len(mediator_names) > 0)}
        
        if self.verbose:
            print(f"\n{'='*50}")
            print("Network Construction Summary:")
            print(f"{'='*50}")
            print(f"Inputs: {final_inputs} included, {summary['inputs_removed']} removed")
            print(f"Mediators: {final_mediators} included, {summary['mediators_removed']} removed")
            print(f"Functions: {final_functions} included, {summary['functions_removed']} removed")
            print(f"Outputs: {final_outputs} included, {summary['outputs_removed']} removed")
            print(f"Isolated nodes: {isolated_inputs} inputs, {isolated_outputs} outputs")
            print(f"\nGraph Statistics:")
            print(f"  Nodes: {n_nodes}")
            print(f"  Edges: {n_edges}")
            print(f"  Density: {density:.4f}")
            print(f"  Avg Degree: {avg_degree:.2f}")
            print(f"  Avg Clustering: {clustering:.4f}")
            print(f"{'='*50}\n")
        
        return data