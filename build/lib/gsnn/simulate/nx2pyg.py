import torch 
from torch_geometric.data import HeteroData
import networkx as nx

def nx2pyg(G, input_nodes, function_nodes, output_nodes, weight_attr=None):

    data = HeteroData()

    # Create edge lists
    input_to_function_edges = []
    function_to_function_edges = []
    function_to_output_edges = []
    
    # Categorize edges based on node types
    for u, v in G.edges:
        if u in input_nodes and v in function_nodes:
            input_to_function_edges.append((input_nodes.index(u), function_nodes.index(v)))
        elif u in function_nodes and v in function_nodes:
            function_to_function_edges.append((function_nodes.index(u), function_nodes.index(v)))
        elif u in function_nodes and v in output_nodes:
            function_to_output_edges.append((function_nodes.index(u), output_nodes.index(v)))

    # Convert to torch LongTensors
    input_edge_index = torch.LongTensor(input_to_function_edges).t().contiguous()
    function_edge_index = torch.LongTensor(function_to_function_edges).t().contiguous()
    output_edge_index = torch.LongTensor(function_to_output_edges).t().contiguous()

    if weight_attr is not None:
        input_to_function_weights = []
        function_to_function_weights = [] 
        function_to_output_weights = []
        for u, v in G.edges():
            if u in input_nodes and v in function_nodes:
                input_to_function_weights.append(G[u][v][weight_attr])
            elif u in function_nodes and v in function_nodes:
                function_to_function_weights.append(G[u][v][weight_attr])
            elif u in function_nodes and v in output_nodes:
                function_to_output_weights.append(G[u][v][weight_attr])
        input_to_function_weights = torch.tensor(input_to_function_weights, dtype=torch.float32)
        function_to_function_weights = torch.tensor(function_to_function_weights, dtype=torch.float32)
        function_to_output_weights = torch.tensor(function_to_output_weights, dtype=torch.float32) 

        edge_weight_dict = {
            ('input', 'to', 'function'): input_to_function_weights,
            ('function', 'to', 'function'): function_to_function_weights,
            ('function', 'to', 'output'): function_to_output_weights,
        }

        data.edge_weight_dict = edge_weight_dict

    # Create node name dictionaries
    input_names = input_nodes
    function_names = function_nodes
    output_names = output_nodes


    # Assign edge index dictionaries
    data.edge_index_dict = {
        ('input', 'to', 'function'): input_edge_index,
        ('function', 'to', 'function'): function_edge_index,
        ('function', 'to', 'output'): output_edge_index,
    }

    # Assign node names dictionaries
    data.node_names_dict = {
        'input': input_names,
        'function': function_names,
        'output': output_names,
    }

    return data

def pyg2nx(data):

    G = nx.DiGraph() 
    input_names = data.node_names_dict['input']
    function_names = data.node_names_dict['function']
    output_names = data.node_names_dict['output']

    for src,dst in data.edge_index_dict['input', 'to', 'function'].T:
        src_name = input_names[src]
        dst_name = function_names[dst]
        G.add_edge(src_name, dst_name)

    for src,dst in data.edge_index_dict['function', 'to', 'function'].T:
        src_name = function_names[src]
        dst_name = function_names[dst]
        G.add_edge(src_name, dst_name)

    for src,dst in data.edge_index_dict['function', 'to', 'output'].T:
        src_name = function_names[src]
        dst_name = output_names[dst]
        G.add_edge(src_name, dst_name)

    return G