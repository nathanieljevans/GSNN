import torch 
from torch_geometric.data import HeteroData

def nx2pyg(G, input_nodes, function_nodes, output_nodes):
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

    # Create node name dictionaries
    input_names = input_nodes
    function_names = function_nodes
    output_names = output_nodes

    # Initialize HeteroData
    data = HeteroData()

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