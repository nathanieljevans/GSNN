import networkx as nx 
import numpy as np 
import torch 
from gsnn.simulate.simulate import simulate, simulate_sde

def simulate_3_in_3_out(n_train, n_test, noise_scale=0.1, device='cpu', zscorey=False): 

    G = nx.DiGraph()

    # Add input nodes, function nodes, and output nodes
    input_nodes = ['in0', 'in1', 'in2']
    function_nodes = ['func0', 'func1', 'func2', 'func3', 'func4']
    output_nodes = ['out0', 'out1', 'out2']

    # Add edges from input nodes to function nodes
    G.add_edges_from([('in0', 'func0'), ('in1', 'func1'), ('in2', 'func2')])

    # Add edges between function nodes
    G.add_edges_from([('func0', 'func3'), ('func1', 'func4'), ('func2', 'func3')])

    # Add edges from function nodes to output nodes
    G.add_edges_from([('func3', 'out0'), ('func4', 'out1'), ('func3', 'out2')])

    # Define positions for each node for plotting
    pos = {
        'in0': (-2, 2), 'in1': (0, 2), 'in2': (2, 2),
        'func0': (-2, 1), 'func1': (0, 1), 'func2': (2, 1),
        'func3': (-1, 0), 'func4': (1, 0),
        'out0': (-2, -1), 'out1': (0, -1), 'out2': (2, -1)
    }

    x_train, x_test, y_train, y_test = simulate(G, n_train=n_train, n_test=n_test, input_nodes=input_nodes, output_nodes=output_nodes, noise_scale=noise_scale,
                                            special_functions={'func1': lambda x: -np.mean(x), 'func2':lambda x: np.sum([np.exp(xx) for xx in x]), 
                                                               'func0': lambda x: np.mean(([(xx-1)**2 for xx in x])), 'func3': lambda x: -np.mean(x) if all([xx > 0 for xx in x]) else np.mean(x)})

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    if zscorey: 
        y_mu = y_train.mean(0); y_std = y_train.std(0)
        y_train = (y_train - y_mu)/(y_std + 1e-8)
        y_test = (y_test - y_mu)/(y_std + 1e-8)

    return G, pos, x_train, x_test, y_train, y_test, input_nodes, function_nodes, output_nodes
        
def simulate_10_in_25_func_10_out_cyclic(n_train, n_test, noise_scale=0.1, device='cpu', zscorey=False, 
                                         dt=0.01, t_final=10.0, seed=None):
    """
    Create a complex cyclic graph with 10 inputs, 25 function nodes, and 10 outputs.
    Maximum path length from input to output is 10. Uses SDE method for data generation.
    
    Args:
        n_train (int): Number of training samples
        n_test (int): Number of test samples  
        noise_scale (float): Noise scale for SDE integration
        device (str): Device to place tensors on
        zscorey (bool): Whether to z-score normalize y values
        dt (float): Time step for SDE integration
        t_final (float): Final time for SDE integration
        seed (int): Random seed for reproducibility
        
    Returns:
        Tuple containing graph, positions, train/test data, and node lists
    """
    
    G = nx.DiGraph()
    
    # Define node names
    input_nodes = [f'in{i}' for i in range(10)]
    function_nodes = [f'func{i}' for i in range(25)]
    output_nodes = [f'out{i}' for i in range(10)]
    
    # Layer structure for controlled path length
    # Layer 0: input_nodes (10 nodes)
    # Layer 1-2: func0-func7 (8 nodes per layer, 16 total)
    # Layer 3-4: func16-func24 (9 nodes in layer 3-4)  
    # Layer 5: output_nodes (10 nodes)
    
    layer1_funcs = [f'func{i}' for i in range(8)]      # func0-func7
    layer2_funcs = [f'func{i}' for i in range(8, 16)]  # func8-func15
    layer3_funcs = [f'func{i}' for i in range(16, 25)] # func16-func24
    
    # Connect inputs to first layer of functions
    for i, inp in enumerate(input_nodes):
        # Each input connects to 2-3 functions in layer 1
        target_funcs = [layer1_funcs[i % 8], layer1_funcs[(i + 1) % 8]]
        if i < 2:  # First two inputs get an extra connection
            target_funcs.append(layer1_funcs[(i + 2) % 8])
        G.add_edges_from([(inp, func) for func in target_funcs])
    
    # Connect layer 1 to layer 2 functions
    for i, func1 in enumerate(layer1_funcs):
        # Each layer 1 func connects to 2-3 layer 2 funcs
        target_funcs = [layer2_funcs[i % 8], layer2_funcs[(i + 1) % 8]]
        if i < 4:  # First half get extra connections
            target_funcs.append(layer2_funcs[(i + 2) % 8])
        G.add_edges_from([(func1, func) for func in target_funcs])
    
    # Connect layer 2 to layer 3 functions
    for i, func2 in enumerate(layer2_funcs):
        # Each layer 2 func connects to 1-2 layer 3 funcs
        target_funcs = [layer3_funcs[i % 9]]
        if i < 5:  # Some get extra connections
            target_funcs.append(layer3_funcs[(i + 1) % 9])
        G.add_edges_from([(func2, func) for func in target_funcs])
    
    # Add cycles within function layers
    # Cycles within layer 1
    G.add_edges_from([
        ('func0', 'func2'), ('func2', 'func4'), ('func4', 'func0'),  # 3-cycle
        ('func1', 'func3'), ('func3', 'func1'),                      # 2-cycle
        ('func5', 'func7'), ('func7', 'func6'), ('func6', 'func5'),  # 3-cycle
    ])
    
    # Cycles within layer 2  
    G.add_edges_from([
        ('func8', 'func10'), ('func10', 'func8'),                    # 2-cycle
        ('func9', 'func11'), ('func11', 'func13'), ('func13', 'func9'), # 3-cycle
        ('func12', 'func14'), ('func14', 'func15'), ('func15', 'func12'), # 3-cycle
    ])
    
    # Cycles within layer 3
    G.add_edges_from([
        ('func16', 'func18'), ('func18', 'func16'),                  # 2-cycle
        ('func17', 'func19'), ('func19', 'func21'), ('func21', 'func17'), # 3-cycle
        ('func20', 'func22'), ('func22', 'func24'), ('func24', 'func20'), # 3-cycle
    ])
    
    # Cross-layer cycles (creates longer cycles)
    G.add_edges_from([
        ('func15', 'func1'),   # layer 2 back to layer 1
        ('func23', 'func9'),   # layer 3 back to layer 2
        ('func24', 'func2'),   # layer 3 back to layer 1
    ])
    
    # Connect layer 3 functions to outputs
    for i, func3 in enumerate(layer3_funcs):
        # Each layer 3 func connects to 1-2 outputs
        target_outs = [output_nodes[i % 10]]
        if i < 5:  # Some get extra connections
            target_outs.append(output_nodes[(i + 5) % 10])
        G.add_edges_from([(func3, out) for out in target_outs])
    
    # Define positions for visualization (layered layout)
    pos = {}
    
    # Input layer
    for i, node in enumerate(input_nodes):
        pos[node] = (i - 4.5, 5)  # Spread across top
    
    # Function layer 1
    for i, node in enumerate(layer1_funcs):
        pos[node] = (i - 3.5, 4)
        
    # Function layer 2  
    for i, node in enumerate(layer2_funcs):
        pos[node] = (i - 3.5, 3)
        
    # Function layer 3
    for i, node in enumerate(layer3_funcs):
        pos[node] = (i - 4, 2)
        
    # Output layer
    for i, node in enumerate(output_nodes):
        pos[node] = (i - 4.5, 1)
    
    # Define some special functions for nonlinear behavior
    special_functions = {
        'func0': lambda x: np.tanh(np.sum(x)),
        'func5': lambda x: np.exp(-np.sum(np.array(x)**2) / len(x)),  # Gaussian-like
        'func10': lambda x: np.sum([xx**3 for xx in x]) / len(x),     # Cubic
        'func15': lambda x: np.sin(np.sum(x)),                        # Sine
        'func20': lambda x: np.sum(x) / (1 + np.abs(np.sum(x))),     # Saturating
        'func24': lambda x: np.sum([xx * np.sign(xx) * np.sqrt(np.abs(xx)) for xx in x]),  # Square root with sign
    }
    
    # Generate data using SDE method
    x_train, y_train, x_test, y_test = simulate_sde(
        G, n_train=n_train, n_test=n_test, 
        input_nodes=input_nodes, output_nodes=output_nodes,
        noise_scale=noise_scale, dt=dt, t_final=t_final,
        special_functions=special_functions, seed=seed
    )
    
    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Optional z-score normalization
    if zscorey:
        y_mu = y_train.mean(0)
        y_std = y_train.std(0)
        y_train = (y_train - y_mu) / (y_std + 1e-8)
        y_test = (y_test - y_mu) / (y_std + 1e-8)
    
    return G, pos, x_train, x_test, y_train, y_test, input_nodes, function_nodes, output_nodes        