import networkx as nx 
import numpy as np 
import torch 
from gsnn.simulate.simulate import simulate

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
        