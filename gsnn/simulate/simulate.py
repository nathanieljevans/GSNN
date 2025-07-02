import pyro
import pyro.distributions as dist
import torch
import numpy as np
import networkx as nx
from typing import Optional, Dict

from gsnn.simulate import utils

def simulate(G, n_train: int, n_test: int, input_nodes, output_nodes, *, noise_scale: float = 1.0,
            special_functions: Optional[Dict] = None):
    r"""Generate samples from a *synthetic* graph-structured data-generation process.

    The function takes a directed NetworkX graph that represents causal
    relationships between **input**, **function**, and **output** nodes.  It
    converts the graph into a *Pyro* probabilistic program (via
    :pyfunc:`gsnn.simulate.utils.nx_to_pyro_model`) and then draws IID samples
    from that model.

    Args:
        G (networkx.DiGraph): Directed graph encoding the Bayesian network
            structure.
        n_train (int): Number of training instances to simulate.
        n_test (int): Number of test instances to simulate.
        input_nodes (list[str]): Ordered list of node names that are treated as
            **inputs** (observed variables).
        output_nodes (list[str]): Ordered list of node names that are treated
            as **outputs** (targets).
        noise_scale (float, optional): Standard deviation of the additive
            Gaussian noise term used for every conditional distribution that
            has no *special function* attached. **Default:** ``1.0``.
        special_functions (dict[str, callable], optional): Mapping from node
            name to a Python callable that overrides the default linear
            relationship for that node.  Each callable must have the signature

            ``f(*parents: Tensor, noise: Tensor) -> Tensor``.

    Shapes:
        - **x_train** – :math:`(n_{\text{train}}, |\text{inputs}|)`
        - **y_train** – :math:`(n_{\text{train}}, |\text{outputs}|)`
        - **x_test** – :math:`(n_{\text{test}}, |\text{inputs}|)`
        - **y_test** – :math:`(n_{\text{test}}, |\text{outputs}|)`

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ``(x_train, y_train, x_test, y_test)`` where each element is a dense
        NumPy array ordered according to ``input_nodes`` / ``output_nodes``.
    """
    
    # Convert the NetworkX graph into a Pyro model with special functions
    pyro_model = utils.nx_to_pyro_model(G, input_nodes, output_nodes, special_functions, noise_scale=noise_scale)
    
    # Helper function to generate samples for a given number of instances
    def generate_samples(n_samples):
        x_samples = []
        y_samples = []
        
        for _ in range(n_samples):
            # Sample x values from a standard normal distribution
            x_values = {node: torch.tensor(np.random.normal(0, 1), dtype=torch.float32) for node in input_nodes}
            
            # Generate the corresponding y values using the Pyro model
            with pyro.plate("data", 1):
                y_values = pyro_model(x_values)
            
            # Append the x and y values to the sample lists
            x_samples.append([x_values[node].item() for node in input_nodes])
            y_samples.append([y_values[node].item() for node in output_nodes])
        
        return np.array(x_samples), np.array(y_samples)
    
    # Generate training and test samples
    x_train, y_train = generate_samples(n_train)
    x_test, y_test = generate_samples(n_test)
    
    return x_train, x_test, y_train, y_test



'''
# Randomly initialize edge weights
def initialize_weights(G):
    weights = {}
    for edge in G.edges:
        weights[edge] = random.normalvariate()
    return weights

# Define the system of ODEs
def ode_system(t, y, G, weights, input_nodes, function_nodes, output_nodes):
    dydt = np.zeros(len(y))  # Derivatives of all nodes (input + function + output)
    
    # Input node derivatives (constant input values, no change)
    for i, node in enumerate(input_nodes):
        dydt[i] = 0
    
    # Function nodes: Derivatives depend on incoming edges and their weights
    for i, node in enumerate(function_nodes):
        node_index = len(input_nodes) + i
        dydt[node_index] = 0  # Initialize derivative
        for pred in G.predecessors(node):
            pred_index = get_node_index(pred, input_nodes, function_nodes, output_nodes)
            dydt[node_index] += weights[(pred, node)] * y[pred_index]
    
    # Output nodes: Derivatives depend on incoming edges from function nodes
    for i, node in enumerate(output_nodes):
        node_index = len(input_nodes) + len(function_nodes) + i
        dydt[node_index] = 0  # Initialize derivative
        for pred in G.predecessors(node):
            pred_index = get_node_index(pred, input_nodes, function_nodes, output_nodes)
            dydt[node_index] += weights[(pred, node)] * y[pred_index]
    
    return dydt

# Get the index of a node in the state vector y
def get_node_index(node, input_nodes, function_nodes, output_nodes):
    if node in input_nodes:
        return input_nodes.index(node)
    elif node in function_nodes:
        return len(input_nodes) + function_nodes.index(node)
    elif node in output_nodes:
        return len(input_nodes) + len(function_nodes) + output_nodes.index(node)

# Solve the ODE and propagate inputs through the network
def solve_network_ode(G, input_values, input_nodes, function_nodes, output_nodes, weights, t_span=[0, 10]):
    num_nodes = len(input_nodes) + len(function_nodes) + len(output_nodes)
    
    # Initial state vector: inputs are set to their values, others to zero
    y0 = np.zeros(num_nodes)
    for i, val in enumerate(input_values):
        y0[i] = val
    
    # Solve the ODE system
    sol = solve_ivp(ode_system, t_span, y0, args=(G, weights, input_nodes, function_nodes, output_nodes), solver='Radau')
    
    # Extract the last time point solution
    final_state = sol.y[:, -1]  # Get the values at the last time point
    
    # Extract the output node values
    output_values = []
    for node in output_nodes:
        output_index = get_node_index(node, input_nodes, function_nodes, output_nodes)
        output_values.append(final_state[output_index])
    
    return output_values

'''