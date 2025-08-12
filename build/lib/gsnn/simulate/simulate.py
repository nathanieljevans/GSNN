import pyro
import pyro.distributions as dist
import torch
import numpy as np
import networkx as nx
from typing import Optional, Dict
from scipy.integrate import solve_ivp

from gsnn.simulate import utils

def simulate(G, n_train: int, n_test: int, input_nodes, output_nodes, *, noise_scale: float = 1.0,
            special_functions: Optional[Dict] = None, signed_edges: Optional[Dict[tuple, int]] = None):
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

            ``f(parent_values: list) -> Tensor`` where parent_values is a list
            of the parent node values.

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
    pyro_model = utils.nx_to_pyro_model(G, input_nodes, output_nodes, special_functions, noise_scale=noise_scale, signed_edges=signed_edges)
    
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



def simulate_sde(G, n_train: int, n_test: int, input_nodes, output_nodes, *, 
                 noise_scale: float = 1.0, dt: float = 0.01, t_final: float = 10.0,
                 special_functions: Optional[Dict] = None, seed: Optional[int] = None,
                 signed_edges: Optional[Dict[tuple, int]] = None):
    r"""Generate samples from a *synthetic* graph-structured data-generation process using stochastic ODEs.

    The function takes a directed NetworkX graph that represents causal
    relationships between **input**, **function**, and **output** nodes.  It
    converts the graph into a system of stochastic ordinary differential equations
    (SODEs) and integrates them using the Euler-Maruyama method to generate
    data samples.

    The stochastic ODE system has the form:
        dX_i(t) = f_i(X_1(t), ..., X_n(t)) dt + σ dW_i(t)
    
    where X_i are the node values, f_i represents the deterministic dynamics
    (weighted sum of parent nodes), σ is the noise scale, and dW_i are
    independent Wiener processes.

    Args:
        G (networkx.DiGraph): Directed graph encoding the network structure.
        n_train (int): Number of training instances to simulate.
        n_test (int): Number of test instances to simulate.
        input_nodes (list[str]): Ordered list of node names that are treated as
            **inputs** (boundary conditions).
        output_nodes (list[str]): Ordered list of node names that are treated
            as **outputs** (targets).
        noise_scale (float, optional): Standard deviation of the stochastic noise
            term (diffusion coefficient). **Default:** ``1.0``.
        dt (float, optional): Time step size for numerical integration.
            **Default:** ``0.01``.
        t_final (float, optional): Final integration time. **Default:** ``10.0``.
        special_functions (dict[str, callable], optional): Mapping from node
            name to a Python callable that overrides the default linear
            relationship for that node. Each callable must have the signature
            ``f(parent_values: list) -> float`` where parent_values is a list
            of the parent node values.
        seed (int, optional): Random seed for reproducibility.

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
    
    if seed is not None:
        np.random.seed(seed)
    
    # Get all nodes and create mapping to indices
    all_nodes = list(nx.topological_sort(G))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n_nodes = len(all_nodes)
    
    # Initialize edge weights randomly
    edge_weights = {}
    for edge in G.edges():
        sign = 1
        if signed_edges is not None:
            sign = signed_edges.get(edge, 1)
        elif G.edges[edge].get('sign') is not None:
            sign = G.edges[edge]['sign']
        edge_weights[edge] = sign * np.random.normal(0, 1)
    
    def sde_system(t, y, noise_scale):
        """
        Define the stochastic ODE system.
        
        Args:
            t: Current time
            y: Current state vector (node values)
            noise_scale: Noise intensity
            
        Returns:
            dydt: Deterministic drift term
            noise: Stochastic diffusion term
        """
        dydt = np.zeros(n_nodes)
        
        # Input nodes: keep constant (no drift)
        for node in input_nodes:
            idx = node_to_idx[node]
            dydt[idx] = 0
        
        # Function and output nodes: dynamics based on weighted sum of parents
        for node in all_nodes:
            if node not in input_nodes:
                idx = node_to_idx[node]
                parent_sum = 0
                
                parents = list(G.predecessors(node))
                if parents:
                    parent_values = [y[node_to_idx[parent]] for parent in parents]
                    
                    # Apply special function if available
                    if special_functions and node in special_functions:
                        parent_sum = special_functions[node](parent_values)
                    else:
                        # Default: weighted sum of parents
                        for parent in parents:
                            parent_idx = node_to_idx[parent]
                            weight = edge_weights.get((parent, node), 1.0)
                            parent_sum += weight * y[parent_idx]
                
                # Simple dynamics: drift towards the weighted sum
                dydt[idx] = -y[idx] + parent_sum
        
        return dydt
    
    def euler_maruyama_step(y, dt, noise_scale):
        """
        Perform one step of Euler-Maruyama integration.
        """
        drift = sde_system(0, y, noise_scale)  # t not used in our system
        noise = np.random.normal(0, np.sqrt(dt) * noise_scale, size=y.shape)
        
        # Input nodes don't get noise
        for node in input_nodes:
            idx = node_to_idx[node]
            noise[idx] = 0
        
        return y + drift * dt + noise
    
    def solve_sde(input_values, n_steps):
        """
        Solve the stochastic ODE for given input values.
        """
        # Initialize state vector
        y = np.zeros(n_nodes)
        
        # Set input node values
        for i, node in enumerate(input_nodes):
            idx = node_to_idx[node]
            y[idx] = input_values[i]
        
        # Integrate over time
        for _ in range(n_steps):
            y = euler_maruyama_step(y, dt, noise_scale)
        
        # Extract output values
        output_values = []
        for node in output_nodes:
            idx = node_to_idx[node]
            output_values.append(y[idx])
        
        return np.array(output_values)
    
    def generate_samples(n_samples):
        """Generate samples using the stochastic ODE system."""
        x_samples = []
        y_samples = []
        n_steps = int(t_final / dt)
        
        for _ in range(n_samples):
            # Sample input values from standard normal distribution
            x_values = np.random.normal(0, 1, len(input_nodes))
            
            # Solve the stochastic ODE system
            y_values = solve_sde(x_values, n_steps)
            
            x_samples.append(x_values)
            y_samples.append(y_values)
        
        return np.array(x_samples), np.array(y_samples)
    
    # Generate training and test samples
    x_train, y_train = generate_samples(n_train)
    x_test, y_test = generate_samples(n_test)
    
    return x_train, y_train, x_test, y_test