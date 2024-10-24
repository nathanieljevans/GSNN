import pyro
import pyro.distributions as dist
import torch
import networkx as nx

def nx_to_pyro_model(G, input_nodes, output_nodes, special_functions=None, noise_scale=1):
    """
    Converts a NetworkX directed graph into a Pyro Bayesian network model with Gaussian distributions
    and allows complex transformations (e.g., squaring inputs, logic gates) specified by the user.
    
    Parameters:
    G (networkx.DiGraph): A directed graph where nodes represent random variables and edges represent dependencies.
    input_nodes (list): A list of input node names.
    output_nodes (list): A list of output node names.
    special_functions (dict): A dictionary where the keys are node names, and the values are lambda functions 
                              that define how to process the parent nodes' values.
    
    Returns:
    model (function): A Pyro model function that takes input values and returns output values.
    """
    
    def model(input_values):
        sampled_values = {}  # Dictionary to store the sampled values of each node

        # First, set the values for the input nodes based on the input_values provided
        for node in input_nodes:
            sampled_values[node] = input_values[node]
        
        # Iterate through the nodes in topological order (ensures we sample parents before children)
        for node in nx.topological_sort(G):
            if node not in input_nodes:
                # Get parent nodes for the current node
                parents = list(G.predecessors(node))
                
                if not parents:
                    # If no parents, assume the node is an independent Gaussian variable
                    sampled_values[node] = pyro.sample(node, dist.Normal(0, 1))
                else:
                    parent_values = [sampled_values[parent] for parent in parents]
                    
                    # Check if the node has a special function
                    if special_functions and (node in special_functions):
                        # Apply the special function to the parent values and ensure it returns a tensor/float
                        transformed_value = special_functions[node](parent_values)
                    else:
                        # Default behavior: sum the parent values
                        transformed_value = sum(parent_values)
                    
                    # Ensure the transformed value is a scalar (tensor or float), not a list
                    if isinstance(transformed_value, list):
                        raise ValueError(f"The special function for {node} returned a list instead of a scalar.")
                    
                    # Sample the node using a Gaussian distribution with the transformed value as the mean
                    sampled_values[node] = pyro.sample(node, dist.Normal(transformed_value, noise_scale))
        
        # Collect the values for the output nodes
        output_values = {node: sampled_values[node] for node in output_nodes}
        
        return output_values

    return model