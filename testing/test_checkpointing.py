

import torch
import pytest
from gsnn.models.GSNN import GSNN
from gsnn.simulate.nx2pyg import nx2pyg
import networkx as nx
import copy 

def make_graph(): 

    # Create a simple directed graph with 3 inputs, 3 outputs, and 5 function nodes
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

    data = nx2pyg(G, input_nodes, function_nodes, output_nodes)

    return data
    

def test_checkpointing():

    data = make_graph()

    x = torch.randn(len(data.node_names_dict['input']), 10)

    model = GSNN(data.edge_index_dict,
                data.node_names_dict, 
                channels=10, 
                layers=10,
                share_layers=True, 
                bias=True,
                add_function_self_edges=False,
                norm='layer')
    
    model_chk = copy.deepcopy(model)
    model_chk.checkpoint = True 

    out = model(x) 
    out.sum().backward() 
    
    out2 = model_chk(x)
    out2.sum().backward()

    for p, p_chk in zip(model.parameters(), model_chk.parameters()): 
        assert torch.allclose(p.grad, p_chk.grad), "Checkpointing does not match."

if __name__ == "__main__":
    pytest.main([__file__])