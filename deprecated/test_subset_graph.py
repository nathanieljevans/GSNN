import pytest
import networkx as nx
from gsnn.proc.lincs.subset import subset_graph

def create_test_graph1():
    """
    Create a directed graph with known paths and depths.
    
    Graph structure:
        A -> B -> C -> D
             |    |
             v    v
             E -> F
    """
    G = nx.DiGraph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'), ('C', 'D'), ('B', 'E'), ('E', 'F'), ('C', 'F')
    ])
    return G

def create_test_graph2():
    """
    Create a directed graph with cycles and multiple paths between nodes.

    Graph structure:
        A -> B -> C -> D -> E
        ^    ^    | 
        |    |    | 
        V    V    V
        F<-> G -> H -> I

    """
    G = nx.DiGraph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), 
        ('A','F'), ('F','A'), ('B','G'), ('G','B'), ('C','H'),
        ('F','G'), ('G','F'), ('G','H'), ('H', 'I')
    ])
    return G

def test_subset_graph():
    G = create_test_graph1()
    roots = ['A']
    leafs = ['D']
    depth = 3

    # Correct subgraph should include paths A->B->C->D
    correct_nodes = {'A', 'B', 'C', 'D'}

    # Subset the graph
    subgraph = subset_graph(G, depth=depth, roots=roots, leafs=leafs, verbose=False)

    # Check if the subgraph contains the correct nodes
    assert set(subgraph.nodes()) == correct_nodes, "Subset does not match expected nodes"

    # Test with depth that should not include any nodes to the leaf 'D' or 'F'
    depth_short = 2
    subgraph_short = subset_graph(G, depth=depth_short, roots=roots, leafs=leafs, verbose=False)

    # Correct subgraph should be empty since no paths A->B->C->D or A->B->C->F can be completed within depth 2
    assert len(subgraph_short.nodes()) == 0, "Subset with insufficient depth should be empty"

    G2 = create_test_graph2()

    subgraph1 = subset_graph(G2, depth=4, roots=['A'], leafs=['E'], verbose=False)
    assert set(subgraph1.nodes()) == {'A', 'B', 'C', 'D', 'E'}, "Scenario 1 failed"

    subgraph2 = subset_graph(G2, depth=8, roots=['A'], leafs=['E'], verbose=False)
    assert set(subgraph2.nodes()) == {'A', 'B', 'C', 'D', 'E', 'F','G'}, "Scenario 2 failed"

    subgraph3 = subset_graph(G2, depth=10, roots=['G'], leafs=['E', 'I'], verbose=False)
    assert set(subgraph3.nodes()) == {'A', 'B', 'C', 'D', 'E', 'F','G', 'H', 'I'}, "Scenario 2 failed"


# Note: pytest requires this if block to run from the command line
if __name__ == "__main__":
    pytest.main()
