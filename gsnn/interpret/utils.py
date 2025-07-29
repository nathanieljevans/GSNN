from tkinter import NONE
import networkx as nx 
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

def plot_edge_importance(res, pos=None, cmap=plt.cm.coolwarm, title='', figsize=(4,3), roots=None, leafs=None): 

    G = nx.from_pandas_edgelist(res, source='source', target='target', edge_attr='score', create_using=nx.DiGraph)

    if roots is not None: 
        # remove all nodes that are not ancestors or descendants of the root 
        subset = set()
        for root in roots: 
            subset.update(nx.descendants(G, root).union([root])) 
        G = nx.subgraph(G, subset).copy()

    if leafs is not None: 
        # remove all nodes that are not ancestors or descendants of the root 
        subset = set()
        for leaf in leafs: 
            subset.update(nx.ancestors(G, leaf).union([leaf]))
        G = nx.subgraph(G, subset).copy()

    edge_color = [G.edges[e]['score'] for e in G.edges] 

    vmin, vmax = min(edge_color), max(edge_color)
    abs_max = max(abs(vmin), abs(vmax))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    if pos is None:
        H = nx.convert_node_labels_to_integers(G, label_attribute="node_label")
        H_layout = nx.nx_pydot.pydot_layout(H, prog="dot")
        pos = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}

    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=3, edge_cmap=cmap, 
                            edge_vmin=-abs_max, edge_vmax=abs_max, arrows=True, 
                            arrowstyle='->', arrowsize=20) 
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)

    # draw rotated node labels 
    for node, position in pos.items():
        plt.text(position[0], position[1], node, fontsize=10, rotation=45, ha='center', va='center') 


    #edge_labels = {e: f"{G.edges[e]['score']:.2f}" for e in G.edges}
    #nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels=edge_labels)

    # add colorbar centered at 0 (white = 0, red = positive, blue = negative)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label='Edge Importance')
    plt.tight_layout() 
    plt.title(title) 

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.show()