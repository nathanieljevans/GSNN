import networkx as nx 
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def plot_edge_importance(res, pos, cmap=plt.cm.RdBu, title='', figsize=(4,3)): 

    G = nx.from_pandas_edgelist(res, source='source', target='target', edge_attr='score', create_using=nx.DiGraph)
    edge_color = [G.edges[e]['score'] for e in G.edges] 

    vmin, vmax = min(edge_color), max(edge_color)
    abs_max = max(abs(vmin), abs(vmax))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=2, edge_cmap=cmap, 
                            edge_vmin=-abs_max, edge_vmax=abs_max, arrows=True, 
                            arrowstyle='->', arrowsize=20) 
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=10)

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