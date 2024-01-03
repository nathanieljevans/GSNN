
import networkx as nx 
from matplotlib import pyplot as plt 
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np

_DRUGCOLOR      = 'r'
_PROTEINCOLOR   = 'b'
_RNACOLOR       = 'g'
_LINCSCOLOR     = 'c'
_OMICSCOLOR     = 'm'

_c1 = 'k'
_c2 = 'r'

def plot_hairball(res, save=None, figsize=(10,10), fontsize=12):

    bigG = nx.from_pandas_edgelist(res, create_using=nx.DiGraph)
    lilG = nx.from_pandas_edgelist(res[lambda x: x.score > 0.5], create_using=nx.DiGraph)

    pos3 = nx.drawing.nx_agraph.graphviz_layout(bigG, prog='neato')

    nim = res[lambda x: x.score > 0.5].shape[0]
    nun = res[lambda x: x.score <= 0.5].shape[0]
    tot = res.shape[0]

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(bigG, pos=pos3, arrowstyle='-', edge_color=_c1, node_size=0, alpha=0.2, width=1.25)
    nx.draw_networkx_nodes(bigG, pos=pos3, node_size=3, node_color='gray', alpha=0.5)
    nx.draw_networkx_edges(lilG, pos=pos3, arrowstyle='-', edge_color=_c2, node_size=0, alpha=0.4, width=1.25)

    red = mpatches.Patch(color=_c2, label=f'Involved Edges (n={nim} [{nim/tot*100:.0f}%])')
    blk = mpatches.Patch(color=_c1, label=f'Uninvolved Edges (n={res[lambda x: x.score <= 0.5].shape[0]} [{nun/tot*100:.0f}%])')

    plt.legend(handles=[red,blk], fontsize=fontsize)
    plt.box(False)
    plt.tight_layout() 

    if save is not None: 
        plt.savefig(save, bbox_inches='tight', pad_inches=0., dpi=300)
    
    plt.show()



def bounding_box_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return (x1_min < x2_max and x1_max > x2_min and
            y1_min < y2_max and y1_max > y2_min)

def adjust_label_positions(pos, labels, scale_factor=300, shift_margin=1, hy=15, max_iters=1000):
    """
    Adjust label positions to avoid overlap using bounding boxes.

    Parameters:
    - pos: dict, node positions
    - labels: dict, node labels
    - scale_factor: float, factor to scale bounding box based on label length
    - shift_margin: float, margin to shift labels up/down to resolve overlap

    Returns:
    - new_pos: dict, adjusted label positions
    """
    label_pos = {node: np.array([x, y]) for node, (x, y) in pos.items() if node in labels}
    shifted = {n:False for n in pos.keys()}

    for jj in range(10): 
        for node_i in label_pos:
            for node_j in label_pos:
                if node_i != node_j:
                    # Define bounding boxes based on label lengths
                    len_i = len(labels[node_i])
                    len_j = len(labels[node_j])

                    xi, yi = label_pos[node_i] 
                    xj, yj = label_pos[node_j]
                    #           x min                   |  y min   |      x max                 | y max
                    box_i = (xi - len_i/2 * scale_factor, yi - hy/2, xi + len_i/2 * scale_factor, yi + hy/2)
                    box_j = (xj - len_j/2 * scale_factor, yj - hy/2, xj + len_j/2 * scale_factor, yj + hy/2)

                    # Check for overlap and adjust position
                    ii = 0 
                    step=False
                    while bounding_box_overlap(box_i, box_j) and (ii < max_iters):
                        #print('adjusting labels: ', labels[node_i], labels[node_j], f'[steps: {ii + 1}]', end='\r')
                        if (ii == 0) and not shifted[node_j]: 
                            label_pos[node_i][1] += 0
                            label_pos[node_j][1] -= 30
                            shifted[node_j] = True
                        else: 
                            #label_pos[node_i][1] += shift_margin
                            label_pos[node_j][1] -= shift_margin
                        # Update bounding box for node_i
                        xi, yi = label_pos[node_i] 
                        xj, yj = label_pos[node_j]
                        box_i = (xi - len_i/2 * scale_factor, yi - hy/2, xi + len_i/2 * scale_factor, yi + hy/2)
                        box_j = (xj - len_j/2 * scale_factor, yj - hy/2, xj + len_j/2 * scale_factor, yj + hy/2)
                        ii+=1
                        step=True
                    #if step: print() 

    return label_pos



def plot_explanation_graph(res, threshold=0.5, num_edges=None, save=None, figsize=(10,10), fontsize=8, node_size=25, extdata_path='../extdata/'):

    if threshold is not None: 
        G = nx.from_pandas_edgelist(res[lambda x: x.score > threshold], create_using=nx.DiGraph)
    elif num_edges is not None: 
        raise ValueError('threshold and num_edges cannot both be passed.')

    if num_edges is not None: 
        print('creating graph by top edges - NOTE: the GSNNExplainer reported R2 score will not reflect this graph.')
        G = nx.from_pandas_edgelist(res.sort_values(by='score', ascending=False).head(num_edges), create_using=nx.DiGraph)
    
    if (num_edges is None) and (threshold is None): 
        raise ValueError('threshold and num_edges cannot both be zero.')


    print('full graph size:', len(G))
    # select only the largest connected component 
    G = G.subgraph(next(iter(nx.connected_components(G.to_undirected()))))
    print('largest comp. subgraph', len(G))

    uni2symb = pd.read_csv(f'{extdata_path}/omnipath_uniprot2genesymb.tsv', sep='\t').set_index('From').to_dict()['To']

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')

    node_colors = [] 

    node_label_dict = {}

    for node in G.nodes(): 
        
            type_ = node.split('__')[0]
            if type_ == 'DRUG': 
                node_colors.append(_DRUGCOLOR)
                node_label_dict[node] = node.split('__')[1]
            elif type_ == 'PROTEIN': 
                node_colors.append(_PROTEINCOLOR)
                node_label_dict[node] = uni2symb[node.split('__')[1]]
            elif type_ == 'RNA': 
                node_colors.append(_RNACOLOR)
            elif type_ == 'LINCS': 
                node_colors.append(_LINCSCOLOR)
            else: 
                # omic node
                node_colors.append(_OMICSCOLOR)

    plt.figure(figsize=(figsize))

    drug_patch = mpatches.Patch(color=_DRUGCOLOR, label='Drug Node')
    prot_patch = mpatches.Patch(color=_PROTEINCOLOR, label='Protein Node')
    _rna_patch = mpatches.Patch(color=_RNACOLOR, label='RNA Node')
    linc_patch = mpatches.Patch(color=_LINCSCOLOR, label='LINCS Node')
    omic_patch = mpatches.Patch(color=_OMICSCOLOR, label='Omic Node')

    label_pos = adjust_label_positions({k:(x, y+15) for k, (x,y) in pos.items()}, node_label_dict)
    # {k:(x, y+15 - (np.random.rand(1).item() > 0.5)*30) for k, (x,y) in pos.items()}

    nx.draw_networkx_edges(G, pos=pos, node_size=node_size, edge_color='lightgray')
    nx.draw_networkx_labels(G, pos=label_pos, font_size=fontsize, labels=node_label_dict, font_color='k', font_weight='bold')
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_size, node_color=node_colors, alpha=1.)
    plt.legend(handles=[drug_patch, prot_patch, _rna_patch, linc_patch, omic_patch], fontsize=fontsize)
    #plt.tight_layout()  # BUG: messes with text positions for some reason.

    if save is not None: 
        plt.savefig(save, bbox_inches='tight', pad_inches=0., dpi=300)
    
    plt.show()