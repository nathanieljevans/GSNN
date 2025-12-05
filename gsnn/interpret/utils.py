from tkinter import NONE
import networkx as nx 
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

def plot_edge_importance(res, pos=None, cmap=plt.cm.coolwarm, title='', figsize=(4,3), roots=None, leafs=None, interactive=False): 

    if interactive:
        try:
            import ipycytoscape
            from IPython.display import display
            print("✅ ipycytoscape imported successfully")
        except ImportError:
            print("❌ ipycytoscape not installed. Install with: pip install ipycytoscape")
            print("For JupyterLab: jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-cytoscape")
            print("Falling back to matplotlib...")
            interactive = False
        
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

    if interactive:
        def get_edge_color(score, abs_max):
            """Get hex color for edge based on score using continuous RdBu colormap"""
            # Assume score is already in [-1, 1] range
            assert -1 <= score <= 1, f'score {score} is not in [-1, 1] range'
            
            # Continuous interpolation: Blue (-1) -> White (0) -> Red (+1)
            if score < 0:
                # Interpolate from dark blue to white
                # Dark blue: (0, 102, 204) -> White: (255, 255, 255)
                t = 1 + score  # t goes from 0 (at -1) to 1 (at 0)
                r = int(0 + t * 255)
                g = int(102 + t * 153)
                b = int(204 + t * 51)
            else:
                # Interpolate from white to dark red
                # White: (255, 255, 255) -> Dark red: (204, 0, 0)
                t = score  # t goes from 0 (at 0) to 1 (at 1)
                r = int(255 - t * 51)
                g = int(255 - t * 255)
                b = int(255 - t * 255)
            
            return f'#{r:02x}{g:02x}{b:02x}'
        
        # Create ipycytoscape widget
        cytoscape_widget = ipycytoscape.CytoscapeWidget()
        cytoscape_widget.graph.clear()
        
        # Set widget size
        cytoscape_widget.layout.width = f'{figsize[0]*100}px'
        cytoscape_widget.layout.height = f'{figsize[1]*100}px'
        
        # Normalize positions to fit in viewport
        if pos:
            # Get position bounds
            x_coords = [pos[node][0] for node in G.nodes()]
            y_coords = [pos[node][1] for node in G.nodes()]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Scale to fit in a reasonable viewport (e.g., 800x600)
            x_range = x_max - x_min if x_max != x_min else 1
            y_range = y_max - y_min if y_max != y_min else 1
            scale_x = 600 / x_range
            scale_y = 400 / y_range
            scale = min(scale_x, scale_y, 1)  # Don't scale up, only down
            
            print(f"🔧 Position bounds: x({x_min:.0f}, {x_max:.0f}), y({y_min:.0f}, {y_max:.0f})")
            print(f"🔧 Scale factor: {scale:.3f}")
        
        # Convert NetworkX graph to Cytoscape JSON format
        nodes = []
        edges = []
        
        # Add nodes with positions
        for node in G.nodes():
            x, y = pos[node]
            # Normalize and center the positions
            norm_x = (x - x_min) * scale - 300  # Center around 0
            norm_y = -((y - y_min) * scale - 200)  # Center around 0, flip Y
            
            nodes.append({
                'data': {'id': str(node), 'label': str(node)},
                'position': {'x': norm_x, 'y': norm_y},
                'classes': 'node'
            })
        
        # Add edges
        for edge in G.edges():
            source, target = edge
            score = G.edges[edge]['score']
            
            edges.append({
                'data': {
                    'id': f'{source}-{target}',
                    'source': str(source),
                    'target': str(target),
                    'score': score,
                    'label': f'{score:.3f}'
                },
                'classes': 'edge'
            })
        
        # Create the JSON structure expected by ipycytoscape
        graph_json = {
            'nodes': nodes,
            'edges': edges
        }
        
        # Set the graph
        cytoscape_widget.graph.add_graph_from_json(graph_json)
        
        # Set stylesheet for appearance
        base_styles = [
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'background-color': '#87CEEB',  # skyblue
                    'border-color': '#000000',
                    'border-width': 2,
                    'width': '30px',
                    'height': '30px',
                    'font-size': '12px',
                    'color': '#000000',
                    'text-outline-color': '#FFFFFF',
                    'text-outline-width': 1
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': '#666666',
                    'line-color': '#666666',
                    'width': 3,
                    'font-size': '8px',
                    'color': '#000000'
                }
            }
        ]
        
        # Apply edge colors individually
        for edge in G.edges():
            source, target = edge
            score = G.edges[edge]['score']
            color = get_edge_color(score, abs_max)
            
            # Use data attribute selector for more robust matching (handles special characters)
            edge_id = f'{source}-{target}'
            base_styles.append({
                'selector': f'[id = "{edge_id}"]',
                'style': {
                    'line-color': color,
                    'target-arrow-color': color,
                }
            })
        
        # Apply all styles at once
        cytoscape_widget.set_style(base_styles)
        
        # Set layout to use preset positions
        cytoscape_widget.set_layout(name='preset')
        
        # Enable tooltips (try-catch in case method signature is different)
        try:
            cytoscape_widget.set_tooltip_source('label')
        except:
            pass  # Skip if tooltip method fails
        
        print(f"📊 Graph created: {len(G.nodes())} nodes, {len(G.edges())} edges")
        print(f"🔧 Position bounds: x({x_min:.0f}, {x_max:.0f}), y({y_min:.0f}, {y_max:.0f})")
        print(f"🔧 Scale factor: {scale:.3f}")
        
        # Color legend with actual score ranges
        edge_scores = [G.edges[e]['score'] for e in G.edges()]
        min_score, max_score = min(edge_scores), max(edge_scores)
        print(f"\n🎨 Edge Color Legend (Score Range: {min_score:.3f} to {max_score:.3f}):")
        print("   Continuous color mapping from -1.0 to +1.0:")
        print("   🔵 -1.0: Dark Blue (#0066cc)")
        print("   ⚪  0.0: White (#ffffff)")
        print("   🔴 +1.0: Dark Red (#cc0000)")
        print("   Colors interpolate smoothly between these values")
        
        print("\n🎉 Interactive Cytoscape Graph with Draggable Nodes!")
        print("✅ Drag nodes to rearrange them")
        print("✅ Zoom with mouse wheel") 
        print("✅ Pan by dragging background")
        print("✅ Hover over edges to see importance scores")
        
        # Display the widget
        try:
            display(cytoscape_widget)
            print("✅ Widget displayed successfully")
            return cytoscape_widget  # Return widget and exit function
        except Exception as e:
            print(f"❌ Error displaying widget: {e}")
            print("Falling back to matplotlib...")
            # Fall through to matplotlib version
    
    # Matplotlib version (original code) - runs if interactive=False or if interactive failed
    if not interactive:
        # Use matplotlib (original code)
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


def plot_node_importance(res, G, pos=None, cmap=plt.cm.coolwarm, title='', figsize=(4,3), roots=None, leafs=None):

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

    # Create node color mapping from res
    node_score_dict = dict(zip(res['node'], res['score']))
    node_color = [node_score_dict.get(node, 0) for node in G.nodes()]

    vmin, vmax = min(node_color), max(node_color)
    abs_max = max(abs(vmin), abs(vmax))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    if pos is None:
        H = nx.convert_node_labels_to_integers(G, label_attribute="node_label")
        H_layout = nx.nx_pydot.pydot_layout(H, prog="dot")
        pos = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}

    plt.figure(figsize=figsize)

    # Draw edges in black (not colored)
    nx.draw_networkx_edges(G, pos, edge_color='black', width=3, arrows=True, 
                            arrowstyle='->', arrowsize=20) 
    # Draw nodes colored by importance
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500, cmap=cmap, 
                          vmin=-abs_max, vmax=abs_max)

    # draw rotated node labels 
    for node, position in pos.items():
        plt.text(position[0], position[1], node, fontsize=10, rotation=45, ha='center', va='center') 

    # add colorbar centered at 0 (white = 0, red = positive, blue = negative)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label='Node Importance')
    plt.tight_layout() 
    plt.title(title) 

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.show()