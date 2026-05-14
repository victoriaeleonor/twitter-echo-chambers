import networkx as nx
import pandas as pd
import random
import os
import sys
from pyvis.network import Network

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def plot_interactive_graph(G, partition, top_n_communities=5, max_nodes=300):
    """
    Creates an interactive HTML graph using Pyvis.
    Open the output file in a browser to explore the network.
    - Zoom in/out with scroll
    - Drag nodes to reposition
    - Hover over nodes to see username
    - Communities shown by color
    """
    print("\n[Pyvis] Building interactive graph...")

    # --- Step 1: Keep only top N communities ---
    community_sizes = pd.Series(partition.values()).value_counts()
    top_communities = community_sizes.head(top_n_communities).index.tolist()

    # --- Step 2: Sample nodes, always including top degree users ---
    global_degrees = dict(G.degree())
    nodes_to_plot = [
        node for node, comm in partition.items()
        if comm in top_communities and node in G.nodes()
    ]

    sampled = []
    for comm in top_communities:
        comm_nodes = [n for n in nodes_to_plot if partition[n] == comm]
        top_comm_nodes = sorted(
            comm_nodes,
            key=lambda n: global_degrees.get(n, 0),
            reverse=True
        )[:10]
        remaining = [n for n in comm_nodes if n not in top_comm_nodes]
        n_sample = int(max_nodes * community_sizes[comm] / community_sizes[top_communities].sum())
        n_random = max(0, n_sample - len(top_comm_nodes))
        random_nodes = random.sample(remaining, min(n_random, len(remaining)))
        sampled.extend(top_comm_nodes + random_nodes)

    subgraph = G.subgraph(sampled)
    print(f"  Nodes: {subgraph.number_of_nodes():,}")
    print(f"  Edges: {subgraph.number_of_edges():,}")

    # --- Step 3: Community colors ---
    colors_list = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
    community_colors = {
        comm: colors_list[i]
        for i, comm in enumerate(top_communities)
    }

    # --- Step 4: Create Pyvis network ---
    net = Network(
        height='750px',
        width='100%',
        bgcolor='#1a1a2e',
        font_color='white',
        notebook=False
    )

    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
            "gravitationalConstant": -150,
            "centralGravity": 0.005,
            "springLength": 200,
            "springConstant": 0.05,
            "damping": 0.4
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "iterations": 500
        }
      },
      "nodes": {
        "shape": "dot",
        "borderWidth": 1.5
      },
      "edges": {
        "smooth": false,
        "color": {
          "opacity": 0.3
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    # --- Step 5: Add nodes ---
    for node in subgraph.nodes():
        comm = partition.get(node, -1)
        color = community_colors.get(comm, '#888888')
        degree = global_degrees.get(node, 1)

        size = 8 + (degree * 1.5)
        size = min(size, 50)

        label = node if degree >= 15 else ''

        net.add_node(
            node,
            label=label,
            title=f"{node}\nCommunity: {comm}\nDegree: {degree}",
            color=color,
            size=size,
            borderColor='white'
        )

    # --- Step 6: Add edges ---
    for source, target, data in subgraph.edges(data=True):
        weight = data.get('weight', 1)
        net.add_edge(
            source, target,
            value=weight,
            color='#ffffff'
        )

    # --- Step 7: Save as HTML ---
    output_file = os.path.join(OUTPUT_PATH, 'user_graph_interactive.html')
    net.save_graph(output_file)
    print(f"  Saved to: data/user_graph_interactive.html")
    print(f"  Open this file in your browser to explore!")


def main():
    print("=" * 40)
    print("INTERACTIVE USER GRAPH VISUALIZATION")
    print("=" * 40)

    # Load graph
    print("\n[1/2] Loading graph and results...")
    graph_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_graph.graphml')
    G = nx.read_graphml(graph_path)

    results_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_graph_results.csv')
    results_df = pd.read_csv(results_path)
    partition = dict(zip(results_df['user'], results_df['louvain_community']))

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    # Build interactive graph
    print("\n[2/2] Building interactive graph...")
    plot_interactive_graph(G, partition, top_n_communities=5, max_nodes=300)

    print("\n" + "=" * 40)
    print("DONE — open data/user_graph_interactive.html in your browser")
    print("=" * 40)


if __name__ == '__main__':
    main()