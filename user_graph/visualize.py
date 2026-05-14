import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
import sys
import random
from pyvis import Network

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..')
    )

# Output folder for visualizations
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

def plot_community_graph(G, partition, top_n_communities=5, max_nodes=500):
    """
    Plots the user interaction graph colored by Louvain community.
    Uses a two-level layout to force communities to be spatially separated.
    """
    print("\n[Graph] Plotting community graph...")

    # --- Step 1: Keep only top N largest communities ---
    community_sizes = pd.Series(partition.values()).value_counts()
    top_communities = community_sizes.head(top_n_communities).index.tolist()

    nodes_to_plot = [
        node for node, comm in partition.items()
        if comm in top_communities and node in G.nodes()
    ]

    if len(nodes_to_plot) > max_nodes:
        sampled = []
        global_degrees = dict(G.degree())
        for comm in top_communities:
            comm_nodes = [n for n in nodes_to_plot if partition[n] == comm]
            
            # Always include top 10 highest degree nodes per community
            top_comm_nodes = sorted(
                comm_nodes,
                key=lambda n: global_degrees.get(n, 0),
                reverse=True
            )[:10]
            
            # Fill the rest with random sampling
            remaining = [n for n in comm_nodes if n not in top_comm_nodes]
            n_sample = int(max_nodes * community_sizes[comm] / community_sizes[top_communities].sum())
            n_random = max(0, n_sample - len(top_comm_nodes))
            random_nodes = random.sample(remaining, min(n_random, len(remaining)))
            
            sampled.extend(top_comm_nodes + random_nodes)
        nodes_to_plot = sampled

    subgraph = G.subgraph(nodes_to_plot)
    print(f"  Plotting {len(subgraph.nodes()):,} nodes from top {top_n_communities} communities")

    # --- Step 2: Colors ---
    colors_list = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
    community_colors = {comm: colors_list[i] for i, comm in enumerate(top_communities)}
    node_colors = [community_colors[partition[node]] for node in subgraph.nodes()]

    # --- Step 3: Two-level layout ---
    print("  Calculating two-level layout...")

    # Position each community center in a circle
    import math
    community_centers = {}
    for i, comm in enumerate(top_communities):
        angle = 2 * math.pi * i / len(top_communities)
        community_centers[comm] = (math.cos(angle) * 3, math.sin(angle) * 3)

    # Position each node around its community center
    pos = {}
    for comm in top_communities:
        comm_nodes = [n for n in subgraph.nodes() if partition[n] == comm]
        comm_subgraph = subgraph.subgraph(comm_nodes)

        if len(comm_nodes) == 1:
            pos[comm_nodes[0]] = community_centers[comm]
            continue

        # Spring layout within community, centered at community center
        sub_pos = nx.spring_layout(
            comm_subgraph,
            seed=42,
            k=0.3,
            scale=0.8
        )

        cx, cy = community_centers[comm]
        for node, (x, y) in sub_pos.items():
            pos[node] = (x + cx, y + cy)

    # --- Step 4: Node sizes ---
    degrees = dict(subgraph.degree())
    node_sizes = [40 + (degrees[node] * 15) for node in subgraph.nodes()]

    # --- Step 5: Draw ---
    fig, ax = plt.subplots(figsize=(16, 12))

    # Draw edges
    nx.draw_networkx_edges(
        subgraph, pos,
        alpha=0.15,
        edge_color="#848484",
        width=0.4,
        ax=ax
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.85,
        ax=ax
    )

   # Labels using global degree from original graph
    global_degrees = dict(G.degree())
    high_degree_nodes = {
        node: node for node in subgraph.nodes()
        if global_degrees.get(node, 0) >= 25
    }

    nx.draw_networkx_labels(
        subgraph, pos,
        labels=high_degree_nodes,
        font_size=7,
        font_weight='bold',
        ax=ax
    )

    # --- Step 6: Draw community hulls (background circles) ---
    for comm in top_communities:
        comm_nodes = [n for n in subgraph.nodes() if partition[n] == comm]
        if len(comm_nodes) < 3:
            continue
        comm_positions = [pos[n] for n in comm_nodes]
        xs = [p[0] for p in comm_positions]
        ys = [p[1] for p in comm_positions]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        radius = max(max(xs) - cx, max(ys) - cy) + 0.3
        circle = plt.Circle(
            (cx, cy), radius,
            color=community_colors[comm],
            alpha=0.08,
            zorder=0
        )
        ax.add_patch(circle)

        # Community label in center
        ax.text(
            cx, cy + radius + 0.1,
            f'Community {comm}\n({community_sizes[comm]} users)',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold',
            color=community_colors[comm]
        )

    # --- Step 7: Legend ---
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=community_colors[comm],
                   markersize=10,
                   label=f'Community {comm} ({community_sizes[comm]} users)')
        for comm in top_communities
    ]
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)

    ax.set_title(
        f'User Interaction Graph — Top {top_n_communities} Communities (Louvain)',
        fontsize=15, fontweight='bold', pad=30
    )
    ax.axis('off')
    plt.tight_layout()

    output_file = os.path.join(OUTPUT_PATH, 'user_graph_communities.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved to: data/user_graph_communities.png")
    plt.show()


def plot_pagerank(pr_df, top_n=15):
    """
    Plots a horizontal bar chart of the top N users by PageRank.
    Colors bars by political leaning based on known accounts.
    """
    print("\n[PageRank] Plotting bar chart...")

    # Keep only top N users
    top_users = pr_df.head(top_n).copy()

    # Manually assign political leaning for known accounts
    democrat_accounts = {
        'JoeBiden', 'KamalaHQ', 'POTUS', 'harryjsisson',
        'mmpadellan', 'RpsAgainstTrump', 'JoeBiden47'
    }
    republican_accounts = {
        'GOP', 'GuntherEagleman', 'SpeakerJohnson',
        'catturd2', 'RepMTG', 'TuckerCarlson', 'realDonaldTrump'
    }

    def assign_color(username):
        if username in democrat_accounts:
            return '#457B9D'   # blue — democrat
        elif username in republican_accounts:
            return '#E63946'   # red — republican
        else:
            return '#6C757D'   # gray — unknown

    top_users['color'] = top_users['user'].apply(assign_color)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(
        top_users['user'],
        top_users['pagerank'],
        color=top_users['color'],
        edgecolor='white',
        height=0.6
    )

    # Add value labels on bars
    for bar, value in zip(bars, top_users['pagerank']):
        ax.text(
            bar.get_width() + 0.00005,
            bar.get_y() + bar.get_height() / 2,
            f'{value:.4f}',
            va='center', ha='left',
            fontsize=8, color='#333333'
        )

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='#457B9D', markersize=10, label='Democrat'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='#E63946', markersize=10, label='Republican'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='#6C757D', markersize=10, label='Unknown'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9)

    ax.set_xlabel('PageRank Score', fontsize=11)
    ax.set_title(
        f'Top {top_n} Most Influential Users — PageRank',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    output_file = os.path.join(OUTPUT_PATH, 'user_pagerank.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved to: data/user_pagerank.png")
    plt.show()


def plot_pagerank_vs_betweenness(results_df, top_n_labels=15):
    """
    Scatter plot of PageRank vs Betweenness Centrality.
    Each point is a user. The four quadrants reveal different roles:
    - High PageRank + High Betweenness = influential bridge users
    - High PageRank + Low Betweenness  = echo chamber leaders
    - Low PageRank  + High Betweenness = hidden bridges
    - Low PageRank  + Low Betweenness  = peripheral users
    """
    print("\n[Scatter] Plotting PageRank vs Betweenness...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Plot all users as small gray dots ---
    ax.scatter(
        results_df['pagerank'],
        results_df['betweenness'],
        alpha=0.3,
        s=20,
        color='#CCCCCC',
        zorder=1
    )

    # --- Highlight top users by pagerank ---
    top_users = results_df.head(top_n_labels).copy()

    democrat_accounts = {
        'JoeBiden', 'KamalaHQ', 'POTUS', 'harryjsisson',
        'mmpadellan', 'RpsAgainstTrump', 'JoeBiden47'
    }
    republican_accounts = {
        'GOP', 'GuntherEagleman', 'SpeakerJohnson',
        'catturd2', 'RepMTG', 'TuckerCarlson', 'realDonaldTrump'
    }

    def assign_color(username):
        if username in democrat_accounts:
            return '#457B9D'
        elif username in republican_accounts:
            return '#E63946'
        else:
            return '#6C757D'

    top_users['color'] = top_users['user'].apply(assign_color)

    ax.scatter(
        top_users['pagerank'],
        top_users['betweenness'],
        color=top_users['color'],
        s=120,
        zorder=2,
        edgecolors='white',
        linewidths=0.8
    )

    # --- Labels for top users ---
    for _, row in top_users.iterrows():
        ax.annotate(
            row['user'],
            xy=(row['pagerank'], row['betweenness']),
            xytext=(6, 4),
            textcoords='offset points',
            fontsize=7.5,
            color='#333333'
        )

    # --- Quadrant lines at median values ---
    median_pr = results_df['pagerank'].median()
    median_bt = results_df['betweenness'].median()

    ax.axvline(x=median_pr, color='#AAAAAA', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(y=median_bt, color='#AAAAAA', linestyle='--', linewidth=0.8, alpha=0.7)

    # --- Quadrant labels ---
    ax.text(
        results_df['pagerank'].max() * 0.75,
        results_df['betweenness'].max() * 0.92,
        'Influential\nBridges',
        fontsize=8, color='#888888', style='italic'
    )
    ax.text(
        results_df['pagerank'].max() * 0.75,
        median_bt * 0.1,
        'Echo Chamber\nLeaders',
        fontsize=8, color='#888888', style='italic'
    )
    ax.text(
        median_pr * 0.05,
        results_df['betweenness'].max() * 0.92,
        'Hidden\nBridges',
        fontsize=8, color='#888888', style='italic'
    )
    ax.text(
        median_pr * 0.05,
        median_bt * 0.1,
        'Peripheral\nUsers',
        fontsize=8, color='#888888', style='italic'
    )

    # --- Legend ---
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#457B9D', markersize=9, label='Democrat'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#E63946', markersize=9, label='Republican'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#6C757D', markersize=9, label='Unknown'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9)

    ax.set_xlabel('PageRank (Influence)', fontsize=11)
    ax.set_ylabel('Betweenness Centrality (Bridge Role)', fontsize=11)
    ax.set_title(
        'PageRank vs Betweenness Centrality — User Roles in the Network',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    output_file = os.path.join(OUTPUT_PATH, 'user_pagerank_vs_betweenness.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved to: data/user_pagerank_vs_betweenness.png")
    plt.show()


def main():
    """
    Runs all visualizations for the user graph.
    Loads the graph and results from data/ folder.
    """
    print("=" * 40)
    print("VISUALIZING USER GRAPH RESULTS")
    print("=" * 40)

    # Step 1: Load graph
    print("\n[1/4] Loading graph...")
    graph_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_graph.graphml')
    G = nx.read_graphml(graph_path)
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    # Step 2: Load results
    print("\n[2/4] Loading algorithm results...")
    results_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_graph_results.csv')
    results_df = pd.read_csv(results_path)

    # Rebuild partition dictionary from results
    partition = dict(zip(results_df['user'], results_df['louvain_community']))
    pr_df = results_df[['user', 'pagerank']].sort_values('pagerank', ascending=False).reset_index(drop=True)

    # Step 3: Plot community graph
    print("\n[3/4] Generating visualizations...")
    plot_community_graph(G, partition, top_n_communities=5, max_nodes=500)

    # Step 4: Plot PageRank bar chart
    plot_pagerank(pr_df, top_n=15)

    # Step 5: Plot PageRank vs Betweenness scatter
    plot_pagerank_vs_betweenness(results_df, top_n_labels=15)

    print("\n" + "=" * 40)
    print("ALL VISUALIZATIONS SAVED TO data/")
    print("=" * 40)


if __name__ == '__main__':
    main()