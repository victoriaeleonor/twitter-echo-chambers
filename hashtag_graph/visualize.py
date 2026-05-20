import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
import sys
import math
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


# ── Plot 1: Community graph ───────────────────────────────────────────────────

def plot_community_graph(G, partition, top_n_communities=5, max_nodes=400):
    """
    Plots the hashtag co-occurrence graph colored by Louvain community.
    Each cluster visually represents an ideological bubble.
    Node size = weighted degree (how central the hashtag is in its cluster).
    Labels shown only for the highest-degree hashtags to avoid clutter.
    """
    print("\n[Graph] Plotting community graph...")

    # Keep only top N largest communities
    community_sizes = pd.Series(partition.values()).value_counts()
    top_communities = community_sizes.head(top_n_communities).index.tolist()

    nodes_to_plot = [
        node for node, comm in partition.items()
        if comm in top_communities and node in G.nodes()
    ]

    # Sample down if too many nodes, keeping high-degree ones
    if len(nodes_to_plot) > max_nodes:
        sampled = []
        weighted_degrees = dict(G.degree(weight='weight'))
        for comm in top_communities:
            comm_nodes = [n for n in nodes_to_plot if partition[n] == comm]

            # Always keep the top 10 most connected hashtags per community
            top_comm_nodes = sorted(
                comm_nodes,
                key=lambda n: weighted_degrees.get(n, 0),
                reverse=True
            )[:10]

            remaining = [n for n in comm_nodes if n not in top_comm_nodes]
            n_sample = int(max_nodes * community_sizes[comm] / community_sizes[top_communities].sum())
            n_random = max(0, n_sample - len(top_comm_nodes))
            random_nodes = random.sample(remaining, min(n_random, len(remaining)))

            sampled.extend(top_comm_nodes + random_nodes)
        nodes_to_plot = sampled

    subgraph = G.subgraph(nodes_to_plot)
    print(f"  Plotting {len(subgraph.nodes()):,} nodes from top {top_n_communities} communities")

    # Colors — same palette as user graph for visual consistency
    colors_list = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
    community_colors = {comm: colors_list[i] for i, comm in enumerate(top_communities)}
    node_colors = [community_colors[partition[node]] for node in subgraph.nodes()]

    # Two-level layout: communities arranged in a circle, nodes spring within
    print("  Calculating two-level layout...")
    community_centers = {}
    for i, comm in enumerate(top_communities):
        angle = 2 * math.pi * i / len(top_communities)
        community_centers[comm] = (math.cos(angle) * 3, math.sin(angle) * 3)

    pos = {}
    for comm in top_communities:
        comm_nodes = [n for n in subgraph.nodes() if partition[n] == comm]
        comm_subgraph = subgraph.subgraph(comm_nodes)

        if len(comm_nodes) == 1:
            pos[comm_nodes[0]] = community_centers[comm]
            continue

        sub_pos = nx.spring_layout(comm_subgraph, seed=42, k=0.4, scale=0.9)
        cx, cy = community_centers[comm]
        for node, (x, y) in sub_pos.items():
            pos[node] = (x + cx, y + cy)

    # Node sizes by weighted degree
    weighted_degrees = dict(G.degree(weight='weight'))
    node_sizes = [30 + (weighted_degrees.get(node, 1) * 8) for node in subgraph.nodes()]

    fig, ax = plt.subplots(figsize=(16, 12))

    nx.draw_networkx_edges(
        subgraph, pos,
        alpha=0.12,
        edge_color='#848484',
        width=0.4,
        ax=ax
    )

    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.85,
        ax=ax
    )

    # Labels for top hashtags (high weighted degree in the full graph)
    label_threshold = sorted(weighted_degrees.values(), reverse=True)[min(30, len(weighted_degrees) - 1)]
    high_degree_labels = {
        node: f'#{node}' for node in subgraph.nodes()
        if weighted_degrees.get(node, 0) >= label_threshold
    }
    nx.draw_networkx_labels(
        subgraph, pos,
        labels=high_degree_labels,
        font_size=7,
        font_weight='bold',
        ax=ax
    )

    # Community background circles
    for comm in top_communities:
        comm_nodes = [n for n in subgraph.nodes() if partition[n] == comm]
        if len(comm_nodes) < 3:
            continue
        comm_positions = [pos[n] for n in comm_nodes]
        xs = [p[0] for p in comm_positions]
        ys = [p[1] for p in comm_positions]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        radius = max(max(xs) - cx, max(ys) - cy) + 0.35
        circle = plt.Circle(
            (cx, cy), radius,
            color=community_colors[comm],
            alpha=0.08,
            zorder=0
        )
        ax.add_patch(circle)
        ax.text(
            cx, cy + radius + 0.1,
            f'Cluster {comm}\n({community_sizes[comm]} hashtags)',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold',
            color=community_colors[comm]
        )

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=community_colors[comm],
                   markersize=10,
                   label=f'Cluster {comm} ({community_sizes[comm]} hashtags)')
        for comm in top_communities
    ]
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)
    ax.set_title(
        f'Hashtag Co-occurrence Graph — Top {top_n_communities} Communities (Louvain)',
        fontsize=15, fontweight='bold', pad=30
    )
    ax.axis('off')
    plt.tight_layout()

    output_file = os.path.join(OUTPUT_PATH, 'hashtag_graph_communities.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved to: data/hashtag_graph_communities.png")
    plt.show()


# ── Plot 2: Top hashtags by PageRank ─────────────────────────────────────────

def plot_pagerank(pr_df, partition, top_n=15):
    """
    Horizontal bar chart of the top N hashtags by PageRank.
    Bars are colored by their Louvain community, so you can immediately
    see which ideological cluster each influential hashtag belongs to.
    """
    print("\n[PageRank] Plotting bar chart...")

    top_tags = pr_df.head(top_n).copy()

    colors_list = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#9B5DE5', '#00BBF9']

    community_sizes = pd.Series(partition.values()).value_counts()
    top_communities = community_sizes.head(len(colors_list)).index.tolist()
    community_colors = {comm: colors_list[i] for i, comm in enumerate(top_communities)}

    def assign_color(tag):
        comm = partition.get(tag, -1)
        return community_colors.get(comm, '#AAAAAA')

    top_tags['color'] = top_tags['hashtag'].apply(assign_color)
    top_tags['label'] = top_tags['hashtag'].apply(lambda t: f'#{t}')

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(
        top_tags['label'],
        top_tags['pagerank'],
        color=top_tags['color'],
        edgecolor='white',
        height=0.6
    )

    for bar, value in zip(bars, top_tags['pagerank']):
        ax.text(
            bar.get_width() + max(top_tags['pagerank']) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f'{value:.5f}',
            va='center', ha='left',
            fontsize=8, color='#333333'
        )

    legend_handles = [
        mpatches.Patch(color=community_colors[comm], label=f'Cluster {comm}')
        for comm in top_communities
        if comm in top_tags['hashtag'].apply(lambda t: partition.get(t, -1)).values
    ]
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9)

    ax.set_xlabel('PageRank Score', fontsize=11)
    ax.set_title(
        f'Top {top_n} Most Central Hashtags — PageRank',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    output_file = os.path.join(OUTPUT_PATH, 'hashtag_pagerank.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved to: data/hashtag_pagerank.png")
    plt.show()


# ── Plot 3: PageRank vs Betweenness scatter ───────────────────────────────────

def plot_pagerank_vs_betweenness(results_df, partition, top_n_labels=20):
    """
    Scatter plot of PageRank vs Betweenness Centrality.
    Each point is a hashtag. The four quadrants reveal different roles:
      - High PR + High BW = dominant cross-cutting hashtags (bridge anchors)
      - High PR + Low BW  = echo chamber core hashtags (bubble leaders)
      - Low PR  + High BW = niche bridge hashtags (hidden connectors)
      - Low PR  + Low BW  = peripheral hashtags
    This is the key plot for characterizing echo chamber structure.
    """
    print("\n[Scatter] Plotting PageRank vs Betweenness...")

    colors_list = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#9B5DE5', '#00BBF9']
    community_sizes = pd.Series(partition.values()).value_counts()
    top_communities = community_sizes.head(len(colors_list)).index.tolist()
    community_colors = {comm: colors_list[i] for i, comm in enumerate(top_communities)}

    def assign_color(tag):
        comm = partition.get(tag, -1)
        return community_colors.get(comm, '#CCCCCC')

    fig, ax = plt.subplots(figsize=(13, 8))

    # All hashtags as small dots, colored by community
    all_colors = results_df['hashtag'].apply(assign_color)
    ax.scatter(
        results_df['pagerank'],
        results_df['betweenness'],
        c=all_colors,
        alpha=0.35,
        s=18,
        zorder=1
    )

    # Highlight and label top hashtags by pagerank
    top_tags = results_df.head(top_n_labels).copy()
    top_colors = top_tags['hashtag'].apply(assign_color)
    ax.scatter(
        top_tags['pagerank'],
        top_tags['betweenness'],
        c=top_colors,
        s=130,
        zorder=2,
        edgecolors='white',
        linewidths=0.8
    )

    for _, row in top_tags.iterrows():
        ax.annotate(
            f'#{row["hashtag"]}',
            xy=(row['pagerank'], row['betweenness']),
            xytext=(6, 4),
            textcoords='offset points',
            fontsize=7.5,
            color='#333333'
        )

    # Quadrant lines at median
    median_pr = results_df['pagerank'].median()
    median_bt = results_df['betweenness'].median()
    ax.axvline(x=median_pr, color='#AAAAAA', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(y=median_bt, color='#AAAAAA', linestyle='--', linewidth=0.8, alpha=0.7)

    # Quadrant labels
    pr_max = results_df['pagerank'].max()
    bt_max = results_df['betweenness'].max()
    ax.text(pr_max * 0.72, bt_max * 0.92, 'Bridge\nAnchors',     fontsize=8, color='#888888', style='italic')
    ax.text(pr_max * 0.72, median_bt * 0.1, 'Bubble\nLeaders',   fontsize=8, color='#888888', style='italic')
    ax.text(median_pr * 0.05, bt_max * 0.92, 'Hidden\nBridges',  fontsize=8, color='#888888', style='italic')
    ax.text(median_pr * 0.05, median_bt * 0.1, 'Peripheral\nTags', fontsize=8, color='#888888', style='italic')

    # Legend by community
    legend_handles = [
        mpatches.Patch(color=community_colors[comm], label=f'Cluster {comm}')
        for comm in top_communities
    ]
    ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9)

    ax.set_xlabel('PageRank (Centrality within co-occurrence network)', fontsize=11)
    ax.set_ylabel('Betweenness Centrality (Bridge role across clusters)', fontsize=11)
    ax.set_title(
        'PageRank vs Betweenness — Hashtag Roles in the Echo Chamber Network',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    output_file = os.path.join(OUTPUT_PATH, 'hashtag_pagerank_vs_betweenness.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved to: data/hashtag_pagerank_vs_betweenness.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 40)
    print("VISUALIZING HASHTAG GRAPH RESULTS")
    print("=" * 40)

    # Load graph
    print("\n[1/4] Loading graph...")
    graph_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hashtag_graph.graphml')
    G = nx.read_graphml(graph_path)
    print(f"  Nodes (hashtags): {G.number_of_nodes():,}")
    print(f"  Edges:            {G.number_of_edges():,}")

    # Load results
    print("\n[2/4] Loading algorithm results...")
    results_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hashtag_graph_results.csv')
    results_df = pd.read_csv(results_path)

    partition = dict(zip(results_df['hashtag'], results_df['louvain_community']))
    pr_df = results_df[['hashtag', 'pagerank']].sort_values('pagerank', ascending=False).reset_index(drop=True)

    # Visualizations
    print("\n[3/4] Generating visualizations...")
    plot_community_graph(G, partition, top_n_communities=5, max_nodes=400)
    plot_pagerank(pr_df, partition, top_n=15)
    plot_pagerank_vs_betweenness(results_df, partition, top_n_labels=20)

    print("\n" + "=" * 40)
    print("ALL VISUALIZATIONS SAVED TO data/")
    print("=" * 40)


if __name__ == '__main__':
    main()
