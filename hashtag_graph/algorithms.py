import networkx as nx
import community as community_louvain
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ── Louvain ───────────────────────────────────────────────────────────────────

def run_louvain(G):
    """
    Detects hashtag communities using the Louvain algorithm.
    Each community = a cluster of hashtags that frequently appear together
    → these map to ideological bubbles (e.g. #MAGA + #Trump2024 + #KAG).

    Louvain optimizes modularity — higher score = better separated clusters.
    """
    print("\n[Louvain] Running community detection...")
    partition = community_louvain.best_partition(G, weight='weight')

    num_communities = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, G, weight='weight')

    print(f"  Communities found: {num_communities}")
    print(f"  Modularity score:  {modularity:.4f}")

    community_sizes = pd.Series(partition.values()).value_counts().sort_index()
    print(f"\n  Top 10 largest communities:")
    print(community_sizes.head(10).to_string())

    return partition, modularity


# ── Label Propagation ─────────────────────────────────────────────────────────

def run_label_propagation(G):
    """
    Detects communities using Label Propagation.
    Each hashtag starts with a unique label and adopts the most common label
    among its neighbors iteratively. Faster than Louvain but non-deterministic.
    """
    print("\n[Label Propagation] Running community detection...")
    communities = nx.community.label_propagation_communities(G)

    partition = {}
    for community_id, community_members in enumerate(communities):
        for node in community_members:
            partition[node] = community_id

    num_communities = len(set(partition.values()))

    community_sets = [
        {node for node, cid in partition.items() if cid == c}
        for c in set(partition.values())
    ]
    modularity = nx.community.modularity(G, community_sets, weight='weight')

    print(f"  Communities found:  {num_communities}")
    print(f"  Modularity score:   {modularity:.4f}")

    community_sizes = pd.Series(partition.values()).value_counts().sort_index()
    print(f"\n  Top 10 largest communities:")
    print(community_sizes.head(10).to_string())

    return partition, modularity


# ── PageRank ──────────────────────────────────────────────────────────────────

def run_pagerank(G):
    """
    Calculates PageRank for each hashtag.
    A hashtag has high PageRank if it co-occurs with many other important hashtags.
    These are the semantic anchors of their ideological bubble —
    the hashtags that tie the whole cluster together.
    """
    print("\n[PageRank] Calculating scores...")
    pagerank_scores = nx.pagerank(G, weight='weight')

    pr_df = pd.DataFrame([
        {'hashtag': node, 'pagerank': score}
        for node, score in pagerank_scores.items()
    ]).sort_values('pagerank', ascending=False).reset_index(drop=True)

    print(f"  Total hashtags ranked: {len(pr_df):,}")
    print(f"\n  Top 10 most central hashtags:")
    print(pr_df.head(10).to_string())

    return pagerank_scores, pr_df


# ── Betweenness Centrality ────────────────────────────────────────────────────

def run_betweenness_centrality(G):
    """
    Calculates Betweenness Centrality for each hashtag.
    A hashtag has high betweenness if it bridges different ideological clusters —
    it appears in contexts spanning multiple bubbles.
    High betweenness = potential cross-cutting hashtag (e.g. #Election2024
    used by both sides), which is key evidence for echo chamber boundaries.
    """
    print("\n[Betweenness Centrality] Calculating scores...")

    # k=500 approximation — exact betweenness is very slow on large graphs
    betweenness_scores = nx.betweenness_centrality(
        G,
        weight='weight',
        normalized=True,
        k=min(500, G.number_of_nodes())   # safe if graph is small
    )

    bc_df = pd.DataFrame([
        {'hashtag': node, 'betweenness': score}
        for node, score in betweenness_scores.items()
    ]).sort_values('betweenness', ascending=False).reset_index(drop=True)

    print(f"  Total hashtags ranked: {len(bc_df):,}")
    print(f"\n  Top 10 bridge hashtags (cross-cutting):")
    print(bc_df.head(10).to_string())

    return betweenness_scores, bc_df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 40)
    print("RUNNING ALGORITHMS ON HASHTAG GRAPH")
    print("=" * 40)

    # Step 1: Load graph
    print("\n[1/5] Loading graph...")
    graph_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hashtag_graph.graphml')
    G = nx.read_graphml(graph_path)
    print(f"  Nodes (hashtags): {G.number_of_nodes():,}")
    print(f"  Edges:            {G.number_of_edges():,}")

    # Step 2: Louvain
    louvain_partition, louvain_modularity = run_louvain(G)

    # Step 3: Label Propagation
    lp_partition, lp_modularity = run_label_propagation(G)

    # Step 4: PageRank
    pagerank_scores, pr_df = run_pagerank(G)

    # Step 5: Betweenness Centrality
    betweenness_scores, bc_df = run_betweenness_centrality(G)

    # Step 6: Combine and save results
    print("\n[5/5] Saving results...")
    results_df = pd.DataFrame({
        'hashtag': list(louvain_partition.keys()),
        'louvain_community': list(louvain_partition.values()),
        'lp_community': [lp_partition.get(node, -1) for node in louvain_partition.keys()],
        'pagerank': [pagerank_scores.get(node, 0) for node in louvain_partition.keys()],
        'betweenness': [betweenness_scores.get(node, 0) for node in louvain_partition.keys()],
    })

    results_df = results_df.sort_values('pagerank', ascending=False).reset_index(drop=True)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hashtag_graph_results.csv')
    results_df.to_csv(output_path, index=False)

    print(f"\n  Results saved to: data/hashtag_graph_results.csv")
    print(f"\n  Sample results:")
    print(results_df.head(10).to_string())

    # Summary
    print("\n" + "=" * 40)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 40)
    print(f"  Louvain    — communities: {len(set(louvain_partition.values()))}, modularity: {louvain_modularity:.4f}")
    print(f"  Label Prop — communities: {len(set(lp_partition.values()))}, modularity: {lp_modularity:.4f}")

    return results_df


if __name__ == '__main__':
    main()
