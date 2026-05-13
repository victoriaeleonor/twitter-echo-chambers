import networkx as nx
import community as community_louvain
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def run_louvain(G):
    """
    Detects communities using the Louvain algorithm.
    Louvain optimizes modularity — a score that measures
    how well-separated the communities are (0 to 1).
    Higher modularity = better defined communities.
    """

    print("\n[Louvain] Running community detection...")
    # returns a dictionary {node: community_id}
    partition = community_louvain.best_partition(G, weight='weight')

    # Count how many communities were found
    num_communities = len(set(partition.values()))

    # Calculate modularity score
    modularity = community_louvain.modularity(partition, G, weight='weight')

    print(f"  Communities found: {num_communities}")
    print(f"  Modularity score: {modularity:.4f}")

    # Count how many users are in each community
    community_sizes = pd.Series(partition.values()).value_counts().sort_index()
    print(f"\n  Top 10 largest communities:")
    print(community_sizes.head(10).to_string())

    return partition, modularity


def run_label_propagation(G):
    """
    Detects communities using Label Propagation.
    Each node starts with a unique label. Then iteratively,
    each node adopts the most common label among its neighbors.
    Faster than Louvain but results may vary between runs.
    """

    print("\n[Label Propagation] Running community detection...")
    # returns a generator of sets
    communities = nx.community.label_propagation_communities(G)

    # Convert generator to a partition dictionary {node: community_id}
    partition = {}
    for community_id, community_members in enumerate(communities):
        for node in community_members:
            partition[node] = community_id

    # Count communities
    num_communities = len(set(partition.values()))

    # Calculate modularity
    community_sets = [
        {node for node, cid in partition.items() if cid == c}
        for c in set(partition.values())
    ]
    modularity = nx.community.modularity(G, community_sets, weight='weight')

    print(f"  Communities found:  {num_communities}")
    print(f"  Modularity score:   {modularity:.4f}")

    # Community sizes
    community_sizes = pd.Series(partition.values()).value_counts().sort_index()
    print(f"\n  Top 10 largest communities:")
    print(community_sizes.head(10).to_string())

    return partition, modularity


def run_pagerank(G):
    """
    Calculates PageRank for each user in the graph.
    A user has high PageRank if many important users interact with them.
    Identifies the most influential users in the network.
    """

    print("\n[PageRank] Calculating scores...")
    pagerank_scores = nx.pagerank(G, weight='weight')

    # Convert to DataFrame and sort by score
    pr_df = pd.DataFrame([
        {'user': node, 'pagerank': score}
        for node, score in pagerank_scores.items()
    ]).sort_values('pagerank', ascending=False).reset_index(drop=True)

    print(f"  Total users ranked: {len(pr_df):,}")
    print(f"\n  Top 10 most influential users:")
    print(pr_df.head(10).to_string())

    return pagerank_scores, pr_df


def run_betweenness_centrality(G):
    """
    Calculates Betweenness Centrality for each user.
    A user has high betweenness if they act as a bridge
    between different communities — they are the connectors
    between echo chambers.
    High betweenness + low community membership = bridge user.
    """
    print("\n[Betweenness Centrality] Calculating scores...")
    # Run Betweenness Centrality
    # k=500 means we use 500 sample nodes for approximation
    # exact calculation on large graphs is very slow
    betweenness_scores = nx.betweenness_centrality(
        G,
        weight='weight',
        normalized=True,
        k=500
    )

    # Convert to DataFrame and sort
    bc_df = pd.DataFrame([
        {'user': node, 'betweenness': score}
        for node, score in betweenness_scores.items()
    ]).sort_values('betweenness', ascending=False).reset_index(drop=True)

    print(f"  Total users ranked: {len(bc_df):,}")
    print(f"\n  Top 10 bridge users:")
    print(bc_df.head(10).to_string())

    return betweenness_scores, bc_df

# runs all algorithms
def main():
    print("=" * 40)
    print("RUNNING ALGORITHMS ON USER GRAPH")
    print("=" * 40)

    # Step 1: Load the graph
    print("\n[1/5] Loading graph...")
    graph_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_graph.graphml')
    G = nx.read_graphml(graph_path)
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    # Step 2: Louvain
    louvain_partition, louvain_modularity = run_louvain(G)

    # Step 3: Label Propagation
    lp_partition, lp_modularity = run_label_propagation(G)

    # Step 4: PageRank
    pagerank_scores, pr_df = run_pagerank(G)

    # Step 5: Betweenness Centrality
    betweenness_scores, bc_df = run_betweenness_centrality(G)

    # Step 6: Combine all results into one DataFrame
    print("\n[5/5] Saving results...")
    results_df = pd.DataFrame({
        'user': list(louvain_partition.keys()),
        'louvain_community': list(louvain_partition.values()),
        'lp_community': [lp_partition.get(node, -1) for node in louvain_partition.keys()],
        'pagerank': [pagerank_scores.get(node, 0) for node in louvain_partition.keys()],
        'betweenness': [betweenness_scores.get(node, 0) for node in louvain_partition.keys()],
    })

    # Sort by pagerank
    results_df = results_df.sort_values('pagerank', ascending=False).reset_index(drop=True)

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_graph_results.csv')
    results_df.to_csv(output_path, index=False)

    print(f"\n  Results saved to: data/user_graph_results.csv")
    print(f"\n  Sample results:")
    print(results_df.head(10).to_string())

    # Summary comparison
    print("\n" + "=" * 40)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 40)
    print(f"  Louvain — communities: {len(set(louvain_partition.values()))}, modularity: {louvain_modularity:.4f}")
    print(f"  Label Prop — communities: {len(set(lp_partition.values()))}, modularity: {lp_modularity:.4f}")

    return results_df


if __name__ == '__main__':
    main()