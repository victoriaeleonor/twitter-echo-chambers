import pandas as pd
import networkx as nx
import itertools
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.loader import load_tweets, parse_hashtags


# ── Step 1: extract co-occurrence edges ──────────────────────────────────────

def build_edges(df):
    """
    For each tweet with ≥2 hashtags, generate all pairwise co-occurrences.
    Example: tweet with [#MAGA, #Trump2024, #KAG] → edges:
        (maga, trump2024), (maga, kag), (trump2024, kag)

    Returns a list of (hashtag_a, hashtag_b) tuples (lowercased, sorted).
    """
    edges = []
    skipped = 0

    for raw in df['hashtags'].dropna():
        tags = parse_hashtags(raw)  # reuse loader.py parser → list of lowercase strings

        if len(tags) < 2:
            skipped += 1
            continue

        # Deduplicate tags within the same tweet to avoid self-loops
        tags = list(set(tags))

        # All pairs from this tweet
        for tag_a, tag_b in itertools.combinations(tags, 2):
            # Sort so (maga, trump2024) and (trump2024, maga) are the same edge
            edges.append(tuple(sorted([tag_a, tag_b])))

    print(f"  Tweets processed:        {len(df['hashtags'].dropna()):,}")
    print(f"  Tweets with <2 hashtags: {skipped:,}")
    print(f"  Total co-occurrence pairs collected: {len(edges):,}")
    return edges


# ── Step 2: build weighted undirected graph ───────────────────────────────────

def build_graph(edges):
    """
    Creates a weighted undirected graph from the edge list.
    Weight = number of tweets where both hashtags appeared together.
    """
    G = nx.Graph()

    for tag_a, tag_b in edges:
        if G.has_edge(tag_a, tag_b):
            G[tag_a][tag_b]['weight'] += 1
        else:
            G.add_edge(tag_a, tag_b, weight=1)

    print(f"Graph created:")
    print(f"  Nodes (hashtags): {G.number_of_nodes():,}")
    print(f"  Edges:            {G.number_of_edges():,}")

    return G


# ── Step 3: filter noise ──────────────────────────────────────────────────────

def filter_graph(G, min_edge_weight=2, min_degree=2):
    """
    Remove low-signal hashtags:
      - Edges with weight < min_edge_weight (appeared together only once = noise)
      - Nodes with degree < min_degree after pruning (isolated or near-isolated)

    This keeps the meaningful semantic clusters and discards one-off tags.
    """
    nodes_before = G.number_of_nodes()
    edges_before = G.number_of_edges()

    # Remove weak edges
    weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_edge_weight]
    G.remove_edges_from(weak_edges)

    # Remove low-degree nodes (hashtags that no longer connect to anything meaningful)
    nodes_to_remove = [n for n, deg in G.degree() if deg < min_degree]
    G.remove_nodes_from(nodes_to_remove)

    print(f"Filtering (min_edge_weight={min_edge_weight}, min_degree={min_degree}):")
    print(f"  Nodes before: {nodes_before:,}  →  after: {G.number_of_nodes():,}")
    print(f"  Edges before: {edges_before:,}  →  after: {G.number_of_edges():,}")
    print(f"  Weak edges removed: {len(weak_edges):,}")
    print(f"  Isolated nodes removed: {len(nodes_to_remove):,}")

    return G


# ── Step 4: quick sanity check ───────────────────────────────────────────────

def print_top_hashtags(G, top_n=15):
    """
    Prints the top hashtags by weighted degree (total co-occurrence strength).
    A high weighted degree means this hashtag co-occurs frequently with many others.
    """
    weighted_degree = dict(G.degree(weight='weight'))
    top = sorted(weighted_degree.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n  Top {top_n} hashtags by co-occurrence strength:")
    for rank, (tag, score) in enumerate(top, 1):
        print(f"    {rank:>2}. #{tag:<30} weighted degree: {score:,}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 40)
    print("BUILDING HASHTAG CO-OCCURRENCE GRAPH")
    print("=" * 40)

    # Step 1: Load dataset
    print("\n[1/5] Loading dataset...")
    df = load_tweets()

    # Step 2: Build edges
    print("\n[2/5] Extracting co-occurrence edges...")
    edges = build_edges(df)

    # Step 3: Create graph
    print("\n[3/5] Creating graph...")
    G = build_graph(edges)

    # Step 4: Filter noise
    print("\n[4/5] Filtering low-signal nodes and edges...")
    G = filter_graph(G, min_edge_weight=2, min_degree=2)

    # Step 5: Sanity check + save
    print("\n[5/5] Top hashtags & saving...")
    print_top_hashtags(G, top_n=15)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hashtag_graph.graphml')
    nx.write_graphml(G, output_path)
    print(f"\nGraph saved to: data/hashtag_graph.graphml")
    print("DONE")

    return G


if __name__ == '__main__':
    main()
