import pandas as pd
import networkx as nx
import ast  # converts text to a pyhton dict 
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.loader import load_tweets

# extracts the username column
def extract_username(raw):
    try:
        match = re.search(r"'username':\s*'([^']+)'", str(raw))
        if match:
            return match.group(1)
    except:
        return None
    return None
    

# build the edges - represents the interaction between two users
# replies: user -> in_reply_to_screen_name
# mentions: user -> mentioned_username
def build_edges(df):
    edges = []

    df['username'] = df['user'].apply(extract_username)  # we create a new column 'username' 
    print(f"\n--- Username extraction sample ---")
    print(df['username'].head(10).to_string())
    print(f"Null usernames: {df['username'].isna().sum():,}")

     # --- Source 1: Replies ---
    replies = df[df['in_reply_to_screen_name'].notna()][['username', 'in_reply_to_screen_name']]

    for _, row in replies.iterrows():
        source = row['username']
        target = row['in_reply_to_screen_name']

        # Skip if source or target is missing
        if pd.isna(source) or pd.isna(target):
            continue
        if source == target:
            continue

        edges.append((str(source), str(target)))

    # --- Source 2: Mentions ---
    mentions_df = df[df['mentionedUsers'].notna()][['username', 'mentionedUsers']]

    for _, row in mentions_df.iterrows():
        try:
            mentioned = ast.literal_eval(row['mentionedUsers'])
            if isinstance(mentioned, list):
                for m in mentioned:
                    if isinstance(m, dict):
                        target = m.get('username', None)
                        source = row['username']
                        if source and target and source != target:
                            edges.append((source, target))
        except:
            continue

    print(f"Total edges collected: {len(edges):,}")
    # Diagnose: how many valid edges after filtering nan?
    valid = [(s, t) for s, t in edges if s != 'nan' and t != 'nan']
    print(f"Valid edges (no nan): {len(valid):,}")
    return valid
    #return edges


# creates weighted undirected graph from the list of edges
def build_graph(edges):
    G = nx.Graph()  #undirected graph

    for source, target in edges:
        if G.has_edge(source, target):
            # If edge already exists, increase the weight
            G[source][target]['weight'] += 1
        else:
            # If edge doesn't exist, create it with weight 1
            G.add_edge(source, target, weight=1)

    print(f"Graph created:")
    print(f"  Nodes (users): {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    return G

# users who interacted once are noise - no real community
def filter_graph(G, min_degree=2):

    nodes_before = G.number_of_nodes()

    # Identify nodes to remove
    nodes_to_remove = [node for node, degree in G.degree() if degree < min_degree]
    
    # Remove them
    G.remove_nodes_from(nodes_to_remove)

    nodes_after = G.number_of_nodes()

    print(f"Filtering (min_degree={min_degree}):")
    print(f"  Nodes before: {nodes_before:,}")
    print(f"  Nodes removed: {len(nodes_to_remove):,}")
    print(f"  Nodes after: {nodes_after:,}")
    print(f"  Edges after: {G.number_of_edges():,}")

    return G

# main function that runs the complete pipeline
def main():
    print("=" * 40)
    print("BUILDING USER INTERACTION GRAPH")
    print("=" * 40)

    # Step 1: Load dataset
    print("\n[1/4] Loading dataset...")
    df = load_tweets()

    # Step 2: Build edges
    print("\n[2/4] Building edges...")
    edges = build_edges(df)

    # Step 3: Create graph
    print("\n[3/4] Creating graph...")
    G = build_graph(edges)

    # Step 4: Filter noise
    print("\n[4/4] Filtering low-degree nodes...")
    G = filter_graph(G, min_degree=2)

    # Step 5: Save graph
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_graph.graphml')
    nx.write_graphml(G, output_path)
    print(f"\nGraph saved to: data/user_graph.graphml")

    print("DONE")
    return G


if __name__ == '__main__':
    main()