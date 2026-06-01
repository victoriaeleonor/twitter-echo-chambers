import pandas as pd
import numpy as np
import networkx as nx
import ast
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import normalized_mutual_info_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.loader import load_tweets

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_results():
    """
    Loads results from both graphs.
    Returns user results, hashtag results, and the original tweets dataframe.
    """
    print("\n[1/4] Loading results...")

    # User graph results
    user_results = pd.read_csv(os.path.join(DATA_PATH, 'user_graph_results.csv'))
    print(f"User graph: {len(user_results):,} users")

    # Hashtag graph results
    hashtag_results = pd.read_csv(os.path.join(DATA_PATH, 'hashtag_graph_results.csv'))
    print(f"Hashtag graph: {len(hashtag_results):,} hashtags")

    # Original tweets for cross-analysis
    df = load_tweets()

    return user_results, hashtag_results, df


def compare_communities_nmi(user_results, hashtag_results):
    """
    Compares community structures between user and hashtag graphs
    using Normalized Mutual Information (NMI).
    
    To compare both graphs we need a common element.
    We use the tweets dataset to link users to hashtags:
    - A user 'belongs' to the hashtag communities they used most.
    - We compare if users in the same user-community
      also used hashtags from the same hashtag-community.
    """
    print("\n[2/4] Comparing communities with NMI...")

    # --- Step 1: Algorithm comparison table ---
    print("\n  === ALGORITHM COMPARISON ===")
    print(f"  {'Metric':<30} {'User Graph':>12} {'Hashtag Graph':>14}")
    print(f"  {'-'*56}")

    # Louvain communities
    user_louvain_n = user_results['louvain_community'].nunique()
    hashtag_louvain_n = hashtag_results['louvain_community'].nunique()
    print(f"  {'Louvain communities':<30} {user_louvain_n:>12} {hashtag_louvain_n:>14}")

    # Label Propagation communities
    user_lp_n = user_results['lp_community'].nunique()
    hashtag_lp_n = hashtag_results['lp_community'].nunique()
    print(f"  {'Label Prop communities':<30} {user_lp_n:>12} {hashtag_lp_n:>14}")

    # PageRank stats
    user_pr_mean = user_results['pagerank'].mean()
    hashtag_pr_mean = hashtag_results['pagerank'].mean()
    print(f"  {'PageRank mean':<30} {user_pr_mean:>12.6f} {hashtag_pr_mean:>14.6f}")

    # Betweenness stats
    user_bt_mean = user_results['betweenness'].mean()
    hashtag_bt_mean = hashtag_results['betweenness'].mean()
    print(f"  {'Betweenness mean':<30} {user_bt_mean:>12.6f} {hashtag_bt_mean:>14.6f}")

    # --- Step 2: NMI between Louvain and Label Propagation within each graph ---
    print(f"\n  === NMI: LOUVAIN vs LABEL PROPAGATION ===")

    # User graph — how consistent are the two algorithms?
    user_nmi = normalized_mutual_info_score(
        user_results['louvain_community'],
        user_results['lp_community']
    )
    print(f"  User graph    (Louvain vs LabelProp): {user_nmi:.4f}")

    # Hashtag graph — same question
    hashtag_nmi = normalized_mutual_info_score(
        hashtag_results['louvain_community'],
        hashtag_results['lp_community']
    )
    print(f"  Hashtag graph (Louvain vs LabelProp): {hashtag_nmi:.4f}")

    # --- Step 3: Interpret NMI ---
    print(f"\n  === INTERPRETATION ===")
    for name, nmi in [("User graph", user_nmi), ("Hashtag graph", hashtag_nmi)]:
        if nmi >= 0.7:
            interpretation = "Very consistent — both algorithms agree strongly"
        elif nmi >= 0.4:
            interpretation = "Moderately consistent — algorithms partially agree"
        else:
            interpretation = "Low consistency — algorithms disagree significantly"
        print(f"  {name}: {interpretation}")

    return user_nmi, hashtag_nmi


def cross_analysis(user_results, hashtag_results, df):
    """
    Cross-analyzes user communities with hashtag usage.
    Answers the key question:
    Do users in the same community use the same hashtags?
    If yes — that's direct evidence of semantic echo chambers.
    """
    print("\n[3/4] Cross-analyzing users and hashtags...")

    # --- Step 1: Parse hashtags from tweets ---
    def parse_hashtags(raw):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [h['text'].lower() for h in parsed if isinstance(h, dict) and 'text' in h]
        except:
            pass
        return []

    import re
    def extract_username(raw):
        try:
            match = re.search(r"'username':\s*'([^']+)'", str(raw))
            if match:
                return match.group(1)
        except:
            return None
        return None

    # Extract username and hashtags from each tweet
    df['username'] = df['user'].apply(extract_username)
    df['hashtag_list'] = df['hashtags'].apply(parse_hashtags)

    # --- Step 2: Map users to their Louvain community ---
    user_community_map = dict(zip(
        user_results['user'],
        user_results['louvain_community']
    ))

    # --- Step 3: Get top communities ---
    top_communities = (
        user_results['louvain_community']
        .value_counts()
        .head(5)
        .index.tolist()
    )

    # --- Step 4: For each community, collect all hashtags used ---
    print(f"\n  === TOP HASHTAGS PER USER COMMUNITY ===\n")

    community_hashtags = {}
    for comm in top_communities:
        # Get all users in this community
        comm_users = set(
            user_results[user_results['louvain_community'] == comm]['user']
        )

        # Get all tweets by those users
        comm_tweets = df[df['username'].isin(comm_users)]

        # Flatten all hashtags
        all_hashtags = (
            comm_tweets['hashtag_list']
            .explode()
            .dropna()
        )
        all_hashtags = all_hashtags[all_hashtags != '']

        # Top 10 hashtags
        top_hashtags = all_hashtags.value_counts().head(10)
        community_hashtags[comm] = top_hashtags

        n_users = len(comm_users)
        n_tweets = len(comm_tweets)
        print(f"  Community {comm} ({n_users} users, {n_tweets} tweets):")
        for hashtag, count in top_hashtags.items():
            print(f"    #{hashtag:<25} {count:>4} uses")
        print()

    # --- Step 5: Measure hashtag overlap between communities ---
    print(f"  === HASHTAG OVERLAP BETWEEN COMMUNITIES ===\n")

    community_ids = list(community_hashtags.keys())
    for i in range(len(community_ids)):
        for j in range(i + 1, len(community_ids)):
            comm_a = community_ids[i]
            comm_b = community_ids[j]

            set_a = set(community_hashtags[comm_a].index)
            set_b = set(community_hashtags[comm_b].index)

            overlap = set_a & set_b
            union = set_a | set_b
            jaccard = len(overlap) / len(union) if union else 0

            print(f"  Community {comm_a} vs Community {comm_b}:")
            print(f"    Shared hashtags: {len(overlap)} — {list(overlap)}")
            print(f"    Jaccard similarity: {jaccard:.3f}")
            print()

    return community_hashtags



def plot_comparison(user_results, hashtag_results, community_hashtags, user_nmi, hashtag_nmi):
    """
    Creates a comparison visualization with four panels:
    1. Community size distribution - user graph
    2. Community size distribution - hashtag graph
    3. Top hashtags per user community (heatmap)
    4. NMI comparison bar chart
    """
    print("\n[4/4] Generating comparison visualizations...")

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    colors_list = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']

    # --- Panel 1: User community size distribution ---
    ax1 = fig.add_subplot(gs[0, 0])
    user_sizes = (
        user_results['louvain_community']
        .value_counts()
        .head(10)
        .sort_values(ascending=True)
    )
    bars = ax1.barh(
        [f'Community {i}' for i in user_sizes.index],
        user_sizes.values,
        color=[colors_list[i % len(colors_list)] for i in range(len(user_sizes))],
        edgecolor='white',
        height=0.6
    )
    for bar, val in zip(bars, user_sizes.values):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=8)
    ax1.set_title('User Graph — Community Sizes (Louvain)', fontweight='bold', pad=10)
    ax1.set_xlabel('Number of users')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Panel 2: Hashtag community size distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    hashtag_sizes = (
        hashtag_results['louvain_community']
        .value_counts()
        .head(10)
        .sort_values(ascending=True)
    )
    bars2 = ax2.barh(
        [f'Community {i}' for i in hashtag_sizes.index],
        hashtag_sizes.values,
        color=[colors_list[i % len(colors_list)] for i in range(len(hashtag_sizes))],
        edgecolor='white',
        height=0.6
    )
    for bar, val in zip(bars2, hashtag_sizes.values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=8)
    ax2.set_title('Hashtag Graph — Community Sizes (Louvain)', fontweight='bold', pad=10)
    ax2.set_xlabel('Number of hashtags')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # --- Panel 3: Hashtag heatmap per user community ---
    ax3 = fig.add_subplot(gs[1, 0])

    # Build matrix: rows = communities, cols = top hashtags
    all_top_hashtags = []
    for comm, hashtags in community_hashtags.items():
        all_top_hashtags.extend(hashtags.head(5).index.tolist())
    all_top_hashtags = list(dict.fromkeys(all_top_hashtags))[:20]

    matrix_data = []
    community_labels = []
    for comm, hashtags in community_hashtags.items():
        row = [hashtags.get(ht, 0) for ht in all_top_hashtags]
        matrix_data.append(row)
        community_labels.append(f'Comm {comm}')

    matrix = np.array(matrix_data, dtype=float)
    # Normalize each row
    row_maxes = matrix.max(axis=1, keepdims=True)
    row_maxes[row_maxes == 0] = 1
    matrix_norm = matrix / row_maxes

    im = ax3.imshow(matrix_norm, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(all_top_hashtags)))
    ax3.set_xticklabels([f'#{ht}' for ht in all_top_hashtags],
                        rotation=45, ha='right', fontsize=7)
    ax3.set_yticks(range(len(community_labels)))
    ax3.set_yticklabels(community_labels, fontsize=9)
    ax3.set_title('Hashtag Usage by User Community\n(normalized per community)',
                  fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax3, shrink=0.8, label='Relative usage')

    # --- Panel 4: NMI comparison ---
    ax4 = fig.add_subplot(gs[1, 1])
    nmi_labels = ['User Graph\n(Louvain vs LabelProp)', 'Hashtag Graph\n(Louvain vs LabelProp)']
    nmi_values = [user_nmi, hashtag_nmi]
    nmi_colors = ['#457B9D', '#2A9D8F']

    bars4 = ax4.bar(
        nmi_labels, nmi_values,
        color=nmi_colors,
        edgecolor='white',
        width=0.4
    )
    for bar, val in zip(bars4, nmi_values):
        ax4.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.4f}',
                ha='center', fontsize=10, fontweight='bold')

    ax4.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.7, label='High consistency (0.7)')
    ax4.axhline(y=0.4, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Moderate consistency (0.4)')
    ax4.set_ylim(0, 1.0)
    ax4.set_ylabel('NMI Score')
    ax4.set_title('Algorithm Consistency\n(Louvain vs Label Propagation)',
                  fontweight='bold', pad=10)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    fig.suptitle(
        'Echo Chamber Detection — User Graph vs Hashtag Graph Comparison',
        fontsize=15, fontweight='bold', y=1.01
    )

    output_file = os.path.join(DATA_PATH, 'comparison_results.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved to: data/comparison_results.png")
    plt.show()


def main():
    print("=" * 50)
    print("ECHO CHAMBER DETECTION — COMPARISON ANALYSIS")
    print("=" * 50)

    # Step 1: Load all results
    user_results, hashtag_results, df = load_results()

    # Step 2: NMI comparison
    user_nmi, hashtag_nmi = compare_communities_nmi(user_results, hashtag_results)

    # Step 3: Cross analysis
    community_hashtags = cross_analysis(user_results, hashtag_results, df)

    # Step 4: Visualization
    plot_comparison(user_results, hashtag_results, community_hashtags, user_nmi, hashtag_nmi)

    print("\n" + "=" * 50)
    print("COMPARISON COMPLETE")
    print("=" * 50)


if __name__ == '__main__':
    main()