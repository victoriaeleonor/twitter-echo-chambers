import pandas as pd
import numpy as np
import ast
import os
import sys
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.loader import load_tweets

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def parse_hashtags(raw):
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [h['text'].lower() for h in parsed if isinstance(h, dict) and 'text' in h]
    except:
        pass
    return []


def extract_username(raw):
    try:
        match = re.search(r"'username':\s*'([^']+)'", str(raw))
        if match:
            return match.group(1)
    except:
        return None
    return None


def build_community_hashtags(user_results, df, top_n=5):
    """
    Builds hashtag frequency per user community.
    Returns community_hashtags dict and matrix data for heatmap.
    """
    df['username'] = df['user'].apply(extract_username)
    df['hashtag_list'] = df['hashtags'].apply(parse_hashtags)

    top_communities = (
        user_results['louvain_community']
        .value_counts()
        .head(top_n)
        .index.tolist()
    )

    community_hashtags = {}
    for comm in top_communities:
        comm_users = set(
            user_results[user_results['louvain_community'] == comm]['user']
        )
        comm_tweets = df[df['username'].isin(comm_users)]
        all_hashtags = comm_tweets['hashtag_list'].explode().dropna()
        all_hashtags = all_hashtags[all_hashtags != '']
        community_hashtags[comm] = all_hashtags.value_counts().head(10)

    return community_hashtags, top_communities


def plot_interactive_comparison(user_results, hashtag_results, user_nmi, hashtag_nmi, community_hashtags, top_communities):
    """
    Creates an interactive HTML with two panels:
    1. Heatmap — hashtag usage per user community
    2. NMI bar chart — algorithm consistency comparison
    """
    print("\n[Plotly] Building interactive comparison...")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Hashtag Usage by User Community',
            'Algorithm Consistency (NMI)'
        ),
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.12
    )

    # --- Panel 1: Heatmap ---
    # Build matrix
    all_top_hashtags = []
    for comm in top_communities:
        all_top_hashtags.extend(
            community_hashtags[comm].head(5).index.tolist()
        )
    all_top_hashtags = list(dict.fromkeys(all_top_hashtags))[:20]

    matrix_data = []
    community_labels = []
    hover_text = []

    for comm in top_communities:
        hashtags = community_hashtags[comm]
        n_users = len(user_results[user_results['louvain_community'] == comm])
        row = [hashtags.get(ht, 0) for ht in all_top_hashtags]
        row_max = max(row) if max(row) > 0 else 1
        row_norm = [v / row_max for v in row]
        matrix_data.append(row_norm)
        community_labels.append(f'Comm {comm}<br>({n_users} users)')

        # Hover shows actual count
        hover_row = [
            f'#{ht}<br>Community {comm}<br>{row[i]} uses'
            for i, ht in enumerate(all_top_hashtags)
        ]
        hover_text.append(hover_row)

    fig.add_trace(
        go.Heatmap(
            z=matrix_data,
            x=[f'#{ht}' for ht in all_top_hashtags],
            y=community_labels,
            text=hover_text,
            hoverinfo='text',
            colorscale='RdYlBu_r',
            colorbar=dict(
                title='Relative<br>Usage',
                x=0.60,
                len=0.8
            ),
            zmin=0,
            zmax=1
        ),
        row=1, col=1
    )

    # --- Panel 2: NMI bar chart ---
    nmi_labels = ['User Graph<br>(Louvain vs<br>LabelProp)', 'Hashtag Graph<br>(Louvain vs<br>LabelProp)']
    nmi_values = [user_nmi, hashtag_nmi]
    nmi_colors = ['#457B9D', '#2A9D8F']
    nmi_hover = [
        f'User Graph NMI: {user_nmi:.4f}<br>Moderately consistent',
        f'Hashtag Graph NMI: {hashtag_nmi:.4f}<br>Very consistent'
    ]

    fig.add_trace(
        go.Bar(
            x=nmi_labels,
            y=nmi_values,
            marker_color=nmi_colors,
            text=[f'{v:.4f}' for v in nmi_values],
            textposition='outside',
            hovertext=nmi_hover,
            hoverinfo='text',
            width=0.35
        ),
        row=1, col=2
    )

    # Reference lines for NMI
    fig.add_shape(
        type='line', row=1, col=2,
        x0=-0.5, x1=1.5, y0=0.7, y1=0.7,
        line=dict(color='green', dash='dash', width=1.5)
    )
    fig.add_shape(
        type='line', row=1, col=2,
        x0=-0.5, x1=1.5, y0=0.4, y1=0.4,
        line=dict(color='orange', dash='dash', width=1.5)
    )

    # Annotations for reference lines
    fig.add_annotation(
        x=1.5, y=0.71, xref='x2', yref='y2',
        text='High (0.7)', showarrow=False,
        font=dict(size=9, color='green'), xanchor='right'
    )
    fig.add_annotation(
        x=1.5, y=0.41, xref='x2', yref='y2',
        text='Moderate (0.4)', showarrow=False,
        font=dict(size=9, color='orange'), xanchor='right'
    )

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text='Echo Chamber Detection — Interactive Comparison<br><sup>Hover over any element for details</sup>',
            font=dict(size=16),
            x=0.5
        ),
        height=550,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        showlegend=False,
        yaxis2=dict(range=[0, 1.0], gridcolor='#333355'),
        xaxis=dict(tickangle=-35),
        xaxis2=dict(tickfont=dict(size=9))
    )

    fig.update_xaxes(tickfont=dict(size=9), row=1, col=1)

    # Save
    output_file = os.path.join(DATA_PATH, 'comparison_interactive.html')
    fig.write_html(output_file)
    print(f"  Saved to: data/comparison_interactive.html")


def main():
    print("=" * 50)
    print("INTERACTIVE COMPARISON VISUALIZATION")
    print("=" * 50)

    # Load results
    print("\n[1/3] Loading results...")
    user_results = pd.read_csv(os.path.join(DATA_PATH, 'user_graph_results.csv'))
    hashtag_results = pd.read_csv(os.path.join(DATA_PATH, 'hashtag_graph_results.csv'))
    df = load_tweets()

    # NMI values from previous analysis
    from sklearn.metrics import normalized_mutual_info_score
    user_nmi = normalized_mutual_info_score(
        user_results['louvain_community'],
        user_results['lp_community']
    )
    hashtag_nmi = normalized_mutual_info_score(
        hashtag_results['louvain_community'],
        hashtag_results['lp_community']
    )

    # Build hashtag data
    print("\n[2/3] Building hashtag data...")
    community_hashtags, top_communities = build_community_hashtags(user_results, df)

    # Plot
    print("\n[3/3] Generating interactive visualization...")
    plot_interactive_comparison(
        user_results, hashtag_results,
        user_nmi, hashtag_nmi,
        community_hashtags, top_communities
    )

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == '__main__':
    main()