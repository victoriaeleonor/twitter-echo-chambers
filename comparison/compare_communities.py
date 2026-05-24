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
    print(f"  User graph:    {len(user_results):,} users")

    # Hashtag graph results
    hashtag_results = pd.read_csv(os.path.join(DATA_PATH, 'hashtag_graph_results.csv'))
    print(f"  Hashtag graph: {len(hashtag_results):,} hashtags")

    # Original tweets for cross-analysis
    df = load_tweets()

    return user_results, hashtag_results, df


