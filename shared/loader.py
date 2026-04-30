import pandas as pd
import os
import ast

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

# Carga el dataset desde la carpeta data/ 
def load_tweets(filename='tweets.csv'):
    filepath = os.path.join(DATA_PATH, filename)
    df = pd.read_csv(filepath, low_memory=False)
    
    print(f"Dataset loaded: {len(df):,} filas, {len(df.columns)} columnas")
    print(f"Available columns: {list(df.columns)}\n")
    
    return df   #retorna un DataFrame limpio

def parse_hashtags(raw):
    """
    Parses the hashtags column from JSON-like strings to a flat list of hashtag texts.
    """
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [h['text'].lower() for h in parsed if isinstance(h, dict) and 'text' in h]
    except:
        pass
    return []


def parse_mentioned_users(raw):
    """
    Parses the mentionedUsers column to extract screen names.
    """
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [u['username'] for u in parsed if isinstance(u, dict) and 'username' in u]
    except:
        pass

# Muestra estadisticas basicas del dataset
def basic_stats(df):
    print("\n=== BASIC STATS ===\n")

    # Unique users
    print(f"Unique users:         {df['user'].nunique():,}")

    # Hashtags - parsed correctly
    hashtags_flat = df['hashtags'].dropna().apply(parse_hashtags).explode()
    hashtags_flat = hashtags_flat[hashtags_flat != '']
    print(f"Unique hashtags:      {hashtags_flat.nunique():,}")
    print(f"\nTop 10 hashtags:")
    print(hashtags_flat.value_counts().head(10).to_string())

    # Replies
    replies = df['in_reply_to_screen_name'].dropna()
    print(f"\nTotal replies:        {len(replies):,}")

    # Mentions - parsed correctly
    mentions_flat = df['mentionedUsers'].dropna().apply(parse_mentioned_users).explode()
    mentions_flat = mentions_flat[mentions_flat != '']
    print(f"Unique mentioned users:{mentions_flat.nunique():,}")

    # Date range
    print(f"\nDate range:")
    print(f"  From: {df['date'].min()}")
    print(f"  To:   {df['date'].max()}")


if __name__ == '__main__':
    df = load_tweets()
    basic_stats(df)