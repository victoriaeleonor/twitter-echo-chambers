import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

# Carga el dataset desde la carpeta data/ 
def load_tweets(filename='tweets.csv'):
    filepath = os.path.join(DATA_PATH, filename)
    df = pd.read_csv(filepath, low_memory=False)
    
    print(f"Dataset cargado: {len(df):,} filas, {len(df.columns)} columnas")
    print(f"Columnas disponibles: {list(df.columns)}\n")
    
    return df   #retorna un DataFrame limpio


# Muestra estadisticas basicas del dataset
def basic_stats(df):
    """
    Displays basic statistics about the dataset.
    """
    print("\n=== BASIC STATS ===\n")

    # Unique users
    print(f"Unique users:        {df['user'].nunique():,}")

    # Hashtags (pre-extracted column)
    hashtags = df['hashtags'].dropna()
    hashtags_flat = hashtags.str.split(',').explode().str.strip().str.lower()
    hashtags_flat = hashtags_flat[hashtags_flat != '']
    print(f"Unique hashtags:     {hashtags_flat.nunique():,}")
    print(f"\nTop 10 hashtags:")
    print(hashtags_flat.value_counts().head(10).to_string())

    # Retweets
    rt = df['retweetedUserID'].dropna()
    print(f"\nTotal retweets:      {len(rt):,}")

    # Mentions
    mentions = df['mentionedUsers'].dropna()
    print(f"Tweets with mentions:{len(mentions):,}")

    # Replies
    replies = df['in_reply_to_screen_name'].dropna()
    print(f"Total replies:       {len(replies):,}")

    # Date range
    print(f"\nDate range:")
    print(f"  From: {df['date'].min()}")
    print(f"  To:   {df['date'].max()}")


if __name__ == '__main__':
    df = load_tweets()
    basic_stats(df)