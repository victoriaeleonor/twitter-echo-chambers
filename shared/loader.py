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
    print("=== BASIC STATS ===\n")
    
    # Users
    if 'user' in df.columns:
        print(f"Unique Users:     {df['user'].nunique():,}")
    
    # Hashtags
    if 'text' in df.columns:
        hashtags = df['text'].dropna().str.findall(r'#\w+')
        total_hashtags = hashtags.explode().dropna()
        print(f"Unique Hashtags:     {total_hashtags.nunique():,}")
        print(f"Top 10 hashtags:")
        print(total_hashtags.value_counts().head(10).to_string())
        print()
    
    # Retweets
    if 'retweeted_user' in df.columns:
        rt = df['retweeted_user'].dropna()
        print(f"\nRetweets totales:    {len(rt):,}")
        print(f"Usuarios más retweteados:")
        print(rt.value_counts().head(10).to_string())
    
    # Rango de fechas
    for col in ['date', 'created_at', 'timestamp']:
        if col in df.columns:
            print(f"\nRango de fechas ({col}):")
            print(f"  Desde: {df[col].min()}")
            print(f"  Hasta: {df[col].max()}")
            break

if __name__ == '__main__':
    df = load_tweets()
    basic_stats(df)