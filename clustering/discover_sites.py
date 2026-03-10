import pandas as pd
import numpy as np
import yaml
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

def discover_sites():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    embed_path = "data/processed/embeddings.csv"
    if not os.path.exists(embed_path):
        print("Error: Embeddings file not found. Run extract_embeddings.py first.")
        return

    df = pd.read_csv(embed_path)
    # Extract just the numeric columns (features)
    features = df.drop('filename', axis=1).values
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print("Running KMeans Clustering...")
    kmeans = KMeans(n_clusters=config['clustering']['n_clusters'], random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(features_scaled)

    print("Running DBSCAN Clustering (Anomalous site discovery)...")
    dbscan = DBSCAN(eps=config['clustering']['dbscan_eps'], min_samples=config['clustering']['dbscan_min_samples'])
    df['dbscan_cluster'] = dbscan.fit_predict(features_scaled)

    # Save results
    df.to_csv("data/processed/clusters.csv", index=False)
    print(f"Clustering complete. Results saved to data/processed/clusters.csv")
    
    # Summary
    print("\nCluster Distribution (KMeans):")
    print(df['kmeans_cluster'].value_counts())
    print("\nCluster Distribution (DBSCAN -1 are noise/outliers):")
    print(df['dbscan_cluster'].value_counts())

if __name__ == "__main__":
    discover_sites()
