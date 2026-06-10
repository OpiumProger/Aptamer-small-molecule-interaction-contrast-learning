"""
Кластеризация 768d aptamer embeddings → negative latent seeds для GRU.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def collect_embeddings_from_model(model, test_loader, device):
    """Собирает эмбеддинги аптамеров (768d) из contrastive-модели."""
    model.eval()

    embeddings = []
    types = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting embeddings"):
            positive_apts = batch['positive_apts'].to(device)
            negative_apts = batch['negative_apts'].to(device)

            z_pos = model.encode_aptamer(positive_apts)
            for i in range(len(z_pos)):
                embeddings.append(z_pos[i].cpu().numpy())
                types.append('positive')

            batch_size, k_neg, _ = negative_apts.shape
            for i in range(batch_size):
                for j in range(k_neg):
                    z_neg = model.encode_aptamer(negative_apts[i, j].unsqueeze(0))
                    embeddings.append(z_neg[0].cpu().numpy())
                    types.append('negative')

    return np.array(embeddings), np.array(types)


def cluster_embeddings(embeddings, types, n_clusters=5):
    """KMeans-кластеризация 768d эмбеддингов."""
    print("\n" + "=" * 60)
    print(f"CLUSTERING EMBEDDINGS (n_clusters={n_clusters})")
    print("=" * 60)

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)

    for i in range(n_clusters):
        mask = cluster_labels == i
        n_pos = (types[mask] == 'positive').sum()
        n_neg = (types[mask] == 'negative').sum()
        print(f"  Cluster {i}: {mask.sum()} points (pos:{n_pos}, neg:{n_neg})")

    return cluster_labels, kmeans, scaler


def full_pipeline(model, test_loader, device, n_clusters=5):
    """
    Сбор 768d эмбеддингов → кластеризация → сохранение cluster_*.npy.
    """
    print("=" * 70)
    print("CLUSTERING PIPELINE: 768d Embeddings → KMeans")
    print("=" * 70)

    print("\nStep 1: Collecting 768d embeddings from model...")
    embeddings_768d, types = collect_embeddings_from_model(model, test_loader, device)
    print(f"  Collected {len(embeddings_768d)} embeddings (768d)")

    print("\nStep 2: Clustering embeddings...")
    cluster_labels, _, _ = cluster_embeddings(embeddings_768d, types, n_clusters)

    np.save('cluster_embeddings_768d.npy', embeddings_768d)
    np.save('cluster_labels_768d.npy', cluster_labels)
    np.save('cluster_types.npy', types)
    print("\nCluster info saved:")
    print("   - cluster_embeddings_768d.npy")
    print("   - cluster_labels_768d.npy")
    print("   - cluster_types.npy")

    return cluster_labels, embeddings_768d, types


def get_negative_cluster_embeddings(embeddings, cluster_labels, types):
    """Возвращает эмбеддинги из кластера с максимальным числом negative точек."""
    n_clusters = len(np.unique(cluster_labels))
    best_cluster = None
    best_neg_count = 0

    for i in range(n_clusters):
        mask = cluster_labels == i
        n_neg = (types[mask] == 'negative').sum()
        if n_neg > best_neg_count:
            best_neg_count = n_neg
            best_cluster = i

    mask = cluster_labels == best_cluster
    return embeddings[mask], best_cluster, best_neg_count
