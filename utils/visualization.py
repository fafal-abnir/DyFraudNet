import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_embeddings(embeddings, labels, epoch, saving_path, mode="Train"):
    reducer_dim = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_emb = reducer_dim.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels, cmap="jet", alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f"{mode} Embeddings - Epoch {epoch}")
    plt.savefig(saving_path)  # Save figure
