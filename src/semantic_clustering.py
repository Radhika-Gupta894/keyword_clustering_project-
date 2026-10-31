"""FR-04 / FR-05: sentence embeddings + HDBSCAN (or agglomerative fallback)."""
import warnings

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import config


def load_embedder(model_name=config.EMBEDDING_MODEL, revision=config.EMBEDDING_REVISION,
                  batch_size=config.EMBEDDING_BATCH_SIZE):
    """Return embed(texts) -> L2-normalised float32 array. Runs LOCALLY on CPU (no external API)."""
    from sentence_transformers import SentenceTransformer  # lazy: heavy import
    model = SentenceTransformer(model_name, revision=revision, device="cpu")

    def embed(texts):
        return np.asarray(model.encode(list(texts), batch_size=batch_size,
                                       normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)
    return embed


def cluster_semantic(emb, algorithm=config.SEMANTIC_ALGORITHM,
                     hdbscan_params=None, agg_threshold=config.SEMANTIC_AGG_THRESHOLD):
    """Cluster normalised embeddings. Noise label is -1 (HDBSCAN only)."""
    n = len(emb)
    if n == 1:
        return np.zeros(1, dtype=int)
    if algorithm == "hdbscan":
        from sklearn.cluster import HDBSCAN  # scikit-learn >= 1.3, no extra dependency
        p = dict(config.HDBSCAN_PARAMS if hdbscan_params is None else hdbscan_params)
        mcs = max(2, p["min_cluster_size"])
        if n < mcs:                      # cannot form any cluster -> everything is noise
            return np.full(n, -1, dtype=int)
        ms = max(1, min(p["min_samples"], n - 1))
        # On L2-normalised vectors euclidean distance is monotonic in cosine distance.
        with warnings.catch_warnings():   # sklearn announces a default change of `copy` in 1.10
            warnings.simplefilter("ignore", FutureWarning)
            return HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean").fit_predict(emb).astype(int)
    if algorithm == "agglomerative":
        return AgglomerativeClustering(n_clusters=None, distance_threshold=agg_threshold,
                                       metric="cosine", linkage="average").fit_predict(emb).astype(int)
    raise ValueError(f"Unknown algorithm: {algorithm}")
