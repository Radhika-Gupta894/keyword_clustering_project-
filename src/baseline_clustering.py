"""FR-03: fixed lexical baseline = word TF-IDF (1,2)-grams + average-linkage agglomerative on cosine distance."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import config


def cluster_baseline(texts, threshold: float, tfidf_params=None):
    """Return (labels, X). X is the sparse TF-IDF matrix (rows = unique normalized keywords).

    API used: AgglomerativeClustering(n_clusters=None, distance_threshold=t, metric="precomputed",
    linkage="average") on a cosine-distance matrix (needs scikit-learn >= 1.2). A dense n x n float64
    matrix is built (about 200 MB at n = 5,000), which is within the assignment's 5,000-keyword limit.
    """
    params = dict(config.TFIDF_PARAMS if tfidf_params is None else tfidf_params)
    X = TfidfVectorizer(**params).fit_transform(texts)
    n = len(texts)
    if n == 1:
        return np.zeros(1, dtype=int), X
    D = cosine_distances(X)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                    metric="precomputed", linkage="average")
    return model.fit_predict(D).astype(int), X
