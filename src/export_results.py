"""FR-06 / FR-07 / A4: exact-schema output files + run metadata."""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import config

ASSIGNMENT_COLUMNS = ["keyword_original", "keyword_normalized", "duplicate_group_id", "method",
                      "cluster_id", "is_noise", "similarity_to_representative"]
SUMMARY_COLUMNS = ["method", "cluster_id", "cluster_size_unique", "representative_keyword", "member_keywords"]
QUALITY_COLUMNS = ["metric", "value", "notes"]
EVALUATION_COLUMNS = ["method", "parameter_config", "n_clusters", "noise_rate", "silhouette_score",
                      "mean_human_coherence", "pct_clusters_rating_4_or_5", "runtime_seconds",
                      "peak_memory_mb"]


def representatives_and_similarity(X, labels):
    """Representative = member closest (cosine) to the cluster centroid.

    Returns (reps {label: unique_index}, sims array; NaN for noise). The similarity is a cosine
    similarity in [0, 1] (negatives clipped) - NOT a probability or calibrated confidence.
    """
    labels = np.asarray(labels)
    sims = np.full(len(labels), np.nan)
    reps = {}
    for lab in sorted(set(labels[labels >= 0].tolist())):
        idx = np.where(labels == lab)[0]
        Xi = X[idx]
        centroid = np.asarray(Xi.mean(axis=0)).reshape(1, -1)
        to_centroid = cosine_similarity(Xi, centroid).ravel()
        rep = int(idx[int(np.argmax(to_centroid))])   # ties -> lowest index (deterministic)
        reps[int(lab)] = rep
        sims[idx] = np.clip(cosine_similarity(Xi, X[[rep]]).ravel(), 0.0, 1.0)
    return reps, sims


def build_assignments(records: pd.DataFrame, method: str, labels, sims) -> pd.DataFrame:
    """One row per original (valid) input record; duplicates inherit their unique keyword's cluster."""
    labels = np.asarray(labels)
    g = records["duplicate_group_id"].to_numpy()
    lab = labels[g]
    sim = np.asarray(sims, dtype=float)[g]
    df = pd.DataFrame({
        "keyword_original": records["keyword_original"].to_numpy(),
        "keyword_normalized": records["keyword_normalized"].to_numpy(),
        "duplicate_group_id": g,
        "method": method,
        "cluster_id": lab.astype(int),
        "is_noise": lab < 0,
        "similarity_to_representative": np.where(lab < 0, np.nan, np.round(sim, 4)),
    })
    return df[ASSIGNMENT_COLUMNS]


def build_summary(unique: pd.DataFrame, method: str, labels, reps) -> pd.DataFrame:
    labels = np.asarray(labels)
    rows = []
    for lab in sorted(reps):
        idx = np.where(labels == lab)[0]
        rows.append({"method": method, "cluster_id": int(lab), "cluster_size_unique": int(len(idx)),
                     "representative_keyword": unique["keyword_display"].iloc[reps[lab]],
                     "member_keywords": ";".join(unique["keyword_display"].iloc[idx])})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_metadata(out_dir: Path, input_path, methods, files):
    pkgs = {}
    for name in ["pandas", "numpy", "scikit-learn", "sentence-transformers", "torch", "transformers"]:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = "not installed"
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(config).items()
           if k.isupper()}
    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_filename": Path(input_path).name, "dataset_sha256": sha256_of(input_path),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": pkgs, "methods_run": methods,
            "embedding_model": config.EMBEDDING_MODEL, "embedding_revision": config.EMBEDDING_REVISION,
            "inference": "local, CPU (sentence-transformers); no external API",
            "config": cfg, "output_files": files}
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
