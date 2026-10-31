"""FR-09: diagnostics, silhouette, tuning grid, human-review sheet, method selection."""
import itertools
import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import config
from src.baseline_clustering import cluster_baseline
from src.semantic_clustering import cluster_semantic


def peak_memory_mb():
    """Process-wide PEAK memory so far (cumulative, not per method). None if unavailable."""
    try:
        import resource
        import sys
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(v / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except ImportError:
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 2**20, 1)
        except Exception:
            return None


def measure(fn):
    t = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t, 3), peak_memory_mb()


def silhouette_or_none(X, labels):
    """Cosine silhouette over NON-NOISE points. None when mathematically invalid (<2 clusters etc.)."""
    labels = np.asarray(labels)
    mask = labels >= 0
    k = len(set(labels[mask].tolist()))
    if k < 2 or mask.sum() <= k:
        return None
    try:
        return float(silhouette_score(X[mask], labels[mask], metric="cosine"))
    except ValueError:
        return None


def diagnostics(labels) -> dict:
    labels = np.asarray(labels)
    n = len(labels)
    clustered = labels[labels >= 0]
    sizes = pd.Series(clustered).value_counts() if len(clustered) else pd.Series(dtype=int)
    return {
        "n_unique_keywords": n,
        "n_clusters": int(len(sizes)),
        "noise_rate": float((labels < 0).sum() / n),
        "size_min": int(sizes.min()) if len(sizes) else 0,
        "size_median": float(sizes.median()) if len(sizes) else 0.0,
        "size_max": int(sizes.max()) if len(sizes) else 0,
        "largest_cluster_share": float(sizes.max() / n) if len(sizes) else 0.0,
        "singleton_clusters": int((sizes == 1).sum()),
        "pct_keywords_in_singletons": float((sizes == 1).sum() / n),
    }


def deterministic_subset(n, size, seed=config.RANDOM_STATE):
    """Same keywords every run (sorted indices); all of them if n <= size."""
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.RandomState(seed).choice(n, size=size, replace=False))


# ---------------------------------------------------------------- tuning (A2)
def _row(method, cfg, labels, X, secs):
    d = diagnostics(labels)
    reasons = []
    if d["noise_rate"] > config.REJECT_NOISE_RATE:
        reasons.append("noise>50%")
    if d["n_unique_keywords"] >= 20 and d["largest_cluster_share"] > config.REJECT_DOMINANT_SHARE:
        reasons.append("dominant cluster>50%")
    sil = silhouette_or_none(X, labels)
    return {"method": method, "parameter_config": cfg, "n_clusters": d["n_clusters"],
            "noise_rate": round(d["noise_rate"], 4),
            "largest_cluster_share": round(d["largest_cluster_share"], 4),
            "singleton_clusters": d["singleton_clusters"],
            "silhouette_score": None if sil is None else round(sil, 4),
            "runtime_seconds": secs, "rejected": bool(reasons), "reject_reason": ",".join(reasons)}


def tune_baseline(texts):
    rows, labelsets = [], {}
    for thr in config.BASELINE_THRESHOLD_GRID:
        (labels, X), secs, _ = measure(lambda: cluster_baseline(texts, thr))
        cfg = f"distance_threshold={thr}"
        rows.append(_row(config.BASELINE_METHOD, cfg, labels, X, secs))
        labelsets[cfg] = labels
    return pd.DataFrame(rows), labelsets


def tune_semantic(emb):
    rows, labelsets, n = [], {}, len(emb)
    if config.SEMANTIC_ALGORITHM == "hdbscan":
        for mcs, ms in itertools.product(config.HDBSCAN_GRID["min_cluster_size"],
                                         config.HDBSCAN_GRID["min_samples"]):
            if mcs > n // 2 or ms > mcs:   # constrain by dataset size / sensible combos
                continue
            p = {"min_cluster_size": mcs, "min_samples": ms}
            labels, secs, _ = measure(lambda: cluster_semantic(emb, "hdbscan", p))
            cfg = f"min_cluster_size={mcs};min_samples={ms}"
            rows.append(_row(config.SEMANTIC_METHOD, cfg, labels, emb, secs))
            labelsets[cfg] = labels
    else:
        for thr in config.SEMANTIC_AGG_GRID:
            labels, secs, _ = measure(lambda: cluster_semantic(emb, "agglomerative", None, thr))
            cfg = f"distance_threshold={thr}"
            rows.append(_row("minilm_agglomerative", cfg, labels, emb, secs))
            labelsets[cfg] = labels
    return pd.DataFrame(rows), labelsets


def preview_clusters(texts, labels, n_clusters=6, n_members=8):
    """Plain-text preview of the largest clusters, for eyeballing during tuning."""
    labels = np.asarray(labels)
    s = pd.Series(labels[labels >= 0]).value_counts().head(n_clusters)
    out = []
    for lab, size in s.items():
        members = [texts[i] for i in np.where(labels == lab)[0][:n_members]]
        out.append(f"  cluster {lab} (size {size}): " + " | ".join(members))
    return "\n".join(out) if out else "  (no clusters)"


# ---------------------------------------------------------------- human review (A3)
REVIEW_COLUMNS = ["method", "cluster_id", "cluster_size_unique", "representative_keyword",
                  "members_shown", "rating_1_to_5", "reviewer_initials_or_role", "reason",
                  "misplaced_keywords"]


def build_review_sheet(results: dict, uniq_texts, eval_idx) -> pd.DataFrame:
    """results: {method: dict(labels=..., reps={label: idx})}. Same eval keywords for every method.

    Candidate clusters = clusters containing >=1 eval keyword. Sorted by cluster id, then (stable) by
    size; REVIEW_N_CLUSTERS picked at evenly spaced positions so small/medium/large clusters are covered.
    """
    eval_set = set(int(i) for i in eval_idx)
    rows = []
    for method, res in results.items():
        labels = np.asarray(res["labels"])
        cand = sorted({int(labels[i]) for i in eval_set if labels[i] >= 0})
        if not cand:
            continue
        sizes = {c: int((labels == c).sum()) for c in cand}
        ordered = sorted(cand, key=lambda c: (sizes[c], c))
        k = min(config.REVIEW_N_CLUSTERS, len(ordered))
        pos = sorted(set(np.linspace(0, len(ordered) - 1, k).round().astype(int).tolist()))
        for c in sorted(ordered[p] for p in pos):
            members = np.where(labels == c)[0]
            shown = [uniq_texts[i] for i in members[: config.REVIEW_MAX_MEMBERS]]
            rows.append({"method": method, "cluster_id": c, "cluster_size_unique": sizes[c],
                         "representative_keyword": uniq_texts[res["reps"][c]],
                         "members_shown": "; ".join(shown), "rating_1_to_5": "",
                         "reviewer_initials_or_role": "", "reason": "", "misplaced_keywords": ""})
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def summarize_ratings(sheet) -> dict:
    """{method: (mean_rating, pct_4_or_5, n_rated)} using only valid 1-5 ratings."""
    out = {}
    if sheet is None or len(sheet) == 0:
        return out
    r = pd.to_numeric(sheet["rating_1_to_5"], errors="coerce")
    ok = sheet.assign(_r=r)[r.between(1, 5)]
    for method, g in ok.groupby("method"):
        out[method] = (float(g["_r"].mean()), float((g["_r"] >= 4).mean()), int(len(g)))
    return out


def select_method(eval_df: pd.DataFrame):
    """A6 selection rule. Returns (method_or_None, explanation)."""
    d = eval_df.copy()
    d["mean_human_coherence"] = pd.to_numeric(d["mean_human_coherence"], errors="coerce")
    d["pct_clusters_rating_4_or_5"] = pd.to_numeric(d["pct_clusters_rating_4_or_5"], errors="coerce")
    if d["mean_human_coherence"].isna().all():
        return None, "No final selection yet - fill reports/human_review_sheet.csv and re-run."
    if d["mean_human_coherence"].isna().any():
        return None, "No final selection yet - human ratings are missing for one or more evaluated methods."
    elig = d[(d["mean_human_coherence"] >= config.ELIGIBLE_MEAN_COHERENCE) &
             (d["pct_clusters_rating_4_or_5"] >= config.ELIGIBLE_PCT_4_OR_5)]
    if elig.empty:
        return None, "Neither method meets the thresholds: report 'not ready for practical use' + recommendations."
    elig = elig.sort_values("mean_human_coherence", ascending=False)
    if len(elig) > 1 and (elig.iloc[0]["mean_human_coherence"] -
                          elig.iloc[1]["mean_human_coherence"]) < config.TIE_MARGIN:
        elig = elig.assign(_cov=1 - pd.to_numeric(elig["noise_rate"])).sort_values(
            ["_cov", "runtime_seconds"], ascending=[False, True])
        return elig.iloc[0]["method"], "Scores within 0.25: chose better coverage / lower runtime."
    return elig.iloc[0]["method"], "Highest mean human coherence among eligible methods."
