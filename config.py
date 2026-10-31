"""Single place for ALL configuration (FR-08 / A8). Do not scatter tuning values elsewhere."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"
# Default input: synthetic sample dataset for testing (TODO: update to data/input/real_keywords.csv once real dataset is approved)
DEFAULT_INPUT = DATA_DIR / "synthetic" / "sample_keywords.csv"

RANDOM_STATE = 42
KEYWORD_COLUMN = "keyword"
MAX_KEYWORD_CHARS = 200          # longer strings are treated as invalid rows

# ---------------- Baseline (fixed by assignment A2) ----------------
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    # Documented deviation: default token pattern drops 1-char tokens ("vitamin c", "5k run").
    token_pattern=r"(?u)\b\w+\b",
)
BASELINE_THRESHOLD_GRID = [0.20, 0.30, 0.40, 0.50]
BASELINE_THRESHOLD = 0.40        # <-- FREEZE after running `tune` + manual inspection

# ---------------- Semantic (A2) ----------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Pin an immutable commit hash. VERIFY on https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/commits/main
EMBEDDING_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
EMBEDDING_BATCH_SIZE = 64
SEMANTIC_ALGORITHM = "hdbscan"   # "hdbscan" or "agglomerative" (fallback)
HDBSCAN_GRID = {"min_cluster_size": [5, 10, 20], "min_samples": [1, 5, 10]}
HDBSCAN_PARAMS = {"min_cluster_size": 5, "min_samples": 1}   # <-- FREEZE after tuning
SEMANTIC_AGG_GRID = [0.20, 0.30, 0.40, 0.50]
SEMANTIC_AGG_THRESHOLD = 0.40    # only used if SEMANTIC_ALGORITHM == "agglomerative"

# ---------------- Tuning / evaluation protocol (A2, A3, A6) ----------------
DEV_SUBSET_SIZE = 1000           # deterministic development subset for tuning
EVAL_SAMPLE_SIZE = 300           # common evaluation set for both methods
REVIEW_N_CLUSTERS = 20
REVIEW_MAX_MEMBERS = 10
REJECT_NOISE_RATE = 0.50         # reject settings leaving >50% unassigned
REJECT_DOMINANT_SHARE = 0.50     # reject settings with one cluster holding >50% of keywords
ELIGIBLE_MEAN_COHERENCE = 3.5
ELIGIBLE_PCT_4_OR_5 = 0.60
TIE_MARGIN = 0.25

BASELINE_METHOD = "tfidf_agglomerative"
SEMANTIC_METHOD = "minilm_hdbscan"  # becomes "minilm_agglomerative" automatically for the fallback
