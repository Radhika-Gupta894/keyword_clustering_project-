"""FR-02 + A5 duplicate policy + data-quality report."""
import re
from dataclasses import dataclass
import pandas as pd
import config
from src.data_loader import KeywordInputError

_WS = re.compile(r"\s+")


def normalize_keyword(text) -> str:
    """Trim, collapse internal whitespace, case-fold. Punctuation/tokens are NOT removed."""
    if text is None:
        return ""
    return _WS.sub(" ", str(text).strip()).casefold()


def is_valid_keyword(norm: str, max_chars: int = config.MAX_KEYWORD_CHARS) -> bool:
    return bool(norm) and len(norm) <= max_chars and any(c.isalnum() for c in norm)


@dataclass
class PreparedData:
    records: pd.DataFrame   # one row per VALID original row
    unique: pd.DataFrame    # one row per unique normalized keyword (index == duplicate_group_id)
    quality: pd.DataFrame   # data_quality_report.csv


def prepare(raw: pd.DataFrame, max_chars: int = config.MAX_KEYWORD_CHARS) -> PreparedData:
    orig = raw["keyword"].astype(str)
    norm = orig.map(normalize_keyword)
    blank = norm == ""
    valid = norm.map(lambda s: is_valid_keyword(s, max_chars))
    invalid = ~blank & ~valid

    rec = pd.DataFrame({"keyword_original": orig[valid], "keyword_normalized": norm[valid]})
    if rec.empty:
        raise KeywordInputError("No valid keywords found (all rows blank or invalid).")
    rec = rec.reset_index(drop=True)
    # 0-based id in order of first appearance -> also the index into the unique-keyword arrays
    codes, uniques = pd.factorize(rec["keyword_normalized"])
    rec["duplicate_group_id"] = codes
    first_orig = rec.groupby("duplicate_group_id")["keyword_original"].first()
    counts = rec.groupby("duplicate_group_id").size()
    unique = pd.DataFrame({"keyword_normalized": list(uniques),
                           "keyword_display": first_orig.values,
                           "n_rows": counts.values})

    n_words = unique["keyword_normalized"].str.split().str.len()
    changed = int((orig[valid].values != rec["keyword_normalized"].values).sum())
    n_valid, n_unique = len(rec), len(unique)
    q = [
        ("input_rows", len(raw), "Data rows in the CSV (header excluded)"),
        ("blank_rows", int(blank.sum()), "Empty / whitespace-only; excluded from clustering and assignments"),
        ("invalid_rows", int(invalid.sum()), f"No letters/digits or > {max_chars} chars; excluded"),
        ("valid_rows", n_valid, "Rows carried through the pipeline"),
        ("exact_duplicate_rows", n_valid - n_unique, "Valid rows that repeat an earlier normalized keyword"),
        ("unique_normalized_keywords", n_unique, "Clustering is fitted on these only (A5)"),
        ("rows_changed_by_normalization", changed, "Case/whitespace differs from the original"),
        ("words_per_keyword_mean", round(float(n_words.mean()), 2), "Over unique keywords"),
        ("words_per_keyword_median", float(n_words.median()), "Over unique keywords"),
        ("words_per_keyword_max", int(n_words.max()), "Over unique keywords"),
    ]
    quality = pd.DataFrame(q, columns=["metric", "value", "notes"])
    return PreparedData(records=rec, unique=unique, quality=quality)
