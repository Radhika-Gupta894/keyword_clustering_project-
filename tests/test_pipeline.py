"""Automated tests (A10). A deterministic hashing 'embedder' replaces the real model so tests run offline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

import config
from run_pipeline import run, main
from src import export_results as ex
from src.data_loader import KeywordInputError, load_keywords
from src.preprocessing import normalize_keyword, prepare

_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=256, norm="l2", alternate_sign=False)


def fake_embedder(texts):
    return _HV.transform(list(texts)).toarray().astype(np.float32)


def write_csv(path, lines, header="keyword"):
    path.write_text("\n".join([header] + lines) + "\n", encoding="utf-8")
    return path


TOPIC_A = ["best running shoes for beginners", "running shoes for new runners", "trail running shoes",
           "cheap running shoes", "marathon running shoes", "running shoes review"]
TOPIC_B = ["best crm for small agency", "crm software for startups", "free crm tool",
           "sales crm software", "crm for freelancers", "cloud crm review"]


def small_csv(tmp_path):
    return write_csv(tmp_path / "k.csv", TOPIC_A + TOPIC_B + ["  Trail RUNNING   shoes ", TOPIC_A[0], "", "!!!"])


def test_valid_csv_loads(tmp_path):
    df = load_keywords(small_csv(tmp_path))
    assert list(df.columns) == ["keyword"] and len(df) >= 12


def test_missing_keyword_column_gives_clear_error(tmp_path):
    p = write_csv(tmp_path / "x.csv", ["a", "b"], header="query")
    with pytest.raises(KeywordInputError, match="Required column 'keyword'"):
        load_keywords(p)


def test_missing_file_and_empty_file_and_header_only(tmp_path):
    with pytest.raises(KeywordInputError, match="not found"):
        load_keywords(tmp_path / "nope.csv")
    empty = tmp_path / "empty.csv"; empty.write_text("")
    with pytest.raises(KeywordInputError, match="empty"):
        load_keywords(empty)
    with pytest.raises(KeywordInputError, match="no data rows"):
        load_keywords(write_csv(tmp_path / "h.csv", []))


def test_empty_and_invalid_rows_handled_safely(tmp_path):
    prep = prepare(load_keywords(small_csv(tmp_path)))
    q = dict(zip(prep.quality["metric"], prep.quality["value"]))
    assert q["blank_rows"] == 1 and q["invalid_rows"] == 1
    assert q["input_rows"] == q["blank_rows"] + q["invalid_rows"] + q["valid_rows"]
    with pytest.raises(KeywordInputError, match="No valid keywords"):
        prepare(pd.DataFrame({"keyword": ["", "   ", "!!"]}))


def test_normalization_is_consistent():
    assert normalize_keyword("  Best   RUNNING\tShoes ") == "best running shoes"
    assert normalize_keyword("Best Running Shoes") == normalize_keyword("best  running shoes")
    assert normalize_keyword("C++ Tutorial!") == "c++ tutorial!"      # punctuation preserved


def test_duplicates_kept_and_share_cluster(tmp_path):
    res = run(small_csv(tmp_path), tmp_path / "o", tmp_path / "r", embedder=fake_embedder, verbose=False)
    a = res["assignments"]
    assert len(a[a.method == config.BASELINE_METHOD]) == len(res["prep"].records)  # nothing dropped
    for m, g in a.groupby("method"):
        assert (g.groupby("keyword_normalized")["cluster_id"].nunique() == 1).all()
        assert (g.groupby("duplicate_group_id")["cluster_id"].nunique() == 1).all()


def test_required_files_and_exact_schemas(tmp_path):
    out = tmp_path / "o"
    run(small_csv(tmp_path), out, tmp_path / "r", embedder=fake_embedder, verbose=False)
    expected = {"keyword_assignments.csv": ex.ASSIGNMENT_COLUMNS, "cluster_summary.csv": ex.SUMMARY_COLUMNS,
                "data_quality_report.csv": ex.QUALITY_COLUMNS, "evaluation_summary.csv": ex.EVALUATION_COLUMNS}
    for fname, cols in expected.items():
        assert list(pd.read_csv(out / fname).columns) == cols
    assert ex.ASSIGNMENT_COLUMNS == ["keyword_original", "keyword_normalized", "duplicate_group_id", "method",
                                      "cluster_id", "is_noise", "similarity_to_representative"]
    assert (out / "run_metadata.json").exists()


def test_noise_convention():
    rec = pd.DataFrame({"keyword_original": ["a", "b", "c"], "keyword_normalized": ["a", "b", "c"],
                        "duplicate_group_id": [0, 1, 2]})
    labels = np.array([0, -1, 0])
    a = ex.build_assignments(rec, "m", labels, np.array([1.0, np.nan, 0.8]))
    assert a.loc[1, "cluster_id"] == -1 and bool(a.loc[1, "is_noise"]) is True
    assert pd.isna(a.loc[1, "similarity_to_representative"])
    assert (a.loc[[0, 2], "is_noise"] == False).all()  # noqa: E712
    unique = pd.DataFrame({"keyword_display": ["a", "b", "c"]})
    s = ex.build_summary(unique, "m", labels, {0: 0})
    assert list(s["cluster_id"]) == [0]                    # noise excluded from summary


def test_repeated_run_is_deterministic(tmp_path):
    r1 = run(small_csv(tmp_path), tmp_path / "o1", tmp_path / "r1", embedder=fake_embedder, verbose=False)
    r2 = run(small_csv(tmp_path), tmp_path / "o2", tmp_path / "r2", embedder=fake_embedder, verbose=False)
    pd.testing.assert_frame_equal(r1["assignments"], r2["assignments"])


def test_tiny_dataset(tmp_path):
    p = write_csv(tmp_path / "t.csv", ["running shoes", "crm software", "vegan recipes"])
    res = run(p, tmp_path / "o", tmp_path / "r", embedder=fake_embedder, verbose=False)
    assert len(res["assignments"]) == 6                    # 3 rows x 2 methods
    assert (res["assignments"].loc[res["assignments"].is_noise, "cluster_id"] == -1).all()


def test_single_unique_keyword(tmp_path):
    p = write_csv(tmp_path / "s.csv", ["running shoes", "Running Shoes", " running  shoes"])
    res = run(p, tmp_path / "o", tmp_path / "r", embedder=fake_embedder, verbose=False)
    a = res["assignments"]
    assert len(a) == 6 and a["cluster_id"].nunique() == 1 and not a["is_noise"].any()
    assert res["evaluation"]["silhouette_score"].eq("NA").all()   # invalid for 1 cluster -> NA


def test_cli_returns_error_code_for_bad_input(tmp_path, capsys):
    assert main(["run", "--input", str(tmp_path / "missing.csv"), "--methods", "baseline"]) == 2
    assert "INPUT ERROR" in capsys.readouterr().err
