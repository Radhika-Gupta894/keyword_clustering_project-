"""Command-line entry point.

    python run_pipeline.py tune --input data/sample_keywords.csv      # parameter search on dev subset
    python run_pipeline.py run  --input data/sample_keywords.csv      # frozen config -> all outputs
    python run_pipeline.py run  --input data/smoke_keywords.csv --methods baseline
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import evaluation as ev
from src import export_results as ex
from src.baseline_clustering import cluster_baseline
from src.data_loader import KeywordInputError, load_keywords
from src.preprocessing import prepare
from src.semantic_clustering import cluster_semantic, load_embedder


def _semantic_name():
    return config.SEMANTIC_METHOD if config.SEMANTIC_ALGORITHM == "hdbscan" else "minilm_agglomerative"


def _semantic_cfg():
    if config.SEMANTIC_ALGORITHM == "hdbscan":
        p = config.HDBSCAN_PARAMS
        return f"min_cluster_size={p['min_cluster_size']};min_samples={p['min_samples']}"
    return f"distance_threshold={config.SEMANTIC_AGG_THRESHOLD}"


def run(input_path, output_dir=config.OUTPUT_DIR, reports_dir=config.REPORTS_DIR,
        methods="both", embedder=None, regenerate_review=False, verbose=True):
    """Full pipeline. `embedder` (texts -> normalised array) can be injected for tests."""
    log = print if verbose else (lambda *a, **k: None)
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw = load_keywords(input_path, config.KEYWORD_COLUMN)
    prep = prepare(raw)
    uniq = prep.unique["keyword_normalized"].tolist()
    n = len(uniq)
    log(f"Loaded {len(raw)} rows -> {len(prep.records)} valid -> {n} unique normalized keywords")
    eval_idx = ev.deterministic_subset(n, config.EVAL_SAMPLE_SIZE)

    results, eval_rows, assign_parts, summary_parts, diag_rows = {}, [], [], [], []

    def finish(name, cfg, labels, X, secs, mem):
        reps, sims = ex.representatives_and_similarity(X, labels)
        results[name] = {"labels": labels, "reps": reps}
        assign_parts.append(ex.build_assignments(prep.records, name, labels, sims))
        summary_parts.append(ex.build_summary(prep.unique, name, labels, reps))
        d = ev.diagnostics(labels)
        diag_rows.append({"method": name, **d})
        sil = ev.silhouette_or_none(X, labels)
        eval_rows.append({"method": name, "parameter_config": cfg, "n_clusters": d["n_clusters"],
                          "noise_rate": round(d["noise_rate"], 4),
                          "silhouette_score": "NA" if sil is None else round(sil, 4),
                          "mean_human_coherence": "NA", "pct_clusters_rating_4_or_5": "NA",
                          "runtime_seconds": secs, "peak_memory_mb": "NA" if mem is None else mem})
        log(f"[{name}] {cfg}: {d['n_clusters']} clusters, noise {d['noise_rate']:.1%}, "
            f"silhouette {sil if sil is None else round(sil, 3)}, {secs}s")

    if methods in ("both", "baseline"):
        (labels, X), secs, mem = ev.measure(lambda: cluster_baseline(uniq, config.BASELINE_THRESHOLD))
        finish(config.BASELINE_METHOD, f"distance_threshold={config.BASELINE_THRESHOLD}", labels, X, secs, mem)

    if methods in ("both", "semantic"):
        def sem():
            emb_fn = embedder or load_embedder()
            emb = emb_fn(uniq)
            return cluster_semantic(emb, config.SEMANTIC_ALGORITHM, config.HDBSCAN_PARAMS,
                                    config.SEMANTIC_AGG_THRESHOLD), emb
        (labels, emb), secs, mem = ev.measure(sem)   # runtime includes embedding time
        finish(_semantic_name(), _semantic_cfg(), labels, emb, secs, mem)

    # ---- human review sheet (never overwrite a sheet that already has ratings)
    sheet_path = reports_dir / "human_review_sheet.csv"
    if regenerate_review or not sheet_path.exists():
        ev.build_review_sheet(results, uniq, eval_idx).to_csv(sheet_path, index=False)
        log(f"Wrote review sheet: {sheet_path}  (fill rating_1_to_5, then re-run)")
    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    ratings = ev.summarize_ratings(sheet)
    for row in eval_rows:
        if row["method"] in ratings:
            mean, pct, n_rated = ratings[row["method"]]
            row["mean_human_coherence"], row["pct_clusters_rating_4_or_5"] = round(mean, 2), round(pct, 3)

    evaluation = pd.DataFrame(eval_rows)[ex.EVALUATION_COLUMNS]
    assignments = pd.concat(assign_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    files = {"keyword_assignments.csv": assignments, "cluster_summary.csv": summary,
             "data_quality_report.csv": prep.quality[ex.QUALITY_COLUMNS],
             "evaluation_summary.csv": evaluation}
    for fname, df in files.items():
        df.to_csv(output_dir / fname, index=False)
    pd.DataFrame(diag_rows).to_csv(reports_dir / "cluster_diagnostics.csv", index=False)
    ex.write_run_metadata(output_dir, input_path, methods, list(files) + ["run_metadata.json"])

    choice, why = ev.select_method(evaluation)
    log(f"Method selection: {choice or 'none yet'} - {why}")
    log(f"Outputs written to {output_dir}")
    return {"prep": prep, "results": results, "evaluation": evaluation,
            "assignments": assignments, "summary": summary}


def tune(input_path, reports_dir=config.REPORTS_DIR, methods="both", embedder=None):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prep = prepare(load_keywords(input_path, config.KEYWORD_COLUMN))
    uniq = prep.unique["keyword_normalized"].tolist()
    dev_idx = ev.deterministic_subset(len(uniq), config.DEV_SUBSET_SIZE)
    dev = [uniq[i] for i in dev_idx]
    print(f"Tuning on deterministic development subset: {len(dev)} unique keywords")
    frames, preview = [], []
    if methods in ("both", "baseline"):
        df, sets = ev.tune_baseline(dev)
        frames.append(df)
        for cfg, lab in sets.items():
            preview.append(f"== {config.BASELINE_METHOD} {cfg}\n{ev.preview_clusters(dev, lab)}")
    if methods in ("both", "semantic"):
        emb = (embedder or load_embedder())(dev)
        df, sets = ev.tune_semantic(emb)
        frames.append(df)
        for cfg, lab in sets.items():
            preview.append(f"== semantic {cfg}\n{ev.preview_clusters(dev, lab)}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(reports_dir / "tuning_results.csv", index=False)
    (reports_dir / "tuning_preview.txt").write_text("\n\n".join(preview), encoding="utf-8")
    print(out.to_string(index=False))
    print("\nNext: read reports/tuning_preview.txt, rate coherence of the non-rejected settings, then FREEZE "
          "your choice in config.py (BASELINE_THRESHOLD, HDBSCAN_PARAMS). Do not re-tune after the final review.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="VyorAI keyword clustering engine")
    ap.add_argument("command", choices=["run", "tune"])
    ap.add_argument("--input", default=str(config.DEFAULT_INPUT))
    ap.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    ap.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    ap.add_argument("--methods", choices=["both", "baseline", "semantic"], default="both")
    ap.add_argument("--regenerate-review", action="store_true",
                    help="overwrite reports/human_review_sheet.csv (destroys ratings!)")
    a = ap.parse_args(argv)
    try:
        if a.command == "run":
            run(a.input, a.output_dir, a.reports_dir, a.methods, regenerate_review=a.regenerate_review)
        else:
            tune(a.input, a.reports_dir, a.methods)
    except KeywordInputError as e:
        print(f"INPUT ERROR: {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"MISSING DEPENDENCY: {e}\nInstall requirements.txt or use --methods baseline.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
