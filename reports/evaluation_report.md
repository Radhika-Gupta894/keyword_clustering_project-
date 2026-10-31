# Evaluation report — FILL IN FROM YOUR OWN RUNS
## 1. Setup
Dataset (name/hash, synthetic vs real), hardware (CPU/RAM/OS), Python + package versions, model + revision, config.
## 2. Data quality (from data_quality_report.csv)
## 3. Parameter selection (dev subset only)
Paste reports/tuning_results.csv, say which settings were rejected and why, which you froze and why.
## 4. Baseline vs semantic (from evaluation_summary.csv + cluster_diagnostics.csv)
| method | n_clusters | noise_rate | singleton share | silhouette | mean coherence | % rated 4-5 | runtime s | peak MB |
## 5. Human review (reviewer role, protocol A3)
Mean coherence, % rated 4-5, >=3 good clusters, >=3 failure cases (with misplaced keywords), second-reviewer status.
## 6. Method selection (A6 rule)
Eligible? (mean >= 3.5 and >= 60% rated 4-5). Winner or "neither ready for practical use".
## 7. Metric limitations
Silhouette vs noise, cosine vs TF-IDF, duplicates, synthetic data, single reviewer.
## 8. Performance tiers (A9): 100 / 1,000 / 5,000 keywords — measured runtime & memory, limiting step.
## 9. Limitations and next steps
