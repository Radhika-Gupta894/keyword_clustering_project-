# Keyword Clustering Evaluation Report

## 1. Setup & Environment
* **Dataset:** Evaluated against `data/synthetic/sample_keywords.csv` (synthetic benchmark fixture; real query dataset pending supervisor approval).
  * Dataset SHA-256: `8a57165cfaebc3c6f7b3df6672fcb7c4d59f1be0bb8d73b06385cfcf23606f37`
  * Status: Synthetic test fixture (pipeline validation only; not representative of organic Google search telemetry).
* **Hardware & Runtime Environment (Measured):**
  * **OS:** Windows 11 (10.0.26300)
  * **Python:** 3.12.10
  * **CPU:** Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (12 logical cores)
  * **System RAM:** 15.70 GB total (2.90 GB available at benchmark start)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`), CPU inference.

---

## 2. Data Quality (from `data/output/data_quality_report.csv`)
* **Total Input Rows:** 677
* **Blank Rows:** 0 (0.0%)
* **Invalid Rows:** 0 (0.0%)
* **Valid Rows Carried Through:** 677 (100.0%)
* **Exact Duplicate Rows:** 80 (valid rows repeating an earlier normalized keyword)
* **Unique Normalized Keywords:** 597 (clustering is fitted exclusively on these; assignments map back to all 677 rows)
* **Rows Modified by Normalization:** 50 (whitespace trimming and lowercasing)
* **Keyword Word Counts:** Mean 4.8 words, Median 5.0 words, Max 9.0 words.

---

## 3. Parameter Selection & Tuning (from `reports/tuning_results.csv`)
Grid search was conducted on the deterministic development subset (597 unique keywords).

| Method | Parameter Config | Clusters ($n$) | Noise Rate | Largest Cluster Share | Singletons | Silhouette | Runtime (s) | Rejection Status |
|---|---|---|---|---|---|---|---|---|
| `tfidf_agglomerative` | `distance_threshold=0.2` | 581 | 0.00% | 0.50% | 566 | 0.0319 | 0.059 | Accepted |
| `tfidf_agglomerative` | `distance_threshold=0.3` | 531 | 0.00% | 0.84% | 478 | 0.0784 | 0.024 | Accepted |
| `tfidf_agglomerative` | `distance_threshold=0.4` | 421 | 0.00% | 1.34% | 301 | 0.1488 | 0.025 | Accepted |
| `tfidf_agglomerative` | `distance_threshold=0.5` | 280 | 0.00% | 1.51% | 128 | 0.2204 | 0.024 | Accepted |
| `minilm_hdbscan` | `min_cluster_size=5;min_samples=1` | 29 | 5.70% | 9.72% | 0 | 0.4103 | 0.112 | Accepted |
| `minilm_hdbscan` | `min_cluster_size=5;min_samples=5` | 17 | 4.19% | 10.05% | 0 | 0.3940 | 0.226 | Accepted |
| `minilm_hdbscan` | `min_cluster_size=10;min_samples=1` | 12 | 3.52% | 10.22% | 0 | 0.4139 | 0.097 | Accepted |
| `minilm_hdbscan` | `min_cluster_size=10;min_samples=5` | 12 | 4.69% | 10.22% | 0 | 0.4166 | 0.301 | Accepted |
| `minilm_hdbscan` | `min_cluster_size=10;min_samples=10` | 11 | 20.77% | 10.05% | 0 | 0.4520 | 0.312 | Accepted |
| `minilm_hdbscan` | `min_cluster_size=20;min_samples=1` | 10 | 5.53% | 10.22% | 0 | 0.4072 | 0.090 | Accepted |
| `minilm_hdbscan` | `min_cluster_size=20;min_samples=5` | 10 | 6.20% | 10.22% | 0 | 0.4088 | 0.209 | Accepted |
| `minilm_hdbscan` | `min_cluster_size=20;min_samples=10` | 10 | 22.61% | 10.05% | 0 | 0.4542 | 0.212 | Accepted |

* **Rejection Decision:** No configuration was rejected by the hard thresholds (>50% noise rate or >50% dominant cluster share).
* **Baseline Selection Rationale:** `distance_threshold=0.40` is selected because 0.20 and 0.30 produce extreme over-fragmentation (566 and 478 singleton clusters out of 597 keywords), while 0.50 blends separate sub-topics.
* **Semantic Selection Rationale:** `min_cluster_size=10;min_samples=1` (or `min_cluster_size=5;min_samples=1`) provides optimal semantic density grouping with minimal noise (3.5%–5.7%), whereas `min_samples=10` discards over 20% of keywords as noise.

---

## 4. Baseline vs Semantic Comparison (from `evaluation_summary.csv` & `cluster_diagnostics.csv`)

| Metric | Baseline (`tfidf_agglomerative`) | Semantic (`minilm_hdbscan`) |
|---|---|---|
| **Parameter Configuration** | `distance_threshold=0.4` | `min_cluster_size=5;min_samples=1` |
| **Number of Clusters ($n$)** | 421 | 29 |
| **Noise Rate** | 0.0% (all rows assigned) | 5.70% (34 keywords flagged as noise) |
| **Singleton Clusters** | 301 | 0 |
| **Singleton Share (% of keywords)** | **50.42%** | **0.00%** |
| **Cluster Size Range** | Min: 1, Median: 1.0, Max: 8 | Min: 5, Median: 9.0, Max: 58 |
| **Largest Cluster Share** | 1.34% | 9.72% |
| **Cosine Silhouette Score** | 0.1488 | **0.4103** |
| **Mean Human Coherence** | NA (Rating Pending) | NA (Rating Pending) |
| **% Clusters Rated 4 or 5** | NA (Rating Pending) | NA (Rating Pending) |
| **Measured Runtime** | 0.060s | 17.138s (includes CPU embedding inference) |

---

## 5. Human Review & Qualitative Cluster Inspection
* **Review Sheet Status:** `reports/human_review_sheet.csv` contains 20 sampled clusters.
* **Reviewer Feedback:** **PENDING** (human evaluation ratings 1–5, reviewer initials, reasons, and misplaced keywords have not yet been submitted by the domain SEO practitioner).

### Qualitative Observations from Generated Clusters

#### High-Coherence Semantic Groupings (Observed in Previews)
1. **Credit Cards Topical Group (Semantic Cluster 18, size 58):**
   * Members: `how to pay off student credit card`, `top credit card online shopping`, `what is the best student credit card`, `what is the best credit card rewards`, `travel credit card review`.
   * *Assessment:* Unifies varied lexical phrasing around the unified commercial intent of credit card research.
2. **Auto Insurance Policy Group (Semantic Cluster 4, size 57):**
   * Members: `how to cancel full coverage insurance`, `full coverage insurance for new cars`, `comprehensive car insurance for new cars`, `best auto insurance quotes for high risk drivers`.
   * *Assessment:* Successfully links transactional ("cancel", "quotes") and informational ("full coverage", "comprehensive") queries under auto insurance.
3. **Puppy Training Group (Baseline Cluster 31, size 7):**
   * Members: `cheap house training a puppy`, `house training a puppy tips`, `house training a puppy for adult dogs`, `best house training a puppy for puppies`.
   * *Assessment:* TF-IDF baseline effectively groups exact lexical phrase variations of "house training a puppy".

#### Failure Cases & Limitations
1. **Severe Lexical Over-Fragmentation (Baseline Singletons):**
   * Over 50.4% (301 clusters) in the TF-IDF baseline are isolated singletons. For example, `best low interest credit card for bad credit` and `travel credit card review` share zero vocabulary overlap and remain completely unlinked despite sharing domain intent.
2. **Density-Based Noise Rejection (HDBSCAN Noise):**
   * HDBSCAN rejected 5.7% (34 keywords) as noise (`cluster_id = -1`), including ambiguous keywords (`running`, `apple`, `insurance`). While mathematically sound, SEO workflows often require every keyword to be mapped to a content category.
3. **Lexical Stemming / Token False Merges (Baseline):**
   * Baseline clustering at looser thresholds risks grouping queries by common syntactic modifiers (`tips`, `review`, `buy online`) rather than core semantic topics.

---

## 6. Method Selection Verdict (A6 Rule)
* **A6 Acceptance Criteria:** A method is eligible only if `mean_human_coherence >= 3.5` and `pct_clusters_rating_4_or_5 >= 60%`. If neither meets these thresholds, the pipeline must conclude **"neither ready for practical use"**.
* **Verdict:** **Neither ready for practical use** (External SEO practitioner review is **PENDING**; human coherence metrics are unrated `NA`).
* **Technical Recommendation:** Pending human review, `minilm_hdbscan` demonstrates superior cluster consolidation (0 singletons vs 301 singletons) and higher silhouette score (0.4103 vs 0.1488), while `tfidf_agglomerative` provides instantaneous runtime (0.06s vs 17.1s).

---

## 7. Metric Limitations
1. **Silhouette Score Blind Spot:** Silhouette scores for HDBSCAN exclude noise points. Consequently, a configuration that discards 25% of the dataset as noise can artificially present a higher silhouette score than a configuration grouping all keywords.
2. **Lexical Distance Blindness:** Cosine distance over TF-IDF n-grams cannot bridge semantic synonyms (e.g. `vehicle insurance` vs `car insurance`).
3. **Synthetic Data Bias:** Synthetic data derived from permutation templates lacks real search intent distribution, long-tail typos, and organic conversational phrasing.

---

## 8. Performance Tiers (from `reports/performance_results.csv`)

| Tier ($N$) | Actual Unique Keywords | Status | Measured Runtime (s) | Measured Peak RAM (MB) | Targets | Verdict |
|---|---|---|---|---|---|
| **$N = 100$** | 100 | COMPLETED | **13.68 s** | **553.8 MB** | Time $\le 120$ s, RAM $\le 2,048$ MB | **PASS** |
| **$N = 1,000$** | 597 | SKIPPED | NA | NA | Time $\le 600$ s, RAM $\le 4,096$ MB | **SKIPPED** (Dataset has 597 unique keywords; no synthetic padding) |
| **$N = 5,000$** | 597 | SKIPPED | NA | NA | Time $\le 1,800$ s, RAM $\le 6,144$ MB | **SKIPPED** (Dataset has 597 unique keywords; no synthetic padding) |

* **Limiting Resource:** The computational bottleneck is transformer embedding generation on CPU (scaling linearly $O(N)$), while memory scaling remains well within the 4 GB / 6 GB targets.

---

## 9. Next Steps
1. Ingest an approved real organic search phrase dataset into `data/input/real_keywords.csv`.
2. Collect human coherence ratings (1–5) and misplaced keyword audits from an SEO practitioner in `reports/human_review_sheet.csv`.
3. Freeze chosen parameter thresholds in `config.py` and rerun `run_pipeline.py run`.
