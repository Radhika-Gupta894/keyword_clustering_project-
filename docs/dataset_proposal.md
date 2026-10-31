# Dataset Proposal (for Supervisor Review)

## Evaluated Dataset Details

1. **Name + source URL:**
   * **Name:** `cleanor-search-index` (Cleanor Research Labs)
   * **URL:** [https://huggingface.co/datasets/cleanorlabs/cleanor-search-index](https://huggingface.co/datasets/cleanorlabs/cleanor-search-index)
   * **Repository:** [https://github.com/cleanorlabs/cleanor-search-index](https://github.com/cleanorlabs/cleanor-search-index)
   * **Local Raw Archive:** `data/input/raw/cleanor_search_index.csv` (unmodified raw download)

2. **Access date:**
   * 2026-10-31

3. **Licence / commercial-use restrictions (quote licence text/link):**
   * **Data Licence:** Creative Commons Attribution 4.0 International (**CC BY 4.0**).
   * **Code Licence:** MIT License.
   * **Commercial Use:** Permitted under CC BY 4.0 provided attribution is given: *"Cleanor Research Labs (cleanorlabs/cleanor-search-index)"*.

4. **Rows, relevant columns:**
   * **Rows:** 3,360 rows (20 countries × 14 AI brands × 12 monthly snapshots from 2025-06 to 2026-05).
   * **Columns Present:** `snapshot`, `category`, `country_code`, `country`, `brand`, `vendor`, `year_month`, `searches`.
   * **Missing Required Fields:** Zero columns containing natural language search phrases, keywords, or queries. The only text entity column is `brand`, with only 14 distinct values.

5. **Why it was evaluated for keyword clustering:**
   * Reflects real-world search volume telemetry derived from Google Keyword Planner for Generative AI tools across 20 countries.

6. **Known quality issues & Evaluation Outcome (EVALUATED AND REJECTED):**
   * **Status:** **REJECTED as pipeline input.**
   * **Reason for Rejection:** The dataset does not contain keyword or search query phrases. It contains aggregate monthly search volume counts grouped by entity (`brand`), country, and month. With only 14 unique brand terms repeated across countries and months, unsupervised NLP clustering algorithms (TF-IDF + Agglomerative and MiniLM + HDBSCAN with `min_cluster_size >= 5`) cannot perform semantic topical clustering.
   * **Zero Tolerance for Data Misrepresentation:** Synthesizing query permutations or treating single-brand volume indexes as natural search phrases would violate data integrity standards.

7. **Suitable for commercial experimentation? (yes/no/unclear):**
   * **Licence Suitability:** **Yes** (CC BY 4.0 allows commercial use).
   * **Methodological Suitability:** **No (Rejected)**. The dataset is an aggregate search volume time-series index, not a keyword phrase corpus.
   * **Recommendation:** Keep `data/input/` empty of pseudo-real data. Retain the raw Cleanor file in `data/input/raw/` for provenance. Seek an approved organic search query dataset (e.g. MS MARCO, AOL query logs, or e-commerce search logs). In the interim, execute all pipeline testing and validation strictly on clearly labeled synthetic datasets.

---

## Fallback Section: Synthetic Dataset in Use

Because no valid organic query dataset is currently approved, this project operates strictly on the bundled synthetic datasets located in `data/synthetic/`:
- `data/synthetic/sample_keywords.csv` (pipeline benchmark input, 597 rows)
- `data/synthetic/smoke_keywords.csv` (rapid smoke test input, 100 rows)
- `data/synthetic/synthetic_gold_topics.csv` (ground-truth topic annotations for evaluation verification)
- `data/synthetic/ai_search_keywords.csv` (synthetic AI queries generated for taxonomy testing)

### Scope of Validity
* **Pipeline & Integration Testing:** Valid for testing data loading, UTF-8 parsing, error handling, normalization, schema export conformity, determinism, and CLI execution.
* **Algorithmic Comparison:** Demonstrates functionality and behavior of baseline (`tfidf_agglomerative`) versus semantic (`minilm_hdbscan`) clustering workflows on known synthetic groupings.

### Limitations & Non-Representation of Real-World SEO
* **Not Real Search Data:** All synthetic keywords were algorithmically generated using fixed permutation templates across predefined topical verticals.
* **Artificial Intent & Vocabulary:** Synthetic queries do not reflect the natural vocabulary distribution, spelling anomalies, semantic nuance, search volume, click-through behavior, or organic intent shifts of real user search engines.
* **Zero Real-World SEO Predictive Value:** The clusters, silhouette scores, and metrics produced from this synthetic data validate pipeline mechanics only. They cannot and do not predict Google rankings, organic search traffic, or real-world SEO performance.
