# 7-Minute Video Demonstration Script
**Project:** AI-Powered Keyword Clustering Engine (VyorAI Assignment)  
**Presenter:** [Your Name / Candidate]  
**Total Target Duration:** ~7 minutes (approx. 130–140 words per minute)

---

### Segment 1: The Problem & Introduction (0:00 – 1:00)
**On Screen:** Show code editor (VS Code) open to the project root with the file tree visible.

**What to Say:**
> "Hello. Today I am presenting the AI-Powered Keyword Clustering Engine developed for VyorAI.
>
> In digital marketing and SEO, content teams routinely export thousands of unorganized search queries from tools like Google Search Console or Ahrefs. Historically, analysts spend hours manually sorting queries in spreadsheets or writing brittle regex filters to decide which topics deserve dedicated landing pages or articles.
>
> Our objective is to engineer an automated, reproducible pipeline that ingests raw keyword exports, validates data quality, and clusters them using two contrasting paradigms: a fast, lexical n-gram baseline, and a dense transformer-based semantic model.
>
> An essential design principle of this project is that clustering is strictly assistive for human editorial review. Clusters do not predict Google rankings, and they must always be verified by an SEO practitioner."

---

### Segment 2: Data Provenance, Licensing, & Synthetic Status (1:00 – 2:00)
**On Screen:** Open `docs/dataset_proposal.md` and highlight the Data Provenance section.

**What to Say:**
> "Before running code, let's address data provenance and licensing.
>
> We evaluated the public Hugging Face dataset `cleanorlabs/cleanor-search-index`, licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Its raw file is preserved untouched in `data/input/raw/cleanor_search_index.csv`.
>
> However, our audit revealed that Cleanor is an aggregate monthly search volume index for 14 AI brand entities across 20 countries. It contains zero natural language query phrases or search keywords. Grouping 14 single-word brand names repeated across months fails the fundamental requirement of an SEO query corpus.
>
> In accordance with strict data integrity standards, we rejected Cleanor as a query corpus and refused to artificially synthesize queries and label them as 'real'. 
>
> Therefore, all runs demonstrated today operate on our bundled synthetic benchmark fixtures in `data/synthetic/sample_keywords.csv`. These fixtures validate software engineering, error handling, and algorithmic mechanics only, not real-world SEO performance. A high-priority TODO remains in our README to ingest an approved organic search query corpus once selected."

---

### Segment 3: Architecture & Data Cleaning (2:00 – 3:00)
**On Screen:** Switch to terminal in VS Code. Run the test suite first:

**Terminal Command:**
```bash
python -m pytest -q
```

**What to Say:**
> "To verify pipeline stability, we run our automated test suite. All 12 unit and integration tests pass offline in approximately 4 seconds using a deterministic hashing embedder, verifying input validation, error handling, normalization, duplicate grouping, and schema conformity.
>
> Our ingestion engine in `src/data_loader.py` enforces strict UTF-8 decoding and halts with clear error codes if the file is missing, empty, or lacks the required 'keyword' column.
>
> Next, `src/preprocessing.py` standardizes text: it trims irregular whitespace, lowercases tokens, and assigns unique `duplicate_group_id` tags to exact and normalized duplicates. 
>
> Crucially, clustering is fitted only on unique normalized keywords, while assignments are mapped back to all valid input rows so no input data is silently dropped."

---

### Segment 4: Baseline vs. Semantic Methodology (3:00 – 4:00)
**On Screen:** Open `config.py` side-by-side with `src/baseline_clustering.py` and `src/semantic_clustering.py`.

**What to Say:**
> "The engine implements two contrasting clustering methodologies:
>
> 1. **The Baseline:** Word-level TF-IDF vectorization with 1-to-2 n-grams coupled with Agglomerative Clustering using cosine distance. Following hyperparameter grid search on our dev subset, we set `distance_threshold = 0.40`. At lower thresholds like 0.20, over 95% of keywords ended up as isolated singletons; 0.40 provides a balanced lexical cutoff.
>
> 2. **The Semantic Method:** We generate 384-dimensional dense sentence embeddings using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`, pinned to an immutable commit hash (`c9745ed1...`) and running entirely offline on CPU. These vectors are clustered using HDBSCAN with `min_cluster_size = 5` and `min_samples = 1`. 
>
> Unlike the baseline, HDBSCAN is a density-based algorithm that explicitly identifies unclusterable outliers as noise (`cluster_id = -1`)."

---

### Segment 5: Live Pipeline Run (4:00 – 5:00)
**On Screen:** Execute the complete pipeline in the terminal:

**Terminal Command:**
```bash
python run_pipeline.py run --input data/synthetic/sample_keywords.csv --methods both
```

**What to Say:**
> "Now let's execute the live pipeline on our 677-row benchmark input.
>
> Watch the console:
> First, it loads 677 rows, detects zero corrupt entries, and identifies 80 exact duplicates, leaving 597 unique normalized keywords.
>
> Second, the TF-IDF baseline executes near-instantaneously in 0.08 seconds, producing 421 clusters with zero noise and a silhouette score of 0.149.
>
> Third, the semantic pipeline loads the local transformer model once, computes embeddings on CPU, and completes HDBSCAN in 13 seconds, producing 29 consolidated clusters with a 5.7% noise rate and a silhouette score of 0.410.
>
> Finally, all outputs are written to `data/output/` alongside `run_metadata.json`."

---

### Segment 6: Results, Good Cluster vs. Failure Case (5:00 – 6:15)
**On Screen:** Open `data/output/cluster_summary.csv` and `data/output/evaluation_summary.csv`.

**What to Say:**
> "Let's inspect the results:
>
> In `evaluation_summary.csv`, the semantic method achieves a silhouette score of 0.410 compared to 0.149 for the baseline. More importantly, `reports/cluster_diagnostics.csv` reveals a massive operational difference:
> The baseline left 50.4% of keywords—over 300 queries—as isolated singletons because lexical TF-IDF cannot connect queries that do not share exact words. HDBSCAN had zero singletons.
>
> **Here is a strong, successful semantic cluster (Cluster 18, size 58):**
> It cleanly groups commercial credit card queries:
> - `how to pay off student credit card`
> - `top credit card online shopping`
> - `what is the best student credit card`
> - `travel credit card review`
> Despite varied syntax, it correctly unifies them around credit card research.
>
> **Now, let's examine a failure case in the baseline:**
> In the baseline, `best low interest credit card for bad credit` and `travel credit card review` were assigned to completely separate singleton clusters. Because one uses 'low interest' and the other uses 'travel', TF-IDF treated them as alien topics despite sharing commercial credit card intent.
>
> **Regarding HDBSCAN noise:**
> HDBSCAN marked 5.7% of keywords as noise (`is_noise = True`), such as short, ambiguous terms like `running`, `apple`, and `insurance`. In production, analysts must decide whether to route these to a fallback category."

---

### Segment 7: Limitations & Conclusion (6:15 – 7:00)
**On Screen:** Open `docs/business_note.md` and show the handover files.

**What to Say:**
> "To conclude, let's address limitations and handover status:
>
> 1. **Human Evaluation Gate (A6):** Our sample review sheet `reports/human_review_sheet.csv` contains 20 sampled clusters ready for scoring. Because human SEO practitioner feedback is strictly marked as PENDING, the formal A6 verdict is 'neither ready for practical use' until manual ratings are logged. We do not invent ratings.
> 2. **Technical Limitations:** High vector similarity does not guarantee identical search engine intent, and HDBSCAN noise rejection requires manual routing.
> 3. **Non-Claims:** We make zero commercial claims regarding Google rankings, organic traffic, CTR, or speculative ROI.
>
> All required deliverables—including pinned requirements, performance benchmarks, and output CSVs—are fully tracked and documented. Thank you."
