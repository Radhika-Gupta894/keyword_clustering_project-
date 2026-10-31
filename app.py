"""Interactive Streamlit Web Application for AI-Powered Keyword Clustering Engine.

Allows users to upload custom keyword CSVs or use the bundled synthetic benchmark,
configure clustering parameters, view interactive cluster tables, and download
the 4 assignment deliverables.
"""
import io
import json
import os
from pathlib import Path
import tempfile
import time

import pandas as pd
import streamlit as st

# Setup page
st.set_page_config(
    page_title="AI-Powered Keyword Clustering Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
DEFAULT_SAMPLE_CSV = SYNTHETIC_DIR / "sample_keywords.csv"

# Import core pipeline modules
import config
from src.data_loader import KeywordInputError, load_keywords
from src.preprocessing import prepare
from src.baseline_clustering import cluster_baseline
from src.semantic_clustering import cluster_semantic, load_embedder
from src import evaluation as ev
from src import export_results as ex


@st.cache_resource(show_spinner="Loading Sentence-Transformers embedding model (CPU)...")
def get_cached_embedder():
    """Load and cache the SentenceTransformer model once in memory."""
    return load_embedder(
        model_name=config.EMBEDDING_MODEL,
        revision=config.EMBEDDING_REVISION,
        batch_size=config.EMBEDDING_BATCH_SIZE
    )


# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #E2E8F0;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #0284C7;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --- Main Title ---
st.markdown('<div class="main-header">🔍 AI-Powered Keyword Clustering Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated lexical and semantic grouping for SEO search query exports and content architecture planning.</div>', unsafe_allow_html=True)


# --- Sidebar Navigation & Input Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    input_source = st.radio(
        "Choose Input Dataset:",
        options=["Bundled Synthetic Benchmark (~600 queries)", "Upload Custom CSV"],
        index=0
    )
    
    uploaded_file = None
    input_path = None
    
    if input_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV with a 'keyword' column:",
            type=["csv"],
            help="Must be a UTF-8 encoded CSV containing at least a 'keyword' column."
        )
    else:
        st.info("Operating on `data/synthetic/sample_keywords.csv` (pipeline testing fixture).")
        input_path = DEFAULT_SAMPLE_CSV

    st.markdown("---")
    st.subheader("Clustering Methods")
    methods_choice = st.selectbox(
        "Select Pipeline Methods:",
        options=["both", "semantic", "baseline"],
        format_func=lambda x: {
            "both": "Both (Baseline TF-IDF + Semantic HDBSCAN)",
            "semantic": "Semantic Only (MiniLM-L6-v2 + HDBSCAN)",
            "baseline": "Baseline Only (Word TF-IDF + Agglomerative)"
        }[x]
    )

    with st.expander("🛠️ Advanced Hyperparameters", expanded=False):
        st.caption("Baseline Parameters")
        baseline_thresh = st.slider(
            "Distance Threshold (Baseline)",
            min_value=0.20, max_value=0.60, value=float(config.BASELINE_THRESHOLD), step=0.05,
            help="Higher values create larger, more relaxed clusters."
        )

        st.caption("Semantic HDBSCAN Parameters")
        min_cluster_size = st.slider(
            "Min Cluster Size",
            min_value=2, max_value=25, value=int(config.HDBSCAN_PARAMS.get("min_cluster_size", 5)), step=1,
            help="Smallest size grouping to qualify as a cluster."
        )
        min_samples = st.slider(
            "Min Samples",
            min_value=1, max_value=10, value=int(config.HDBSCAN_PARAMS.get("min_samples", 1)), step=1,
            help="Conservative factor. Higher values classify more border points as noise."
        )

    run_button = st.button("🚀 Run Clustering Pipeline", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("VyorAI Intern Assignment • Handover Demo")


# --- Pipeline Execution Logic ---
if "pipeline_results" not in st.session_state:
    st.session_state["pipeline_results"] = None

if run_button:
    # Resolve input source
    temp_input_file = None
    if input_source == "Upload Custom CSV":
        if uploaded_file is None:
            st.error("Please upload a CSV file before running.")
            st.stop()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tfile.write(uploaded_file.getvalue())
        tfile.flush()
        temp_input_file = Path(tfile.name)
        active_input_path = temp_input_file
    else:
        active_input_path = DEFAULT_SAMPLE_CSV

    if not active_input_path.exists():
        st.error(f"Input file not found at: {active_input_path}")
        st.stop()

    try:
        with st.spinner("Processing keywords through clustering pipeline..."):
            start_total = time.perf_counter()

            # 1. Ingest
            raw = load_keywords(active_input_path, config.KEYWORD_COLUMN)
            prep = prepare(raw)
            uniq = prep.unique["keyword_normalized"].tolist()
            n = len(uniq)

            # Temporary output dirs for isolation
            with tempfile.TemporaryDirectory() as tmp_out_dir:
                tmp_out = Path(tmp_out_dir)
                tmp_rep = tmp_out / "reports"
                tmp_rep.mkdir(parents=True, exist_ok=True)

                eval_rows, assign_parts, summary_parts, diag_rows = [], [], [], []
                results = {}

                def finish(name, cfg, labels, X, secs, mem):
                    reps, sims = ex.representatives_and_similarity(X, labels)
                    results[name] = {"labels": labels, "reps": reps}
                    assign_parts.append(ex.build_assignments(prep.records, name, labels, sims))
                    summary_parts.append(ex.build_summary(prep.unique, name, labels, reps))
                    d = ev.diagnostics(labels)
                    diag_rows.append({"method": name, **d})
                    sil = ev.silhouette_or_none(X, labels)
                    eval_rows.append({
                        "method": name, "parameter_config": cfg, "n_clusters": d["n_clusters"],
                        "noise_rate": round(d["noise_rate"], 4),
                        "silhouette_score": "NA" if sil is None else round(sil, 4),
                        "mean_human_coherence": "NA", "pct_clusters_rating_4_or_5": "NA",
                        "runtime_seconds": secs, "peak_memory_mb": "NA" if mem is None else mem
                    })

                # Baseline execution
                if methods_choice in ("both", "baseline"):
                    (labels_b, X_b), secs_b, mem_b = ev.measure(lambda: cluster_baseline(uniq, baseline_thresh))
                    finish(config.BASELINE_METHOD, f"distance_threshold={baseline_thresh}", labels_b, X_b, secs_b, mem_b)

                # Semantic execution
                if methods_choice in ("both", "semantic"):
                    embedder_fn = get_cached_embedder()
                    def sem():
                        emb = embedder_fn(uniq)
                        hdb_p = {"min_cluster_size": min_cluster_size, "min_samples": min_samples}
                        return cluster_semantic(emb, config.SEMANTIC_ALGORITHM, hdb_p, config.SEMANTIC_AGG_THRESHOLD), emb
                    (labels_s, emb_s), secs_s, mem_s = ev.measure(sem)
                    finish(config.SEMANTIC_METHOD, f"min_cluster_size={min_cluster_size};min_samples={min_samples}", labels_s, emb_s, secs_s, mem_s)

                evaluation_df = pd.DataFrame(eval_rows)[ex.EVALUATION_COLUMNS]
                assignments_df = pd.concat(assign_parts, ignore_index=True)
                summary_df = pd.concat(summary_parts, ignore_index=True)
                quality_df = prep.quality[ex.QUALITY_COLUMNS]
                diagnostics_df = pd.DataFrame(diag_rows)

                # Generate run_metadata dictionary
                metadata_dict = {
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "dataset": active_input_path.name,
                    "methods_run": methods_choice,
                    "total_valid_rows": len(prep.records),
                    "unique_keywords": n,
                    "model": config.EMBEDDING_MODEL
                }

                st.session_state["pipeline_results"] = {
                    "assignments": assignments_df,
                    "summary": summary_df,
                    "quality": quality_df,
                    "evaluation": evaluation_df,
                    "diagnostics": diagnostics_df,
                    "metadata": metadata_dict,
                    "unique_count": n,
                    "valid_count": len(prep.records),
                    "total_runtime": round(time.perf_counter() - start_total, 2)
                }

    except KeywordInputError as e:
        st.error(f"⚠️ Input Validation Error: {e}")
    except Exception as e:
        st.error(f"❌ Execution Failure: {e}")
    finally:
        if temp_input_file and temp_input_file.exists():
            temp_input_file.unlink()


# --- Display Dashboard Results ---
results = st.session_state.get("pipeline_results")

if results:
    st.success(f"Clustering completed in {results['total_runtime']}s across {results['unique_count']} unique keywords.")

    # KPI Banner
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Valid Rows", f"{results['valid_count']:,}")
    with col2:
        st.metric("Unique Keywords", f"{results['unique_count']:,}")
    with col3:
        n_clusters_total = results['summary']['cluster_id'].nunique()
        st.metric("Clusters Formed", f"{n_clusters_total:,}")
    with col4:
        if "noise_rate" in results['evaluation'].columns:
            avg_noise = results['evaluation']['noise_rate'].mean()
            st.metric("Avg Noise Rate", f"{avg_noise:.1%}")
    with col5:
        st.metric("Runtime", f"{results['total_runtime']}s")

    st.markdown("---")

    # Tabs for Outputs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Cluster Summary",
        "🏷️ Keyword Assignments",
        "📈 Evaluation & Metrics",
        "🧹 Data Quality Audit",
        "📥 Download Deliverables"
    ])

    with tab1:
        st.subheader("Topic Clusters & Central Representatives")
        method_filter = st.selectbox(
            "Filter by Method:",
            options=results["summary"]["method"].unique().tolist()
        )
        filtered_summary = results["summary"][results["summary"]["method"] == method_filter]
        
        # Search inside clusters
        query = st.text_input("🔍 Search clusters by representative or member keyword:", "")
        if query:
            filtered_summary = filtered_summary[
                filtered_summary["representative_keyword"].str.contains(query, case=False, na=False) |
                filtered_summary["members"].str.contains(query, case=False, na=False)
            ]

        st.dataframe(
            filtered_summary[["cluster_id", "cluster_size", "representative_keyword", "members"]],
            use_container_width=True,
            height=380
        )

        st.caption("Top 15 Largest Clusters:")
        chart_data = filtered_summary.sort_values(by="cluster_size", ascending=False).head(15)
        st.bar_chart(chart_data.set_index("representative_keyword")["cluster_size"])

    with tab2:
        st.subheader("Keyword Assignments Table")
        st.caption("Maps each input keyword to its assigned cluster, representative similarity, and noise flag.")
        
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            method_assign_filter = st.selectbox(
                "Select Method:",
                options=results["assignments"]["method"].unique().tolist(),
                key="assign_filter"
            )
        with col_f2:
            search_keyword = st.text_input("Search Keyword:", "", key="kw_search")

        filtered_assign = results["assignments"][results["assignments"]["method"] == method_assign_filter]
        if search_keyword:
            filtered_assign = filtered_assign[
                filtered_assign["keyword_original"].str.contains(search_keyword, case=False, na=False)
            ]

        st.dataframe(filtered_assign, use_container_width=True, height=450)

    with tab3:
        st.subheader("Algorithmic Evaluation & Diagnostics")
        st.dataframe(results["evaluation"], use_container_width=True)

        st.markdown("#### Diagnostic Distribution (Cluster Sizes & Singletons)")
        st.dataframe(results["diagnostics"], use_container_width=True)
        st.info("Note: Human Coherence ratings reflect 'NA' because qualitative reviews by an SEO practitioner are pending.")

    with tab4:
        st.subheader("Data Quality Report")
        st.dataframe(results["quality"], use_container_width=True)

    with tab5:
        st.subheader("Download Section 16 Handover Files")
        st.markdown("Download all standardized output CSVs and JSON run metadata generated during this session:")
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="📄 Download keyword_assignments.csv",
                data=results["assignments"].to_csv(index=False).encode("utf-8"),
                file_name="keyword_assignments.csv",
                mime="text/csv"
            )
            st.download_button(
                label="📄 Download cluster_summary.csv",
                data=results["summary"].to_csv(index=False).encode("utf-8"),
                file_name="cluster_summary.csv",
                mime="text/csv"
            )
            st.download_button(
                label="📄 Download data_quality_report.csv",
                data=results["quality"].to_csv(index=False).encode("utf-8"),
                file_name="data_quality_report.csv",
                mime="text/csv"
            )
        with c2:
            st.download_button(
                label="📄 Download evaluation_summary.csv",
                data=results["evaluation"].to_csv(index=False).encode("utf-8"),
                file_name="evaluation_summary.csv",
                mime="text/csv"
            )
            st.download_button(
                label="⚙️ Download run_metadata.json",
                data=json.dumps(results["metadata"], indent=2).encode("utf-8"),
                file_name="run_metadata.json",
                mime="application/json"
            )

else:
    st.info("👈 Select your dataset in the sidebar and click **'🚀 Run Clustering Pipeline'** to begin.")
