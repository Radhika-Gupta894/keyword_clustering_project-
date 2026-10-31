"""Interactive Streamlit Web Application for AI-Powered Keyword Clustering Engine.

Allows users to upload custom keyword CSVs or use the bundled synthetic benchmark,
configure clustering parameters, view interactive cluster tables, complete the
qualitative human review sheet (reports/human_review_sheet.csv), and download
all Section 16 assignment deliverables.
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
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_SAMPLE_CSV = SYNTHETIC_DIR / "sample_keywords.csv"
HUMAN_REVIEW_CSV = REPORTS_DIR / "human_review_sheet.csv"

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


def load_existing_results():
    """Load existing run outputs from disk if present so dashboard starts populated."""
    assign_path = OUTPUT_DIR / "keyword_assignments.csv"
    summary_path = OUTPUT_DIR / "cluster_summary.csv"
    eval_path = OUTPUT_DIR / "evaluation_summary.csv"
    quality_path = OUTPUT_DIR / "data_quality_report.csv"
    diag_path = REPORTS_DIR / "cluster_diagnostics.csv"
    meta_path = OUTPUT_DIR / "run_metadata.json"

    if assign_path.exists() and summary_path.exists():
        try:
            assignments_df = pd.read_csv(assign_path)
            summary_df = pd.read_csv(summary_path)
            eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
            quality_df = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
            diag_df = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            method_first = assignments_df["method"].iloc[0] if len(assignments_df) else None
            valid_c = len(assignments_df[assignments_df["method"] == method_first]) if method_first else len(assignments_df)
            uniq_c = len(assignments_df["keyword_normalized"].unique()) if "keyword_normalized" in assignments_df.columns else valid_c

            return {
                "assignments": assignments_df,
                "summary": summary_df,
                "quality": quality_df,
                "evaluation": eval_df,
                "diagnostics": diag_df,
                "metadata": meta,
                "unique_count": uniq_c,
                "valid_count": valid_c,
                "total_runtime": meta.get("runtime_seconds", 0.0),
                "is_preloaded": True
            }
        except Exception:
            return None
    return None


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
    .cluster-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
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


# --- Initialize Session State ---
if "pipeline_results" not in st.session_state:
    st.session_state["pipeline_results"] = load_existing_results()


# --- Sidebar Navigation & Input Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")

    input_source = st.radio(
        "Choose Input Dataset:",
        options=["Bundled Synthetic Benchmark (~600 queries)", "Upload Custom CSV"],
        index=0
    )

    uploaded_file = None
    if input_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV with a 'keyword' column:",
            type=["csv"],
            help="Must be a UTF-8 encoded CSV containing at least a 'keyword' column."
        )
    else:
        st.info("Operating on `data/synthetic/sample_keywords.csv` (pipeline testing fixture).")
        active_input_path = DEFAULT_SAMPLE_CSV

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
if run_button:
    temp_input_file = None
    if input_source == "Upload Custom CSV":
        if uploaded_file is None:
            st.error("Please upload a CSV file before running.")
            st.stop()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tfile:
            tfile.write(uploaded_file.getvalue())
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

            # Incorporate human ratings if present
            if HUMAN_REVIEW_CSV.exists():
                sheet = pd.read_csv(HUMAN_REVIEW_CSV, dtype=str, keep_default_na=False)
                ratings = ev.summarize_ratings(sheet)
                for row in eval_rows:
                    if row["method"] in ratings:
                        mean, pct, n_rated = ratings[row["method"]]
                        row["mean_human_coherence"], row["pct_clusters_rating_4_or_5"] = round(mean, 2), round(pct, 3)

            evaluation_df = pd.DataFrame(eval_rows)[ex.EVALUATION_COLUMNS]
            assignments_df = pd.concat(assign_parts, ignore_index=True)
            summary_df = pd.concat(summary_parts, ignore_index=True)
            quality_df = prep.quality[ex.QUALITY_COLUMNS]
            diagnostics_df = pd.DataFrame(diag_rows)

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
                "total_runtime": round(time.perf_counter() - start_total, 2),
                "is_preloaded": False
            }

    except KeywordInputError as e:
        st.error(f"⚠️ Input Validation Error: {e}")
    except Exception as e:
        st.error(f"❌ Execution Failure: {e}")
    finally:
        if temp_input_file and temp_input_file.exists():
            try:
                temp_input_file.unlink(missing_ok=True)
            except Exception:
                pass


# --- Display Dashboard Results ---
results = st.session_state.get("pipeline_results")

if results:
    if results.get("is_preloaded"):
        st.info("💡 Displaying pre-computed pipeline deliverables from `data/output/`. You can adjust parameters in the sidebar and click **'🚀 Run Clustering Pipeline'** anytime to recompute.")
    else:
        st.success(f"Clustering completed in {results['total_runtime']}s across {results['unique_count']} unique keywords.")

    # KPI Banner
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Valid Rows", f"{results['valid_count']:,}")
    with col2:
        st.metric("Unique Keywords", f"{results['unique_count']:,}")
    with col3:
        n_clusters_total = results['summary']['cluster_id'].nunique() if 'summary' in results and 'cluster_id' in results['summary'] else 0
        st.metric("Clusters Formed", f"{n_clusters_total:,}")
    with col4:
        if "noise_rate" in results['evaluation'].columns and len(results['evaluation']):
            try:
                avg_noise = pd.to_numeric(results['evaluation']['noise_rate'], errors='coerce').mean()
                st.metric("Avg Noise Rate", f"{avg_noise:.1%}")
            except Exception:
                st.metric("Avg Noise Rate", "N/A")
        else:
            st.metric("Avg Noise Rate", "0.0%")
    with col5:
        st.metric("Runtime", f"{results['total_runtime']}s")

    st.markdown("---")

    # Tabs for Outputs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Cluster Summary",
        "🏷️ Keyword Assignments",
        "📈 Evaluation & Metrics",
        "📝 Human Review Sheet",
        "🧹 Data Quality Audit",
        "📥 Download Deliverables"
    ])

    # --- TAB 1: Cluster Summary ---
    with tab1:
        st.subheader("Topic Clusters & Central Representatives")
        if "summary" in results and not results["summary"].empty:
            method_filter = st.selectbox(
                "Filter by Method:",
                options=results["summary"]["method"].unique().tolist()
            )
            filtered_summary = results["summary"][results["summary"]["method"] == method_filter]

            query = st.text_input("🔍 Search clusters by representative or member keyword:", "")
            if query:
                filtered_summary = filtered_summary[
                    filtered_summary["representative_keyword"].str.contains(query, case=False, na=False) |
                    filtered_summary["member_keywords"].str.contains(query, case=False, na=False)
                ]

            st.dataframe(
                filtered_summary[["cluster_id", "cluster_size_unique", "representative_keyword", "member_keywords"]],
                use_container_width=True,
                height=380
            )

            st.caption("Top 15 Largest Clusters:")
            chart_data = filtered_summary.sort_values(by="cluster_size_unique", ascending=False).head(15)
            st.bar_chart(chart_data.set_index("representative_keyword")["cluster_size_unique"])
        else:
            st.info("No cluster summary available.")

    # --- TAB 2: Keyword Assignments ---
    with tab2:
        st.subheader("Keyword Assignments Table")
        st.caption("Maps each input keyword to its assigned cluster, representative similarity, and noise flag.")

        if "assignments" in results and not results["assignments"].empty:
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
        else:
            st.info("No assignments data available.")

    # --- TAB 3: Evaluation & Metrics ---
    with tab3:
        st.subheader("Algorithmic Evaluation & Diagnostics")
        st.dataframe(results["evaluation"], use_container_width=True)

        # Automated Method Selection Decision
        choice, why = ev.select_method(results["evaluation"])
        if choice:
            st.success(f"🏆 **Selected Method:** `{choice}` — {why}")
        else:
            st.info(f"ℹ️ **Method Selection Decision:** {why}")

        st.markdown("#### Diagnostic Distribution (Cluster Sizes & Singletons)")
        st.dataframe(results["diagnostics"], use_container_width=True)

    # --- TAB 4: Human Review Sheet (reports/human_review_sheet.csv) ---
    with tab4:
        st.subheader("📝 Qualitative Human Review (`reports/human_review_sheet.csv`)")
        st.markdown(
            "Per assignment guidelines (FR-09 / Section 16), human evaluation measures practitioner coherence, "
            "cluster intrusion detection, and semantic quality across small, medium, and large sampled clusters."
        )

        if not HUMAN_REVIEW_CSV.exists():
            st.warning(f"File `{HUMAN_REVIEW_CSV}` does not exist yet. Run the pipeline to generate it.")
        else:
            df_review = pd.read_csv(HUMAN_REVIEW_CSV, dtype=str, keep_default_na=False)

            # Ensure all expected columns are present
            for col in ev.REVIEW_COLUMNS:
                if col not in df_review.columns:
                    df_review[col] = ""

            total_clusters = len(df_review)
            numeric_ratings = pd.to_numeric(df_review["rating_1_to_5"], errors="coerce")
            rated_mask = numeric_ratings.between(1, 5)
            rated_count = int(rated_mask.sum())
            pending_count = total_clusters - rated_count
            avg_rating = round(float(numeric_ratings[rated_mask].mean()), 2) if rated_count > 0 else "N/A"

            # Review KPI Cards
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.metric("Total Sampled Clusters", f"{total_clusters}")
            with rc2:
                st.metric("Reviewed Clusters", f"{rated_count} / {total_clusters}")
            with rc3:
                st.metric("Pending Review", f"{pending_count}")
            with rc4:
                st.metric("Mean Coherence Rating", f"{avg_rating}")

            st.progress(rated_count / total_clusters if total_clusters > 0 else 0.0)

            # Review Mode Toggle
            review_mode = st.radio(
                "Review View Mode:",
                options=["🎴 Step-by-Step Cluster Reviewer Card", "📋 Full Spreadsheet Table Editor"],
                horizontal=True
            )

            # MODE 1: Step-by-Step Card Reviewer
            if review_mode == "🎴 Step-by-Step Cluster Reviewer Card":
                st.markdown("---")
                
                # Format cluster selector label
                def format_row(idx):
                    r = df_review.iloc[idx]
                    status = f"✅ Rated: {r['rating_1_to_5']}/5" if r['rating_1_to_5'] in ["1","2","3","4","5"] else "⏳ Pending"
                    return f"Cluster #{idx + 1} (ID: {r['cluster_id']}) — '{r['representative_keyword']}' [{status}]"

                selected_idx = st.selectbox(
                    "Select Cluster to Review:",
                    options=range(total_clusters),
                    format_func=format_row,
                    key="card_cluster_selector"
                )

                row = df_review.iloc[selected_idx]

                with st.container():
                    st.markdown(f"### Cluster #{selected_idx + 1} of {total_clusters}")
                    
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        st.markdown(f"**Method:** `{row['method']}`")
                    with cc2:
                        st.markdown(f"**Cluster ID:** `{row['cluster_id']}`")
                    with cc3:
                        st.markdown(f"**Cluster Size:** `{row['cluster_size_unique']}` unique keywords")

                    st.markdown(f"**Representative Keyword:** `{row['representative_keyword']}`")
                    
                    st.markdown("#### Member Keywords Shown:")
                    members = [m.strip() for m in str(row["members_shown"]).split(";") if m.strip()]
                    for m_i, member_text in enumerate(members, 1):
                        st.markdown(f"{m_i}. `{member_text}`")

                    st.markdown("---")
                    st.markdown("#### Enter Practitioner Evaluation:")

                    with st.form(key=f"cluster_review_form_{selected_idx}"):
                        fc1, fc2 = st.columns([1, 1])
                        with fc1:
                            curr_rating = row["rating_1_to_5"]
                            default_idx = [1, 2, 3, 4, 5].index(int(curr_rating)) if curr_rating in ["1", "2", "3", "4", "5"] else 4
                            rating_val = st.radio(
                                "Coherence Rating (1-5):",
                                options=[1, 2, 3, 4, 5],
                                index=default_idx,
                                horizontal=True,
                                help="1 = Incoherent / unrelated, 5 = Highly coherent single topic."
                            )
                        with fc2:
                            reviewer_val = st.text_input(
                                "Reviewer Initials / Role:",
                                value=row["reviewer_initials_or_role"] if row["reviewer_initials_or_role"] else "RG",
                                help="e.g. RG (intern)"
                            )

                        reason_val = st.text_area(
                            "Reason / Topic Justification:",
                            value=row["reason"],
                            placeholder="e.g. Clean technical SEO checklist intent; terms share identical search need."
                        )

                        misplaced_val = st.text_input(
                            "Misplaced Keywords (or 'none'):",
                            value=row["misplaced_keywords"] if row["misplaced_keywords"] else "none",
                            help="List any intruder keywords that don't belong, or 'none'."
                        )

                        save_submitted = st.form_submit_button(
                            f"💾 Save Review for Cluster {row['cluster_id']}",
                            type="primary",
                            use_container_width=True
                        )

                        if save_submitted:
                            df_review.at[selected_idx, "rating_1_to_5"] = str(rating_val)
                            df_review.at[selected_idx, "reviewer_initials_or_role"] = reviewer_val.strip()
                            df_review.at[selected_idx, "reason"] = reason_val.strip()
                            df_review.at[selected_idx, "misplaced_keywords"] = misplaced_val.strip()
                            df_review.to_csv(HUMAN_REVIEW_CSV, index=False, encoding="utf-8")
                            st.success(f"✅ Successfully saved review for Cluster ID {row['cluster_id']} to `reports/human_review_sheet.csv`!")
                            st.rerun()

            # MODE 2: Full Table Editor
            else:
                st.markdown("---")
                st.markdown("#### Full Table View & Inline Cell Editor")
                st.caption("You can directly edit ratings, reviewer role, reasons, and misplaced keywords in the table below:")

                edited_df = st.data_editor(
                    df_review,
                    column_config={
                        "method": st.column_config.TextColumn("Method", disabled=True),
                        "cluster_id": st.column_config.NumberColumn("Cluster ID", disabled=True),
                        "cluster_size_unique": st.column_config.NumberColumn("Size", disabled=True),
                        "representative_keyword": st.column_config.TextColumn("Representative", disabled=True),
                        "members_shown": st.column_config.TextColumn("Members Shown", disabled=True, width="large"),
                        "rating_1_to_5": st.column_config.SelectboxColumn("Rating (1-5)", options=["1", "2", "3", "4", "5", ""], required=False),
                        "reviewer_initials_or_role": st.column_config.TextColumn("Reviewer Role"),
                        "reason": st.column_config.TextColumn("Reason", width="medium"),
                        "misplaced_keywords": st.column_config.TextColumn("Misplaced Keywords", width="medium"),
                    },
                    use_container_width=True,
                    num_rows="fixed",
                    key="review_sheet_data_editor"
                )

                eb_col1, eb_col2 = st.columns([1, 1])
                with eb_col1:
                    if st.button("💾 Save All Table Edits to File", type="primary", use_container_width=True):
                        edited_df.to_csv(HUMAN_REVIEW_CSV, index=False, encoding="utf-8")
                        st.success("✅ Successfully saved table updates to `reports/human_review_sheet.csv`!")
                        st.rerun()
                with eb_col2:
                    st.download_button(
                        label="📥 Download human_review_sheet.csv",
                        data=df_review.to_csv(index=False).encode("utf-8"),
                        file_name="human_review_sheet.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # --- TAB 5: Data Quality Audit ---
    with tab5:
        st.subheader("Data Quality Report")
        if "quality" in results and not results["quality"].empty:
            st.dataframe(results["quality"], use_container_width=True)
        else:
            st.info("No data quality report available.")

    # --- TAB 6: Download Deliverables ---
    with tab6:
        st.subheader("Download Section 16 Handover Files")
        st.markdown("Download all standardized output CSVs and JSON run metadata generated during this session:")

        c1, c2 = st.columns(2)
        with c1:
            if "assignments" in results and not results["assignments"].empty:
                st.download_button(
                    label="📄 Download keyword_assignments.csv",
                    data=results["assignments"].to_csv(index=False).encode("utf-8"),
                    file_name="keyword_assignments.csv",
                    mime="text/csv"
                )
            if "summary" in results and not results["summary"].empty:
                st.download_button(
                    label="📄 Download cluster_summary.csv",
                    data=results["summary"].to_csv(index=False).encode("utf-8"),
                    file_name="cluster_summary.csv",
                    mime="text/csv"
                )
            if "quality" in results and not results["quality"].empty:
                st.download_button(
                    label="📄 Download data_quality_report.csv",
                    data=results["quality"].to_csv(index=False).encode("utf-8"),
                    file_name="data_quality_report.csv",
                    mime="text/csv"
                )
        with c2:
            if "evaluation" in results and not results["evaluation"].empty:
                st.download_button(
                    label="📄 Download evaluation_summary.csv",
                    data=results["evaluation"].to_csv(index=False).encode("utf-8"),
                    file_name="evaluation_summary.csv",
                    mime="text/csv"
                )
            if "metadata" in results and results["metadata"]:
                st.download_button(
                    label="⚙️ Download run_metadata.json",
                    data=json.dumps(results["metadata"], indent=2).encode("utf-8"),
                    file_name="run_metadata.json",
                    mime="application/json"
                )
            if HUMAN_REVIEW_CSV.exists():
                st.download_button(
                    label="📝 Download human_review_sheet.csv",
                    data=HUMAN_REVIEW_CSV.read_bytes(),
                    file_name="human_review_sheet.csv",
                    mime="text/csv"
                )

else:
    st.info("👈 Select your dataset in the sidebar and click **'🚀 Run Clustering Pipeline'** to begin.")
