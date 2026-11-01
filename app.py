"""
app.py - XAI-Guard Streamlit Interactive Cockpit
Course: NTU AI6127 (2026) — Question 2 Deliverable.

An Explainable, Retrieval-Augmented Safety Moderation system conditioning decisions
on statutory policy clauses (Reddit, Meta, X, Singapore IMDA) and historical precedents
(HateXplain) with SenticNet affective primitives and live latency telemetry (ms).
"""

import json
import os
import sys
import time
from pathlib import Path

# Force stable threading on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalize import normalize_text
from src.retriever import DualChannelRetriever, RetrievalConfig
from src.sentic import extract_sentic_features

# Set page configuration
st.set_page_config(
    page_title="XAI-Guard: Explainable Safety RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Benchmark queries for Question 2
SAMPLE_QUERIES = {
    "Select a benchmark query...": "",
    "Q1: Explicit Hate / Xenophobia": "go back to your own country, nobody wants you here",
    "Q2: Implicit Sarcasm / Stereotyping": "Oh wow, another DEI hire pretending to understand engineering, truly inspiring.",
    "Q3: Benign Identity / Counter-speech": "I am proud to be a gay man and no one can take my dignity away.",
    "Q4: Obfuscated Leetspeak Harassment": "u r so d*mb, go k1ll urself u worthless piece of tr@sh",
    "Q5: Neutral Everyday Conversation": "Can someone recommend a good beginner tutorial for Python and machine learning?",
}


@st.cache_resource(show_spinner="Initializing DualChannelRetriever & SenticNet models...")
def load_resources():
    retriever = DualChannelRetriever()
    cfg_policy = RetrievalConfig(top_k=3, use_bm25=True, use_dense=True, use_rrf=True, use_rerank=True)
    cfg_prec = RetrievalConfig(top_k=5, use_bm25=True, use_dense=True, use_rrf=True, use_rerank=True)
    return retriever, cfg_policy, cfg_prec


@st.cache_data
def load_corpus_stats():
    stats_file = PROJECT_ROOT / "data" / "processed" / "corpus_stats.json"
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "total_documents": 17322,
        "total_words": 407736,
        "unique_vocab_types": 25973
    }


def main():
    retriever, cfg_policy, cfg_prec = load_resources()
    stats = load_corpus_stats()

    # --- Sidebar ---
    st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=70)
    st.sidebar.title("XAI-Guard")
    st.sidebar.caption("Explainable Safety Moderation RAG\nNTU AI6127 (2026)")
    st.sidebar.markdown("---")

    st.sidebar.subheader("📊 Knowledge Base Scale")
    st.sidebar.metric("Total Passages", f"{stats['total_documents']:,}")
    st.sidebar.metric("Total Word Count", f"{stats['total_words']:,}")
    st.sidebar.metric("Unique Vocab Types", f"{stats['unique_vocab_types']:,}")
    st.sidebar.markdown("---")

    st.sidebar.subheader("⚙️ Retrieval Settings")
    use_rerank = st.sidebar.checkbox("Enable Cross-Encoder Re-ranker", value=True)
    top_k_prec = st.sidebar.slider("Channel B Precedents (K)", min_value=1, max_value=10, value=5)
    cfg_prec.use_rerank = use_rerank
    cfg_prec.top_k = top_k_prec

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**XAI-Guard** conditions all decisions on **Channel A** (Statutory & Platform Policy Rules) and "
        "**Channel B** (17k Precedent Cases from HateXplain) with SenticNet 7 neurosymbolic primitives."
    )

    # --- Main Header ---
    st.title("🛡️ XAI-Guard: Explainable Safety Cockpit")
    st.markdown(
        "An explainable, neurosymbolic Retrieval-Augmented Generation system for content moderation. "
        "Every moderation decision is grounded in codified policies and human-adjudicated precedents."
    )

    # Benchmark query selector
    col_bench, col_clear = st.columns([4, 1])
    with col_bench:
        selected_bench = st.selectbox(
            "⚡ Quick-Load Benchmark Queries (Question 2):",
            list(SAMPLE_QUERIES.keys()),
            index=0
        )
    with col_clear:
        st.write("")
        st.write("")
        if st.button("Reset Input"):
            st.session_state["query_input"] = ""

    default_text = SAMPLE_QUERIES[selected_bench] if selected_bench != "Select a benchmark query..." else ""
    if "query_input" not in st.session_state or default_text:
        st.session_state["query_input"] = default_text

    user_text = st.text_area(
        "Enter text to evaluate for safety violations:",
        value=st.session_state["query_input"],
        height=100,
        placeholder="Type or paste social media text here..."
    )

    if st.button("🛡️ Run Safety Audit", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text to inspect.")
            return

        with st.spinner("Executing Microtext Normalization, Dual-Channel Hybrid Search, and SenticNet Analysis..."):
            # 1. Normalization
            t0 = time.perf_counter()
            cleaned_text = normalize_text(user_text)
            t_norm = (time.perf_counter() - t0) * 1000.0

            # 2. SenticNet Features
            t0 = time.perf_counter()
            sentic_prof = extract_sentic_features(cleaned_text)
            t_sentic = (time.perf_counter() - t0) * 1000.0

            # 3. Channel A: Policy Retrieval
            t0 = time.perf_counter()
            policy_hits = retriever.retrieve(cleaned_text, channel="policy", config=cfg_policy)
            t_policy = (time.perf_counter() - t0) * 1000.0

            # 4. Channel B: Precedent Retrieval
            t0 = time.perf_counter()
            prec_hits = retriever.retrieve(cleaned_text, channel="precedent", config=cfg_prec)
            t_prec = (time.perf_counter() - t0) * 1000.0

            total_ms = t_norm + t_sentic + t_policy + t_prec

        # Compute Precedent Consensus
        labels = [p.category_or_section for p in prec_hits if p.category_or_section]
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        majority_label = max(counts, key=counts.get) if counts else "normal"
        consensus_ratio = (counts.get(majority_label, 0) / len(labels)) if labels else 0.0

        # Latency Telemetry Row
        st.markdown("### ⏱️ Live Latency Telemetry")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Latency", f"{total_ms:.1f} ms")
        m2.metric("Normalization", f"{t_norm:.2f} ms")
        m3.metric("SenticNet 7", f"{t_sentic:.2f} ms")
        m4.metric("Channel A (Policy)", f"{t_policy:.1f} ms")
        m5.metric("Channel B (Precedent)", f"{t_prec:.1f} ms")

        st.markdown("---")

        # Top Verdict Row
        v_col1, v_col2, v_col3 = st.columns([2, 2, 2])

        with v_col1:
            st.markdown("#### Primary Decision")
            if majority_label == "hatespeech":
                st.error("🚨 **HATE SPEECH DETECTED**")
            elif majority_label in ["offensive", "harassment"]:
                st.warning("⚠️ **OFFENSIVE / HARASSMENT**")
            else:
                st.success("✅ **SAFE / NORMAL CONTENT**")
            st.caption(f"Normalized Input: *\"{cleaned_text}\"*")

        with v_col2:
            st.markdown("#### Evidence Consensus")
            pct = int(consensus_ratio * 100)
            st.progress(consensus_ratio)
            st.write(f"**{pct}% Precedent Consensus** ({counts.get(majority_label, 0)} of {len(labels)} precedents agree)")
            if consensus_ratio < 0.60:
                st.info("⚠️ Low consensus — critic flags for human review.")

        with v_col3:
            st.markdown("#### Affective Polarity")
            st.write(f"**Dominant Emotion:** {sentic_prof.dominant_emotion}")
            st.write(f"**Polarity Score:** `{sentic_prof.polarity:.2f}` (Range: -1.0 to +1.0)")
            if sentic_prof.matched_concepts:
                st.write(f"**Trigger Concepts:** {', '.join(sentic_prof.matched_concepts)}")
            else:
                st.write("**Trigger Concepts:** None detected")

        st.markdown("---")

        # Evidence Tabs
        tab_policy, tab_precedents, tab_sentic = st.tabs([
            "📜 Channel A: Policy Grounding",
            "👥 Channel B: Precedent Grounding",
            "🧠 Neurosymbolic Affective Radar",
        ])

        with tab_policy:
            st.markdown("#### Retrieved Statutory & Platform Policy Clauses")
            st.write(
                "These codified rules were retrieved from the Knowledge Base via hybrid BM25 + dense FAISS + re-ranking:"
            )
            for i, hit in enumerate(policy_hits, 1):
                with st.expander(f"Clause {i}: [{hit.source.upper()}] {hit.category_or_section} (Relevance Score: {hit.score})", expanded=(i == 1)):
                    st.markdown(f"**Clause Text:**\n> {hit.text}")
                    st.markdown(f"**Identifier:** `{hit.id}` | **Length:** {hit.word_count} words")

        with tab_precedents:
            st.markdown(f"#### Retrieved Top-{len(prec_hits)} Human-Adjudicated Precedents (HateXplain)")
            st.write(
                "Similar real-world social media cases showing how human annotators adjudicated comparable posts:"
            )
            for i, hit in enumerate(prec_hits, 1):
                lbl = hit.category_or_section.upper()
                badge = "🔴 HATE" if "HATE" in lbl else ("🟡 OFFENSIVE" if "OFFENSIVE" in lbl else "🟢 NORMAL")
                with st.container():
                    st.markdown(f"**{i}. {badge}** (Match Score: `{hit.score}`) — Post ID: `{hit.id}`")
                    st.markdown(f"> *\"{hit.text}\"*")
                    st.caption(f"Annotated Targets: `{hit.title_or_targets}`")
                    st.markdown("")

        with tab_sentic:
            st.markdown("#### SenticNet 7 Hourglass of Emotions")
            st.write(
                "Auxiliary neurosymbolic affective features decomposed into four affective dimensions:"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pleasantness (Attitude)", f"{sentic_prof.pleasantness:+.2f}")
            c2.metric("Attention (Introspection)", f"{sentic_prof.attention:+.2f}")
            c3.metric("Sensitivity (Fear)", f"{sentic_prof.sensitivity:+.2f}")
            c4.metric("Aptitude (Temper/Rage)", f"{sentic_prof.aptitude:+.2f}")

            st.markdown(f"**Affective Synthesis:** {sentic_prof.summary_text}")


if __name__ == "__main__":
    main()
