"""
app.py - XAI-Guard Streamlit UI
Course: NTU AI6127 (2026) — Question 2 Deliverable.

Paste a post and get an explainable moderation decision:
- verdict (Safe / Hate Speech / Harassment-Offensive, or "Needs Human Review" when the critic abstains)
- the policy clause it violates, with a link to the official source
- the most similar human-labelled precedents (HateXplain) and their consensus
- the trigger words highlighted in the post, subjectivity / polarity / sarcasm, SenticNet emotions
- a respectful rewrite suggestion, the critic's checks, and per-stage latency

Run: streamlit run app.py
"""

import html
import json
import os
import re
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import ExplainableSafetyReasoner, ModerationConfig

st.set_page_config(page_title="XAI-Guard", page_icon="🛡️", layout="wide")

SAMPLE_QUERIES = {
    "— choose an example —": "",
    "Q1 Explicit xenophobia": "go back to your own country, nobody wants you here",
    "Q2 Sarcastic stereotyping": "Oh wow, another DEI hire pretending to understand engineering, truly inspiring.",
    "Q3 Benign identity statement": "I am proud to be a gay man and no one can take my dignity away.",
    "Q4 Obfuscated harassment": "u r so d*mb, go k1ll urself u worthless piece of tr@sh",
    "Q5 Neutral question": "Can someone recommend a good beginner tutorial for Python and machine learning?",
}
VERDICT_COLOURS = {"Safe": "#1b7f3b", "Hate Speech": "#b3261e", "Harassment-Offensive": "#c26a00",
                   "Needs Human Review": "#5b5b8f"}


@st.cache_resource(show_spinner="Loading retriever, re-ranker and NLI models…")
def load_reasoner():
    reasoner = ExplainableSafetyReasoner()
    _ = reasoner.retriever
    return reasoner


@st.cache_data
def load_policy_urls():
    urls = {}
    with open(PROJECT_ROOT / "data" / "processed" / "policy_chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            urls[d["id"]] = (d["url"], d["title"])
    return urls


@st.cache_data
def load_corpus_stats():
    return json.loads((PROJECT_ROOT / "data" / "processed" / "corpus_stats.json").read_text())


def highlight(text: str, triggers):
    out = html.escape(text)
    for t in sorted(set(triggers), key=len, reverse=True):
        if t.strip():
            out = re.sub(re.escape(html.escape(t)), lambda m: f"<mark>{m.group(0)}</mark>", out, flags=re.IGNORECASE)
    return out


def main():
    stats = load_corpus_stats()
    st.sidebar.title("🛡️ XAI-Guard")
    st.sidebar.caption("Explainable retrieval-augmented content moderation · NTU AI6127")
    st.sidebar.metric("Indexed chunks", f"{stats['total_chunks']:,}")
    st.sidebar.caption(f"{stats['policy_clauses_count']} policy clauses from {stats['policy_pages_count']} official pages · "
                       f"{stats['precedent_posts_count']:,} human-labelled precedents")
    st.sidebar.subheader("Pipeline switches")
    cfg = ModerationConfig(
        use_hyde=st.sidebar.checkbox("HyDE policy query", value=True),
        use_sentic=st.sidebar.checkbox("SenticNet features", value=True),
        use_cot=st.sidebar.checkbox("Suitcase chain-of-thought", value=True),
        use_critic=st.sidebar.checkbox("Grounding critic (may abstain)", value=True),
    )

    st.title("XAI-Guard: why is this post (not) allowed?")
    choice = st.selectbox("Example posts", list(SAMPLE_QUERIES))
    text = st.text_area("Post to moderate", value=SAMPLE_QUERIES[choice], height=90,
                        placeholder="Paste a social-media post…")
    if not st.button("Moderate", type="primary") or not text.strip():
        return

    reasoner = load_reasoner()
    with st.spinner("Retrieving policies and precedents, asking the LLM judge, checking the evidence…"):
        out = reasoner.reason(text, cfg)

    final = out.final_verdict
    colour = VERDICT_COLOURS.get(final, "#444")
    st.markdown(f"<div style='padding:14px;border-radius:8px;background:{colour};color:white;font-size:1.4em'>"
                f"<b>{final}</b>{'' if final == out.category else f' &nbsp;(LLM verdict: {out.category})'}</div>",
                unsafe_allow_html=True)
    if out.critic_decision["abstention_reason"]:
        st.warning("Sent to human review — " + out.critic_decision["abstention_reason"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjectivity", out.subjectivity)
    c2.metric("Polarity", out.polarity)
    c3.metric("Sarcastic", "Yes" if out.is_sarcastic else "No")
    pc = out.precedent_consensus
    c4.metric("Precedent agreement", f"{pc['confidence']:.0%}", help=f"Majority label of the 5 nearest precedents: {pc['label']}")

    st.markdown("**Post (trigger words highlighted)**")
    st.markdown(f"<div style='font-size:1.1em'>{highlight(text, out.trigger_words)}</div>", unsafe_allow_html=True)
    st.markdown(f"**Why:** {out.explanation}")
    if out.violation_claim:
        st.caption(f"Claim checked by the critic: {out.violation_claim}  ·  target: {out.target_group}")
    if out.detox_rephrase and final != "Safe":
        st.info(f"**Suggested rewrite:** {out.detox_rephrase}")

    left, right = st.columns(2)
    urls = load_policy_urls()
    with left:
        st.subheader("Policy clauses (Channel A)")
        for p in out.retrieved_policies:
            url, title = urls.get(p["id"], ("", p["title_or_targets"]))
            cited = " ✅ cited" if p["id"] == out.cited_clause else ""
            with st.expander(f"{p['source']} — {p['category_or_section']}{cited}", expanded=bool(cited)):
                st.write(" ".join(p["text"].split()[:180]) + " …")
                st.caption(f"[{title}]({url}) · id `{p['id']}` · re-rank score {p['score']:.2f}")
        if out.hyde_query:
            st.caption(f"HyDE search query: {out.hyde_query}")
    with right:
        st.subheader("Similar labelled posts (Channel B)")
        for p in out.retrieved_precedents:
            cited = " ✅ cited" if p["id"] in out.cited_precedents else ""
            targets = ", ".join(p["title_or_targets"]) if p["title_or_targets"] else "none"
            st.markdown(f"**{p['category_or_section']}**{cited} · targets: {targets}  \n> {html.escape(p['text'][:220])}")

    with st.expander("SenticNet 9 emotions, reasoning steps, critic checks and latency"):
        s = out.sentic_profile
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Polarity", f"{s['polarity']:+.2f}")
        e2.metric("Introspection", f"{s['introspection']:+.2f}")
        e3.metric("Temper", f"{s['temper']:+.2f}")
        e4.metric("Attitude", f"{s['attitude']:+.2f}")
        e5.metric("Sensitivity", f"{s['sensitivity']:+.2f}")
        st.caption(f"Dominant emotion: {s['dominant_emotion']} · concepts: {', '.join(s['matched_concepts']) or 'none'}")
        if out.reasoning:
            st.json(out.reasoning)
        st.json(out.critic_decision)
        st.json(out.latency_ms)


if __name__ == "__main__":
    main()
