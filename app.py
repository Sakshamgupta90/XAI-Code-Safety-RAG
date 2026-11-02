"""
app.py - XAI-Guard Streamlit UI
Course: NTU AI6127 (2026) — Question 2 Deliverable.

Paste a post and get an explainable moderation decision:
- a verdict card (Safe / Hate Speech / Harassment-Offensive, or "Needs human review" when the critic abstains)
- the post with its trigger words highlighted, a short reason, and a respectful rewrite
- tab "Matching rules": the policy clauses retrieved from Meta / X / Reddit / YouTube / IMDA, the cited one first,
  each with a link to the official page
- tab "Similar past cases": the closest human-labelled posts (HateXplain) and how they were labelled
- tab "How it decided": reasoning steps, the critic's checks, SenticNet emotions and per-stage timing

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

st.set_page_config(page_title="XAI-Guard", page_icon="🛡️", layout="centered")

EXAMPLES = {
    "Xenophobia": "go back to your own country, nobody wants you here",
    "Sarcasm": "Oh wow, another DEI hire pretending to understand engineering, truly inspiring.",
    "Benign identity": "I am proud to be a gay man and no one can take my dignity away.",
    "Obfuscated abuse": "u r so d*mb, go k1ll urself u worthless piece of tr@sh",
    "Neutral": "Can someone recommend a good beginner tutorial for Python and machine learning?",
}
VERDICT_STYLE = {
    "Safe": ("#047857", "#ecfdf5", "✓", "No policy violation found"),
    "Hate Speech": ("#b91c1c", "#fef2f2", "!", "Attacks people for who they are"),
    "Harassment-Offensive": ("#b45309", "#fffbeb", "!", "Insulting, abusive or threatening"),
    "Needs Human Review": ("#4338ca", "#eef2ff", "?", "Evidence does not support a confident verdict"),
}
LABEL_CHIP = {"Safe": ("#047857", "#d1fae5"), "Hate Speech": ("#b91c1c", "#fee2e2"),
              "Harassment-Offensive": ("#b45309", "#fef3c7")}
SOURCE_CHIP = {"Meta": "#1d4ed8", "X": "#111827", "Reddit": "#ea580c", "YouTube": "#dc2626", "Singapore IMDA": "#0f766e"}

CSS = """
<style>
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 820px; }
.hero h1 { font-size: 2.0rem; font-weight: 750; letter-spacing: -0.02em; margin: 0; color: #0f172a; }
.hero p { color: #64748b; margin: 4px 0 18px 0; font-size: 1.0rem; }
.card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 18px; margin: 10px 0;
        box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.verdict { border-radius: 16px; padding: 18px 20px; margin: 18px 0 8px 0; display: flex; gap: 16px; align-items: center; }
.verdict .icon { width: 46px; height: 46px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
                 font-size: 1.5rem; font-weight: 800; color: white; flex-shrink: 0; }
.verdict .title { font-size: 1.45rem; font-weight: 750; line-height: 1.2; }
.verdict .sub { color: #475569; font-size: .95rem; margin-top: 2px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px 0; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px; font-size: .82rem;
        background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0; white-space: nowrap; }
.chip b { color: #0f172a; font-weight: 650; }
.label { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 650; }
.src { display: inline-block; padding: 2px 9px; border-radius: 6px; font-size: .75rem; font-weight: 700; color: white; }
.section-title { font-size: .78rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: .06em; margin: 18px 0 6px; }
.post { font-size: 1.12rem; line-height: 1.6; color: #0f172a; }
.post mark { background: #fde68a; padding: 1px 3px; border-radius: 4px; }
.reason { color: #334155; line-height: 1.6; }
.rewrite { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 12px 16px; color: #14532d; }
.rule-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.rule-title { font-weight: 650; color: #0f172a; margin: 8px 0 4px; }
.rule-text { color: #475569; font-size: .92rem; line-height: 1.55; }
.cited { background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe; padding: 2px 9px; border-radius: 999px; font-size: .75rem; font-weight: 700; }
.case-text { color: #1e293b; font-size: .95rem; line-height: 1.5; margin-top: 6px; }
.muted { color: #94a3b8; font-size: .8rem; }
a.link { color: #4338ca; text-decoration: none; font-size: .85rem; font-weight: 600; }
.stButton > button[kind="primary"] { border-radius: 10px; padding: .55rem 1.4rem; font-weight: 650; }
.stTextArea textarea { border-radius: 12px; font-size: 1.02rem; }
</style>
"""


@st.cache_resource(show_spinner="Loading search indexes and models (first run only)…")
def load_reasoner():
    reasoner = ExplainableSafetyReasoner()
    _ = reasoner.retriever
    return reasoner


@st.cache_data
def load_policy_meta():
    meta = {}
    with open(PROJECT_ROOT / "data" / "processed" / "policy_chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta[d["id"]] = d
    return meta


def esc(text) -> str:
    return html.escape(str(text or ""))


def highlight(text: str, triggers) -> str:
    out = esc(text)
    for t in sorted(set(triggers), key=len, reverse=True):
        if t.strip():
            out = re.sub(re.escape(esc(t)), lambda m: f"<mark>{m.group(0)}</mark>", out, flags=re.IGNORECASE)
    return out


def label_chip(label: str) -> str:
    fg, bg = LABEL_CHIP.get(label, ("#334155", "#f1f5f9"))
    return f"<span class='label' style='color:{fg};background:{bg}'>{esc(label)}</span>"


def render_result(text: str, out, show_details: bool):
    final = out.final_verdict
    colour, bg, icon, sub = VERDICT_STYLE.get(final, VERDICT_STYLE["Needs Human Review"])
    llm_note = f" · model's verdict was <b>{esc(out.category)}</b>" if final != out.category else ""
    st.markdown(f"""<div class='verdict' style='background:{bg};border:1px solid {colour}33'>
        <div class='icon' style='background:{colour}'>{icon}</div>
        <div><div class='title' style='color:{colour}'>{esc(final)}</div>
        <div class='sub'>{sub}{llm_note}</div></div></div>""", unsafe_allow_html=True)

    pc = out.precedent_consensus
    st.markdown(f"""<div class='chips'>
        <span class='chip'>Tone <b>{esc(out.subjectivity)}{'' if out.polarity == 'None' else ' · ' + esc(out.polarity)}</b></span>
        <span class='chip'>Sarcasm <b>{'Yes' if out.is_sarcastic else 'No'}</b></span>
        <span class='chip'>Target <b>{esc(out.target_group if out.target_group.lower() != 'none' else '—')}</b></span>
        <span class='chip'>Similar cases agreeing <b>{pc['confidence']:.0%}</b></span>
        <span class='chip'>Evidence check <b>{'passed' if not out.critic_decision['requires_human_review'] else 'flagged'}</b></span>
        </div>""", unsafe_allow_html=True)
    if out.critic_decision["abstention_reason"]:
        st.info("Sent to a human moderator: " + out.critic_decision["abstention_reason"])

    st.markdown("<div class='section-title'>The post</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card post'>{highlight(text, out.trigger_words)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Why</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='reason'>{esc(out.explanation)}</div>", unsafe_allow_html=True)
    if out.detox_rephrase and final != "Safe" and out.category != "Safe":
        st.markdown("<div class='section-title'>A respectful way to say it</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='rewrite'>{esc(out.detox_rephrase)}</div>", unsafe_allow_html=True)

    st.write("")
    tab_rules, tab_cases, tab_how = st.tabs(["📜 Matching rules", "👥 Similar past cases", "🔍 How it decided"])
    meta = load_policy_meta()
    with tab_rules:
        st.caption("Official platform and government rules found for this post (searched with a policy-style rewrite of it).")
        rules = sorted(out.retrieved_policies, key=lambda p: p["id"] != out.cited_clause)
        for p in rules:
            m = meta.get(p["id"], {})
            words = p["text"].split(": ", 1)[-1].split()
            excerpt = " ".join(words[:55]) + (" …" if len(words) > 55 else "")
            cited = "<span class='cited'>Cited in the decision</span>" if p["id"] == out.cited_clause else ""
            st.markdown(f"""<div class='card'>
                <div class='rule-head'><span class='src' style='background:{SOURCE_CHIP.get(p['source'], '#334155')}'>{esc(p['source'])}</span>{cited}</div>
                <div class='rule-title'>{esc(m.get('title', ''))} — {esc(p['category_or_section'])}</div>
                <div class='rule-text'>{esc(excerpt)}</div>
                <div style='margin-top:8px'><a class='link' href='{esc(m.get('url', ''))}' target='_blank'>Read the official policy ↗</a></div>
                </div>""", unsafe_allow_html=True)
    with tab_cases:
        st.caption(f"The 5 most similar posts that human annotators already labelled (HateXplain). "
                   f"Most common label: {pc['label']} ({pc['confidence']:.0%}).")
        for p in out.retrieved_precedents:
            targets = ", ".join(p["title_or_targets"]) if p["title_or_targets"] else "no target"
            cited = " <span class='cited'>Cited</span>" if p["id"] in out.cited_precedents else ""
            st.markdown(f"""<div class='card'>{label_chip(p['category_or_section'])}{cited}
                <span class='muted'>&nbsp; targets: {esc(targets)}</span>
                <div class='case-text'>{esc(p['text'][:260])}</div></div>""", unsafe_allow_html=True)
    with tab_how:
        if out.reasoning:
            st.markdown("**Step-by-step reasoning**")
            for k, v in out.reasoning.items():
                st.markdown(f"- **{k.split('_', 1)[-1].replace('_', ' ').capitalize()}**: {v}")
        st.markdown("**Evidence check (critic)**")
        crit = out.critic_decision
        st.markdown(f"- Citations point to retrieved evidence: {'✅' if crit['citation_valid'] else '❌'}  \n"
                    f"- Claim checked against the post: *{esc(out.violation_claim or '—')}*  \n"
                    f"- NLI scores: {', '.join(f'{k} {v:.2f}' for k, v in crit['nli_scores'].items()) or '—'}")
        s = out.sentic_profile
        st.markdown(f"**Emotions (SenticNet 9)**: polarity {s['polarity']:+.2f}, dominant *{s['dominant_emotion']}*; "
                    f"introspection {s['introspection']:+.2f}, temper {s['temper']:+.2f}, attitude {s['attitude']:+.2f}, "
                    f"sensitivity {s['sensitivity']:+.2f}")
        lat = out.latency_ms
        st.markdown("**Timing**: " + " · ".join(f"{k.replace('_ms', '').replace('_', ' ')} {v / 1000:.1f}s"
                                                for k, v in lat.items()))
        if show_details and out.hyde_query:
            st.markdown(f"**Policy-search query written by the model (HyDE)**: {out.hyde_query}")


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class='hero'><h1>🛡️ XAI-Guard</h1>
        <p>Paste a social-media post. Get a moderation decision, the rule behind it, and similar cases humans already judged.</p></div>""",
                unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Settings")
        cfg = ModerationConfig(
            use_hyde=st.toggle("Policy-style search (HyDE)", value=True),
            use_sentic=st.toggle("Emotion features (SenticNet)", value=True),
            use_cot=st.toggle("Step-by-step reasoning", value=True),
            use_critic=st.toggle("Evidence check (may defer to a human)", value=True),
        )
        show_details = st.toggle("Show technical details", value=False)
        st.caption("Knowledge base: 142 clauses from 20 official policy pages (Meta, X, Reddit, YouTube, Singapore IMDA) "
                   "and 17,305 human-labelled posts (HateXplain).")

    choice = st.pills("Try an example", list(EXAMPLES), selection_mode="single")
    text = st.text_area("Post", value=EXAMPLES.get(choice, ""), height=110, label_visibility="collapsed",
                        placeholder="Paste a post to check…")
    if st.button("Check post", type="primary", disabled=not text.strip()):
        reasoner = load_reasoner()
        with st.spinner("Searching rules and similar cases, asking the model, checking the evidence…"):
            st.session_state["result"] = (text, reasoner.reason(text, cfg))
    if "result" in st.session_state:
        render_result(*st.session_state["result"], show_details=show_details)


if __name__ == "__main__":
    main()
