"""
src/annotate_app.py - Manual annotation tool for eval.xls (human annotator).
Course: NTU AI6127 (2026) — Question 4.

Shows one post at a time from the label-free pool (data/raw/eval_pool/candidates.jsonl, main + reserve splits,
pre-shuffled). The annotator picks the sentiment (Objective / Subjective-Positive / Subjective-Negative), ticks
sarcasm if it applies, then clicks the category button, which saves the row and moves to the next post.
Labels are appended to data/annotation/sheets/human_annotator.csv after every click, so the session can be
closed and resumed at any time. Stop once every class has >= 333 posts (counts are shown in the sidebar).

Run: streamlit run src/annotate_app.py
"""

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POOL_FILE = PROJECT_ROOT / "data" / "raw" / "eval_pool" / "candidates.jsonl"
OUT_FILE = PROJECT_ROOT / "data" / "annotation" / "sheets" / "human_annotator.csv"
GUIDELINES = PROJECT_ROOT / "data" / "annotation" / "ANNOTATION_GUIDELINES.md"
FIELDS = ["id", "text", "category", "subjectivity", "polarity", "is_sarcastic", "labelled_at"]
TARGET = 333
SENTIMENT = {"Objective (no opinion)": ("Objective", "None"),
             "Subjective – positive": ("Subjective", "Positive"),
             "Subjective – negative": ("Subjective", "Negative")}

st.set_page_config(page_title="XAI-Guard annotation", page_icon="🏷️", layout="centered")


@st.cache_data
def load_pool():
    with open(POOL_FILE, encoding="utf-8") as f:
        return [json.loads(l) for l in f if json.loads(l)["split"] in ("main", "reserve")]


def load_done():
    if not OUT_FILE.exists():
        return {}
    with open(OUT_FILE, encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def save(row):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = load_done()
    rows[row["id"]] = row
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow(r)


pool = load_pool()
done = load_done()
counts = Counter(r["category"] for r in done.values())

st.sidebar.header("Progress")
st.sidebar.metric("Labelled", f"{len(done)} / {len(pool)}")
for c in ("Safe", "Hate Speech", "Harassment-Offensive"):
    st.sidebar.progress(min(1.0, counts.get(c, 0) / TARGET), text=f"{c}: {counts.get(c, 0)} / {TARGET}")
if all(counts.get(c, 0) >= TARGET for c in ("Safe", "Hate Speech", "Harassment-Offensive")):
    st.sidebar.success("Every class has ≥ 333 posts — you can stop (more labels are fine).")
with st.sidebar.expander("Guidelines"):
    st.markdown(GUIDELINES.read_text() if GUIDELINES.exists() else "")

if "redo" in st.session_state and st.session_state["redo"]:
    todo = [p for p in pool if p["id"] == st.session_state["redo"]]
else:
    todo = [p for p in pool if p["id"] not in done]
if not todo:
    st.success("All posts labelled.")
    st.stop()
post = todo[0]

st.caption(f"Post {post['id']}")
st.markdown(f"<div style='font-size:1.35em;padding:14px;border:1px solid #ccc;border-radius:8px'>{post['text']}</div>",
            unsafe_allow_html=True)
st.write("")
sentiment = st.radio("1. Sentiment", list(SENTIMENT), index=None, horizontal=True, key=f"s_{post['id']}")
sarcastic = st.checkbox("Sarcastic (literal meaning is the opposite of the intended one)", key=f"x_{post['id']}")
st.write("2. Category (saves and goes to the next post)")
cols = st.columns(3)
for col, cat in zip(cols, ("Safe", "Harassment-Offensive", "Hate Speech")):
    if col.button(cat, use_container_width=True, key=f"{cat}_{post['id']}"):
        if sentiment is None:
            st.warning("Pick the sentiment first.")
        else:
            subj, pol = SENTIMENT[sentiment]
            save({"id": post["id"], "text": post["text"], "category": cat, "subjectivity": subj, "polarity": pol,
                  "is_sarcastic": "Yes" if sarcastic else "No",
                  "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            st.session_state["last"] = post["id"]
            st.session_state["redo"] = None
            st.rerun()

if st.session_state.get("last"):
    if st.button(f"↩ Redo previous post ({st.session_state['last']})"):
        st.session_state["redo"] = st.session_state["last"]
        st.rerun()
