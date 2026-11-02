"""
src/data_prep.py
Corpus Ingestion, Structural Chunking, and Corpus Statistics Calculation.
Course: NTU AI6127 (2026) — Question 1 Deliverables.

Actions:
1. Downloads HateXplain dataset and split divisions.
2. Ingests the policy pages collected by src/collect_policies.py (Meta, X, Reddit, YouTube, Singapore IMDA).
3. Performs structure-aware chunking: split by section heading, then pack paragraphs to ~200 words.
4. Normalizes all text using src/normalize.py.
5. Saves:
   - data/processed/policy_chunks.jsonl (Channel A Rulebook)
   - data/processed/precedent_docs.jsonl (Channel B HateXplain train+val)
   - data/processed/hatexplain_test.jsonl (Held-out test split reserved for rationale evaluation)
   - data/processed/corpus_stats.json (Verified documents, chunks, words, and unique type count)
"""

import json
import os
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.normalize import normalize_text
except ImportError:
    from normalize import normalize_text

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Source URLs for HateXplain official repository
HATEXPLAIN_DATA_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
HATEXPLAIN_SPLITS_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"

def download_file_with_retry(url: str, target_path: Path, max_retries: int = 3) -> bool:
    """Downloads a file over HTTP with retry logic."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 1000:
        print(f"[OK] Cache found: {target_path} ({target_path.stat().st_size} bytes)")
        return True

    print(f"Downloading {url} -> {target_path} ...")
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
                out_file.write(response.read())
            print(f"[OK] Downloaded successfully: {target_path} ({target_path.stat().st_size} bytes)")
            return True
        except Exception as e:
            print(f"[WARN] Attempt {attempt}/{max_retries} failed for {url}: {e}")
    return False


# Lines that mark the end of the policy body (site footers, related links, feedback widgets)
POLICY_FOOTER_MARKERS = (
    "We have the same policies around the world",
    "## >x< Platform", "## \\>x\\< Platform",
    "Was this helpful", "Need more help", "Related articles", "Give feedback about this article",
    "Read more about", "Other resources",
)
# Navigation / change-log lines that carry no policy content
POLICY_NOISE = re.compile(
    r"^(CHANGE LOG|Today|Policy details|User Experiences|Data|"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}|\*\*[A-Z][a-z]+ \d{4}\*\*)$"
)
CHUNK_TARGET_WORDS = 200
CHUNK_MIN_WORDS = 30


def parse_policy_markdown(path: Path) -> Tuple[Dict[str, str], List[Tuple[str, List[str]]]]:
    """
    Splits a collected policy file into (metadata, [(section heading, paragraphs)]).
    Markdown headings and short title-like lines (e.g. 'Tier 1', 'Do not post:') open a new section.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    meta: Dict[str, str] = {"title": lines[0].lstrip("# ").strip()}
    body_start = 1
    for i, line in enumerate(lines[1:6], 1):
        m = re.match(r"^(Source|Source URL|Retrieved|Terms): (.*)$", line)
        if m:
            meta[m.group(1).lower().replace(" ", "_")] = m.group(2).strip()
            body_start = i + 1

    sections: List[Tuple[str, List[str]]] = []
    heading, paras, buf = meta["title"], [], []

    def flush_para():
        if buf:
            paras.append(" ".join(buf).strip())
            buf.clear()

    for raw in lines[body_start:]:
        line = raw.strip()
        if any(line.startswith(m) for m in POLICY_FOOTER_MARKERS):
            break
        if not line:
            flush_para()
            continue
        if POLICY_NOISE.match(line):
            continue
        plain = re.sub(r"[*#>\\]", "", line).strip()
        is_md_heading = line.startswith("#")
        is_title_like = (
            len(plain.split()) <= 8 and not re.search(r"[.;,]$", plain)
            and plain[:1].isupper() and not line.startswith(("-", "*", "|"))
        )
        if is_md_heading or is_title_like:
            flush_para()
            if paras:
                sections.append((heading, paras))
                paras = []
            if plain:
                heading = plain
            continue
        buf.append(re.sub(r"\*\*|\\", "", line.lstrip("-* ").strip()) if line.startswith(("- ", "* ")) else re.sub(r"\*\*|\\", "", line))
    flush_para()
    if paras:
        sections.append((heading, paras))
    return meta, sections


def chunk_policy_sections(sections: List[Tuple[str, List[str]]]) -> List[Tuple[str, str]]:
    """
    Packs paragraphs of each section into passages of about CHUNK_TARGET_WORDS words.
    Sections shorter than CHUNK_MIN_WORDS are merged into the following section so that
    every chunk is a self-contained, citable clause.
    """
    chunks: List[Tuple[str, str]] = []
    carry_heading, carry_text = None, ""
    for heading, paras in sections:
        text = " ".join(paras)
        if carry_text:
            heading = f"{carry_heading} / {heading}"
            paras = [carry_text] + paras
            carry_heading, carry_text = None, ""
        if len(" ".join(paras).split()) < CHUNK_MIN_WORDS:
            carry_heading, carry_text = heading, " ".join(paras)
            continue
        current: List[str] = []
        for para in paras:
            current.append(para)
            if len(" ".join(current).split()) >= CHUNK_TARGET_WORDS:
                chunks.append((heading, " ".join(current)))
                current = []
        if current:
            tail = " ".join(current)
            if len(tail.split()) < CHUNK_MIN_WORDS and chunks and chunks[-1][0] == heading:
                chunks[-1] = (heading, chunks[-1][1] + " " + tail)
            else:
                chunks.append((heading, tail))
    if carry_text and chunks:
        chunks[-1] = (chunks[-1][0], chunks[-1][1] + f" {carry_heading}: {carry_text}")
    return chunks


def build_policy_corpus() -> List[Dict[str, Any]]:
    """
    Chunks the collected policy pages (src/collect_policies.py) into citable clauses.
    Structure-aware: split on section headings first, then pack paragraphs to ~200 words.
    """
    print("\n--- Ingesting & Chunking Policy Corpus ---")
    policy_dir = DATA_RAW / "policies"
    files = sorted(policy_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No policy files in {policy_dir}. Run src/collect_policies.py first.")

    policy_chunks = []
    for path in files:
        meta, sections = parse_policy_markdown(path)
        for idx, (heading, text) in enumerate(chunk_policy_sections(sections), 1):
            cleaned_text = normalize_text(text)
            policy_chunks.append({
                "id": f"{path.stem}_{idx:03d}",
                "source": meta.get("source", ""),
                "title": meta["title"],
                "section": heading,
                "url": meta.get("source_url", ""),
                "retrieved_on": meta.get("retrieved", ""),
                "text": f"{heading}: {cleaned_text}",
                "word_count": len(cleaned_text.split()),
            })

    by_source = Counter(c["source"] for c in policy_chunks)
    print(f"[OK] Processed {len(policy_chunks)} policy clauses from {len(files)} pages: {dict(by_source)}")
    return policy_chunks


def resolve_hatexplain_majority_label(annotators: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """
    Computes majority label from annotators.
    HateXplain uses labels: 'normal', 'offensive', 'hatespeech'.
    Maps to:
      'normal'      -> 'Safe'
      'offensive'   -> 'Harassment-Offensive'
      'hatespeech'  -> 'Hate Speech'
    Returns (raw_label, mapped_label) or None if a 3-way split occurs.
    """
    labels = [a.get("label") for a in annotators if "label" in a]
    if not labels:
        return None

    counts = Counter(labels)
    # Check if there is a majority (at least 2 of 3 annotators agree)
    top_label, top_count = counts.most_common(1)[0]
    if top_count >= 2:
        mapping = {
            "normal": "Safe",
            "offensive": "Harassment-Offensive",
            "hatespeech": "Hate Speech",
        }
        return (top_label, mapping.get(top_label, "Safe"))

    # 3-way split (e.g. 1 normal, 1 offensive, 1 hatespeech) -> discard
    return None


def extract_rationales(rationale_masks: List[List[int]], num_tokens: int) -> List[int]:
    """
    HateXplain stores one token mask per annotator who judged the post toxic (post-level "rationales" field).
    A token is a gold rationale if at least half of those annotators marked it (majority, as in Mathew et al.).
    """
    masks = [m for m in rationale_masks if m and len(m) == num_tokens]
    if not masks:
        return [0] * num_tokens
    return [1 if sum(m[i] for m in masks) * 2 >= len(masks) else 0 for i in range(num_tokens)]


def process_hatexplain_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Downloads and processes the HateXplain dataset.
    Splits into:
    - precedent_docs (train + val splits) -> Channel B index (~17k posts)
    - hatexplain_test (test split) -> held out for rationale evaluation
    """
    DATA_RAW_HX = DATA_RAW / "hatexplain"
    DATA_RAW_HX.mkdir(parents=True, exist_ok=True)

    data_file = DATA_RAW_HX / "dataset.json"
    splits_file = DATA_RAW_HX / "post_id_divisions.json"

    # 1. Download dataset files
    download_file_with_retry(HATEXPLAIN_DATA_URL, data_file)
    download_file_with_retry(HATEXPLAIN_SPLITS_URL, splits_file)

    if not data_file.exists() or not splits_file.exists():
        raise FileNotFoundError("Failed to download HateXplain dataset files.")

    # 2. Load JSON files
    with open(data_file, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = set(splits.get("train", []))
    val_ids = set(splits.get("val", []))
    test_ids = set(splits.get("test", []))
    precedent_ids = train_ids.union(val_ids)

    print(f"\nHateXplain Released Splits: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    precedent_docs = []
    test_docs = []
    dropped_no_majority = 0
    label_distribution = Counter()

    for post_id, data in raw_dataset.items():
        tokens = data.get("post_tokens", [])
        raw_text = " ".join(tokens)
        cleaned_text = normalize_text(raw_text)

        # Resolve majority agreement label
        resolved = resolve_hatexplain_majority_label(data.get("annotators", []))
        if resolved is None:
            dropped_no_majority += 1
            continue

        raw_label, mapped_label = resolved
        rationales = extract_rationales(data.get("rationales", []), len(tokens))

        # Extract target communities if annotated
        targets = []
        for ann in data.get("annotators", []):
            targets.extend(ann.get("target", []))
        unique_targets = list(set([t for t in targets if t and t.lower() != "none"]))

        doc_obj = {
            "id": post_id,
            "source": "HateXplain",
            "text": cleaned_text,
            "raw_text": raw_text,
            "tokens": tokens,
            "rationales": rationales,
            "gold_label_raw": raw_label,
            "category": mapped_label,
            "targets": unique_targets,
            "word_count": len(cleaned_text.split()),
        }

        if post_id in precedent_ids:
            label_distribution[mapped_label] += 1
            precedent_docs.append(doc_obj)
        elif post_id in test_ids:
            test_docs.append(doc_obj)
        else:
            # If not in split file, default to precedent
            label_distribution[mapped_label] += 1
            precedent_docs.append(doc_obj)

    stats = {
        "total_posts_parsed": len(raw_dataset),
        "precedent_count": len(precedent_docs),
        "test_count": len(test_docs),
        "dropped_no_majority": dropped_no_majority,
        "label_distribution": dict(label_distribution),
    }

    print(f"[OK] Precedents (Train + Val): {len(precedent_docs)}")
    print(f"[OK] Held-Out Test Set:        {len(test_docs)}")
    print(f"[OK] Discarded (No Majority): {dropped_no_majority}")
    print(f"[OK] Precedent Labels:        {dict(label_distribution)}")

    return precedent_docs, test_docs, stats


def compute_vocabulary_types(texts: List[str]) -> int:
    """Computes total count of unique lowercase alphanumeric word types."""
    vocab: Set[str] = set()
    token_pattern = re.compile(r"\b[a-zA-Z0-9]+\b")
    for text in texts:
        tokens = token_pattern.findall(text.lower())
        vocab.update(tokens)
    return len(vocab)


def run_data_preparation():
    """Main execution orchestrating Milestone M1."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("XAI-Guard Milestone M1: Data Ingestion & Knowledge Base Construction")
    print("=" * 80)

    # 1. Build and chunk policy corpus (Channel A)
    policy_chunks = build_policy_corpus()
    policy_file = DATA_PROCESSED / "policy_chunks.jsonl"
    with open(policy_file, "w", encoding="utf-8") as f:
        for chunk in policy_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[SAVED] Policy chunks -> {policy_file}")

    # 2. Ingest HateXplain precedent corpus (Channel B)
    precedent_docs, test_docs, hx_stats = process_hatexplain_data()

    precedent_file = DATA_PROCESSED / "precedent_docs.jsonl"
    with open(precedent_file, "w", encoding="utf-8") as f:
        for doc in precedent_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"[SAVED] Precedent docs -> {precedent_file}")

    test_file = DATA_PROCESSED / "hatexplain_test.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        for doc in test_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"[SAVED] Held-out HateXplain test set -> {test_file}")

    # 3. Compute deterministic corpus statistics (Question 1)
    print("\n--- Computing Official Corpus Statistics (Question 1) ---")
    all_kb_texts = [p["text"] for p in policy_chunks] + [d["text"] for d in precedent_docs]
    total_chunks = len(all_kb_texts)
    total_docs = len({p["url"] for p in policy_chunks}) + len(precedent_docs)
    total_words = sum(len(t.split()) for t in all_kb_texts)
    unique_types = compute_vocabulary_types(all_kb_texts)

    corpus_stats = {
        "generated_on": datetime.now(timezone.utc).isoformat(),
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "total_word_count": total_words,
        "unique_vocabulary_types": unique_types,
        "policy_clauses_count": len(policy_chunks),
        "precedent_posts_count": len(precedent_docs),
        "held_out_test_posts": len(test_docs),
        "dropped_no_majority_posts": hx_stats["dropped_no_majority"],
        "precedent_label_distribution": hx_stats["label_distribution"],
        "policy_pages_count": len({p["url"] for p in policy_chunks}),
        "policy_word_count": sum(len(p["text"].split()) for p in policy_chunks),
        "source_breakdown": {
            **dict(Counter(p["source"] for p in policy_chunks)),
            "HateXplain": len(precedent_docs),
        },
        "assignment_minimum_compliance": {
            "required_documents": 10000,
            "actual_documents": total_docs,
            "documents_met": total_docs >= 10000,
            "required_words": 100000,
            "actual_words": total_words,
            "words_met": total_words >= 100000,
        },
    }

    stats_file = DATA_PROCESSED / "corpus_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(corpus_stats, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] Official corpus statistics -> {stats_file}")

    # Display Question 1 summary table
    print("\n" + "=" * 80)
    print("OFFICIAL CORPUS STATISTICS FOR QUESTION 1 REPORT")
    print("=" * 80)
    print(f"Total Documents:          {total_docs:,}  (Threshold: >= 10,000) -> {'PASS' if total_docs >= 10000 else 'FAIL'}")
    print(f"Total Chunks (indexed):   {total_chunks:,}")
    print(f"Total Word Count:         {total_words:,} (Threshold: >= 100,000) -> {'PASS' if total_words >= 100000 else 'FAIL'}")
    print(f"Unique Types (Vocab):     {unique_types:,}")
    print(f"Source Breakdown:")
    for src, cnt in corpus_stats["source_breakdown"].items():
        print(f"  - {src:16s}: {cnt:,} passages")
    print("=" * 80)


if __name__ == "__main__":
    run_data_preparation()
