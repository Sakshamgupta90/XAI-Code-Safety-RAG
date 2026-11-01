"""
src/build_eval_pool.py
Held-Out Evaluation Pool Construction and Annotation Sheet Generation.
Course: NTU AI6127 (2026) — Question 1 & Question 4 Deliverables.

Actions:
1. Downloads an external, KB-disjoint tweet collection (Davidson et al., 2017, MIT licence).
   Only the tweet TEXT is used as raw material; the original crowd labels are never shown to our
   annotators and never copied into eval.xls. They are used solely to stratify sampling so the pool
   contains enough candidate hate posts (hate is <6% of random tweets) — this is disclosed in the report.
2. Cleans the text (HTML entities, retweet prefixes) and applies microtext normalization.
3. Removes exact and near duplicates inside the pool (cosine >= 0.95) and against the Knowledge Base
   (exact match or cosine >= 0.85 to any HateXplain precedent, BGE embeddings + FAISS) => zero leakage.
4. Samples 50 pilot posts + 1,500 main posts (+150 reserve posts with likely hate) and splits the main pool
   across 3 annotator pairs (A, B, C). Each pair double-labels its own 500 posts independently.
5. Writes blank, label-free annotation workbooks with dropdown validation to data/annotation/sheets/.
"""

import csv
import html
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

# Stable CPU threading on macOS (torch + faiss OpenMP runtimes)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

torch.set_num_threads(1)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalize import normalize_text

DATA_RAW_POOL = PROJECT_ROOT / "data" / "raw" / "eval_pool"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SHEETS_DIR = PROJECT_ROOT / "data" / "annotation" / "sheets"
INDEXES_DIR = PROJECT_ROOT / "indexes"

EXTERNAL_DATA_URL = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"

SEED = 6127
PILOT_SIZES = {"hate": 17, "offensive": 17, "neither": 16}
PAIRS = ["A", "B", "C"]
PER_PAIR = 500
# Stratification quotas over Davidson's crowd classes (hidden from annotators)
QUOTAS = {"hate": 700, "offensive": 400, "neither": 400}
RESERVE_HATE = 150
KB_DUP_COSINE = 0.85
POOL_DUP_COSINE = 0.95

LABEL_COLUMNS = ["category", "subjectivity", "polarity", "is_sarcastic", "notes"]
CHOICES = {
    "category": ["Safe", "Hate Speech", "Harassment-Offensive"],
    "subjectivity": ["Objective", "Subjective"],
    "polarity": ["Positive", "Negative", "None"],
    "is_sarcastic": ["Yes", "No"],
}


def download_file(url: str, target: Path) -> bool:
    """Downloads remote file once and caches it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1000:
        print(f"[OK] Cache found: {target} ({target.stat().st_size:,} bytes)")
        return True
    print(f"Downloading external dataset from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(target, "wb") as f:
        f.write(resp.read())
    print(f"[OK] Downloaded: {target} ({target.stat().st_size:,} bytes)")
    return True


def clean_tweet(raw: str) -> str:
    """Undoes HTML escaping and strips retweet chains before normalization."""
    text = html.unescape(html.unescape(raw)).strip()
    text = re.sub(r'^[!"\s]*(RT\s+)?(@\w+:?\s*)+', "", text)
    text = re.sub(r"\s+RT\s+@\w+:?", " ", text)
    return normalize_text(text)


def load_candidates() -> Dict[str, List[Dict[str, Any]]]:
    raw_csv = DATA_RAW_POOL / "davidson_labeled_data.csv"
    download_file(EXTERNAL_DATA_URL, raw_csv)
    strata: Dict[str, List[Dict[str, Any]]] = {"hate": [], "offensive": [], "neither": []}
    names = {"0": "hate", "1": "offensive", "2": "neither"}
    seen = set()
    with open(raw_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = clean_tweet(row.get("tweet", ""))
            n_words = len(text.split())
            if n_words < 5 or n_words > 60 or text.lower() in seen:
                continue
            seen.add(text.lower())
            strata[names[row["class"]]].append({
                "source_row": int(row[""]),
                "text": text,
                "stratum": names[row["class"]],
            })
    print(f"[OK] Usable candidates per stratum: { {k: len(v) for k, v in strata.items()} }")
    return strata


def filter_leakage_and_near_duplicates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drops posts that (near-)duplicate the Knowledge Base or each other, using BGE + FAISS cosine."""
    from sentence_transformers import SentenceTransformer
    import faiss

    kb_texts = set()
    with open(DATA_PROCESSED / "precedent_docs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            kb_texts.add(json.loads(line)["text"].lower())

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = encoder.encode([it["text"] for it in items], batch_size=64, normalize_embeddings=True,
                          show_progress_bar=False).astype("float32")
    kb_index = faiss.read_index(str(INDEXES_DIR / "precedent_faiss.bin"))
    kb_sims, _ = kb_index.search(vecs, 1)

    kept, kept_vecs = [], []
    dropped_kb, dropped_pool = 0, 0
    for it, v, sim in zip(items, vecs, kb_sims[:, 0]):
        if it["text"].lower() in kb_texts or sim >= KB_DUP_COSINE:
            dropped_kb += 1
            continue
        if kept_vecs and float(np.max(np.stack(kept_vecs) @ v)) >= POOL_DUP_COSINE:
            dropped_pool += 1
            continue
        it["max_kb_cosine"] = round(float(sim), 4)
        kept.append(it)
        kept_vecs.append(v)
    print(f"[OK] Leakage filter: dropped {dropped_kb} KB near-duplicates (cos >= {KB_DUP_COSINE}), "
          f"{dropped_pool} in-pool near-duplicates (cos >= {POOL_DUP_COSINE}); kept {len(kept)}")
    return kept


def write_sheet(path: Path, rows: List[Dict[str, Any]], title: str):
    """Writes a blank annotation workbook: id + text visible, label columns with dropdowns."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    ws.append(["id", "text"] + LABEL_COLUMNS)
    for r in rows:
        ws.append([r["id"], r["text"]] + [""] * len(LABEL_COLUMNS))

    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E79")
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 90
    for col in "CDEF":
        ws.column_dimensions[col].width = 22
    ws.column_dimensions["G"].width = 30
    for row in ws.iter_rows(min_row=2, max_col=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"

    n = len(rows) + 1
    for col, field in zip("CDEF", ["category", "subjectivity", "polarity", "is_sarcastic"]):
        dv = DataValidation(type="list", formula1='"' + ",".join(CHOICES[field]) + '"', allow_blank=True)
        dv.error = f"Choose one of: {', '.join(CHOICES[field])}"
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{n}")

    info = wb.create_sheet("README")
    info.append([title])
    info.append(["Label every row independently. Do not discuss posts with your pair partner until both sheets are complete."])
    info.append(["Follow data/annotation/ANNOTATION_GUIDELINES.md. Use the dropdowns; 'notes' is optional."])
    info.append(["polarity: Positive/Negative only when subjectivity = Subjective; use None when Objective."])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def build_evaluation_pool():
    """Main pipeline: sample, de-duplicate, split across pairs, write blank sheets."""
    print("=" * 80)
    print("XAI-Guard: Held-Out Evaluation Pool & Annotation Sheets")
    print("=" * 80)
    rng = random.Random(SEED)
    strata = load_candidates()

    # Oversample each stratum before leakage filtering, then trim to quota
    sampled: List[Dict[str, Any]] = []
    for name, quota in QUOTAS.items():
        need = quota + PILOT_SIZES[name] + (RESERVE_HATE if name == "hate" else 0)
        pool = strata[name][:]
        rng.shuffle(pool)
        sampled.extend(pool[: int(need * 1.3)])
    kept = filter_leakage_and_near_duplicates(sampled)

    pilot: List[Dict[str, Any]] = []
    main: List[Dict[str, Any]] = []
    reserve: List[Dict[str, Any]] = []
    for name, quota in QUOTAS.items():
        items = [it for it in kept if it["stratum"] == name]
        n_pilot = PILOT_SIZES[name]
        if len(items) < n_pilot + quota:
            raise RuntimeError(f"Not enough '{name}' candidates after filtering: {len(items)}")
        pilot.extend(items[:n_pilot])
        main.extend(items[n_pilot:n_pilot + quota])
        if name == "hate":
            reserve.extend(items[n_pilot + quota:n_pilot + quota + RESERVE_HATE])
    for group in (pilot, main, reserve):
        rng.shuffle(group)

    for i, it in enumerate(pilot, 1):
        it["id"] = f"pilot_{i:03d}"
    for i, it in enumerate(main, 1):
        it["id"] = f"ev_{i:04d}"
        it["pair"] = PAIRS[(i - 1) // PER_PAIR]
    for i, it in enumerate(reserve, 1):
        it["id"] = f"res_{i:03d}"

    # Internal manifest (keeps stratum + source row for the report; never given to annotators)
    manifest_file = DATA_RAW_POOL / "candidates.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for group, split in ((pilot, "pilot"), (main, "main"), (reserve, "reserve")):
            for it in group:
                f.write(json.dumps({**it, "split": split}, ensure_ascii=False) + "\n")
    print(f"[SAVED] Pool manifest ({len(pilot)} pilot, {len(main)} main, {len(reserve)} reserve) -> {manifest_file}")

    # Blank sheets: pilot for all 6 annotators, main for each pair member, reserve if hate falls short
    for k in range(1, 7):
        write_sheet(SHEETS_DIR / f"pilot_annotator{k}.xlsx", pilot,
                    f"Pilot round — annotator {k} (all six annotators label the same 50 posts)")
    for pair in PAIRS:
        rows = [it for it in main if it["pair"] == pair]
        for member in (1, 2):
            write_sheet(SHEETS_DIR / f"pair{pair}_annotator{member}.xlsx", rows,
                        f"Main round — pair {pair}, annotator {member} ({len(rows)} posts)")
    for member in (1, 2):
        write_sheet(SHEETS_DIR / f"reserve_annotator{member}.xlsx", reserve,
                    "Reserve — only label if the main round yields < 333 Hate Speech posts")
    print(f"[SAVED] Blank annotation workbooks -> {SHEETS_DIR}")
    return pilot, main, reserve


if __name__ == "__main__":
    build_evaluation_pool()
