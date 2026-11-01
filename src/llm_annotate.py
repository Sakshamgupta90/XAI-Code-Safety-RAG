"""
src/llm_annotate.py - LLM annotators 2 and 3 for eval.xls.
Course: NTU AI6127 (2026) — Question 4.

The group has one human annotator (annotator 1, src/annotate_app.py, whose labels are the gold labels).
To measure agreement, two LLMs from *different model families than the generator* (Qwen) label the same
posts independently with the same written guidelines: DeepSeek-V4.1-Flash and Gemma-3-27B-it (temperature 0).
This is disclosed in the report; LLM labels are never used as gold.

Writes data/annotation/sheets/llm_annotator_{deepseek,gemma}.csv (same columns as the human sheet).
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_eval_pool import CHOICES
from src.llm import get_client, parallel_map

POOL_FILE = PROJECT_ROOT / "data" / "raw" / "eval_pool" / "candidates.jsonl"
SHEETS_DIR = PROJECT_ROOT / "data" / "annotation" / "sheets"
GUIDELINES = PROJECT_ROOT / "data" / "annotation" / "ANNOTATION_GUIDELINES.md"
ANNOTATORS = {"deepseek": "deepseek-ai/DeepSeek-V4.1-Flash", "gemma": "google/gemma-3-27b-it"}

PROMPT = """You are an annotator for a content-moderation dataset. Follow these guidelines exactly.

{guidelines}

Post: "{post}"

Reply with JSON only:
{{"category": "Safe" | "Hate Speech" | "Harassment-Offensive",
  "subjectivity": "Objective" | "Subjective",
  "polarity": "Positive" | "Negative" | "None",
  "is_sarcastic": "Yes" | "No"}}"""


def guideline_text() -> str:
    text = GUIDELINES.read_text()
    return text.split("## Workflow")[0]


def canonical(field, value):
    v = str(value or "").strip().lower()
    for c in CHOICES[field]:
        if v == c.lower():
            return c
    return ""


def annotate(name: str, model: str):
    client = get_client(model)
    posts = [json.loads(l) for l in open(POOL_FILE, encoding="utf-8")]
    posts = [p for p in posts if p["split"] in ("main", "reserve")]
    guide = guideline_text()
    print(f"Annotator {name} ({model}) on {len(posts)} posts")

    def one(p):
        out = client.chat_json([{"role": "user", "content": PROMPT.format(guidelines=guide, post=p["text"])}], max_tokens=300)
        row = {"id": p["id"], "text": p["text"]}
        for f in ("category", "subjectivity", "polarity", "is_sarcastic"):
            row[f] = canonical(f, out.get(f))
        if row["subjectivity"] == "Objective":
            row["polarity"] = "None"
        return row

    rows = parallel_map(one, posts, workers=8, desc=name)
    out_file = SHEETS_DIR / f"llm_annotator_{name}.csv"
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "category", "subjectivity", "polarity", "is_sarcastic"])
        w.writeheader()
        w.writerows(rows)
    bad = sum(1 for r in rows if not all(r[k] for k in ("category", "subjectivity", "polarity", "is_sarcastic")))
    print(f"[SAVED] {out_file} ({bad} rows with an unparseable field) | usage {client.usage}")


if __name__ == "__main__":
    for name, model in ANNOTATORS.items():
        annotate(name, model)
