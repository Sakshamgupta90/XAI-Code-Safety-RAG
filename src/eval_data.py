"""
src/eval_data.py
Evaluation datasets shared by the retrieval and downstream evaluation scripts.
Course: NTU AI6127 (2026) — Questions 3, 4 & 5.

- "eval_xls"        : our own 1,000 manually labelled posts (data/eval.xls) — the official Question 4 test set.
- "hatexplain_dev"  : 300 posts (100 per class) from the HateXplain *test* split, used only to calibrate the
                      critic thresholds (never reported as a test result).
- "hatexplain_test" : 900 further posts (300 per class) from the HateXplain test split, disjoint from dev.
                      A second, independently human-labelled benchmark (3 annotators per post, majority vote).
The HateXplain test split is never indexed in the Knowledge Base (see src/data_prep.py).
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SPLITS_FILE = DATA_PROCESSED / "hatexplain_eval_splits.json"
EVAL_XLS = PROJECT_ROOT / "data" / "eval.xls"

SEED = 6127
CLASSES = ["Safe", "Hate Speech", "Harassment-Offensive"]


def _hatexplain_test() -> List[Dict[str, Any]]:
    with open(DATA_PROCESSED / "hatexplain_test.jsonl", "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_hatexplain_splits(dev_per_class: int = 100, test_per_class: int = 300) -> Dict[str, List[str]]:
    """Stratified, seeded dev/test split of the HateXplain test posts (written once, then reused)."""
    if SPLITS_FILE.exists():
        return json.loads(SPLITS_FILE.read_text())
    rng = random.Random(SEED)
    posts = [p for p in _hatexplain_test() if len(p["text"].split()) >= 3]
    splits: Dict[str, List[str]] = {"dev": [], "test": []}
    for cls in CLASSES:
        ids = [p["id"] for p in posts if p["category"] == cls]
        rng.shuffle(ids)
        splits["dev"] += ids[:dev_per_class]
        splits["test"] += ids[dev_per_class:dev_per_class + test_per_class]
    SPLITS_FILE.write_text(json.dumps(splits, indent=2))
    return splits


def load_dataset(name: str) -> List[Dict[str, Any]]:
    """Returns records with id, text, raw_text, label and any auxiliary gold fields."""
    if name == "eval_xls":
        if not EVAL_XLS.exists():
            raise FileNotFoundError("data/eval.xls not built yet — finish annotation and run `python src/agreement.py finalize`.")
        df = pd.read_excel(EVAL_XLS, keep_default_na=False)
        return [{"id": str(r["id"]), "text": str(r["text"]), "raw_text": str(r["text"]), "label": r["label"],
                 "subjectivity": r["subjectivity"], "polarity": r["polarity"],
                 "is_sarcastic": r["is_sarcastic"] == "Yes"} for _, r in df.iterrows()]

    split = {"hatexplain_dev": "dev", "hatexplain_test": "test"}[name]
    ids = set(make_hatexplain_splits()[split])
    out = []
    for p in _hatexplain_test():
        if p["id"] in ids:
            gold_words = [t for t, m in zip(p["tokens"], p["rationales"]) if m == 1]
            out.append({"id": p["id"], "text": p["text"], "raw_text": p["raw_text"], "label": p["category"],
                        "targets": p.get("targets", []), "rationale_tokens": gold_words})
    out.sort(key=lambda r: r["id"])
    return out


if __name__ == "__main__":
    for n in ("hatexplain_dev", "hatexplain_test"):
        d = load_dataset(n)
        print(n, len(d), pd.Series([r["label"] for r in d]).value_counts().to_dict())
