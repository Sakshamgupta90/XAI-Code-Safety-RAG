"""
src/agreement.py
Inter-Annotator Agreement Calculation, Adjudication, and eval.xls Generation.
Course: NTU AI6127 (2026) — Question 4 Deliverables.

Two group members label the 1,650-post pool independently with src/annotate_app.py
(data/annotation/sheets/human_annotator_<name>.csv). This script validates both sheets, reports percent agreement and
Cohen's kappa for every field, and builds data/eval.xls from the posts whose category both annotators chose.
Nothing here generates or guesses labels: an incomplete sheet stops the script with the rows that still need labels.

Usage: python src/agreement.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xlwt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_eval_pool import CHOICES

DATA_DIR = PROJECT_ROOT / "data"
SHEETS_DIR = DATA_DIR / "annotation" / "sheets"
DATA_ANNOTATION = DATA_DIR / "annotation"
EVAL_XLS_PATH = DATA_DIR / "eval.xls"
REPORT_FILE = DATA_ANNOTATION / "agreement_report.json"

FIELDS = ["category", "subjectivity", "polarity", "is_sarcastic"]
SEED = 6127


def calculate_percent_agreement(labels_1: List[str], labels_2: List[str]) -> float:
    """Calculates observed percentage agreement Po."""
    assert len(labels_1) == len(labels_2)
    matches = sum(1 for a, b in zip(labels_1, labels_2) if a == b)
    return (matches / len(labels_1)) * 100.0


def calculate_cohens_kappa(labels_1: List[str], labels_2: List[str]) -> float:
    """
    Calculates Cohen's Kappa coefficient (kappa) for two annotators:
    kappa = (Po - Pe) / (1 - Pe)
    """
    assert len(labels_1) == len(labels_2)
    n = len(labels_1)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(labels_1, labels_2) if a == b) / n
    counts_1, counts_2 = Counter(labels_1), Counter(labels_2)
    pe = sum((counts_1[c] / n) * (counts_2[c] / n) for c in set(labels_1) | set(labels_2))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _canonical(field: str, value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    v = str(value).strip().lower()
    for choice in CHOICES[field]:
        if v == choice.lower():
            return choice
    aliases = {"hate": "Hate Speech", "offensive": "Harassment-Offensive", "harassment": "Harassment-Offensive",
               "true": "Yes", "false": "No", "n/a": "None", "na": "None", "neutral": "None"}
    return aliases.get(v) if aliases.get(v) in CHOICES[field] else None


def load_sheet(stem: str, required: bool = True) -> Optional[pd.DataFrame]:
    """Loads a completed sheet (xlsx or CSV export) and validates every label."""
    path = next((SHEETS_DIR / f"{stem}{ext}" for ext in (".xlsx", ".csv") if (SHEETS_DIR / f"{stem}{ext}").exists()), None)
    if path is None:
        if required:
            raise FileNotFoundError(f"Missing sheet {stem}.xlsx/.csv in {SHEETS_DIR}")
        return None
    # keep_default_na=False: the polarity label "None" must not be read as a missing value
    read_kw = {"keep_default_na": False, "na_values": [""]}
    df = pd.read_csv(path, **read_kw) if path.suffix == ".csv" else pd.read_excel(path, sheet_name="labels", **read_kw)
    df["id"] = df["id"].astype(str)
    problems = []
    for field in FIELDS:
        df[field] = [_canonical(field, v) for v in df[field]]
        bad = df.loc[df[field].isna(), "id"].tolist()
        if bad:
            problems.append(f"{field}: {len(bad)} empty/invalid (e.g. {', '.join(bad[:5])})")
    incoherent = df[(df["subjectivity"] == "Objective") & (df["polarity"] != "None")]["id"].tolist()
    if incoherent:
        print(f"[WARN] {stem}: {len(incoherent)} Objective rows have polarity != None; treated as None")
        df.loc[df["subjectivity"] == "Objective", "polarity"] = "None"
    if problems:
        raise ValueError(f"{path.name} is incomplete:\n  " + "\n  ".join(problems))
    return df


def save_report(report: Dict[str, Any]):
    DATA_ANNOTATION.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[SAVED] Agreement report -> {REPORT_FILE}")


def export_genuine_legacy_xls(records: List[Dict[str, Any]], target_file: Path, columns: List[str]):
    """Exports the evaluation dataset to genuine legacy .xls (BIFF8) using xlwt, as the brief requires."""
    workbook = xlwt.Workbook(encoding="utf-8")
    sheet = workbook.add_sheet("eval")
    header_style = xlwt.easyxf("font: bold on;")
    for col_idx, col_name in enumerate(columns):
        sheet.write(0, col_idx, col_name, header_style)
    for row_idx, item in enumerate(records, start=1):
        for col_idx, col_name in enumerate(columns):
            sheet.write(row_idx, col_idx, str(item.get(col_name, "")))
    workbook.save(str(target_file))
    print(f"[OK] Exported {len(records):,} records to genuine legacy .xls -> {target_file}")


# ---------------------------------------------------------------- two human annotators (workflow used for eval.xls)
HUMAN_ANNOTATORS = {"annotator_1": "human_annotator_saksham", "annotator_2": "human_annotator_hariharan"}
EVAL_SIZE = 1000


def _balanced_quota(available: Dict[str, int], total: int) -> Dict[str, int]:
    """Splits `total` as evenly as possible over the classes, capped by what each class has (water-filling)."""
    quota, left = {}, total
    classes = sorted(available, key=available.get)  # smallest first, so its shortfall moves to the larger classes
    for k, c in enumerate(classes):
        quota[c] = min(available[c], left // (len(classes) - k))
        left -= quota[c]
    return quota


def _resolve_auxiliary(category: str, a: pd.Series, b: pd.Series) -> Tuple[str, str, str]:
    """Subjectivity, polarity and sarcasm for a post whose category both annotators agree on.
    Agreed values are kept. On disagreement: sarcasm is Yes only if both said Yes (guideline: rude is not sarcastic);
    a harmful post is Subjective/Negative (guideline: insults are subjective); for a Safe post, an opinion seen by one
    annotator wins over Objective, and a Positive/Negative split keeps annotator 1's reading."""
    sarcastic = "Yes" if a["is_sarcastic"] == b["is_sarcastic"] == "Yes" else "No"
    if (a["subjectivity"], a["polarity"]) == (b["subjectivity"], b["polarity"]):
        return a["subjectivity"], a["polarity"], sarcastic
    if category != "Safe":
        return "Subjective", "Negative", sarcastic
    opinions = [x["polarity"] for x in (a, b) if x["subjectivity"] == "Subjective"]
    return "Subjective", opinions[0], sarcastic


def run_two_humans():
    """Two group members label the whole 1,650-post pool independently (src/annotate_app.py). Agreement is reported on
    the whole pool; eval.xls keeps only posts whose category both annotators chose, as balanced as the pool allows."""
    sheets = {k: load_sheet(stem).set_index("id") for k, stem in HUMAN_ANNOTATORS.items()}
    a, b = sheets["annotator_1"], sheets["annotator_2"]
    ids = sorted(set(a.index) & set(b.index))
    a, b = a.loc[ids], b.loc[ids]
    n = len(ids)

    def pair(x: List[str], y: List[str]) -> Dict[str, float]:
        return {"n": len(x), "percent_agreement": round(calculate_percent_agreement(x, y), 2),
                "cohens_kappa": round(calculate_cohens_kappa(x, y), 4)}

    pool = {f: pair(a[f].tolist(), b[f].tolist()) for f in FIELDS}
    unsafe_a, unsafe_b = [str(v != "Safe") for v in a["category"]], [str(v != "Safe") for v in b["category"]]
    both_unsafe = [(x, y) for x, y in zip(a["category"], b["category"]) if x != "Safe" and y != "Safe"]
    block = 150
    drift = [round(100 * float(np.mean(a["category"].iloc[s:s + block].values == b["category"].iloc[s:s + block].values)), 1)
             for s in range(0, n, block)]
    confusion = pd.crosstab(a["category"], b["category"]).reindex(index=CHOICES["category"], columns=CHOICES["category"],
                                                                  fill_value=0)

    gold = []
    for i in ids:
        ra, rb = a.loc[i], b.loc[i]
        rec = {"id": i, "text": ra["text"].strip(), "annotator_1": ra["category"], "annotator_2": rb["category"],
               "agreed": "Yes" if ra["category"] == rb["category"] else "No"}
        if rec["agreed"] == "Yes":
            rec["category"] = ra["category"]
            rec["subjectivity"], rec["polarity"], rec["is_sarcastic"] = _resolve_auxiliary(ra["category"], ra, rb)
            rec["auxiliary_agreed"] = "Yes" if all(ra[f] == rb[f] for f in FIELDS[1:]) else "No"
        gold.append(rec)

    agreed = [g for g in gold if g["agreed"] == "Yes"]
    available = Counter(g["category"] for g in agreed)
    if sum(available.values()) < EVAL_SIZE:
        raise ValueError(f"Only {sum(available.values())} posts have an agreed category; {EVAL_SIZE} are needed.")
    quota = _balanced_quota({c: available.get(c, 0) for c in CHOICES["category"]}, EVAL_SIZE)
    rng = random.Random(SEED)
    final = []
    for cat in CHOICES["category"]:
        pool_c = [g for g in agreed if g["category"] == cat]
        rng.shuffle(pool_c)
        final.extend(pool_c[:quota[cat]])
    final.sort(key=lambda g: g["id"])
    for g in final:
        g["label"] = g["category"]

    columns = ["id", "text", "label", "subjectivity", "polarity", "is_sarcastic", "annotator_1", "annotator_2",
               "auxiliary_agreed"]
    export_genuine_legacy_xls(final, EVAL_XLS_PATH, columns)
    with open(DATA_ANNOTATION / "gold_all_annotated.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    fin = {f: pair([a.loc[g["id"], f] for g in final], [b.loc[g["id"], f] for g in final]) for f in FIELDS}
    report = {
        "human_annotation": {
            "annotators": {"annotator_1": "Saksham Gupta (group member)", "annotator_2": "Hariharan (group member)"},
            "guidelines": "data/annotation/ANNOTATION_GUIDELINES.md",
            "posts_annotated": n,
            "pool_agreement": pool,
            "safe_vs_unsafe": pair(unsafe_a, unsafe_b),
            "hate_vs_harassment_when_both_unsafe": pair([x for x, _ in both_unsafe], [y for _, y in both_unsafe]),
            "category_confusion": {"rows_annotator_1": CHOICES["category"], "cols_annotator_2": CHOICES["category"],
                                   "matrix": confusion.values.tolist()},
            "category_agreement_per_150_posts": drift,
            "posts_with_agreed_category": len(agreed),
            "agreed_by_class": dict(available),
        },
        "eval_xls": {
            "total_records": len(final),
            "selection": "posts whose category both annotators chose; classes as balanced as the agreed posts allow",
            "agreement_on_final_set": fin,
            "auxiliary_fields_resolved": sum(g["auxiliary_agreed"] == "No" for g in final),
            "class_balance": dict(Counter(g["label"] for g in final)),
            "subjectivity_distribution": dict(Counter(g["subjectivity"] for g in final)),
            "polarity_distribution": dict(Counter(g["polarity"] for g in final)),
            "sarcasm_distribution": dict(Counter(g["is_sarcastic"] for g in final)),
        },
    }
    save_report(report)

    print(f"\nPOOL ({n} posts, 2 annotators)")
    for f in FIELDS:
        print(f"  {f:<13} {pool[f]['percent_agreement']:.1f}% | Cohen's kappa {pool[f]['cohens_kappa']:.3f}")
    print(f"  safe/unsafe   {report['human_annotation']['safe_vs_unsafe']['percent_agreement']:.1f}%")
    print(f"  per {block} posts: {drift}")
    print(f"\nAgreed category on {len(agreed)} posts {dict(available)} -> eval.xls quota {quota}")
    for f in FIELDS[1:]:
        print(f"  final-set {f:<13} {fin[f]['percent_agreement']:.1f}% | kappa {fin[f]['cohens_kappa']:.3f}")
    df_check = pd.read_excel(EVAL_XLS_PATH, keep_default_na=False)
    print(f"[VERIFIED] {EVAL_XLS_PATH.name}: {len(df_check)} rows, {df_check['id'].nunique()} unique ids, "
          f"{df_check['text'].nunique()} unique texts")
    print(df_check["label"].value_counts().to_string())


if __name__ == "__main__":
    run_two_humans()
