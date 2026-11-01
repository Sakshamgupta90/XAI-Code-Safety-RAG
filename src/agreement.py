"""
src/agreement.py
Inter-Annotator Agreement Calculation, Adjudication, and eval.xls Generation.
Course: NTU AI6127 (2026) — Question 4 Deliverables.

Reads the workbooks our annotators filled in (data/annotation/sheets/, written blank by
src/build_eval_pool.py). Nothing here generates or guesses labels: if a sheet is missing or
incomplete the script stops and says which rows still need labels.

Usage:
  python src/agreement.py pilot      # 50-post pilot, all 6 annotators: pairwise % agreement + Fleiss' kappa
  python src/agreement.py main       # per-pair and pooled % agreement + Cohen's kappa; writes disagreements.xlsx
  python src/agreement.py finalize   # merges adjudications, balances classes, exports legacy data/eval.xls
"""

import json
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import xlwt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_eval_pool import CHOICES, PAIRS

DATA_DIR = PROJECT_ROOT / "data"
SHEETS_DIR = DATA_DIR / "annotation" / "sheets"
DATA_ANNOTATION = DATA_DIR / "annotation"
EVAL_XLS_PATH = DATA_DIR / "eval.xls"
REPORT_FILE = DATA_ANNOTATION / "agreement_report.json"
DISAGREEMENTS_FILE = DATA_ANNOTATION / "disagreements.xlsx"

FIELDS = ["category", "subjectivity", "polarity", "is_sarcastic"]
TARGET_PER_CLASS = {"Safe": 334, "Hate Speech": 333, "Harassment-Offensive": 333}
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


def calculate_fleiss_kappa(label_matrix: List[List[str]]) -> float:
    """Fleiss' kappa for N items each labelled by the same number of annotators (rows = items)."""
    categories = sorted({lab for row in label_matrix for lab in row})
    n_items, n_raters = len(label_matrix), len(label_matrix[0])
    if n_items == 0 or n_raters < 2:
        return 0.0
    counts = [[row.count(c) for c in categories] for row in label_matrix]
    p_j = [sum(r[j] for r in counts) / (n_items * n_raters) for j in range(len(categories))]
    p_i = [(sum(x * x for x in r) - n_raters) / (n_raters * (n_raters - 1)) for r in counts]
    p_bar, pe = sum(p_i) / n_items, sum(p * p for p in p_j)
    return 1.0 if pe >= 1.0 else (p_bar - pe) / (1.0 - pe)


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


def field_agreement(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    merged = a.merge(b, on="id", suffixes=("_1", "_2"))
    out = {}
    for field in FIELDS:
        l1, l2 = merged[f"{field}_1"].tolist(), merged[f"{field}_2"].tolist()
        out[field] = {
            "n": len(l1),
            "percent_agreement": round(calculate_percent_agreement(l1, l2), 2),
            "cohens_kappa": round(calculate_cohens_kappa(l1, l2), 4),
        }
    return out


def load_report() -> Dict[str, Any]:
    return json.loads(REPORT_FILE.read_text()) if REPORT_FILE.exists() else {}


def save_report(report: Dict[str, Any]):
    DATA_ANNOTATION.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[SAVED] Agreement report -> {REPORT_FILE}")


def run_pilot():
    sheets = {k: load_sheet(f"pilot_annotator{k}", required=False) for k in range(1, 7)}
    sheets = {k: v for k, v in sheets.items() if v is not None}
    if len(sheets) < 2:
        raise FileNotFoundError("At least two completed pilot sheets are needed.")
    ids = sorted(set.intersection(*[set(s["id"]) for s in sheets.values()]))
    result: Dict[str, Any] = {"annotators": sorted(sheets), "n_posts": len(ids)}
    for field in FIELDS:
        matrix = [[sheets[k].set_index("id").loc[i, field] for k in sorted(sheets)] for i in ids]
        pairwise = [calculate_percent_agreement([r[x] for r in matrix], [r[y] for r in matrix])
                    for x, y in combinations(range(len(sheets)), 2)]
        result[field] = {
            "mean_pairwise_percent_agreement": round(sum(pairwise) / len(pairwise), 2),
            "fleiss_kappa": round(calculate_fleiss_kappa(matrix), 4),
        }
    report = load_report()
    report.setdefault("pilot_rounds", []).append(result)
    save_report(report)
    cat = result["category"]
    print(f"\nPILOT ({len(sheets)} annotators, {len(ids)} posts): category agreement "
          f"{cat['mean_pairwise_percent_agreement']:.1f}% | Fleiss kappa {cat['fleiss_kappa']:.3f} -> "
          f"{'PASS' if cat['mean_pairwise_percent_agreement'] >= 80 else 'BELOW 80%: discuss, revise guidelines, redo'}")
    for field in FIELDS[1:]:
        print(f"  {field:<13} {result[field]['mean_pairwise_percent_agreement']:.1f}% | Fleiss {result[field]['fleiss_kappa']:.3f}")


def _pair_sheets(include_reserve: bool) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    groups = [(f"pair{p}", load_sheet(f"pair{p}_annotator1"), load_sheet(f"pair{p}_annotator2")) for p in PAIRS]
    if include_reserve:
        r1, r2 = load_sheet("reserve_annotator1", required=False), load_sheet("reserve_annotator2", required=False)
        if r1 is not None and r2 is not None:
            groups.append(("reserve", r1, r2))
    return groups


def run_main():
    groups = _pair_sheets(include_reserve=True)
    per_group, pooled_1, pooled_2, rows = {}, [], [], []
    for name, a, b in groups:
        per_group[name] = field_agreement(a, b)
        merged = a.merge(b, on="id", suffixes=("_1", "_2"))
        pooled_1.append(merged[[f"{f}_1" for f in FIELDS]].rename(columns=lambda c: c[:-2]))
        pooled_2.append(merged[[f"{f}_2" for f in FIELDS]].rename(columns=lambda c: c[:-2]))
        for _, r in merged.iterrows():
            disagree = [f for f in FIELDS if r[f"{f}_1"] != r[f"{f}_2"]]
            if disagree:
                row = {"id": r["id"], "group": name, "text": r["text_1"], "disagreed_on": ", ".join(disagree)}
                for f in FIELDS:
                    row[f"{f}_1"], row[f"{f}_2"] = r[f"{f}_1"], r[f"{f}_2"]
                    row[f"adjudicated_{f}"] = r[f"{f}_1"] if f not in disagree else ""
                row["notes_1"], row["notes_2"] = r.get("notes_1", ""), r.get("notes_2", "")
                rows.append(row)
        c = per_group[name]["category"]
        print(f"[{name}] category: {c['percent_agreement']:.1f}% (kappa {c['cohens_kappa']:.3f}) over {c['n']} posts")

    p1, p2 = pd.concat(pooled_1), pd.concat(pooled_2)
    pooled = {f: {"n": len(p1), "percent_agreement": round(calculate_percent_agreement(p1[f].tolist(), p2[f].tolist()), 2),
                  "cohens_kappa": round(calculate_cohens_kappa(p1[f].tolist(), p2[f].tolist()), 4)} for f in FIELDS}

    if DISAGREEMENTS_FILE.exists():
        print(f"[KEEP] {DISAGREEMENTS_FILE.name} already exists (it may hold adjudications); delete it to regenerate.")
    else:
        pd.DataFrame(rows).to_excel(DISAGREEMENTS_FILE, index=False)
        print(f"[SAVED] {len(rows)} disagreements for adjudication -> {DISAGREEMENTS_FILE}")

    report = load_report()
    report["main_round"] = {"per_pair": per_group, "pooled": pooled, "disagreement_rows": len(rows)}
    save_report(report)
    print("\nPOOLED AGREEMENT (all pairs)")
    for f in FIELDS:
        print(f"  {f:<13} {pooled[f]['percent_agreement']:.1f}% | Cohen's kappa {pooled[f]['cohens_kappa']:.3f}"
              f"{'  -> PASS (>= 80%)' if f == 'category' and pooled[f]['percent_agreement'] >= 80 else ''}")


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


def run_finalize():
    groups = _pair_sheets(include_reserve=True)
    if not DISAGREEMENTS_FILE.exists():
        raise FileNotFoundError("Run `python src/agreement.py main` first.")
    adj = pd.read_excel(DISAGREEMENTS_FILE, keep_default_na=False, na_values=[""])
    adj["id"] = adj["id"].astype(str)
    missing = []
    for f in FIELDS:
        adj[f"adjudicated_{f}"] = [_canonical(f, v) for v in adj[f"adjudicated_{f}"]]
        missing += adj.loc[adj[f"adjudicated_{f}"].isna(), "id"].tolist()
    if missing:
        raise ValueError(f"{len(set(missing))} disagreements still need adjudication in {DISAGREEMENTS_FILE.name} "
                         f"(e.g. {', '.join(sorted(set(missing))[:5])})")
    adj = adj.set_index("id")

    gold = []
    for name, a, b in groups:
        merged = a.merge(b, on="id", suffixes=("_1", "_2"))
        for _, r in merged.iterrows():
            rec = {"id": r["id"], "text": r["text_1"], "group": name,
                   "annotator_1": r["category_1"], "annotator_2": r["category_2"],
                   "adjudicated": "Yes" if r["id"] in adj.index else "No"}
            for f in FIELDS:
                rec[f] = adj.loc[r["id"], f"adjudicated_{f}"] if r["id"] in adj.index else r[f"{f}_1"]
            gold.append(rec)

    counts = Counter(g["category"] for g in gold)
    print(f"[OK] Gold labels for {len(gold)} posts: {dict(counts)}")
    short = {c: n - counts.get(c, 0) for c, n in TARGET_PER_CLASS.items() if counts.get(c, 0) < n}
    if short:
        raise ValueError(f"Not enough posts to balance 1,000 records, short by {short}. "
                         f"Label the reserve sheets (likely hate) and rerun `main` then `finalize`.")

    rng = random.Random(SEED)
    final = []
    for cat, n in TARGET_PER_CLASS.items():
        pool = [g for g in gold if g["category"] == cat]
        rng.shuffle(pool)
        final.extend(pool[:n])
    final.sort(key=lambda g: g["id"])
    for g in final:
        g["label"] = g["category"]

    # Primary label first, as in standard sentiment benchmarks (text, label), then auxiliary labels
    columns = ["id", "text", "label", "subjectivity", "polarity", "is_sarcastic",
               "annotator_1", "annotator_2", "adjudicated", "group"]
    export_genuine_legacy_xls(final, EVAL_XLS_PATH, columns)
    with open(DATA_ANNOTATION / "gold_all_annotated.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    l1 = [g["annotator_1"] for g in final]
    l2 = [g["annotator_2"] for g in final]
    report = load_report()
    report["eval_xls"] = {
        "total_records": len(final),
        "category_agreement_on_final_1000": {
            "percent_agreement": round(calculate_percent_agreement(l1, l2), 2),
            "cohens_kappa": round(calculate_cohens_kappa(l1, l2), 4),
        },
        "adjudicated_records": sum(1 for g in final if g["adjudicated"] == "Yes"),
        "class_balance": dict(Counter(g["category"] for g in final)),
        "subjectivity_distribution": dict(Counter(g["subjectivity"] for g in final)),
        "polarity_distribution": dict(Counter(g["polarity"] for g in final)),
        "sarcasm_distribution": dict(Counter(g["is_sarcastic"] for g in final)),
    }
    save_report(report)

    df_check = pd.read_excel(EVAL_XLS_PATH)
    print(f"[VERIFIED] {EVAL_XLS_PATH.name}: {len(df_check)} rows, columns {list(df_check.columns)}")
    print(df_check["label"].value_counts().to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"pilot": run_pilot, "main": run_main, "finalize": run_finalize}[mode]()
