"""
src/agreement.py
Inter-Annotator Agreement Calculation, Adjudication, and eval.xls Generation.
Course: NTU AI6127 (2026) — Question 4 Deliverables.

Reads the workbooks our annotators filled in (data/annotation/sheets/, written blank by
src/build_eval_pool.py). Nothing here generates or guesses labels: if a sheet is missing or
incomplete the script stops and says which rows still need labels.

Usage (team workflow, several human annotators):
  python src/agreement.py pilot      # 50-post pilot, all 6 annotators: pairwise % agreement + Fleiss' kappa
  python src/agreement.py main       # per-pair and pooled % agreement + Cohen's kappa; writes disagreements.xlsx
  python src/agreement.py finalize   # merges adjudications, balances classes, exports legacy data/eval.xls

Usage (workflow actually used: one human annotator + two LLM annotators, src/annotate_app.py + src/llm_annotate.py):
  python src/agreement.py human           # human vs LLM agreement; writes review_sheet.xlsx (posts both LLMs dispute)
  python src/agreement.py finalize_human  # applies the human's optional review, balances, exports data/eval.xls

Usage (LLM-annotator workflow, permitted by the instructor; human labels override where they exist):
  python src/agreement.py llm             # 3 LLM annotators, majority vote; agreement report; exports data/eval.xls
"""

import json
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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


# ---------------------------------------------------------------- one human + two LLM annotators
LLM_SHEETS = {"deepseek": "llm_annotator_deepseek", "gemma": "llm_annotator_gemma"}
REVIEW_FILE = DATA_ANNOTATION / "review_sheet.xlsx"


def _load_human_and_llms():
    human = load_sheet("human_annotator")
    llms = {}
    for name, stem in LLM_SHEETS.items():
        df = pd.read_csv(SHEETS_DIR / f"{stem}.csv", keep_default_na=False, na_values=[""])
        df["id"] = df["id"].astype(str)
        for f in FIELDS:
            df[f] = [_canonical(f, v) or "INVALID" for v in df[f]]
        llms[name] = df[df["id"].isin(set(human["id"]))]
    return human, llms


def run_human():
    human, llms = _load_human_and_llms()
    merged = human[["id", "text"] + FIELDS]
    for name, df in llms.items():
        merged = merged.merge(df[["id"] + FIELDS].rename(columns={f: f"{f}_{name}" for f in FIELDS}), on="id")
    n = len(merged)
    report_fields = {}
    for f in FIELDS:
        h, a, b = merged[f].tolist(), merged[f"{f}_deepseek"].tolist(), merged[f"{f}_gemma"].tolist()
        report_fields[f] = {
            "human_vs_deepseek": {"percent_agreement": round(calculate_percent_agreement(h, a), 2), "cohens_kappa": round(calculate_cohens_kappa(h, a), 4)},
            "human_vs_gemma": {"percent_agreement": round(calculate_percent_agreement(h, b), 2), "cohens_kappa": round(calculate_cohens_kappa(h, b), 4)},
            "deepseek_vs_gemma": {"percent_agreement": round(calculate_percent_agreement(a, b), 2), "cohens_kappa": round(calculate_cohens_kappa(a, b), 4)},
            "fleiss_kappa_all_three": round(calculate_fleiss_kappa([[x, y, z] for x, y, z in zip(h, a, b)]), 4),
            "majority_agreement_with_human": round(100 * sum(1 for x, y, z in zip(h, a, b) if (y == x or z == x)) / n, 2),
        }
    disputed = merged[(merged["category_deepseek"] != merged["category"]) & (merged["category_gemma"] != merged["category"])]
    if not REVIEW_FILE.exists():
        review = disputed[["id", "text", "category", "category_deepseek", "category_gemma"]].copy()
        review["final_category"] = review["category"]
        review.to_excel(REVIEW_FILE, index=False)
        print(f"[SAVED] {len(review)} posts where both LLMs disagree with you -> {REVIEW_FILE} "
              "(optional: re-read them and edit final_category only where you made a mistake)")
    report = load_report()
    report["single_human_round"] = {"annotators": {"annotator_1": "human (group member)", "annotator_2": "DeepSeek-V4.1-Flash",
                                                   "annotator_3": "Gemma-3-27B-it"},
                                    "posts": n, "fields": report_fields, "posts_disputed_by_both_llms": len(disputed)}
    save_report(report)
    for f in FIELDS:
        r = report_fields[f]
        print(f"  {f:<13} human-DeepSeek {r['human_vs_deepseek']['percent_agreement']:.1f}% (k {r['human_vs_deepseek']['cohens_kappa']:.2f}) | "
              f"human-Gemma {r['human_vs_gemma']['percent_agreement']:.1f}% (k {r['human_vs_gemma']['cohens_kappa']:.2f}) | "
              f"Fleiss {r['fleiss_kappa_all_three']:.2f}")


def run_finalize_human():
    human, llms = _load_human_and_llms()
    changes = {}
    if REVIEW_FILE.exists():
        rv = pd.read_excel(REVIEW_FILE, keep_default_na=False, na_values=[""])
        for _, r in rv.iterrows():
            final = _canonical("category", r["final_category"])
            if final and final != r["category"]:
                changes[str(r["id"])] = final
    gold = []
    for _, r in human.iterrows():
        rec = {"id": r["id"], "text": r["text"], "annotator_1": r["category"],
               "annotator_2": llms["deepseek"].set_index("id").loc[r["id"], "category"],
               "annotator_3": llms["gemma"].set_index("id").loc[r["id"], "category"],
               "reviewed": "changed" if r["id"] in changes else "No"}
        for f in FIELDS:
            rec[f] = r[f]
        rec["category"] = changes.get(r["id"], r["category"])
        gold.append(rec)
    counts = Counter(g["category"] for g in gold)
    short = {c: n - counts.get(c, 0) for c, n in TARGET_PER_CLASS.items() if counts.get(c, 0) < n}
    if short:
        raise ValueError(f"Keep labelling: still short of {short} for a balanced 1,000-post set (have {dict(counts)}).")
    rng = random.Random(SEED)
    final = []
    for cat, n in TARGET_PER_CLASS.items():
        pool = [g for g in gold if g["category"] == cat]
        rng.shuffle(pool)
        final.extend(pool[:n])
    final.sort(key=lambda g: g["id"])
    for g in final:
        g["label"] = g["category"]
    columns = ["id", "text", "label", "subjectivity", "polarity", "is_sarcastic",
               "annotator_1", "annotator_2", "annotator_3", "reviewed"]
    export_genuine_legacy_xls(final, EVAL_XLS_PATH, columns)
    with open(DATA_ANNOTATION / "gold_all_annotated.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    h = [g["annotator_1"] for g in final]
    report = load_report()
    report["eval_xls"] = {
        "total_records": len(final),
        "labelled_by_human_total": len(gold),
        "review_changes": len(changes),
        "category_agreement_on_final_1000": {
            "human_vs_deepseek": round(calculate_percent_agreement(h, [g["annotator_2"] for g in final]), 2),
            "human_vs_gemma": round(calculate_percent_agreement(h, [g["annotator_3"] for g in final]), 2),
            "fleiss_kappa": round(calculate_fleiss_kappa([[g["annotator_1"], g["annotator_2"], g["annotator_3"]] for g in final]), 4),
        },
        "class_balance": dict(Counter(g["label"] for g in final)),
        "subjectivity_distribution": dict(Counter(g["subjectivity"] for g in final)),
        "polarity_distribution": dict(Counter(g["polarity"] for g in final)),
        "sarcasm_distribution": dict(Counter(g["is_sarcastic"] for g in final)),
    }
    save_report(report)
    df_check = pd.read_excel(EVAL_XLS_PATH, keep_default_na=False)
    print(f"[VERIFIED] {EVAL_XLS_PATH.name}: {len(df_check)} rows, columns {list(df_check.columns)}")
    print(df_check["label"].value_counts().to_string())


# ---------------------------------------------------------------- three LLM annotators (majority vote)
LLM_ANNOTATORS = ["deepseek", "gemma", "gptoss"]


def run_llm_majority():
    sheets = {}
    for name in LLM_ANNOTATORS:
        df = pd.read_csv(SHEETS_DIR / f"llm_annotator_{name}.csv", keep_default_na=False, na_values=[""])
        df["id"] = df["id"].astype(str)
        for f in FIELDS:
            df[f] = [_canonical(f, v) or "INVALID" for v in df[f]]
        sheets[name] = df.set_index("id")
    human = load_sheet("human_annotator", required=False)
    human = human.set_index("id") if human is not None else None
    ids = sorted(set.intersection(*[set(df.index) for df in sheets.values()]))
    texts = sheets[LLM_ANNOTATORS[0]]["text"]

    def majority(votes):
        top, n = Counter(votes).most_common(1)[0]
        return top if n >= 2 and top != "INVALID" else None

    report_fields = {}
    for f in FIELDS:
        cols = {n: [sheets[n].loc[i, f] for i in ids] for n in LLM_ANNOTATORS}
        pairs = {}
        for a, b in combinations(LLM_ANNOTATORS, 2):
            pairs[f"{a}_vs_{b}"] = {"percent_agreement": round(calculate_percent_agreement(cols[a], cols[b]), 2),
                                    "cohens_kappa": round(calculate_cohens_kappa(cols[a], cols[b]), 4)}
        matrix = [[cols[n][k] for n in LLM_ANNOTATORS] for k in range(len(ids))]
        maj = [majority(r) for r in matrix]
        report_fields[f] = {
            "pairwise": pairs,
            "mean_pairwise_percent_agreement": round(float(np.mean([v["percent_agreement"] for v in pairs.values()])), 2),
            "fleiss_kappa": round(calculate_fleiss_kappa(matrix), 4),
            "posts_with_majority_pct": round(100 * sum(m is not None for m in maj) / len(ids), 2),
            "unanimous_pct": round(100 * sum(len(set(r)) == 1 for r in matrix) / len(ids), 2),
            # agreement of each annotator with the majority label (posts that have one)
            "annotator_vs_majority_pct": {n: round(100 * np.mean([cols[n][k] == maj[k] for k in range(len(ids)) if maj[k]]), 2)
                                          for n in LLM_ANNOTATORS},
        }

    gold = []
    for k, i in enumerate(ids):
        votes = {n: sheets[n].loc[i, "category"] for n in LLM_ANNOTATORS}
        rec = {"id": i, "text": texts.loc[i], **{f"annotator_{n}": v for n, v in votes.items()},
               "annotator_human": human.loc[i, "category"] if human is not None and i in human.index else ""}
        if rec["annotator_human"]:
            for f in FIELDS:
                rec[f] = human.loc[i, f]
            rec["label_source"] = "human"
        else:
            for f in FIELDS:
                rec[f] = majority([sheets[n].loc[i, f] for n in LLM_ANNOTATORS])
            rec["label_source"] = "llm_majority"
            if rec["category"] is None:
                continue  # three-way split: no reliable label (dropped, as HateXplain drops no-majority posts)
            if rec["subjectivity"] == "Objective" or rec["polarity"] is None:
                rec["polarity"] = "None" if rec["subjectivity"] == "Objective" else (rec["polarity"] or "Negative")
            rec["subjectivity"] = rec["subjectivity"] or "Subjective"
            rec["is_sarcastic"] = rec["is_sarcastic"] or "No"
        rec["unanimous"] = "Yes" if len(set(votes.values())) == 1 else "No"
        gold.append(rec)

    counts = Counter(g["category"] for g in gold)
    short = {c: n - counts.get(c, 0) for c, n in TARGET_PER_CLASS.items() if counts.get(c, 0) < n}
    if short:
        raise ValueError(f"Not enough labelled posts per class: short by {short} (have {dict(counts)})")
    rng = random.Random(SEED)
    final = []
    for cat, n in TARGET_PER_CLASS.items():
        pool = [g for g in gold if g["category"] == cat]
        rng.shuffle(pool)
        # Agreement-based selection: human-labelled posts first, then posts all three annotators agree on, then
        # 2-of-3 majority posts (stable sort keeps the seeded random order inside each group)
        pool.sort(key=lambda g: (g["label_source"] != "human", g["unanimous"] != "Yes"))
        final.extend(pool[:n])
    final.sort(key=lambda g: g["id"])
    for g in final:
        g["label"] = g["category"]
    columns = ["id", "text", "label", "subjectivity", "polarity", "is_sarcastic", "label_source",
               "annotator_human", "annotator_deepseek", "annotator_gemma", "annotator_gptoss", "unanimous"]
    export_genuine_legacy_xls(final, EVAL_XLS_PATH, columns)
    with open(DATA_ANNOTATION / "gold_all_annotated.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    fin_matrix = [[g[f"annotator_{n}"] for n in LLM_ANNOTATORS] for g in final]
    report = load_report()
    report["llm_annotation"] = {
        "annotators": {"deepseek": "deepseek-ai/DeepSeek-V4.1-Flash", "gemma": "google/gemma-3-27b-it",
                       "gptoss": "openai/gpt-oss-120b"},
        "note": "LLM annotation permitted by the course instructor; human labels (annotate_app.py) override where present",
        "posts_annotated": len(ids), "fields": report_fields,
        "dropped_three_way_split": len(ids) - len(gold),
    }
    report["eval_xls"] = {
        "total_records": len(final),
        "label_source": dict(Counter(g["label_source"] for g in final)),
        "category_on_final_set": {
            "mean_pairwise_percent_agreement": round(float(np.mean([calculate_percent_agreement(
                [r[a] for r in fin_matrix], [r[b] for r in fin_matrix]) for a, b in combinations(range(3), 2)])), 2),
            "fleiss_kappa": round(calculate_fleiss_kappa(fin_matrix), 4),
            "unanimous_pct": round(100 * sum(g["unanimous"] == "Yes" for g in final) / len(final), 2),
            "selection": "unanimous posts first, then 2-of-3 majority posts, per class",
        },
        "all_fields_on_final_set": {
            f: round(float(np.mean([calculate_percent_agreement(
                [sheets[a].loc[g["id"], f] for g in final], [sheets[b].loc[g["id"], f] for g in final])
                for a, b in combinations(LLM_ANNOTATORS, 2)])), 2) for f in FIELDS},
        "class_balance": dict(Counter(g["label"] for g in final)),
        "subjectivity_distribution": dict(Counter(g["subjectivity"] for g in final)),
        "polarity_distribution": dict(Counter(g["polarity"] for g in final)),
        "sarcasm_distribution": dict(Counter(g["is_sarcastic"] for g in final)),
    }
    save_report(report)
    c = report_fields["category"]
    print(f"\nCategory: mean pairwise agreement {c['mean_pairwise_percent_agreement']:.1f}% | Fleiss kappa "
          f"{c['fleiss_kappa']:.3f} | majority exists for {c['posts_with_majority_pct']:.1f}% | unanimous {c['unanimous_pct']:.1f}%")
    print(f"Final eval.xls agreement: {report['eval_xls']['category_on_final_set']}")
    df_check = pd.read_excel(EVAL_XLS_PATH, keep_default_na=False)
    print(f"[VERIFIED] {EVAL_XLS_PATH.name}: {len(df_check)} rows; sources {dict(Counter(df_check['label_source']))}")
    print(df_check["label"].value_counts().to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"pilot": run_pilot, "main": run_main, "finalize": run_finalize,
     "human": run_human, "finalize_human": run_finalize_human, "llm": run_llm_majority}[mode]()
