#!/usr/bin/env python3
"""
src/evaluate_downstream.py - Downstream Evaluation & Ablation Study (Questions 4 & 5, report Table 4)
Course: NTU AI6127 (2026).

Runs every system configuration on a benchmark and scores it with the same code:

  G0   TF-IDF + Logistic Regression (classical baseline, trained on the HateXplain precedents)
  G1   zero-shot LLM, no retrieval
  G2   dual-channel RAG, plain prompt
  G3   G2 + SenticNet 9 features            G4   G2 + suitcase chain-of-thought
  G5   G2 + critic (abstention)             G3+4, G3+5, G4+5 (pairs)
  G6   full system: RAG + SenticNet + CoT + critic

Only 5 LLM runs are needed (G1, G2, G3, G4, G3+4); the critic variants re-use the same LLM outputs.

Stages (all cached, so re-running is free):
  1. Evidence: HyDE policy query (LLM) + Channel A top-3 (hybrid RRF + cross-encoder) + Channel B top-5 (hybrid RRF)
  2. LLM judge for each configuration (parallel API calls)
  3. Critic (DeBERTa NLI) for the critic configurations
  4. Metrics: per-class P/R/F1, macro-F1, accuracy, coverage, selective/fallback F1, attribute F1 (eval.xls),
     rationale token-F1 (HateXplain), latency, LLM tokens and cost.

Usage:
  python src/evaluate_downstream.py --dataset hatexplain_dev --calibrate   # pick critic thresholds on dev
  python src/evaluate_downstream.py --dataset hatexplain_test
  python src/evaluate_downstream.py --dataset eval_xls
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.critic import THRESHOLDS_FILE, SafetyCritic, load_thresholds
from src.eval_data import CLASSES, load_dataset
from src.generator import ExplainableSafetyReasoner, ModerationConfig, generate_hyde_query
from src.llm import GENERATOR_MODEL, get_client, parallel_map, usage_cost
from src.normalize import normalize_text
from src.retriever import RetrievalConfig, SearchResult

RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"

LLM_CONFIGS = {
    "G1": ModerationConfig(use_rag=False, use_sentic=False, use_cot=False, use_critic=False),
    "G2": ModerationConfig(use_rag=True, use_sentic=False, use_cot=False, use_critic=False),
    "G3": ModerationConfig(use_rag=True, use_sentic=True, use_cot=False, use_critic=False),
    "G4": ModerationConfig(use_rag=True, use_sentic=False, use_cot=True, use_critic=False),
    "G3+4": ModerationConfig(use_rag=True, use_sentic=True, use_cot=True, use_critic=False),
}
# Table 2 rows: (row id, description, LLM run, critic on/off)
TABLE2_ROWS = [
    ("G0", "TF-IDF + Logistic Regression baseline", None, False),
    ("G1", "Zero-shot LLM (no retrieval)", "G1", False),
    ("G2", "Dual-channel RAG, plain prompt", "G2", False),
    ("G3", "G2 + SenticNet 9", "G3", False),
    ("G4", "G2 + suitcase CoT", "G4", False),
    ("G5", "G2 + critic", "G2", True),
    ("G3+4", "G2 + SenticNet + CoT", "G3+4", False),
    ("G3+5", "G2 + SenticNet + critic", "G3", True),
    ("G4+5", "G2 + CoT + critic", "G4", True),
    ("G6", "Full system (SenticNet + CoT + critic)", "G3+4", True),
]


# ---------------------------------------------------------------- stage 1: evidence
def build_evidence(dataset: str, records: List[Dict[str, Any]], reasoner: ExplainableSafetyReasoner) -> Dict[str, Any]:
    cache_file = CACHE_DIR / f"evidence_{dataset}.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    todo = [r for r in records if r["id"] not in cache]
    if todo:
        print(f"\n[Stage 1] Evidence for {len(todo)} posts: HyDE (LLM) + dual-channel retrieval")
        texts = [normalize_text(r["text"]) for r in todo]
        hydes = parallel_map(generate_hyde_query, texts, workers=8, desc="HyDE")
        t0 = time.perf_counter()
        for i, (r, text, hyde) in enumerate(zip(todo, texts, hydes), 1):
            t_a = time.perf_counter()
            pol = reasoner.retriever.retrieve(hyde, channel="policy", config=RetrievalConfig(top_k=3, use_rerank=True, normalize=False))
            t_b = time.perf_counter()
            prec = reasoner.retriever.retrieve(text, channel="precedent", config=RetrievalConfig(top_k=5, use_rerank=False))
            t_c = time.perf_counter()
            cache[r["id"]] = {"hyde": hyde, "policies": [p.to_dict() for p in pol], "precedents": [p.to_dict() for p in prec],
                              "policy_ms": (t_b - t_a) * 1000, "precedent_ms": (t_c - t_b) * 1000}
            if i % 100 == 0 or i == len(todo):
                print(f"  [retrieval] {i}/{len(todo)} ({time.perf_counter() - t0:.0f}s)", flush=True)
                cache_file.write_text(json.dumps(cache))
        cache_file.write_text(json.dumps(cache))
    return cache


# ---------------------------------------------------------------- stage 2: LLM judge
def run_llm_config(name: str, cfg: ModerationConfig, records, evidence, reasoner) -> List[Dict[str, Any]]:
    print(f"\n[Stage 2] LLM judge {name} ({cfg.name()}) on {len(records)} posts")

    def one(r):
        ev = evidence[r["id"]]
        pol = [SearchResult(**p) for p in ev["policies"]]
        prec = [SearchResult(**p) for p in ev["precedents"]]
        t0 = time.perf_counter()
        out = reasoner.reason(r["text"], cfg, policies=pol, precedents=prec, hyde_query=ev["hyde"])
        d = out.to_dict()
        d["wall_ms"] = (time.perf_counter() - t0) * 1000
        return d

    return parallel_map(one, records, workers=8, desc=name)


# ---------------------------------------------------------------- stage 3: critic
def apply_critic(outputs: List[Dict[str, Any]], critic: SafetyCritic) -> List[Dict[str, Any]]:
    decisions = []
    for o in outputs:
        t0 = time.perf_counter()
        dec = critic.evaluate(
            predicted_category=o["category"],
            post_text=o["_clean_text"],
            cited_clause_id=o["cited_clause"],
            cited_precedent_ids=o["cited_precedents"],
            violation_claim=o["violation_claim"],
            retrieved_policies={p["id"]: p["text"] for p in o["retrieved_policies"]},
            precedent_labels={p["id"]: p["category_or_section"] for p in o["retrieved_precedents"]},
        ).to_dict()
        dec["critic_ms"] = (time.perf_counter() - t0) * 1000
        decisions.append(dec)
    return decisions


# ---------------------------------------------------------------- stage 4: metrics
def classification_metrics(gold: List[str], pred: List[str]) -> Dict[str, Any]:
    if not gold:
        return {"n": 0}
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=CLASSES, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(gold, pred, labels=CLASSES, average="macro", zero_division=0)
    return {
        "n": len(gold),
        "accuracy": round(accuracy_score(gold, pred), 4),
        "macro_precision": round(mp, 4), "macro_recall": round(mr, 4), "macro_f1": round(mf, 4),
        "per_class": {c: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"labels": CLASSES, "matrix": confusion_matrix(gold, pred, labels=CLASSES).tolist()},
    }


def selective_metrics(gold, llm_pred, final, fallback) -> Dict[str, Any]:
    covered = [i for i, f in enumerate(final) if f != "Needs Human Review"]
    abstained = [i for i in range(len(final)) if i not in set(covered)]
    fb = [final[i] if i in set(covered) else fallback[i] for i in range(len(final))]
    sel = classification_metrics([gold[i] for i in covered], [final[i] for i in covered])
    return {
        "coverage": round(len(covered) / len(gold), 4),
        "abstained": len(abstained),
        "selective": sel,
        "fallback": classification_metrics(gold, fb),
        # Was the critic catching errors? LLM accuracy on accepted vs abstained posts
        "llm_accuracy_on_accepted": round(np.mean([llm_pred[i] == gold[i] for i in covered]), 4) if covered else None,
        "llm_accuracy_on_abstained": round(np.mean([llm_pred[i] == gold[i] for i in abstained]), 4) if abstained else None,
    }


def rationale_f1(pred_phrases: List[str], gold_tokens: List[str]) -> Optional[float]:
    if not gold_tokens:
        return None
    pred = {t for ph in pred_phrases for t in re.findall(r"[a-z0-9']+", ph.lower())}
    gold = {t for g in gold_tokens for t in re.findall(r"[a-z0-9']+", g.lower())}
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def attribute_metrics(records, outputs) -> Dict[str, Any]:
    res = {}
    if "subjectivity" in records[0]:
        g = [r["subjectivity"] for r in records]
        res["subjectivity_macro_f1"] = round(f1_score(g, [o["subjectivity"] for o in outputs], average="macro", zero_division=0), 4)
        g = [r["polarity"] for r in records]
        res["polarity_macro_f1"] = round(f1_score(g, [o["polarity"] for o in outputs], average="macro", zero_division=0), 4)
        g = [r["is_sarcastic"] for r in records]
        res["sarcasm_macro_f1"] = round(f1_score(g, [bool(o["is_sarcastic"]) for o in outputs], average="macro", zero_division=0), 4)
    if "rationale_tokens" in records[0]:
        scores = [rationale_f1(o["trigger_words"], r["rationale_tokens"]) for r, o in zip(records, outputs)
                  if r["label"] != "Safe"]
        scores = [s for s in scores if s is not None]
        res["rationale_token_f1"] = round(float(np.mean(scores)), 4) if scores else None
        res["rationale_posts"] = len(scores)
    return res


def latency_summary(values: List[float]) -> Dict[str, float]:
    return {"mean_ms": round(float(np.mean(values)), 1), "p95_ms": round(float(np.percentile(values, 95)), 1)}


# ---------------------------------------------------------------- calibration on dev
def calibrate(records, outputs_by_cfg, min_coverage: float = 0.80) -> Dict[str, Any]:
    """Grid search of critic thresholds on the dev split: maximise selective macro-F1 with coverage >= min_coverage."""
    base = outputs_by_cfg["G3+4"]
    gold = [r["label"] for r in records]
    fallback = [o["precedent_consensus"]["label"] or o["category"] for o in base]
    llm_pred = [o["category"] for o in base]
    grid = []
    for claim, rule, safe in product([0.0, 0.1, 0.2, 0.3, 0.5], [0.0, 0.3, 0.5], [0.8, 0.9, 0.95, 1.01]):
        t = {"claim_entail_min": claim, "rule_entail_min": rule, "safe_attack_max": safe, "consensus_min_votes": 4}
        final = [d["final_verdict"] for d in apply_critic(base, SafetyCritic(thresholds=t))]
        m = selective_metrics(gold, llm_pred, final, fallback)
        grid.append({"thresholds": t, "coverage": m["coverage"], "selective_macro_f1": m["selective"].get("macro_f1", 0.0)})
    ok = [g for g in grid if g["coverage"] >= min_coverage] or grid
    best = max(ok, key=lambda g: (g["selective_macro_f1"], g["coverage"]))
    payload = {"chosen_on": "hatexplain_dev", "rule": f"max selective macro-F1 s.t. coverage >= {min_coverage}",
               "thresholds": best["thresholds"], "dev_coverage": best["coverage"],
               "dev_selective_macro_f1": best["selective_macro_f1"], "grid": grid}
    Path(THRESHOLDS_FILE).write_text(json.dumps(payload, indent=2))
    print(f"[CALIBRATED] {best['thresholds']} -> dev coverage {best['coverage']:.3f}, selective F1 {best['selective_macro_f1']:.4f}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hatexplain_test", choices=["hatexplain_dev", "hatexplain_test", "eval_xls"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print(f"XAI-Guard downstream evaluation on {args.dataset}")
    print("=" * 78)
    records = load_dataset(args.dataset)
    if args.limit:
        records = records[: args.limit]
    gold = [r["label"] for r in records]
    print(f"{len(records)} posts: {dict(Counter(gold))}")

    reasoner = ExplainableSafetyReasoner()
    evidence = build_evidence(args.dataset, records, reasoner)
    client = get_client(GENERATOR_MODEL)
    pred_dir = RESULTS_DIR / "predictions" / args.dataset
    pred_dir.mkdir(parents=True, exist_ok=True)

    outputs_by_cfg: Dict[str, List[Dict[str, Any]]] = {}
    usage_by_cfg: Dict[str, Dict[str, int]] = {}
    for name, cfg in LLM_CONFIGS.items():
        before = dict(client.usage)
        t_run = time.perf_counter()
        outs = run_llm_config(name, cfg, records, evidence, reasoner)
        wall_s = time.perf_counter() - t_run
        for r, o in zip(records, outs):
            o["_clean_text"] = normalize_text(r["text"])
        outputs_by_cfg[name] = outs
        pt = client.usage["prompt_tokens"] + client.usage.get("cached_prompt_tokens", 0) - before["prompt_tokens"] - before.get("cached_prompt_tokens", 0)
        ct = client.usage["completion_tokens"] + client.usage.get("cached_completion_tokens", 0) - before["completion_tokens"] - before.get("cached_completion_tokens", 0)
        usage_by_cfg[name] = {"prompt_tokens": pt, "completion_tokens": ct, "wall_s": round(wall_s, 1),
                              "fresh_calls": client.usage["calls"] - before["calls"],
                              # throughput with 8 parallel API requests; only meaningful when the calls were not cached
                              "posts_per_s_8_parallel": round(len(records) / wall_s, 2)}

    if args.calibrate:
        calibrate(records, outputs_by_cfg)

    critic = SafetyCritic(thresholds=load_thresholds())
    print(f"\n[Stage 3] Critic with thresholds {critic.t}")
    critic_by_cfg = {name: apply_critic(outputs_by_cfg[name], critic) for name in ("G2", "G3", "G4", "G3+4")}

    # G0 baseline
    import joblib
    g0 = joblib.load(PROJECT_ROOT / "models" / "g0_tfidf_lr.joblib")
    t0 = time.perf_counter()
    g0_pred = list(g0.predict([normalize_text(r["text"]) for r in records]))
    g0_ms = (time.perf_counter() - t0) * 1000 / len(records)

    hyde_ms = [0.0] * len(records)  # HyDE latency is part of the LLM wall time measured below
    rows, details = [], {}
    for row_id, desc, llm_name, use_critic in TABLE2_ROWS:
        if llm_name is None:
            m = classification_metrics(gold, g0_pred)
            rows.append({"row": row_id, "description": desc, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                         "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"], "coverage": 1.0,
                         "selective_f1": m["macro_f1"], "fallback_f1": m["macro_f1"], "mean_latency_ms": round(g0_ms, 2),
                         "cost_per_1k_usd": 0.0})
            details[row_id] = {"metrics": m}
            continue
        outs = outputs_by_cfg[llm_name]
        llm_pred = [o["category"] for o in outs]
        m = classification_metrics(gold, llm_pred)
        lat = [o["wall_ms"] + (evidence[r["id"]]["policy_ms"] + evidence[r["id"]]["precedent_ms"] if LLM_CONFIGS[llm_name].use_rag else 0)
               for r, o in zip(records, outs)]
        usage = usage_by_cfg[llm_name]
        cost = usage_cost(GENERATOR_MODEL, usage["prompt_tokens"], usage["completion_tokens"]) / len(records) * 1000
        row = {"row": row_id, "description": desc, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
               "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"], "coverage": 1.0,
               "selective_f1": m["macro_f1"], "fallback_f1": m["macro_f1"], "mean_latency_ms": latency_summary(lat)["mean_ms"],
               "cost_per_1k_usd": round(cost, 4), "parse_errors": sum(o["parse_error"] for o in outs)}
        detail = {"metrics": m, "attributes": attribute_metrics(records, outs), "latency": latency_summary(lat), "usage": usage}
        if use_critic:
            decs = critic_by_cfg[llm_name]
            final = [d["final_verdict"] for d in decs]
            fallback = [o["precedent_consensus"]["label"] or o["category"] for o in outs]
            sm = selective_metrics(gold, llm_pred, final, fallback)
            lat = [l + d["critic_ms"] for l, d in zip(lat, decs)]
            row.update({"coverage": sm["coverage"], "selective_f1": sm["selective"].get("macro_f1"),
                        "fallback_f1": sm["fallback"]["macro_f1"], "macro_f1": sm["fallback"]["macro_f1"],
                        "accuracy": sm["fallback"]["accuracy"], "mean_latency_ms": latency_summary(lat)["mean_ms"]})
            detail.update({"selective": sm, "latency": latency_summary(lat),
                           "abstention_reasons": Counter(r.split(":")[0] for d in decs if d["abstention_reason"]
                                                         for r in d["abstention_reason"].split("; ")).most_common()})
        rows.append(row)
        details[row_id] = detail

    # Save predictions (one JSONL per LLM run, with critic decisions attached where used)
    for name, outs in outputs_by_cfg.items():
        with open(pred_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i, (r, o) in enumerate(zip(records, outs)):
                rec = {"id": r["id"], "gold": r["label"], **{k: v for k, v in o.items() if not k.startswith("_")}}
                if name in critic_by_cfg:
                    rec["critic_decision"] = critic_by_cfg[name][i]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(pred_dir / "G0.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(records, g0_pred):
            f.write(json.dumps({"id": r["id"], "gold": r["label"], "category": p}) + "\n")

    summary = {"dataset": args.dataset, "n": len(records), "generator_model": GENERATOR_MODEL,
               "critic_thresholds": critic.t, "table2": rows, "details": details,
               "llm_usage_total": client.usage}
    out_file = RESULTS_DIR / f"downstream_{args.dataset}.json"
    out_file.write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 110)
    print(f"Table 2 — {args.dataset} (n={len(records)})")
    print("=" * 110)
    print(f"{'Row':<6} {'Description':<40} {'MacroF1':>8} {'Acc':>6} {'Cover':>6} {'SelF1':>6} {'Lat(ms)':>8} {'$/1k':>7}")
    for r in rows:
        print(f"{r['row']:<6} {r['description']:<40} {r['macro_f1']:>8.4f} {r['accuracy']:>6.3f} {r['coverage']:>6.3f} "
              f"{(r['selective_f1'] or 0):>6.3f} {r['mean_latency_ms']:>8.0f} {r['cost_per_1k_usd']:>7.3f}")
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
