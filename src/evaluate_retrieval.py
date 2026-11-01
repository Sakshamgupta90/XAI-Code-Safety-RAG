#!/usr/bin/env python3
"""
src/evaluate_retrieval.py - Rank-Aware Retrieval Evaluation & Table 1 (Question 3)
Course: NTU AI6127 (2026).

Configurations (each a strict superset of a previous one, so every innovation is isolated):
  R1  BM25 only
  R2  Dense only (BGE-small + FAISS inner product)
  R3  Hybrid: Reciprocal Rank Fusion of R1 and R2 (k = 60)
  R4  R3 + cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) of the top 20
  R5  R4 + HyDE: the query is an LLM-written policy-style description of the post   (Channel A only)
  R6  R4 without microtext normalisation (raw post text)

Channel A (policy clauses): 90 real posts (30 per class) from the held-out HateXplain test split; graded qrels
  (0/1/2) from src/build_qrels.py. Metrics: P@3, R@3, MRR@10, nDCG@3, nDCG@10.
Channel B (precedents): 450 held-out HateXplain test posts (150 per class). A precedent is relevant if its human
  label equals the query's human label (grade 1), grade 2 if it also shares a target group (or both are Safe).
  Metrics: P@5, MRR@10, nDCG@5. (Recall is not meaningful here: thousands of precedents share each label.)

Before/after case studies are mined automatically from the per-query results (no hand-written outcomes).
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_data import CLASSES, load_dataset
from src.generator import generate_hyde_query
from src.normalize import normalize_text
from src.retriever import DualChannelRetriever, RetrievalConfig

RESULTS_DIR = PROJECT_ROOT / "results"
QRELS_FILE = PROJECT_ROOT / "data" / "processed" / "qrels_policy.json"
SEED = 6127

RETRIEVAL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "R1: BM25": {"cfg": dict(use_bm25=True, use_dense=False, use_rrf=False, use_rerank=False), "norm": True, "hyde": False},
    "R2: Dense (BGE + FAISS)": {"cfg": dict(use_bm25=False, use_dense=True, use_rrf=False, use_rerank=False), "norm": True, "hyde": False},
    "R3: Hybrid RRF": {"cfg": dict(use_bm25=True, use_dense=True, use_rrf=True, use_rerank=False), "norm": True, "hyde": False},
    "R4: RRF + cross-encoder": {"cfg": dict(use_bm25=True, use_dense=True, use_rrf=True, use_rerank=True), "norm": True, "hyde": False},
    "R5: R4 + HyDE": {"cfg": dict(use_bm25=True, use_dense=True, use_rrf=True, use_rerank=True), "norm": True, "hyde": True},
    "R6: R4 w/o normalisation": {"cfg": dict(use_bm25=True, use_dense=True, use_rrf=True, use_rerank=True), "norm": False, "hyde": False},
}


def policy_queries(n_per_class: int = 30) -> List[Dict[str, Any]]:
    """90 real held-out posts used as Channel A queries (seeded, stratified)."""
    rng = random.Random(SEED)
    recs = load_dataset("hatexplain_test")
    out = []
    for cls in CLASSES:
        group = [r for r in recs if r["label"] == cls and len(r["text"].split()) >= 6]
        out += rng.sample(group, n_per_class)
    return out


def run_config(retriever: DualChannelRetriever, query: Dict[str, Any], channel: str, setting: Dict[str, Any], top_k: int):
    """Returns (retrieved ids, latency ms) for one query under one configuration."""
    t0 = time.perf_counter()
    text = normalize_text(query["raw_text"]) if setting["norm"] else query["raw_text"]
    if setting["hyde"]:
        text = generate_hyde_query(normalize_text(query["raw_text"]))
    cfg = RetrievalConfig(top_k=top_k, normalize=False, **setting["cfg"])
    hits = retriever.retrieve(text, channel=channel, config=cfg)
    return [h.id for h in hits], (time.perf_counter() - t0) * 1000


def dcg(grades: List[float], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg(grades: List[float], ideal: List[float], k: int) -> float:
    idcg = dcg(sorted(ideal, reverse=True), k)
    return dcg(grades, k) / idcg if idcg > 0 else 0.0


def mrr(grades: List[float], k: int = 10) -> float:
    return next((1.0 / (i + 1) for i, g in enumerate(grades[:k]) if g >= 1), 0.0)


def evaluate_channel_a(retriever) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    qrels = {q["query_id"]: q for q in json.loads(QRELS_FILE.read_text())}
    queries = [q for q in qrels.values() if any(g >= 1 for g in q["qrels"].values())]
    print(f"\nChannel A: {len(queries)} queries with >= 1 relevant clause (of {len(qrels)} judged)")
    rows, per_query = [], {}
    for name, setting in RETRIEVAL_CONFIGS.items():
        p3, r3, rr, n3, n10, lat, detail = [], [], [], [], [], [], []
        for q in queries:
            ids, ms = run_config(retriever, {"raw_text": q["query_raw"]}, "policy", setting, top_k=10)
            grades = [q["qrels"].get(i, 0) for i in ids]
            n_rel = sum(1 for g in q["qrels"].values() if g >= 1)
            p3.append(sum(1 for g in grades[:3] if g >= 1) / 3)
            r3.append(sum(1 for g in grades[:3] if g >= 1) / n_rel)
            rr.append(mrr(grades))
            n3.append(ndcg(grades, list(q["qrels"].values()), 3))
            n10.append(ndcg(grades, list(q["qrels"].values()), 10))
            lat.append(ms)
            detail.append({"query_id": q["query_id"], "label": q["label"], "query": q["query_raw"], "top3": ids[:3],
                           "grades_top3": grades[:3], "ndcg@3": round(n3[-1], 4)})
        rows.append({"Configuration": name, "P@3": round(np.mean(p3), 4), "R@3": round(np.mean(r3), 4),
                     "MRR@10": round(np.mean(rr), 4), "nDCG@3": round(np.mean(n3), 4), "nDCG@10": round(np.mean(n10), 4),
                     "Latency (ms)": round(np.mean(lat), 1)})
        per_query[name] = detail
        print(f"  {name:<28} P@3 {rows[-1]['P@3']:.3f}  R@3 {rows[-1]['R@3']:.3f}  MRR {rows[-1]['MRR@10']:.3f}  "
              f"nDCG@3 {rows[-1]['nDCG@3']:.3f}  ({rows[-1]['Latency (ms)']:.0f} ms)")
    return rows, per_query


def evaluate_channel_b(retriever, n_per_class: int = 150) -> List[Dict[str, Any]]:
    rng = random.Random(SEED + 1)
    recs = load_dataset("hatexplain_test")
    queries = []
    for cls in CLASSES:
        queries += rng.sample([r for r in recs if r["label"] == cls], n_per_class)
    docs = retriever.precedent_docs
    print(f"\nChannel B: {len(queries)} held-out HateXplain posts as queries")

    def grade(q, doc_id):
        d = docs[doc_id]
        if d["category"] != q["label"]:
            return 0
        if q["label"] == "Safe" or set(d.get("targets", [])) & set(q.get("targets", [])):
            return 2
        return 1

    rows = []
    for name, setting in RETRIEVAL_CONFIGS.items():
        if setting["hyde"]:
            continue
        p5, rr, n5, lat = [], [], [], []
        for q in queries:
            ids, ms = run_config(retriever, q, "precedent", setting, top_k=10)
            grades = [grade(q, i) for i in ids]
            p5.append(sum(1 for g in grades[:5] if g >= 1) / 5)
            rr.append(mrr(grades))
            n5.append(ndcg(grades, [2] * 10, 5))  # ideal: the index holds >= 10 grade-2 precedents for every query class
            lat.append(ms)
        rows.append({"Configuration": name, "P@5": round(np.mean(p5), 4), "MRR@10": round(np.mean(rr), 4),
                     "nDCG@5": round(np.mean(n5), 4), "Latency (ms)": round(np.mean(lat), 1)})
        print(f"  {name:<28} P@5 {rows[-1]['P@5']:.3f}  MRR {rows[-1]['MRR@10']:.3f}  nDCG@5 {rows[-1]['nDCG@5']:.3f}  "
              f"({rows[-1]['Latency (ms)']:.0f} ms)")
    return rows


def mine_case_studies(per_query: Dict[str, List[Dict[str, Any]]], n: int = 3) -> List[Dict[str, Any]]:
    """Queries where an earlier configuration failed (nDCG@3 = 0 or low) and the full pipeline (R5) succeeds."""
    full = {d["query_id"]: d for d in per_query["R5: R4 + HyDE"]}
    cases = []
    for before in ("R1: BM25", "R2: Dense (BGE + FAISS)", "R4: RRF + cross-encoder"):
        for d in per_query[before]:
            f = full[d["query_id"]]
            if d["ndcg@3"] <= 0.3 and f["ndcg@3"] >= 0.8 and all(c["query_id"] != d["query_id"] for c in cases):
                cases.append({"before_config": before, "query_id": d["query_id"], "label": d["label"], "query": d["query"],
                              "before_top3": d["top3"], "before_grades": d["grades_top3"], "before_ndcg@3": d["ndcg@3"],
                              "after_top3": f["top3"], "after_grades": f["grades_top3"], "after_ndcg@3": f["ndcg@3"]})
                break
    return cases[:n]


def main():
    print("=" * 78)
    print("XAI-Guard Question 3: Rank-Aware Retrieval Evaluation (Table 1)")
    print("=" * 78)
    retriever = DualChannelRetriever()
    rows_a, per_query = evaluate_channel_a(retriever)
    rows_b = evaluate_channel_b(retriever)
    cases = mine_case_studies(per_query)
    out = {"channel_a_policy": rows_a, "channel_b_precedent": rows_b, "case_studies": cases,
           "channel_a_per_query": per_query}
    out_file = RESULTS_DIR / "table1_retrieval_ablations.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"\nCase studies found: {len(cases)}")
    for c in cases:
        print(f"  [{c['before_config']} -> R5] {c['query'][:80]!r}: nDCG@3 {c['before_ndcg@3']} -> {c['after_ndcg@3']}")
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
