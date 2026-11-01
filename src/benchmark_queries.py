#!/usr/bin/env python3
"""
src/benchmark_queries.py - Benchmark Query Profiler for Question 2

Executes 5 representative benchmark queries through the full XAI-Guard retrieval
and neurosymbolic pipeline, recording warm-start stage-by-stage latencies (ms)
and decision outputs.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force stable threading for CPU execution
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

from src.normalize import normalize_text
from src.retriever import DualChannelRetriever, RetrievalConfig
from src.sentic import extract_sentic_features

BENCHMARK_QUERIES = [
    {
        "id": "Q1_EXPLICIT_HATE",
        "category": "Explicit Hate Speech / Xenophobia",
        "raw_text": "go back to your own country, nobody wants you here",
        "expected_intent": "hatespeech",
        "description": "Direct exclusion based on origin/nationality. Tests core policy and precedent match."
    },
    {
        "id": "Q2_IMPLICIT_SARCASM",
        "category": "Implicit Sarcasm / Stereotyping",
        "raw_text": "Oh wow, another DEI hire pretending to understand engineering, truly inspiring.",
        "expected_intent": "offensive",
        "description": "Subtle sarcasm and gender/diversity stereotyping without explicit profanity."
    },
    {
        "id": "Q3_BENIGN_IDENTITY",
        "category": "Benign Identity / Counter-speech (False Positive Test)",
        "raw_text": "I am proud to be a gay man and no one can take my dignity away.",
        "expected_intent": "normal",
        "description": "Contains sensitive identity terms ('gay') often falsely flagged by unigram keyword filters."
    },
    {
        "id": "Q4_OBFUSCATED_HARASSMENT",
        "category": "Obfuscated / Leetspeak Harassment",
        "raw_text": "u r so d*mb, go k1ll urself u worthless piece of tr@sh",
        "expected_intent": "hatespeech",
        "description": "Masked slurs and suicide incitement ('k1ll', 'tr@sh', 'd*mb') testing microtext normalization."
    },
    {
        "id": "Q5_SAFE_NEUTRAL",
        "category": "Neutral Everyday Conversation",
        "raw_text": "Can someone recommend a good beginner tutorial for Python and machine learning?",
        "expected_intent": "normal",
        "description": "Standard safe query acting as a true negative baseline."
    }
]


def run_benchmark():
    print("=" * 70)
    print("XAI-Guard Question 2: 5 Benchmark Queries Profiler")
    print("=" * 70)

    # Pre-warm retriever and SenticNet
    print("Loading and warming up models...")
    retriever = DualChannelRetriever()

    cfg_policy = RetrievalConfig(top_k=3, use_bm25=True, use_dense=True, use_rrf=True, use_rerank=True)
    cfg_prec = RetrievalConfig(top_k=5, use_bm25=True, use_dense=True, use_rrf=True, use_rerank=True)

    # Warm-up run to avoid cold-start bias
    _ = normalize_text("warmup query text")
    _ = retriever.retrieve("warmup query", channel="policy", config=cfg_policy)
    _ = retriever.retrieve("warmup query", channel="precedent", config=cfg_prec)
    _ = extract_sentic_features("warmup query")
    print("Models pre-warmed successfully.\n")

    results = []

    for q in BENCHMARK_QUERIES:
        print(f"--- Running: {q['id']} ({q['category']}) ---")
        raw_text = q["raw_text"]

        # Stage 1: Text Normalization
        t0 = time.perf_counter()
        clean_text = normalize_text(raw_text)
        t_norm_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 2: SenticNet Neurosymbolic Analysis
        t0 = time.perf_counter()
        sentic_res = extract_sentic_features(clean_text)
        t_sentic_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 3: Channel A Retrieval (Policy Clauses)
        t0 = time.perf_counter()
        policy_results = retriever.retrieve(clean_text, channel="policy", config=cfg_policy)
        t_policy_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 4: Channel B Retrieval (Precedents)
        t0 = time.perf_counter()
        precedent_results = retriever.retrieve(clean_text, channel="precedent", config=cfg_prec)
        t_prec_ms = (time.perf_counter() - t0) * 1000.0

        # Calculate consensus from precedents
        labels = [p.category_or_section for p in precedent_results if p.category_or_section]
        label_counts = {}
        for l in labels:
            label_counts[l] = label_counts.get(l, 0) + 1
        majority_label = max(label_counts, key=label_counts.get) if label_counts else "normal"
        consensus_confidence = (label_counts.get(majority_label, 0) / len(labels)) if labels else 0.0

        total_latency_ms = t_norm_ms + t_sentic_ms + t_policy_ms + t_prec_ms

        entry = {
            "query_id": q["id"],
            "category": q["category"],
            "description": q["description"],
            "raw_text": raw_text,
            "clean_text": clean_text,
            "expected_intent": q["expected_intent"],
            "derived_label": majority_label,
            "precedent_consensus": label_counts,
            "consensus_confidence": round(consensus_confidence, 4),
            "sentic_polarity": sentic_res.polarity,
            "dominant_emotion": sentic_res.dominant_emotion,
            "pleasantness": sentic_res.pleasantness,
            "aptitude": sentic_res.aptitude,
            "sensitivity": sentic_res.sensitivity,
            "matched_concepts": sentic_res.matched_concepts,
            "retrieved_policies": [
                {
                    "clause_id": p.id,
                    "source": p.source,
                    "section": p.category_or_section,
                    "score": round(float(p.score), 4)
                }
                for p in policy_results
            ],
            "top_precedents": [
                {
                    "post_id": p.id,
                    "label": p.category_or_section,
                    "text": p.text[:120] + "...",
                    "score": round(float(p.score), 4)
                }
                for p in precedent_results[:3]
            ],
            "latency_breakdown_ms": {
                "normalization": round(t_norm_ms, 2),
                "senticnet": round(t_sentic_ms, 2),
                "channel_a_policy": round(t_policy_ms, 2),
                "channel_b_precedent": round(t_prec_ms, 2),
                "total_retrieval_and_feature_pipeline": round(total_latency_ms, 2)
            }
        }
        results.append(entry)

        print(f"Clean Text: \"{clean_text}\"")
        print(f"Sentic: {sentic_res.summary_text}")
        print(f"Precedent Consensus: {label_counts} -> {majority_label} ({consensus_confidence*100:.1f}%)")
        print(f"Top Policy: {policy_results[0].id} ({policy_results[0].source} - {policy_results[0].category_or_section})")
        print(f"Total Pipeline Latency: {total_latency_ms:.2f} ms\n")

    # Save to results/question2_benchmarks.json
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "question2_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print(f"Benchmark profiling complete! Saved results to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
