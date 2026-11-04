#!/usr/bin/env python3
"""
src/evaluate_ragas.py - Answer-Level RAG Metrics (RAGAS, Es et al., 2023)
Course: NTU AI6127 (2026) — Question 4 (faithfulness, answer relevance, context precision) and report Table 6.

Implements the three RAGAS metric definitions directly (the `ragas` package pins numpy/langchain versions
that conflict with this project's FAISS/torch stack), using an LLM judge that is different from and larger
than the generator (Qwen3-235B-A22B-Instruct) and the project's BGE embeddings:

- Faithfulness      = (# answer statements supported by the retrieved context) / (# statements)
                      step 1: the judge splits the answer into atomic statements;
                      step 2: the judge gives a 0/1 support verdict for each statement given the context.
- Answer relevance  = mean cosine( embed(question), embed(q_i) ) over 3 questions the judge generates from the
                      answer; 0 if the judge marks the answer as noncommittal.
- Context precision = average precision over the ranked contexts, where the judge marks each context as useful
                      (1) or not (0) for reaching the answer (RAGAS "without reference" variant; all contexts are
                      judged in one call).

Evaluated on a fixed, stratified 200-post subset of the benchmark for each LLM configuration.

Usage: python src/evaluate_ragas.py --dataset hatexplain_test
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_data import CLASSES
from src.llm import EVALUATOR_MODEL, get_client, parallel_map

RESULTS_DIR = PROJECT_ROOT / "results"
SUBSET_SIZE = 200
SEED = 6127

QUESTION_TEMPLATE = ("Does the following social-media post violate a content policy? Classify it as Safe, "
                     "Hate Speech or Harassment-Offensive and explain why.\nPost: \"{post}\"")

STATEMENTS_PROMPT = """Given a question and an answer, break the answer down into simple, self-contained atomic statements
(each understandable without pronouns). Do not add information.

Question: {question}
Answer: {answer}

Reply with JSON only: {{"statements": ["...", "..."]}}"""

VERDICT_PROMPT = """Your task is to judge the faithfulness of statements based on a given context. For each statement,
return verdict 1 if the statement can be directly inferred from the context, or 0 if it cannot.
The post being discussed is part of the context.

Context:
{context}

Statements:
{statements}

Reply with JSON only: {{"verdicts": [{{"statement": "...", "reason": "...", "verdict": 0 | 1}}, ...]}}"""

QUESTION_GEN_PROMPT = """Generate 3 different questions for which the given answer would be a good answer, and say whether
the answer is noncommittal (evasive, vague, "I don't know"): noncommittal = 1 if so, else 0.

Answer: {answer}

Reply with JSON only: {{"questions": ["...", "...", "..."], "noncommittal": 0 | 1}}"""

CONTEXT_PRECISION_PROMPT = """Given a question, an answer and a ranked list of retrieved contexts, decide for EACH context
whether it was useful in arriving at the given answer (verdict 1) or not (verdict 0).

Question: {question}
Answer: {answer}

Contexts:
{contexts}

Reply with JSON only: {{"verdicts": [0 | 1, ...]}}   (one verdict per context, in order)"""


def stratified_subset(ids_labels: List[tuple], n: int = SUBSET_SIZE) -> List[str]:
    rng = random.Random(SEED)
    per = n // len(CLASSES)
    chosen = []
    for cls in CLASSES:
        ids = sorted(i for i, l in ids_labels if l == cls)
        rng.shuffle(ids)
        chosen += ids[:per + (1 if cls == CLASSES[0] and n % len(CLASSES) else 0)]
    return sorted(chosen)


def contexts_of(pred: Dict[str, Any]) -> List[str]:
    ctx = [f"Policy clause [{p['id']}] ({p['source']}): {' '.join(p['text'].split()[:170])}" for p in pred["retrieved_policies"]]
    ctx += [f"Precedent [{p['id']}] human label {p['category_or_section']}: \"{' '.join(p['text'].split()[:60])}\""
            for p in pred["retrieved_precedents"]]
    return ctx


class RagasJudge:
    def __init__(self):
        self.client = get_client(EVALUATOR_MODEL)
        self._encoder = None

    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._encoder

    def faithfulness(self, question: str, answer: str, post: str, contexts: List[str]) -> float:
        st = self.client.chat_json([{"role": "user", "content": STATEMENTS_PROMPT.format(question=question, answer=answer)}],
                                   max_tokens=500).get("statements") or []
        st = [s for s in st if isinstance(s, str) and s.strip()]
        if not st:
            return 0.0
        ctx = f"Post: \"{post}\"\n" + "\n".join(contexts)
        verdicts = self.client.chat_json([{"role": "user", "content": VERDICT_PROMPT.format(
            context=ctx, statements="\n".join(f"{i + 1}. {s}" for i, s in enumerate(st)))}], max_tokens=1200).get("verdicts") or []
        vals = [1 if str(v.get("verdict")) == "1" else 0 for v in verdicts if isinstance(v, dict)][:len(st)]
        return sum(vals) / len(st)

    def questions(self, answer: str) -> Dict[str, Any]:
        return self.client.chat_json([{"role": "user", "content": QUESTION_GEN_PROMPT.format(answer=answer)}], max_tokens=300)

    def context_precision(self, question: str, answer: str, contexts: List[str]) -> float:
        out = self.client.chat_json([{"role": "user", "content": CONTEXT_PRECISION_PROMPT.format(
            question=question, answer=answer, contexts="\n".join(f"{i + 1}. {c}" for i, c in enumerate(contexts)))}],
            max_tokens=200).get("verdicts") or []
        v = [1 if str(x) == "1" else 0 for x in out][:len(contexts)]
        if sum(v) == 0:
            return 0.0
        ap, hits = 0.0, 0
        for k, x in enumerate(v, 1):
            if x:
                hits += 1
                ap += hits / k
        return ap / sum(v)


def evaluate_config(judge: RagasJudge, preds: List[Dict[str, Any]], use_context: bool) -> Dict[str, Any]:
    def one(p):
        post = p.get("_post") or ""
        question = QUESTION_TEMPLATE.format(post=post)
        answer = f"Verdict: {p['category']}. {p['explanation']}"
        ctx = contexts_of(p) if use_context else []
        q = judge.questions(answer)
        return {
            "faithfulness": judge.faithfulness(question, answer, post, ctx) if use_context else None,
            "context_precision": judge.context_precision(question, answer, ctx) if use_context else None,
            "generated_questions": q.get("questions") or [],
            "noncommittal": int(q.get("noncommittal") or 0),
            "question": question,
        }

    res = parallel_map(one, preds, workers=8, desc="ragas")
    enc = judge.encoder()
    rel = []
    for r in res:
        if r["noncommittal"] or not r["generated_questions"]:
            rel.append(0.0)
            continue
        v = enc.encode([r["question"]] + r["generated_questions"], normalize_embeddings=True)
        rel.append(float(np.mean(v[1:] @ v[0])))
    out = {"n": len(preds), "answer_relevance": round(float(np.mean(rel)), 4)}
    if use_context:
        out["faithfulness"] = round(float(np.mean([r["faithfulness"] for r in res])), 4)
        out["context_precision"] = round(float(np.mean([r["context_precision"] for r in res])), 4)
    out["per_post"] = [{"faithfulness": r["faithfulness"], "context_precision": r["context_precision"], "answer_relevance": a}
                       for r, a in zip(res, rel)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hatexplain_test")
    ap.add_argument("--configs", default="G1,G2,G3,G4,G3+4")
    args = ap.parse_args()

    pred_dir = RESULTS_DIR / "predictions" / args.dataset
    from src.eval_data import load_dataset
    records = {r["id"]: r for r in load_dataset(args.dataset)}
    subset = stratified_subset([(i, r["label"]) for i, r in records.items()])
    judge = RagasJudge()
    results: Dict[str, Any] = {"dataset": args.dataset, "subset_size": len(subset), "judge_model": EVALUATOR_MODEL, "configs": {}}

    for name in args.configs.split(","):
        preds = {json.loads(l)["id"]: json.loads(l) for l in open(pred_dir / f"{name}.jsonl", encoding="utf-8")}
        rows = []
        for i in subset:
            p = preds[i]
            p["_post"] = records[i]["text"]
            rows.append(p)
        print(f"\nRAGAS for {name} on {len(rows)} posts")
        m = evaluate_config(judge, rows, use_context=name != "G1")
        if name != "G1" and "critic_decision" in rows[0]:
            acc = [k for k, p in enumerate(rows) if p["critic_decision"]["final_verdict"] != "Needs Human Review"]
            abst = [k for k in range(len(rows)) if k not in set(acc)]
            m["faithfulness_critic_accepted"] = round(float(np.mean([m["per_post"][k]["faithfulness"] for k in acc])), 4) if acc else None
            m["faithfulness_critic_abstained"] = round(float(np.mean([m["per_post"][k]["faithfulness"] for k in abst])), 4) if abst else None
        results["configs"][name] = m
        print({k: v for k, v in m.items() if k != "per_post"})

    results["evaluator_usage"] = judge.client.usage
    out = RESULTS_DIR / f"ragas_{args.dataset}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
