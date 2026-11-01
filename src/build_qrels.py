#!/usr/bin/env python3
"""
src/build_qrels.py - Graded Relevance Judgements (qrels) for Channel A Policy Retrieval
Course: NTU AI6127 (2026) — Question 3.

1. Queries: 90 real held-out HateXplain test posts (30 Safe / 30 Hate Speech / 30 Harassment-Offensive).
2. Pooling (TREC style): union of the top-10 clauses returned by every configuration R1–R6.
3. Grading: every (post, clause) pair in the pool is graded 0/1/2 by an LLM relevance assessor
   (Qwen3-235B-A22B-Instruct, a different and larger model than the generator; UMBRELA-style prompt,
   Upadhyay et al., 2024):
     2 = the clause directly governs this post (a moderator would apply it to decide this post)
     1 = related topic, partially applicable
     0 = not relevant
4. Human verification: `python src/build_qrels.py sheet` writes a blind grading sheet (20 queries, pooled
   top-5) for two team members; `python src/build_qrels.py check` reports human–human and human–LLM agreement.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agreement import calculate_cohens_kappa, calculate_percent_agreement
from src.evaluate_retrieval import QRELS_FILE, RETRIEVAL_CONFIGS, policy_queries, run_config
from src.llm import EVALUATOR_MODEL, get_client, parallel_map
from src.retriever import DualChannelRetriever

SHEETS_DIR = PROJECT_ROOT / "data" / "annotation" / "sheets"
CHECK_QUERIES = 20

ASSESSOR_PROMPT = """You are a trust-and-safety policy expert judging search results.
A moderator must decide whether the social-media post below breaks a platform rule. A search engine returned one
passage from a content-policy rulebook. Grade how useful this passage is for deciding THIS post:

2 = directly applicable: this is a rule a moderator would apply to judge this post (whether or not the post actually violates it)
1 = related: same general topic, but not the rule that would decide this post
0 = not relevant

Post: "{post}"

Policy passage ({source} — {section}):
"{clause}"

Reply with JSON only: {{"grade": 0 | 1 | 2, "reason": "<one short sentence>"}}"""


def build_pool(retriever: DualChannelRetriever) -> List[Dict[str, Any]]:
    queries = policy_queries()
    pools = []
    for i, q in enumerate(queries, 1):
        pool = []
        for setting in RETRIEVAL_CONFIGS.values():
            ids, _ = run_config(retriever, q, "policy", setting, top_k=10)
            pool += [x for x in ids if x not in pool]
        pools.append({"query_id": f"Q{i:03d}", "post_id": q["id"], "label": q["label"], "query_raw": q["raw_text"], "pool": pool})
    return pools


def grade_pool(pools: List[Dict[str, Any]], docs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    client = get_client(EVALUATOR_MODEL)
    pairs = [(p, cid) for p in pools for cid in p["pool"]]
    print(f"Grading {len(pairs)} (post, clause) pairs with {EVALUATOR_MODEL}")

    def judge(pair):
        p, cid = pair
        d = docs[cid]
        out = client.chat_json([{"role": "user", "content": ASSESSOR_PROMPT.format(
            post=p["query_raw"], source=d["source"], section=d["section"], clause=" ".join(d["text"].split()[:220]))}],
            max_tokens=120)
        try:
            return int(out.get("grade", 0))
        except (TypeError, ValueError):
            return 0

    grades = parallel_map(judge, pairs, workers=8, desc="qrels")
    by_q: Dict[str, Dict[str, int]] = {}
    for (p, cid), g in zip(pairs, grades):
        by_q.setdefault(p["query_id"], {})[cid] = max(0, min(2, g))
    return [{**{k: v for k, v in p.items() if k != "pool"}, "judged": p["pool"], "qrels": by_q[p["query_id"]]} for p in pools]


def write_check_sheet():
    """Blind sheet: 20 queries x their pooled top-5 clauses, for two human graders."""
    qrels = json.loads(QRELS_FILE.read_text())[:: max(1, 90 // CHECK_QUERIES)][:CHECK_QUERIES]
    docs = {json.loads(l)["id"]: json.loads(l) for l in open(PROJECT_ROOT / "data" / "processed" / "policy_chunks.jsonl")}
    rows = []
    for q in qrels:
        for cid in q["judged"][:5]:
            rows.append({"query_id": q["query_id"], "post": q["query_raw"], "clause_id": cid,
                         "clause": f"{docs[cid]['source']} — {docs[cid]['section']}: {' '.join(docs[cid]['text'].split()[:150])}",
                         "grade (0/1/2)": ""})
    for k in (1, 2):
        pd.DataFrame(rows).to_excel(SHEETS_DIR / f"qrels_check_annotator{k}.xlsx", index=False)
    print(f"[SAVED] {len(rows)} pairs -> {SHEETS_DIR}/qrels_check_annotator{{1,2}}.xlsx "
          "(grade 2 = rule that decides this post, 1 = related, 0 = irrelevant)")


def check_agreement():
    qrels = {q["query_id"]: q["qrels"] for q in json.loads(QRELS_FILE.read_text())}
    sheets = [pd.read_excel(SHEETS_DIR / f"qrels_check_annotator{k}.xlsx") for k in (1, 2)]
    h1 = [str(int(g)) for g in sheets[0]["grade (0/1/2)"]]
    h2 = [str(int(g)) for g in sheets[1]["grade (0/1/2)"]]
    llm = [str(qrels[q][c]) for q, c in zip(sheets[0]["query_id"], sheets[0]["clause_id"])]
    report = {
        "pairs": len(h1),
        "human1_vs_human2": {"percent": round(calculate_percent_agreement(h1, h2), 2), "kappa": round(calculate_cohens_kappa(h1, h2), 4)},
        "human1_vs_llm": {"percent": round(calculate_percent_agreement(h1, llm), 2), "kappa": round(calculate_cohens_kappa(h1, llm), 4)},
        "human2_vs_llm": {"percent": round(calculate_percent_agreement(h2, llm), 2), "kappa": round(calculate_cohens_kappa(h2, llm), 4)},
    }
    out = PROJECT_ROOT / "data" / "annotation" / "qrels_agreement.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "sheet":
        return write_check_sheet()
    if mode == "check":
        return check_agreement()
    retriever = DualChannelRetriever()
    pools = build_pool(retriever)
    qrels = grade_pool(pools, retriever.policy_docs)
    QRELS_FILE.write_text(json.dumps(qrels, indent=2, ensure_ascii=False))
    n_rel = sum(1 for q in qrels if any(g >= 1 for g in q["qrels"].values()))
    print(f"[SAVED] {len(qrels)} judged queries ({n_rel} with >= 1 relevant clause) -> {QRELS_FILE}")
    write_check_sheet()


if __name__ == "__main__":
    main()
