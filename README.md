# XAI-Guard: Explainable Retrieval-Augmented Content Moderation

NTU AI6127 Deep Neural Networks for NLP (2026), group assignment.

XAI-Guard moderates a social-media post and explains its decision. It retrieves
**(A)** the platform or statutory policy clauses that apply (Meta, X, Reddit, YouTube, Singapore IMDA) and
**(B)** the most similar human-labelled posts (HateXplain). An open-weight LLM (Qwen3-30B-A3B-Instruct) then decides
**Safe / Hate Speech / Harassment-Offensive**, along with subjectivity, polarity and sarcasm, cites its evidence, and suggests a
respectful rewrite. A DeBERTa-NLI critic checks that the verdict is supported by the evidence. When it is not, the post goes
to **human review** instead of receiving an unsupported verdict.

```
post ─► microtext normalisation ─► SenticNet 9 features
          │
          ├─► HyDE policy query (LLM) ─► Channel A: BM25 + BGE/FAISS ─► RRF ─► cross-encoder ─► top-3 clauses
          └─► Channel B: BM25 + BGE/FAISS ─► RRF ─────────────────────────────────────────► top-5 precedents
                                    │
                LLM judge (suitcase CoT: subjectivity → polarity → sarcasm → target → policy) ─► JSON verdict
                                    │
     critic: citation check + DeBERTa NLI (claim ⊨ post, clause ⊨ category) + precedent consensus
                                    │
                    verdict + citations + highlighted triggers   |   "Needs Human Review"
```

## 1. Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium      # only needed to re-collect the policy pages
echo "NEBIUS_API_KEY=<your key>" > .env    # Nebius Token Factory (OpenAI-compatible); never commit this file
```

Models are downloaded automatically on first use: `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2` and
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. The LLMs are `Qwen/Qwen3-30B-A3B-Instruct-2507` (generator) and
`Qwen/Qwen3-235B-A22B-Instruct-2507` (relevance assessor and RAGAS judge), both served by Nebius. Override them with
`XAI_GENERATOR_MODEL` / `XAI_EVALUATOR_MODEL`. Every LLM response is cached in `results/cache/llm_cache.sqlite`, so
re-running an evaluation reproduces the same outputs at no cost.

**SenticNet 9:** download `https://sentic.net/senticnet.zip` (non-commercial licence) and convert it with the snippet in
`data/raw/senticnet/README.md`. The converted `senticnet9.tsv` is already included in the data zip.

## 2. Build order

| Step | Command | Output |
|---|---|---|
| 1 | `python src/collect_policies.py` | `data/raw/policies/*.md` (20 official pages, with URL and retrieval time), `provenance.json` |
| 2 | `python src/data_prep.py` | `data/processed/policy_chunks.jsonl` (142 clauses), `precedent_docs.jsonl` (17,305), `hatexplain_test.jsonl`, `corpus_stats.json` |
| 3 | `python src/build_indexes.py` | `indexes/` (BM25 pickles, BGE embeddings, FAISS) |
| 4 | `python src/train_baseline.py` | `models/g0_tfidf_lr.joblib` |
| 5 | `python src/build_eval_pool.py` | blank annotation workbooks in `data/annotation/sheets/` |
| 6 | **Human annotation**, see `data/annotation/ANNOTATION_GUIDELINES.md` | completed workbooks |
| 7 | `python src/agreement.py pilot` → `main` → (adjudicate `disagreements.xlsx`) → `finalize` | `data/eval.xls`, `agreement_report.json` |
| 8 | `python src/build_qrels.py` (then `sheet` / `check` for the human verification) | `data/processed/qrels_policy.json` |
| 9 | `python src/evaluate_retrieval.py` | `results/table1_retrieval_ablations.json` (Table 1) |
| 10 | `python src/evaluate_downstream.py --dataset hatexplain_dev --calibrate` | `results/critic_thresholds.json` |
| 11 | `python src/evaluate_downstream.py --dataset hatexplain_test` and `--dataset eval_xls` | `results/downstream_*.json`, `results/predictions/` (Table 2) |
| 12 | `python src/evaluate_ragas.py --dataset hatexplain_test` and `--dataset eval_xls` | `results/ragas_*.json` |
| 13 | `python src/benchmark_queries.py` | `results/question2_benchmarks.json` (5 queries and latency) |
| 14 | `streamlit run app.py` | the UI |

## 3. Code map

| File | Role |
|---|---|
| `src/collect_policies.py` | Fetches the policy pages (Playwright for JavaScript pages, Trafilatura extraction, pypdf for IMDA) |
| `src/data_prep.py` | HateXplain ingestion (majority label, rationales, targets), structure-aware policy chunking, corpus statistics |
| `src/normalize.py` | Microtext normalisation (URLs, mentions, emoji, elongation, hashtags, slang, leetspeak) |
| `src/build_indexes.py`, `src/retriever.py` | BM25, BGE + FAISS, RRF fusion, cross-encoder re-ranking, for both channels |
| `src/llm.py` | Cached Nebius chat client, token usage and cost |
| `src/generator.py` | HyDE, the suitcase prompt, the LLM judge, the end-to-end pipeline |
| `src/sentic.py` | SenticNet 9 concept parsing and Hourglass features |
| `src/critic.py` | Citation check, DeBERTa NLI gates, precedent consensus, abstention |
| `src/build_eval_pool.py`, `src/agreement.py` | Annotation pool, sheets, Cohen's and Fleiss' kappa, adjudication, `eval.xls` |
| `src/build_qrels.py`, `src/evaluate_retrieval.py` | Pooled graded qrels and Table 1 |
| `src/eval_data.py`, `src/evaluate_downstream.py`, `src/evaluate_ragas.py` | Benchmarks, Table 2 ablation, RAGAS metrics |
| `src/benchmark_queries.py`, `app.py` | Question 2 queries and the Streamlit UI |

## 4. Data formats

- `data/eval.xls` (legacy Excel, one row per post): `id, text, label, subjectivity, polarity, is_sarcastic,
  annotator_1, annotator_2, adjudicated, group`. `label` ∈ {Safe, Hate Speech, Harassment-Offensive}.
- `policy_chunks.jsonl`: `id, source, title, section, url, retrieved_on, text, word_count`.
- `precedent_docs.jsonl`: `id, text, raw_text, tokens, rationales, category, targets, …`.
- `qrels_policy.json`: per query `query_id, post_id, label, query_raw, judged, qrels{clause_id: 0|1|2}`.
- `results/predictions/<dataset>/<config>.jsonl`: one JSON object per post with the verdict, citations, triggers,
  explanation, critic decision and latency.
