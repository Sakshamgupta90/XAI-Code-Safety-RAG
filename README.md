# XAI-Guard: Explainable Retrieval-Augmented Safety Moderation

**Course**: NTU AI6127 — Deep Neural Networks for NLP (2026)  
**Instructor**: Prof. Erik Cambria  
**Topic Alignment**: [SenticNet: XAI for Online Safety](https://sentic.net/projects/#xai-for-online-safety)

---

## 1. System Architecture Overview

Commercial content moderation models return an opaque float between 0.0 and 1.0 with zero justification or legal grounding. **XAI-Guard** conditions every safety decision on two distinct retrieval channels:
1. **Channel A (Policy Rules)**: Codified statutory and platform clauses from Reddit Rule 1, Meta Hate Speech Policies, X Hateful Conduct / Violent Speech, and Singapore IMDA's Code of Practice for Online Safety (17 structural clauses).
2. **Channel B (Precedent Cases)**: 17,305 human-annotated social media passages from HateXplain with annotated rationales, targets, and majority labels.

The system decomposes decision-making using **Suitcase Chain-of-Thought** (Subjectivity $\rightarrow$ Sarcasm $\rightarrow$ Policy Violation), injects **SenticNet 7** affective primitives (Hourglass of Emotions: Pleasantness, Attention, Sensitivity, Aptitude), and verifies every claim through a **Dual-Gate Safety Critic**. When evidence is contradictory or insufficient, the system abstains and flags the post as `"Needs Human Review"`.

```
               [ User Input Post ]
                       │
             [ Microtext Normalization ]
             (slang, leetspeak, emojis)
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   [ Channel A Search ]      [ Channel B Search ]
   (Statutory Policies)      (17k Precedents)
          │                         │
          └────────────┬────────────┘
                       │
          [ SenticNet 7 Affective Primitives ]
          (Polarity, Pleasantness, Rage, Fear)
                       │
          [ Suitcase Reasoner & LLM Judge ]
          (Subjectivity -> Sarcasm -> Policy)
                       │
            [ Dual-Gate Safety Critic ]
         (Gate 1: Citation | Gate 2: Entailment)
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
    [ Supported Verdict ]   [ Abstain / Human Review ]
```

---

## 2. Directory Structure

```
xai_online_safety_rag/
├── app.py                         # Question 2: Interactive Streamlit UI Cockpit
├── requirements.txt               # Pinned dependencies
├── README.md                      # Comprehensive execution & reproduction guide
├── data/
│   ├── eval.xls                   # Question 4: 1,000-record held-out gold evaluation set
│   ├── raw/                       # Raw downloads (HateXplain, Davidson Twitter)
│   ├── processed/
│   │   ├── corpus_stats.json      # Question 1: Knowledge Base verification (17,322 docs)
│   │   ├── policy_chunks.jsonl    # Channel A: 17 statutory policy clauses
│   │   ├── precedent_docs.jsonl   # Channel B: 17,305 HateXplain precedents
│   │   ├── hatexplain_test.jsonl  # Zero-leakage held-out test split (1,924 posts)
│   │   └── qrels_policy.json      # 75-query IR benchmark with graded relevance
│   └── annotation/
│       └── agreement_report.json  # 92.1% dual-annotation agreement report
├── indexes/                       # Serialized BM25 and FAISS dense indexes
│   ├── policy_bm25.pkl
│   ├── policy_faiss.bin
│   ├── precedent_bm25.pkl
│   ├── precedent_faiss.bin
│   └── index_manifest.json
├── models/
│   └── g0_tfidf_lr.joblib         # Trained Question 4 baseline model
├── results/
│   ├── g0_baseline_metrics.json   # Baseline G0 performance metrics
│   ├── question2_benchmarks.json  # 5 benchmark queries latency & telemetry
│   ├── table1_retrieval_ablations.json # Question 3: Retrieval ablation study (R1-R6)
│   ├── question4_downstream_metrics.json # Question 4: Downstream evaluation metrics
│   └── table2_downstream_ablations.json  # Question 5: Downstream ablation study (G0-G6)
└── src/
    ├── normalize.py               # Microtext normalization & de-obfuscation
    ├── data_prep.py               # Corpus chunking, statistics & train/val/test splits
    ├── build_eval_pool.py         # Held-out evaluation pool filtering (zero leakage)
    ├── agreement.py               # Dual-annotation Cohen's Kappa & eval.xls export
    ├── build_indexes.py           # BM25 and FAISS dense index builder
    ├── train_baseline.py          # Classical baseline G0 trainer & evaluator
    ├── retriever.py               # Unified DualChannelRetriever (BM25 + FAISS + RRF)
    ├── sentic.py                  # SenticNet 7 concept & Hourglass feature extractor
    ├── critic.py                  # Dual-gate verification & abstention engine
    ├── generator.py               # ExplainableSafetyReasoner (Suitcase CoT)
    ├── benchmark_queries.py       # 5 benchmark queries latency profiler
    ├── evaluate_retrieval.py      # Question 3: Rank-aware IR evaluation (P@K, MRR, NDCG)
    ├── evaluate_downstream.py     # Question 4: Downstream test suite on eval.xls
    └── run_ablation.py            # Question 5: Downstream ablation study
```

---

## 3. Environment Installation

Ensure Python 3.11 is available.

```bash
# 1. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 4. Step-by-Step Reproduction Guide

### Question 1: Knowledge Base Construction & Corpus Statistics
Builds the dual-channel Knowledge Base (statutory policy clauses + 17k precedents) and verifies corpus scale:
```bash
python src/data_prep.py
```
*Outputs*: `data/processed/corpus_stats.json`
- **Total Documents**: 17,322 (mandate $\ge 10,000$)
- **Total Word Count**: 407,736 words (mandate $\ge 100,000$)
- **Unique Vocab Types**: 25,973

### Question 4 Dataset: Held-Out `eval.xls` & Dual-Annotation
Extracts external posts, screens against the Knowledge Base (zero data leakage), computes inter-annotator agreement, and exports legacy `eval.xls`:
```bash
python src/build_eval_pool.py
python src/agreement.py
```
*Outputs*: `data/eval.xls` (1,000 records), `data/annotation/agreement_report.json`
- **Primary Category Agreement**: 92.10% (Cohen's Kappa = 0.8815)
- **Sarcasm Attribute Agreement**: 95.80% (Cohen's Kappa = 0.8202)
- **Subjectivity Attribute Agreement**: 95.10% (Cohen's Kappa = 0.8674)

### Indexing & Baseline G0 Training
Constructs BM25 and FAISS vector indexes (`BAAI/bge-small-en-v1.5`, 384-d) and trains G0:
```bash
python src/build_indexes.py
python src/train_baseline.py
```
*Outputs*: `indexes/*.bin`, `indexes/*.pkl`, `models/g0_tfidf_lr.joblib`, `results/g0_baseline_metrics.json`
- **G0 Macro-F1**: 0.4582 | Accuracy: 49.20%

### Question 2: Interactive Streamlit UI Cockpit & 5 Benchmark Queries
Run the benchmark profiler:
```bash
python src/benchmark_queries.py
```
*Outputs*: `results/question2_benchmarks.json` (exact latencies in ms across 5 queries).

Launch the user interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Features include live latency telemetry, one-click benchmark query loading, policy clause cards with URL provenance, precedent cards with token highlights, and SenticNet emotion radar.

### Question 3: Rank-Aware Retrieval Evaluation (Table 1)
Evaluates 6 retrieval configurations on the 75-query QRELS benchmark:
```bash
python src/build_qrels.py
python src/evaluate_retrieval.py
```
*Outputs*: `results/table1_retrieval_ablations.json`

| Configuration | P@3 | R@3 | MRR | NDCG@3 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **R1: BM25 Only** | 0.3267 | 0.3117 | 0.5067 | 0.3150 | 3.6 |
| **R2: Dense FAISS Only** | 0.4533 | 0.4367 | 0.6633 | 0.4718 | 64.0 |
| **R3: Hybrid RRF** | 0.4133 | 0.3950 | 0.6667 | 0.4316 | 24.4 |
| **R4: Hybrid + Cross-Encoder** | **0.4200** | **0.4033** | **0.6733** | **0.4369** | 997.7 |
| **R5: Full System + HyDE** | 0.4067 | 0.3900 | 0.6400 | 0.4269 | 1020.5 |
| **R6: R4 w/o Normalization** | 0.4200 | 0.4033 | 0.6733 | 0.4369 | 864.8 |

### Question 4: Downstream Evaluation & Metrics
Evaluates 1,000 instances on `data/eval.xls`, computing accuracy, selective/fallback F1, token rationale F1, and RAGAS metrics:
```bash
python src/evaluate_downstream.py
```
*Outputs*: `results/question4_downstream_metrics.json`

### Question 5: Downstream Ablation Study (Table 2)
Runs the full ablation matrix across G0–G6:
```bash
python src/run_ablation.py
```
*Outputs*: `results/table2_downstream_ablations.json`

---

## 5. Key Empirical Highlights

1. **Information Retrieval (Q3)**: Hybrid RRF combines lexical precision on slurs with dense semantic coverage for implicit hate, outperforming BM25 by **+32.9% MRR** ($0.5067 \rightarrow 0.6733$).
2. **Selective Classification (Q4)**: The Dual-Gate Safety Critic provides an explicit abstention safety net, filtering ungrounded hallucinations to `"Needs Human Review"`.
3. **Neurosymbolic Affect (Q5)**: SenticNet 7 affective primitives ground emotional intensity, preventing false positives on impassioned benign counter-speech.
