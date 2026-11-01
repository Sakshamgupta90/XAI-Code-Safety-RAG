"""
src/retriever.py
Unified Dual-Channel Hybrid Retrieval Pipeline (BM25 + Dense FAISS + RRF + Cross-Encoder).
Course: NTU AI6127 (2026) — Questions 2 & 3.

Channels:
- Channel A ('policy'): Codified statutory and platform safety clauses (Reddit, Meta, X, Singapore IMDA).
- Channel B ('precedent'): Labeled historical precedents from HateXplain (17,305 passages).

Supported Modular Configurations:
- BM25 lexical search only
- Dense vector semantic search only
- Reciprocal Rank Fusion (RRF) hybrid blending
- Cross-Encoder re-ranking (ms-marco-MiniLM-L-6-v2)
"""

import json
import os
import pickle
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Stabilize thread execution on macOS to avoid OpenMP segfaults and thread barrier deadlocks
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
# Intra-op threads for the encoders (OpenMP/MKL pools stay single-threaded to avoid macOS deadlocks)
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalize import normalize_text

INDEXES_DIR = PROJECT_ROOT / "indexes"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    rank: int
    source: str
    channel: str
    category_or_section: str
    title_or_targets: Any
    word_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalConfig:
    top_k: int = 5
    use_bm25: bool = True
    use_dense: bool = True
    use_rrf: bool = True
    use_rerank: bool = False
    rrf_k: int = 60
    candidate_pool_size: int = 20
    normalize: bool = True


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text for BM25 matching."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


class DualChannelRetriever:
    """
    Manages indexes and retrieval across both Channel A (Policy) and Channel B (Precedents).
    """

    def __init__(self, indexes_dir: Path = INDEXES_DIR):
        self.indexes_dir = Path(indexes_dir)
        # Pre-initialize PyTorch encoder first to establish OpenMP runtime safely
        self.encoder: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.reranker: Optional[CrossEncoder] = None

        # Channel A (Policy) artifacts
        self.policy_bm25: Optional[BM25Okapi] = None
        self.policy_faiss: Optional[faiss.IndexFlatIP] = None
        self.policy_ids: Optional[np.ndarray] = None
        self.policy_docs: Dict[str, Dict[str, Any]] = {}

        # Channel B (Precedent) artifacts
        self.precedent_bm25: Optional[BM25Okapi] = None
        self.precedent_faiss: Optional[faiss.IndexFlatIP] = None
        self.precedent_ids: Optional[np.ndarray] = None
        self.precedent_docs: Dict[str, Dict[str, Any]] = {}

        self._load_indexes()

    def _load_indexes(self):
        """Loads serialized BM25 and FAISS indexes from disk."""
        # 1. Load Channel A (Policy)
        policy_bm25_file = self.indexes_dir / "policy_bm25.pkl"
        policy_faiss_file = self.indexes_dir / "policy_faiss.bin"
        policy_ids_file = self.indexes_dir / "policy_ids.npy"

        if policy_bm25_file.exists():
            with open(policy_bm25_file, "rb") as f:
                data = pickle.load(f)
                self.policy_bm25 = data["bm25"]
                self.policy_docs = {d["id"]: d for d in data["documents"]}

        if policy_faiss_file.exists() and policy_ids_file.exists():
            self.policy_faiss = faiss.read_index(str(policy_faiss_file))
            self.policy_ids = np.load(policy_ids_file, allow_pickle=True)

        # 2. Load Channel B (Precedents)
        precedent_bm25_file = self.indexes_dir / "precedent_bm25.pkl"
        precedent_faiss_file = self.indexes_dir / "precedent_faiss.bin"
        precedent_ids_file = self.indexes_dir / "precedent_ids.npy"

        if precedent_bm25_file.exists():
            with open(precedent_bm25_file, "rb") as f:
                data = pickle.load(f)
                self.precedent_bm25 = data["bm25"]
                self.precedent_docs = {d["id"]: d for d in data["documents"]}

        if precedent_faiss_file.exists() and precedent_ids_file.exists():
            self.precedent_faiss = faiss.read_index(str(precedent_faiss_file))
            self.precedent_ids = np.load(precedent_ids_file, allow_pickle=True)

        # The BM25 pickle (which also stores the documents) and the FAISS id map must describe the same corpus
        for name, docs, ids in (("policy", self.policy_docs, self.policy_ids), ("precedent", self.precedent_docs, self.precedent_ids)):
            if ids is not None and list(ids) != list(docs.keys()):
                raise RuntimeError(f"{name} indexes are out of sync (BM25 vs FAISS ids) — rerun src/build_indexes.py")

    def _get_encoder(self) -> SentenceTransformer:
        """Lazily initializes the dense encoder."""
        if self.encoder is None:
            self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self.encoder

    def _get_reranker(self) -> CrossEncoder:
        """Lazily initializes the cross-encoder re-ranker."""
        if self.reranker is None:
            # 256 tokens = query + the first ~180 words of a clause; 2.8x faster than 512 on CPU, same top-1 in tests
            self.reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=256)
        return self.reranker

    def _retrieve_bm25(self, query: str, channel: str, top_n: int) -> List[Tuple[str, float, int]]:
        """
        Executes BM25 search over target channel.
        Returns list of (doc_id, bm25_score, rank_position).
        """
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        if channel == "policy":
            bm25 = self.policy_bm25
            doc_ids = list(self.policy_docs.keys())
        else:
            bm25 = self.precedent_bm25
            doc_ids = list(self.precedent_docs.keys())

        if bm25 is None:
            return []

        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0.0:
                results.append((doc_ids[idx], float(scores[idx]), rank))
        return results

    def _retrieve_dense(self, query: str, channel: str, top_n: int) -> List[Tuple[str, float, int]]:
        """
        Executes dense FAISS cosine search over target channel.
        Returns list of (doc_id, cosine_similarity, rank_position).
        """
        if channel == "policy":
            index = self.policy_faiss
            id_map = self.policy_ids
        else:
            index = self.precedent_faiss
            id_map = self.precedent_ids

        if index is None or id_map is None:
            return []

        encoder = self._get_encoder()
        query_vector = encoder.encode([query], normalize_embeddings=True).astype("float32")

        scores, indices = index.search(query_vector, min(top_n, index.ntotal))

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx != -1:
                results.append((str(id_map[idx]), float(score), rank))
        return results

    def retrieve(
        self,
        query: str,
        channel: str = "policy",
        config: Optional[RetrievalConfig] = None,
    ) -> List[SearchResult]:
        """
        Main retrieval function executing configurable single/hybrid/re-ranked search.
        """
        if config is None:
            config = RetrievalConfig()

        assert channel in ("policy", "precedent"), f"Invalid channel: {channel}"

        # 1. Optional microtext normalization
        search_query = normalize_text(query) if config.normalize else query
        if not search_query.strip():
            return []

        docs_map = self.policy_docs if channel == "policy" else self.precedent_docs
        candidate_n = max(config.top_k, config.candidate_pool_size)

        # 2. Execute requested retrievers
        bm25_results = self._retrieve_bm25(search_query, channel, candidate_n) if config.use_bm25 else []
        dense_results = self._retrieve_dense(search_query, channel, candidate_n) if config.use_dense else []

        # 3. Blending / Scoring
        ranked_candidates: List[Tuple[str, float]] = []

        if config.use_bm25 and config.use_dense and config.use_rrf:
            # Reciprocal Rank Fusion (RRF)
            rrf_scores: Dict[str, float] = {}
            for doc_id, _, rank in bm25_results:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (config.rrf_k + rank))
            for doc_id, _, rank in dense_results:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (config.rrf_k + rank))

            sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            ranked_candidates = sorted_rrf[:candidate_n]

        elif config.use_dense:
            # Dense only
            ranked_candidates = [(doc_id, score) for doc_id, score, _ in dense_results[:candidate_n]]
        elif config.use_bm25:
            # BM25 only
            ranked_candidates = [(doc_id, score) for doc_id, score, _ in bm25_results[:candidate_n]]
        else:
            return []

        # 4. Optional Cross-Encoder Re-Ranking
        if config.use_rerank and ranked_candidates:
            reranker = self._get_reranker()
            pairs = []
            valid_ids = []
            for doc_id, _ in ranked_candidates:
                doc = docs_map.get(doc_id)
                if doc:
                    pairs.append([search_query, doc["text"]])
                    valid_ids.append(doc_id)

            if pairs:
                rerank_scores = reranker.predict(pairs)
                re_ranked = sorted(zip(valid_ids, [float(s) for s in rerank_scores]), key=lambda x: x[1], reverse=True)
                ranked_candidates = re_ranked

        # 5. Format Top-K Results
        top_results: List[SearchResult] = []
        for rank, (doc_id, score) in enumerate(ranked_candidates[:config.top_k], 1):
            doc = docs_map.get(doc_id)
            if not doc:
                continue

            if channel == "policy":
                sec = doc.get("section", "")
                title = doc.get("title", "")
            else:
                sec = doc.get("category", "")
                title = doc.get("targets", [])

            res = SearchResult(
                id=doc_id,
                text=doc.get("text", ""),
                score=round(float(score), 4),
                rank=rank,
                source=doc.get("source", channel),
                channel=channel,
                category_or_section=sec,
                title_or_targets=title,
                word_count=doc.get("word_count", len(doc.get("text", "").split())),
            )
            top_results.append(res)

        return top_results

    def retrieve_batch(
        self,
        queries: List[str],
        channel: str = "policy",
        config: Optional[RetrievalConfig] = None,
        batch_size: int = 64
    ) -> List[List[SearchResult]]:
        """
        High-throughput vectorized batch retrieval for large evaluation splits.
        Encodes query vectors in batches and executes batch matrix FAISS search.
        """
        if config is None:
            config = RetrievalConfig()

        docs_map = self.policy_docs if channel == "policy" else self.precedent_docs
        index = self.policy_faiss if channel == "policy" else self.precedent_faiss
        id_map = self.policy_ids if channel == "policy" else self.precedent_ids
        bm25_model = self.policy_bm25 if channel == "policy" else self.precedent_bm25
        # BM25 scores follow the corpus order the index was built with (= docs_map insertion order)
        doc_ids = list(docs_map.keys())

        candidate_n = max(config.top_k, config.candidate_pool_size)

        # 1. Normalize
        clean_queries = [normalize_text(q) if config.normalize else q for q in queries]

        # 2. Batch Dense Embedding & Search
        if config.use_dense and index is not None:
            encoder = self._get_encoder()
            query_vectors = encoder.encode(clean_queries, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False).astype("float32")
            all_scores, all_indices = index.search(query_vectors, min(candidate_n, index.ntotal))
        else:
            all_scores, all_indices = None, None

        batch_results = []
        for q_idx, q_text in enumerate(clean_queries):
            # Dense hits for this query
            dense_results = []
            if all_scores is not None and all_indices is not None:
                for rank, (idx, score) in enumerate(zip(all_indices[q_idx], all_scores[q_idx]), 1):
                    if idx != -1:
                        dense_results.append((str(id_map[idx]), float(score), rank))

            # BM25 hits for this query
            bm25_results = []
            if config.use_bm25 and bm25_model is not None and len(doc_ids) > 0:
                tokens = tokenize_for_bm25(q_text)
                scores = bm25_model.get_scores(tokens) if tokens else np.zeros(len(doc_ids))
                top_indices = np.argsort(scores)[::-1][:candidate_n]
                for rank, idx in enumerate(top_indices, 1):
                    if scores[idx] > 0.0:
                        bm25_results.append((doc_ids[idx], float(scores[idx]), rank))

            # RRF Fusion
            if config.use_bm25 and config.use_dense and config.use_rrf:
                rrf_scores: Dict[str, float] = {}
                for doc_id, _, rank in bm25_results:
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (config.rrf_k + rank))
                for doc_id, _, rank in dense_results:
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (config.rrf_k + rank))
                sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                ranked_candidates = sorted_rrf[:candidate_n]
            elif config.use_dense:
                ranked_candidates = [(doc_id, score) for doc_id, score, _ in dense_results[:candidate_n]]
            else:
                ranked_candidates = [(doc_id, score) for doc_id, score, _ in bm25_results[:candidate_n]]

            if config.use_rerank and ranked_candidates:
                valid = [d for d, _ in ranked_candidates if d in docs_map]
                rerank_scores = self._get_reranker().predict([[q_text, docs_map[d]["text"]] for d in valid])
                ranked_candidates = sorted(zip(valid, [float(x) for x in rerank_scores]), key=lambda x: x[1], reverse=True)

            top_results: List[SearchResult] = []
            for rank, (doc_id, score) in enumerate(ranked_candidates[:config.top_k], 1):
                doc = docs_map.get(doc_id)
                if not doc:
                    continue
                sec = doc.get("section", "") if channel == "policy" else doc.get("category", "")
                title = doc.get("title", "") if channel == "policy" else doc.get("targets", [])
                res = SearchResult(
                    id=doc_id,
                    text=doc.get("text", ""),
                    score=round(float(score), 4),
                    rank=rank,
                    source=doc.get("source", channel),
                    channel=channel,
                    category_or_section=sec,
                    title_or_targets=title,
                    word_count=doc.get("word_count", len(doc.get("text", "").split())),
                )
                top_results.append(res)
            batch_results.append(top_results)

        return batch_results


# Module-level convenience function complying with Section 4.3 of Build Plan
_GLOBAL_RETRIEVER: Optional[DualChannelRetriever] = None


def retrieve(query: str, channel: str = "policy", config: Optional[RetrievalConfig] = None) -> List[SearchResult]:
    """Global retrieval entry point for UI and evaluation scripts."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = DualChannelRetriever()
    return _GLOBAL_RETRIEVER.retrieve(query=query, channel=channel, config=config)


if __name__ == "__main__":
    print("Testing DualChannelRetriever interface ...")
    test_query = "Nobody wants immigrants here, go back to your country"
    retriever = DualChannelRetriever()

    print("\n--- Channel A: Top-3 Policy Clauses (Hybrid RRF + Re-Rank) ---")
    cfg_policy = RetrievalConfig(top_k=3, use_bm25=True, use_dense=True, use_rrf=True, use_rerank=True)
    policy_hits = retriever.retrieve(test_query, channel="policy", config=cfg_policy)
    for hit in policy_hits:
        print(f"[{hit.rank}] ID: {hit.id} | Score: {hit.score} | Section: {hit.category_or_section}")
        print(f"    Excerpt: {hit.text[:120]}...\n")

    print("\n--- Channel B: Top-3 Precedents (Hybrid RRF) ---")
    cfg_prec = RetrievalConfig(top_k=3, use_bm25=True, use_dense=True, use_rrf=True, use_rerank=False)
    prec_hits = retriever.retrieve(test_query, channel="precedent", config=cfg_prec)
    for hit in prec_hits:
        print(f"[{hit.rank}] ID: {hit.id} | Score: {hit.score} | Category: {hit.category_or_section}")
        print(f"    Excerpt: {hit.text[:120]}...\n")
