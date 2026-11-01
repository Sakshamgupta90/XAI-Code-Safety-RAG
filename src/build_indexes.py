"""
src/build_indexes.py
Dual-Channel Hybrid Retrieval Index Builder (BM25 + FAISS Dense Embeddings).
Course: NTU AI6127 (2026) — Section 3.2 & Question 3.

Architectural Design:
To prevent macOS OpenMP thread conflicts between PyTorch and FAISS,
dense vector encoding (Torch) and index creation (FAISS) are strictly decoupled:
1. Dense embeddings are generated using BGE-small and saved as NumPy arrays (.npy).
2. FAISS is imported afterwards to load the embeddings and build IndexFlatIP stores (.bin).
"""

import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Stabilize thread execution on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import torch
torch.set_num_threads(2)

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
INDEXES_DIR = PROJECT_ROOT / "indexes"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text for BM25 matching."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Loads a JSONL file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def build_bm25_index(documents: List[Dict[str, Any]], output_path: Path, channel_name: str) -> BM25Okapi:
    """Builds and serializes a BM25Okapi index (always rebuilt: it takes seconds and must match the documents)."""

    print(f"\nBuilding BM25 index for {channel_name} ({len(documents):,} docs) ...")
    corpus_tokens = [tokenize_for_bm25(doc["text"]) for doc in documents]
    bm25 = BM25Okapi(corpus_tokens)

    payload = {
        "bm25": bm25,
        "doc_ids": [doc["id"] for doc in documents],
        "documents": documents,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[SAVED] {channel_name} BM25 index -> {output_path} ({output_path.stat().st_size:,} bytes)")
    return bm25


def encode_dense_embeddings(
    documents: List[Dict[str, Any]],
    model: SentenceTransformer,
    output_npy: Path,
    ids_npy: Path,
    channel_name: str,
    slice_size: int = 1000,
    batch_size: int = 64
) -> np.ndarray:
    """
    Generates L2-normalized dense embeddings in stable batches and saves as NumPy arrays.
    Does not import or touch FAISS to keep PyTorch runtime completely isolated.
    """
    texts = [doc["text"][:512] for doc in documents]
    doc_ids = [doc["id"] for doc in documents]
    hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    hash_file = output_npy.with_suffix(".sha1.json")

    # Incremental cache: re-encode only passages whose text changed since the embeddings were computed
    if output_npy.exists() and ids_npy.exists() and hash_file.exists() \
            and list(np.load(ids_npy, allow_pickle=True)) == doc_ids:
        embeddings = np.load(output_npy)
        old = json.loads(hash_file.read_text())
        changed = [i for i, (h_old, h_new) in enumerate(zip(old, hashes)) if h_old != h_new]
        if changed:
            print(f"[CACHE] {output_npy.name}: re-encoding {len(changed)} changed passages")
            embeddings[changed] = model.encode([texts[i] for i in changed], normalize_embeddings=True,
                                               convert_to_numpy=True).astype("float32")
            np.save(output_npy, embeddings)
            hash_file.write_text(json.dumps(hashes))
        else:
            print(f"[CACHE] Found {output_npy.name}, all {len(texts):,} passages unchanged.")
        return embeddings

    print(f"\nEncoding {channel_name} ({len(documents):,} docs) with {MODEL_NAME} ...")

    start_time = time.time()
    embeddings_list = []

    total = len(texts)
    with torch.no_grad():
        for i in range(0, total, slice_size):
            chunk_texts = texts[i : i + slice_size]
            chunk_embs = model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype("float32")
            embeddings_list.append(chunk_embs)
            current_done = min(i + slice_size, total)
            print(f"  -> Encoded {current_done:,}/{total:,} passages ({current_done/total*100:.1f}%)")

    all_embeddings = np.vstack(embeddings_list)
    encoding_time = time.time() - start_time
    print(f"[OK] Encoded {len(texts):,} passages in {encoding_time:.2f}s ({len(texts)/encoding_time:.1f} passages/sec)")

    np.save(output_npy, all_embeddings)
    np.save(ids_npy, np.array(doc_ids))
    hash_file.write_text(json.dumps(hashes))
    print(f"[SAVED] {channel_name} Embeddings -> {output_npy} (Shape: {all_embeddings.shape})")
    print(f"[SAVED] {channel_name} ID Map     -> {ids_npy}")
    return all_embeddings


def create_faiss_indexes():
    """
    Imports FAISS in a clean execution phase to convert pre-computed NumPy embeddings
    into IndexFlatIP binary files.
    """
    print("\n--- Initializing FAISS Vector Stores ---")
    import faiss

    # 1. Policy FAISS
    p_embs_path = INDEXES_DIR / "policy_embs.npy"
    p_faiss_path = INDEXES_DIR / "policy_faiss.bin"
    if p_embs_path.exists():
        p_embs = np.load(p_embs_path)
        p_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        p_index.add(p_embs)
        faiss.write_index(p_index, str(p_faiss_path))
        print(f"[SAVED] Policy FAISS Store -> {p_faiss_path} ({p_index.ntotal:,} vectors)")

    # 2. Precedent FAISS
    pr_embs_path = INDEXES_DIR / "precedent_embs.npy"
    pr_faiss_path = INDEXES_DIR / "precedent_faiss.bin"
    if pr_embs_path.exists():
        pr_embs = np.load(pr_embs_path)
        pr_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        pr_index.add(pr_embs)
        faiss.write_index(pr_index, str(pr_faiss_path))
        print(f"[SAVED] Precedent FAISS Store -> {pr_faiss_path} ({pr_index.ntotal:,} vectors)")


def build_all_indexes():
    """Master build workflow."""
    print("=" * 80)
    print("XAI-Guard Milestone M3: Building Dual-Channel Hybrid Retrieval Indexes")
    print("=" * 80)

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    policies = load_jsonl(DATA_PROCESSED / "policy_chunks.jsonl")
    precedents = load_jsonl(DATA_PROCESSED / "precedent_docs.jsonl")
    print(f"[OK] Ingested {len(policies):,} Policy Clauses and {len(precedents):,} Precedent Posts.")

    # Phase 1: Sparse Inverted Indexes
    build_bm25_index(policies, INDEXES_DIR / "policy_bm25.pkl", "Channel A (Policies)")
    build_bm25_index(precedents, INDEXES_DIR / "precedent_bm25.pkl", "Channel B (Precedents)")

    # Phase 2: Dense Neural Embeddings (Torch / SentenceTransformer)
    print(f"\nLoading Dense Neural Encoder: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    encode_dense_embeddings(
        policies, model,
        INDEXES_DIR / "policy_embs.npy",
        INDEXES_DIR / "policy_ids.npy",
        "Channel A (Policies)",
        slice_size=100, batch_size=32
    )

    encode_dense_embeddings(
        precedents, model,
        INDEXES_DIR / "precedent_embs.npy",
        INDEXES_DIR / "precedent_ids.npy",
        "Channel B (Precedents)",
        slice_size=1000, batch_size=64
    )

    # Phase 3: FAISS Vector Stores (Safe delayed import)
    create_faiss_indexes()

    # Phase 4: Index Manifest
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "channel_a_policies": {
            "document_count": len(policies),
            "bm25_index": "indexes/policy_bm25.pkl",
            "faiss_index": "indexes/policy_faiss.bin",
            "ids_file": "indexes/policy_ids.npy"
        },
        "channel_b_precedents": {
            "document_count": len(precedents),
            "bm25_index": "indexes/precedent_bm25.pkl",
            "faiss_index": "indexes/precedent_faiss.bin",
            "ids_file": "indexes/precedent_ids.npy"
        }
    }
    manifest_path = INDEXES_DIR / "index_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[SAVED] Retrieval Index Manifest -> {manifest_path}")

    print("=" * 80)
    print("ALL DUAL-CHANNEL RETRIEVAL INDEXES BUILT SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    build_all_indexes()
