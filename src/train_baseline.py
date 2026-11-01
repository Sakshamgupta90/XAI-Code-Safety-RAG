"""
src/train_baseline.py
Classical Baseline (G0): TF-IDF + Logistic Regression trained on the HateXplain precedents.
Course: NTU AI6127 (2026) — Question 4 baseline & Table 2 row G0.

Actions:
1. Loads training precedents from data/processed/precedent_docs.jsonl (HateXplain train + val).
2. Trains a scikit-learn TF-IDF (word 1-2 grams) + class-balanced Logistic Regression classifier.
3. Saves the trained model to models/g0_tfidf_lr.joblib.
G0 is evaluated on every benchmark (eval.xls, HateXplain test) by src/evaluate_downstream.py, so all
systems are scored by exactly the same code.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

PRECEDENTS_PATH = DATA_DIR / "processed" / "precedent_docs.jsonl"
MODEL_SAVE_PATH = MODELS_DIR / "g0_tfidf_lr.joblib"


def load_training_data() -> Tuple[List[str], List[str]]:
    """Loads HateXplain train+val precedents from data/processed/precedent_docs.jsonl."""
    if not PRECEDENTS_PATH.exists():
        raise FileNotFoundError(f"Missing {PRECEDENTS_PATH}. Run src/data_prep.py first.")

    texts, labels = [], []
    with open(PRECEDENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["category"])

    print(f"[OK] Loaded {len(texts):,} training precedents.")
    print(f"     Label Distribution: {dict(Counter(labels))}")
    return texts, labels


def train_baseline():
    """Trains and saves the G0 classifier."""
    print("=" * 80)
    print("XAI-Guard: Training Classical Baseline G0 (TF-IDF + Logistic Regression)")
    print("=" * 80)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    X_train, y_train = load_training_data()

    # 2. Build Pipeline
    # Unigrams + bigrams, sublinear TF scaling, balanced class weights
    print("\n--- Training Pipeline: TfidfVectorizer + LogisticRegression ---")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs"
        ))
    ])

    pipeline.fit(X_train, y_train)
    print(f"[OK] Model successfully trained on {len(X_train):,} samples.")

    # 3. Save Model Checkpoint
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"[SAVED] Baseline model checkpoint -> {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_baseline()
