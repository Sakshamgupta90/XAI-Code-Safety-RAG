"""
src/train_baseline.py
Classical SOTA Baseline (G0): TF-IDF + Logistic Regression on HateXplain Precedents.
Course: NTU AI6127 (2026) — Question 4 Baseline & Table 2 Row G0.

Actions:
1. Loads training precedents from data/processed/precedent_docs.jsonl (17,305 passages).
2. Trains a scikit-learn TF-IDF + Logistic Regression classifier.
3. Saves the trained model to models/g0_tfidf_lr.joblib.
4. Evaluates performance on the official held-out test set (data/eval.xls - 1,000 records).
5. Computes: Macro Precision, Macro Recall, Macro F1, Accuracy, and Confusion Matrix.
6. Saves official benchmark results to results/g0_baseline_metrics.json.
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
EVAL_XLS_PATH = DATA_DIR / "eval.xls"
MODEL_SAVE_PATH = MODELS_DIR / "g0_tfidf_lr.joblib"
METRICS_SAVE_PATH = RESULTS_DIR / "g0_baseline_metrics.json"


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


def load_evaluation_data() -> Tuple[List[str], List[str]]:
    """Loads 1,000 gold evaluation records from data/eval.xls."""
    if not EVAL_XLS_PATH.exists():
        raise FileNotFoundError(f"Missing {EVAL_XLS_PATH}. Run src/agreement.py first.")

    df = pd.read_excel(EVAL_XLS_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["category"].astype(str).tolist()

    print(f"[OK] Loaded {len(texts):,} evaluation records from {EVAL_XLS_PATH.name}.")
    print(f"     Evaluation Class Balance: {dict(Counter(labels))}")
    return texts, labels


def train_and_evaluate_baseline():
    """Trains G0 classifier and evaluates on eval.xls."""
    print("=" * 80)
    print("XAI-Guard Milestone M3: Training Classical SOTA Baseline (G0)")
    print("=" * 80)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    X_train, y_train = load_training_data()
    X_test, y_test = load_evaluation_data()

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

    # 4. Evaluate on eval.xls
    print("\n--- Evaluating G0 on Official Held-Out Test Set (eval.xls) ---")
    y_pred = pipeline.predict(X_test)

    classes = ["Safe", "Hate Speech", "Harassment-Offensive"]
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    acc = float(np.mean(np.array(y_test) == np.array(y_pred)))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    macro_prec = float(precision_score(y_test, y_pred, average="macro"))
    macro_rec = float(recall_score(y_test, y_pred, average="macro"))

    report_dict = classification_report(y_test, y_pred, labels=classes, output_dict=True)

    results_payload = {
        "model_name": "G0_TFIDF_LogisticRegression",
        "trained_on": "HateXplain (Train + Val splits)",
        "train_samples": len(X_train),
        "test_dataset": "eval.xls",
        "test_samples": len(X_test),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "coverage": 1.0,  # 100% coverage (G0 does not abstain)
        },
        "per_class_metrics": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall": round(report_dict[cls]["recall"], 4),
                "f1_score": round(report_dict[cls]["f1-score"], 4),
                "support": report_dict[cls]["support"],
            }
            for cls in classes
        },
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm.tolist()
        }
    }

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] Official G0 baseline metrics -> {METRICS_SAVE_PATH}")

    # 5. Display Summary Table for Report
    print("\n" + "=" * 80)
    print("OFFICIAL G0 SOTA BASELINE BENCHMARK FOR QUESTION 4 & TABLE 2 (ROW G0)")
    print("=" * 80)
    print(f"Overall Accuracy:         {acc * 100:.2f}%")
    print(f"Macro F1-Score:           {macro_f1:.4f}")
    print(f"Macro Precision:          {macro_prec:.4f}")
    print(f"Macro Recall:             {macro_rec:.4f}")
    print(f"Coverage:                 100.0% (No Abstention)")
    print("\nPer-Class Breakdown:")
    for cls in classes:
        p = results_payload["per_class_metrics"][cls]["precision"]
        r = results_payload["per_class_metrics"][cls]["recall"]
        f = results_payload["per_class_metrics"][cls]["f1_score"]
        print(f"  - {cls:22s} -> Prec: {p:.4f} | Rec: {r:.4f} | F1: {f:.4f}")

    print("\nConfusion Matrix (Rows: Gold, Cols: Predicted):")
    print(f"                 {'Safe':>10} {'Hate Speech':>14} {'Harassment':>14}")
    for idx, row in enumerate(cm):
        print(f"{classes[idx]:<16} {row[0]:>10} {row[1]:>14} {row[2]:>14}")
    print("=" * 80)


if __name__ == "__main__":
    train_and_evaluate_baseline()
