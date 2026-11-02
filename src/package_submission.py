"""
src/package_submission.py — builds the two cloud-upload archives named on the report's front page.

  data.zip : knowledge base (policy pages with provenance, processed chunks and precedents), eval.xls, annotation sheets
             and agreement report, eval pool, qrels and queries, all results (retrieved results, predictions, RAGAS,
             ablation tables, benchmarks) and SenticNet 9 with its non-commercial licence.
  code.zip : source code, UI, report generator, README and requirements. Never includes .env, .venv, indexes or caches.

Usage: python src/package_submission.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_INCLUDE = ["data/eval.xls", "data/annotation", "data/processed", "data/raw/policies", "data/raw/eval_pool",
                "data/raw/senticnet", "results"]
CODE_INCLUDE = ["src", "app.py", "report", "README.md", "requirements.txt", ".streamlit", ".gitignore"]
SKIP_PARTS = {".venv", "__pycache__", "cache", ".git", "indexes"}
SKIP_NAMES = {".env", ".DS_Store"}
SKIP_SUFFIXES = {".pkl", ".pdf"}  # pickles are rebuilt by build_indexes.py; the report PDF is submitted separately


def add(zf: zipfile.ZipFile, rel: str, skip_suffixes=()):
    path = ROOT / rel
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        r = f.relative_to(ROOT)
        if SKIP_PARTS & set(r.parts) or f.name in SKIP_NAMES or f.suffix in skip_suffixes:
            continue
        zf.write(f, Path("xai_online_safety_rag") / r)


def build(name, include, skip_suffixes=()):
    out = ROOT / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            add(zf, rel, skip_suffixes)
        names = zf.namelist()
    assert not any(n.endswith("/.env") or "/.venv/" in n for n in names)
    print(f"[SAVED] {out} ({len(names)} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build("data.zip", DATA_INCLUDE, skip_suffixes=(".pkl",))
    build("code.zip", CODE_INCLUDE, skip_suffixes=tuple(SKIP_SUFFIXES))
