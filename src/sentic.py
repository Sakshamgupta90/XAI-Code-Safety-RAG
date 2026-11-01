"""
src/sentic.py
Neurosymbolic Affective Feature Extractor using the SenticNet 9 knowledge base.
Course: NTU AI6127 (2026) — Section 4.2 & Questions 4/5.

Resource: the official English SenticNet 9 release (https://sentic.net/downloads/, non-commercial licence,
293,207 concepts), converted once to data/raw/senticnet/senticnet9.tsv. Each concept carries the four
Hourglass of Emotions dimensions (introspection, temper, attitude, sensitivity), primary/secondary
emotion labels, and a polarity value in [-1, 1].

Concept parsing: greedy longest match of 3-, 2- and 1-word concepts (SenticNet joins multiword
concepts with '_', e.g. 'go_back', 'take_life'); unigram stopwords are skipped.

Used as:
1. An auxiliary feature block in the LLM judge prompt (tested in ablation G3).
2. The emotion panel in the Streamlit UI (app.py).

Citations: Cambria et al., SenticNet 9 (IEEE TCSS 2026); Susanto et al., The Hourglass Model Revisited (2020).
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENTICNET_FILE = PROJECT_ROOT / "data" / "raw" / "senticnet" / "senticnet9.tsv"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "is", "am", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "do", "does", "did", "have", "has", "had", "so", "as", "just", "all", "not", "no", "can", "will", "would",
    "user", "url", "rt", "s", "t", "u", "im", "dont", "lol",
}


@dataclass
class SenticProfile:
    polarity: float
    dominant_emotion: str
    emotion_intensity: float
    introspection: float  # joy <-> sadness
    temper: float         # calmness <-> anger
    attitude: float       # pleasantness <-> disgust
    sensitivity: float    # eagerness <-> fear
    matched_concepts: List[str]
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_KB: Optional[Dict[str, Tuple[float, float, float, float, str, float]]] = None


def load_senticnet() -> Dict[str, Tuple[float, float, float, float, str, float]]:
    """Loads SenticNet 9 once: concept -> (introspection, temper, attitude, sensitivity, primary_emotion, polarity)."""
    global _KB
    if _KB is None:
        if not SENTICNET_FILE.exists():
            raise FileNotFoundError(f"{SENTICNET_FILE} missing — see README (SenticNet download step).")
        kb = {}
        with open(SENTICNET_FILE, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                c, i, t, a, s, prim, _sec, pol = line.rstrip("\n").split("\t")
                kb[c] = (float(i), float(t), float(a), float(s), prim.lstrip("#"), float(pol))
        _KB = kb
    return _KB


def extract_sentic_features(text: str) -> SenticProfile:
    """
    Extracts SenticNet affective features: mean polarity and mean Hourglass dimensions over the
    matched concepts, plus the dominant primary emotion (weighted by |polarity|).
    """
    kb = load_senticnet()
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    matched: List[str] = []
    values: List[Tuple[float, float, float, float, str, float]] = []

    i = 0
    while i < len(tokens):
        hit = None
        for n in (3, 2, 1):
            if i + n > len(tokens):
                continue
            concept = "_".join(tokens[i:i + n])
            if n == 1 and (concept in STOPWORDS or len(concept) < 3):
                continue
            if concept in kb:
                hit = (concept, n)
                break
        if hit:
            matched.append(hit[0])
            values.append(kb[hit[0]])
            i += hit[1]
        else:
            i += 1

    if values:
        k = len(values)
        intro, temper, attitude, sens = (sum(v[d] for v in values) / k for d in range(4))
        polarity = sum(v[5] for v in values) / k
        weights: Dict[str, float] = {}
        for v in values:
            weights[v[4]] = weights.get(v[4], 0.0) + abs(v[5])
        dominant = max(weights.items(), key=lambda x: x[1])[0]
        intensity = weights[dominant] / k
    else:
        intro = temper = attitude = sens = polarity = intensity = 0.0
        dominant = "none"

    negative = sorted(((c, v[5]) for c, v in zip(matched, values) if v[5] < -0.3), key=lambda x: x[1])[:6]
    summary = (
        f"polarity {polarity:+.2f}; dominant emotion: {dominant} (intensity {intensity:.2f}); "
        f"Hourglass introspection {intro:+.2f}, temper {temper:+.2f}, attitude {attitude:+.2f}, "
        f"sensitivity {sens:+.2f}; most negative concepts: "
        f"{', '.join(f'{c} ({p:+.2f})' for c, p in negative) or 'none'}"
    )
    return SenticProfile(
        polarity=round(polarity, 3),
        dominant_emotion=dominant,
        emotion_intensity=round(intensity, 3),
        introspection=round(intro, 3),
        temper=round(temper, 3),
        attitude=round(attitude, 3),
        sensitivity=round(sens, 3),
        matched_concepts=matched,
        summary_text=summary,
    )


if __name__ == "__main__":
    sample_texts = [
        "Trans people are disgusting freaks who should be locked up",
        "The municipal tax reform is completely incompetent and a disaster",
        "Oh look, another absolute genius in our department",
        "Please enjoy this wonderful sunny morning with friends",
    ]
    for s in sample_texts:
        prof = extract_sentic_features(s)
        print(f"Text:    {s}")
        print(f"Summary: {prof.summary_text}\n")
