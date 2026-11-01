"""
src/sentic.py
Neurosymbolic Affective Feature Extractor using SenticNet Concept Primitives.
Course: NTU AI6127 (2026) — Section 3.2, 4.2 & Question 4.

Provides symbolic affective dimensions based on the Hourglass of Emotions:
- Polarity: Float in [-1.0, +1.0]
- Hourglass Dimensions: Introspection (Grief), Temper (Rage), Attitude (Disgust), Sensitivity (Fear)
- Symbolic Concepts: Matched affective primitives

Used as:
1. An auxiliary feature prompt input to the downstream LLM reasoner (tested in Ablation G3).
2. A visual emotion radar badge in the Streamlit UI (app.py).
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class SenticProfile:
    polarity: float
    dominant_emotion: str
    emotion_intensity: float
    pleasantness: float  # Attitude
    attention: float     # Introspection
    sensitivity: float   # Sensitivity / Fear
    aptitude: float      # Temper / Rage
    matched_concepts: List[str]
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Domain-specific SenticNet 7 affective primitive dictionary for safety moderation
SENTIC_KB: Dict[str, Dict[str, Any]] = {
    "hate": {"polarity": -0.85, "emotion": "Rage", "pleasantness": -0.8, "aptitude": 0.9, "sensitivity": 0.4},
    "kill": {"polarity": -0.95, "emotion": "Rage", "pleasantness": -0.9, "aptitude": 1.0, "sensitivity": 0.8},
    "die": {"polarity": -0.80, "emotion": "Grief", "pleasantness": -0.8, "aptitude": 0.6, "sensitivity": 0.7},
    "disgusting": {"polarity": -0.88, "emotion": "Disgust", "pleasantness": -1.0, "aptitude": 0.5, "sensitivity": 0.3},
    "freak": {"polarity": -0.75, "emotion": "Disgust", "pleasantness": -0.8, "aptitude": 0.7, "sensitivity": 0.4},
    "ugly": {"polarity": -0.65, "emotion": "Disgust", "pleasantness": -0.7, "aptitude": 0.4, "sensitivity": 0.2},
    "idiot": {"polarity": -0.70, "emotion": "Contempt", "pleasantness": -0.7, "aptitude": 0.6, "sensitivity": 0.2},
    "stupid": {"polarity": -0.60, "emotion": "Contempt", "pleasantness": -0.6, "aptitude": 0.5, "sensitivity": 0.2},
    "clown": {"polarity": -0.50, "emotion": "Contempt", "pleasantness": -0.5, "aptitude": 0.4, "sensitivity": 0.1},
    "cockroach": {"polarity": -0.92, "emotion": "Disgust", "pleasantness": -1.0, "aptitude": 0.8, "sensitivity": 0.6},
    "vermin": {"polarity": -0.95, "emotion": "Disgust", "pleasantness": -1.0, "aptitude": 0.9, "sensitivity": 0.7},
    "pest": {"polarity": -0.80, "emotion": "Disgust", "pleasantness": -0.8, "aptitude": 0.6, "sensitivity": 0.4},
    "infestation": {"polarity": -0.85, "emotion": "Fear", "pleasantness": -0.8, "aptitude": 0.7, "sensitivity": 0.9},
    "cancer": {"polarity": -0.90, "emotion": "Fear", "pleasantness": -0.9, "aptitude": 0.6, "sensitivity": 0.95},
    "virus": {"polarity": -0.85, "emotion": "Fear", "pleasantness": -0.8, "aptitude": 0.5, "sensitivity": 0.9},
    "trash": {"polarity": -0.70, "emotion": "Disgust", "pleasantness": -0.7, "aptitude": 0.5, "sensitivity": 0.3},
    "filth": {"polarity": -0.88, "emotion": "Disgust", "pleasantness": -0.9, "aptitude": 0.7, "sensitivity": 0.4},
    "threat": {"polarity": -0.82, "emotion": "Fear", "pleasantness": -0.7, "aptitude": 0.8, "sensitivity": 0.9},
    "attack": {"polarity": -0.78, "emotion": "Rage", "pleasantness": -0.7, "aptitude": 0.85, "sensitivity": 0.6},
    "incompetent": {"polarity": -0.45, "emotion": "Disapproval", "pleasantness": -0.5, "aptitude": 0.3, "sensitivity": 0.2},
    "disaster": {"polarity": -0.55, "emotion": "Grief", "pleasantness": -0.6, "aptitude": 0.4, "sensitivity": 0.5},
    "fail": {"polarity": -0.40, "emotion": "Sadness", "pleasantness": -0.4, "aptitude": 0.3, "sensitivity": 0.3},
    "genius": {"polarity": 0.75, "emotion": "Joy", "pleasantness": 0.8, "aptitude": 0.2, "sensitivity": 0.1},
    "brilliant": {"polarity": 0.80, "emotion": "Joy", "pleasantness": 0.8, "aptitude": 0.1, "sensitivity": 0.1},
    "good": {"polarity": 0.50, "emotion": "Pleasantness", "pleasantness": 0.5, "aptitude": 0.1, "sensitivity": 0.0},
    "clean": {"polarity": 0.40, "emotion": "Pleasantness", "pleasantness": 0.4, "aptitude": 0.0, "sensitivity": 0.0},
    "safe": {"polarity": 0.60, "emotion": "Calmness", "pleasantness": 0.6, "aptitude": 0.0, "sensitivity": -0.5},
}


def extract_sentic_features(text: str) -> SenticProfile:
    """
    Extracts SenticNet symbolic affective features from input text.
    Computes aggregated polarity and dominant Hourglass of Emotions dimension.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    matched = []

    pol_scores = []
    pleasantness_scores = []
    aptitude_scores = []
    sensitivity_scores = []
    emotion_counter: Dict[str, float] = {}

    # Check unigrams and bigrams
    for i in range(len(tokens)):
        unigram = tokens[i]
        if unigram in SENTIC_KB:
            data = SENTIC_KB[unigram]
            matched.append(unigram)
            pol_scores.append(data["polarity"])
            pleasantness_scores.append(data.get("pleasantness", 0.0))
            aptitude_scores.append(data.get("aptitude", 0.0))
            sensitivity_scores.append(data.get("sensitivity", 0.0))
            em = data["emotion"]
            emotion_counter[em] = emotion_counter.get(em, 0.0) + abs(data["polarity"])

        if i < len(tokens) - 1:
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if bigram in SENTIC_KB:
                data = SENTIC_KB[bigram]
                matched.append(bigram)
                pol_scores.append(data["polarity"])
                pleasantness_scores.append(data.get("pleasantness", 0.0))
                aptitude_scores.append(data.get("aptitude", 0.0))
                sensitivity_scores.append(data.get("sensitivity", 0.0))
                em = data["emotion"]
                emotion_counter[em] = emotion_counter.get(em, 0.0) + abs(data["polarity"])

    if matched:
        avg_pol = sum(pol_scores) / len(pol_scores)
        avg_pleasant = sum(pleasantness_scores) / len(pleasantness_scores)
        avg_aptitude = sum(aptitude_scores) / len(aptitude_scores)
        avg_sens = sum(sensitivity_scores) / len(sensitivity_scores)
        dom_em = max(emotion_counter.items(), key=lambda x: x[1])[0]
        intensity = emotion_counter[dom_em] / len(matched)
    else:
        avg_pol = 0.0
        avg_pleasant = 0.0
        avg_aptitude = 0.0
        avg_sens = 0.0
        dom_em = "Neutral"
        intensity = 0.0

    summary = (
        f"Dominant Emotion: {dom_em} (Intensity: {intensity:.2f}) | "
        f"Affective Polarity: {avg_pol:.2f} | Matched Sentic Primitives: {matched or ['none']}"
    )

    return SenticProfile(
        polarity=round(avg_pol, 2),
        dominant_emotion=dom_em,
        emotion_intensity=round(intensity, 2),
        pleasantness=round(avg_pleasant, 2),
        attention=0.0,
        sensitivity=round(avg_sens, 2),
        aptitude=round(avg_aptitude, 2),
        matched_concepts=matched,
        summary_text=summary
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
