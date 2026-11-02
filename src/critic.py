#!/usr/bin/env python3
"""
src/critic.py - Evidence-Grounding Critic & Abstention Mechanism
Course: NTU AI6127 (2026) — Section 6.3 & Questions 4/5.

Checks the LLM judge's verdict against the retrieved evidence; any failed gate turns the verdict
into "Needs Human Review" (selective classification).

- Gate 1 (citation, deterministic): a violation verdict must cite a policy clause that was actually
  retrieved, and every cited precedent ID must be among the retrieved precedents.
- Gate 2 (NLI, DeBERTa-v3-base MNLI/FEVER/ANLI):
    2a  premise = 'A social media post says: "<post>"', hypothesis = the judge's literal violation claim
        ("The post calls someone a ...") -> the claim must be entailed by the post
    2b  premise = cited clause, hypothesis = "<category> content ... is not allowed" -> the cited rule must cover the category
    2c  (Safe verdicts) premise = framed post, hypothesis = "The post insults, threatens or attacks a person or group."
        -> strongly entailed means a likely missed violation
- Gate 3 (precedent consensus): abstain if >= 4 of 5 retrieved precedents carry one other label and none agree.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Hypothesis wordings were chosen on hand-made sanity examples (not on any evaluation data)
CATEGORY_HYPOTHESES = {
    "Hate Speech": "Hateful content attacking people for protected characteristics is not allowed.",
    "Harassment-Offensive": "Content that insults or harasses people is not allowed.",
}
SAFE_HYPOTHESIS = "The post insults, threatens or attacks a person or group."

# Default thresholds; src/evaluate_downstream.py --calibrate picks them on the HateXplain *dev* split
# and writes results/critic_thresholds.json, which is loaded automatically when present.
DEFAULT_THRESHOLDS = {
    "claim_entail_min": 0.50,   # Gate 2a
    "rule_entail_min": 0.50,    # Gate 2b
    "safe_attack_max": 0.90,    # Gate 2c
    "consensus_min_votes": 4,   # Gate 3
}
THRESHOLDS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "critic_thresholds.json")


def load_thresholds() -> Dict[str, float]:
    t = dict(DEFAULT_THRESHOLDS)
    if os.path.exists(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE) as f:
            t.update(json.load(f).get("thresholds", {}))
    return t


def frame_post(post_text: str) -> str:
    """NLI models judge claims *about* a post far better when the premise reports the post."""
    return f'A social media post says: "{post_text}"'


@dataclass
class CriticDecision:
    is_grounded: bool
    citation_valid: bool
    support_score: float
    requires_human_review: bool
    abstention_reason: Optional[str]
    final_verdict: str
    nli_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NLIScorer:
    """Lazy DeBERTa-v3 NLI model returning (entailment, neutral, contradiction) probabilities."""

    def __init__(self, model_name: str = NLI_MODEL_NAME):
        self.model_name = model_name
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
            labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
            self._idx = (labels["entailment"], labels["neutral"], labels["contradiction"])

    @lru_cache(maxsize=50000)
    def score(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        self._load()
        import torch
        enc = self._tok(premise, hypothesis, truncation="only_first", max_length=512, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self._model(**enc).logits[0], dim=-1).tolist()
        e, n, c = self._idx
        return round(probs[e], 4), round(probs[n], 4), round(probs[c], 4)


_NLI: Optional[NLIScorer] = None


def get_nli() -> NLIScorer:
    global _NLI
    if _NLI is None:
        _NLI = NLIScorer()
    return _NLI


class SafetyCritic:
    """Evaluates whether the generated verdict is grounded in the retrieved evidence."""

    def __init__(self, use_nli: bool = True, thresholds: Optional[Dict[str, float]] = None):
        self.use_nli = use_nli
        self.t = thresholds or load_thresholds()

    def evaluate(
        self,
        predicted_category: str,
        post_text: str,
        cited_clause_id: Optional[str],
        cited_precedent_ids: List[str],
        violation_claim: Optional[str],
        retrieved_policies: Dict[str, str],
        precedent_labels: Dict[str, str],
    ) -> CriticDecision:
        """
        retrieved_policies: {clause_id: clause_text}; precedent_labels: {precedent_id: gold label}.
        """
        reasons: List[str] = []
        nli: Dict[str, float] = {}

        # Gate 1: citations must point at retrieved evidence
        citation_valid = True
        if predicted_category != "Safe":
            if not cited_clause_id or cited_clause_id not in retrieved_policies:
                citation_valid = False
                reasons.append(f"Gate 1: cited clause '{cited_clause_id}' was not retrieved")
        bad_prec = [p for p in cited_precedent_ids if p not in precedent_labels]
        if bad_prec:
            citation_valid = False
            reasons.append(f"Gate 1: cited precedents {bad_prec} were not retrieved")

        # Gate 2: NLI entailment checks
        if self.use_nli:
            scorer = get_nli()
            if predicted_category != "Safe":
                if violation_claim:
                    e, _, c = scorer.score(frame_post(post_text), violation_claim)
                    nli["claim_entailment"], nli["claim_contradiction"] = e, c
                    if e < self.t["claim_entail_min"]:
                        reasons.append(f"Gate 2a: violation claim not entailed by the post (p={e:.2f})")
                else:
                    reasons.append("Gate 2a: no violation claim given")
                if citation_valid and cited_clause_id:
                    e, _, c = scorer.score(retrieved_policies[cited_clause_id], CATEGORY_HYPOTHESES[predicted_category])
                    nli["rule_entailment"], nli["rule_contradiction"] = e, c
                    if e < self.t["rule_entail_min"]:
                        reasons.append(f"Gate 2b: cited clause does not cover '{predicted_category}' (p={e:.2f})")
            else:
                e, _, _ = scorer.score(frame_post(post_text), SAFE_HYPOTHESIS)
                nli["attack_entailment"] = e
                if e >= self.t["safe_attack_max"]:
                    reasons.append(f"Gate 2c: post very likely attacks someone (p={e:.2f}) but verdict is Safe")

        # Gate 3: precedent consensus
        labels = list(precedent_labels.values())
        support = sum(1 for l in labels if l == predicted_category) / len(labels) if labels else 0.0
        if labels:
            top_label = max(set(labels), key=labels.count)
            if support == 0.0 and labels.count(top_label) >= self.t["consensus_min_votes"]:
                reasons.append(f"Gate 3: no retrieved precedent supports the verdict ({labels.count(top_label)}/5 say {top_label})")

        abstain = bool(reasons)
        return CriticDecision(
            is_grounded=citation_valid and not abstain,
            citation_valid=citation_valid,
            support_score=round(support, 4),
            requires_human_review=abstain,
            abstention_reason="; ".join(reasons) or None,
            final_verdict="Needs Human Review" if abstain else predicted_category,
            nli_scores=nli,
        )


if __name__ == "__main__":
    critic = SafetyCritic()
    policies = {"meta_hateful_conduct_003": "Do not post content targeting a person or group on the basis of "
                "their protected characteristics or immigration status with dehumanizing speech or calls for exclusion."}
    dec = critic.evaluate("Hate Speech", "go back to your own country, nobody wants you here",
                          "meta_hateful_conduct_003", [], "The post tells someone to go back to their own country.",
                          policies, {"p1": "Hate Speech", "p2": "Hate Speech", "p3": "Safe"})
    print("Grounded:", dec)
    dec = critic.evaluate("Hate Speech", "I love sunny mornings with my friends",
                          "meta_hateful_conduct_003", [], "The post demeans immigrants.",
                          policies, {"p1": "Safe", "p2": "Safe", "p3": "Safe", "p4": "Safe", "p5": "Safe"})
    print("Ungrounded:", dec)
