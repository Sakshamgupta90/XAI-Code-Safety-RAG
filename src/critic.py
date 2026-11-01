#!/usr/bin/env python3
"""
src/critic.py - Dual-Gate Critic & Abstention Mechanism
Course: NTU AI6127 (2026) — Section 2.4, 5.3 & Question 4/5.

Verifies generated safety verdicts against retrieved evidence:
- Gate 1: Deterministic Citation Verification (confirms cited clause ID was in retrieved context).
- Gate 2: Entailment & Support Check (ensures evidence supports the category verdict).
- Abstention: If verification fails or evidence consensus is contradictory, flags for Human Review.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class CriticDecision:
    is_grounded: bool
    citation_valid: bool
    support_score: float
    requires_human_review: bool
    abstention_reason: Optional[str]
    final_verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafetyCritic:
    """Evaluates whether the generated verdict is strictly grounded in retrieved evidence."""

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        predicted_category: str,
        cited_clause_id: Optional[str],
        retrieved_clause_ids: List[str],
        precedent_labels: List[str],
        trigger_words: List[str],
        post_text: str,
        sentic_polarity: float = 0.0
    ) -> CriticDecision:
        """
        Runs dual-gate verification on the candidate verdict.
        """
        # Gate 1: Citation Verification
        # For non-safe categories, a valid retrieved policy clause must be cited
        citation_valid = True
        abstention_reason = None

        if predicted_category in ["Hate Speech", "Harassment-Offensive"]:
            if not cited_clause_id or cited_clause_id not in retrieved_clause_ids:
                citation_valid = False
                abstention_reason = f"Gate 1 Failed: Cited clause '{cited_clause_id}' was not in retrieved context {retrieved_clause_ids}"

        # Gate 2: Support & Consensus Check
        # Measure alignment with retrieved precedent cases
        if precedent_labels:
            matching_precedents = sum(1 for l in precedent_labels if l.lower() in predicted_category.lower() or predicted_category.lower() in l.lower())
            support_score = matching_precedents / len(precedent_labels)
        else:
            support_score = 0.5

        # Check for contradiction / severe uncertainty
        # If predicted is Hate Speech but all precedents are Safe, or vice versa
        contradiction = False
        if predicted_category == "Hate Speech" and all("safe" in l.lower() for l in precedent_labels):
            contradiction = True
            abstention_reason = "Gate 2 Failed: Severe contradiction with retrieved precedent consensus (all Safe)."
        elif predicted_category == "Safe" and sum(1 for l in precedent_labels if "hate" in l.lower()) >= 4:
            contradiction = True
            abstention_reason = "Gate 2 Failed: Flagged Safe despite 80%+ hate speech precedent consensus."

        # Abstention decision
        requires_human_review = (not citation_valid) or contradiction or (support_score < self.confidence_threshold and predicted_category != "Safe")

        if requires_human_review:
            final_verdict = "Needs Human Review"
        else:
            final_verdict = predicted_category

        return CriticDecision(
            is_grounded=citation_valid and not contradiction,
            citation_valid=citation_valid,
            support_score=round(support_score, 4),
            requires_human_review=requires_human_review,
            abstention_reason=abstention_reason,
            final_verdict=final_verdict
        )


if __name__ == "__main__":
    critic = SafetyCritic()

    # Test 1: Grounded Hate Speech decision
    dec1 = critic.evaluate(
        predicted_category="Hate Speech",
        cited_clause_id="meta_hate_tier_3",
        retrieved_clause_ids=["meta_hate_tier_3", "reddit_rule_1_harassment"],
        precedent_labels=["Hate Speech", "Hate Speech", "Harassment-Offensive"],
        trigger_words=["go back"],
        post_text="go back to your country"
    )
    print("Test 1 (Valid):", dec1)

    # Test 2: Hallucinated clause citation (Gate 1 failure)
    dec2 = critic.evaluate(
        predicted_category="Hate Speech",
        cited_clause_id="fictional_unretrieved_clause",
        retrieved_clause_ids=["meta_hate_tier_3", "reddit_rule_1_harassment"],
        precedent_labels=["Hate Speech", "Hate Speech"],
        trigger_words=["go back"],
        post_text="go back to your country"
    )
    print("Test 2 (Gate 1 Hallucination):", dec2)
