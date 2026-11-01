#!/usr/bin/env python3
"""
src/generator.py - Explainable Safety Reasoner & Suitcase CoT Generator
Course: NTU AI6127 (2026) — Questions 4 & 5.

Decomposes online safety reasoning into:
1. Subjectivity & Intent Analysis
2. Sarcasm & Rhetorical Tone Detection
3. Grounded Policy Violation & Precedent Alignment
4. SenticNet 7 Neurosymbolic Affective Features
5. Critic Verification & Abstention

Emits structured JSON output with explanations, cited clauses, and detoxified rephrasing.
"""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.critic import SafetyCritic, CriticDecision
from src.normalize import normalize_text
from src.retriever import DualChannelRetriever, RetrievalConfig, SearchResult
from src.sentic import extract_sentic_features, SenticProfile


@dataclass
class ModerationOutput:
    category: str
    is_sarcastic: bool
    subjectivity: str
    cited_clause: Optional[str]
    trigger_words: List[str]
    explanation: str
    detox_rephrase: str
    critic_decision: Dict[str, Any]
    sentic_profile: Dict[str, Any]
    retrieved_policies: List[Dict[str, Any]]
    retrieved_precedents: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExplainableSafetyReasoner:
    """End-to-end downstream reasoner executing the dual-channel RAG safety workflow."""

    def __init__(self, confidence_threshold: float = 0.50):
        self.retriever = DualChannelRetriever()
        self.critic = SafetyCritic(confidence_threshold=confidence_threshold)
        self.cfg_policy = RetrievalConfig(top_k=3, use_bm25=True, use_dense=True, use_rrf=True, use_rerank=False)
        self.cfg_prec = RetrievalConfig(top_k=5, use_bm25=True, use_dense=True, use_rrf=True, use_rerank=False)

    def format_suitcase_prompt(
        self,
        text: str,
        sentic: SenticProfile,
        policies: List[SearchResult],
        precedents: List[SearchResult]
    ) -> str:
        """Constructs the structured Suitcase Chain-of-Thought prompt."""
        policy_context = "\n".join([f"- [{p.id}] {p.source} - {p.category_or_section}: {p.text[:200]}..." for p in policies])
        precedent_context = "\n".join([f"- Case [{p.id}] Label: {p.category_or_section} | Target: {p.title_or_targets} | Post: \"{p.text[:120]}...\"" for p in precedents])

        prompt = f"""You are XAI-Guard, an explainable content safety moderation system.
Analyze the following social media post using the provided evidence and affective features.

Input Post: "{text}"

[Neurosymbolic Affective Features (SenticNet 7)]
- Affective Polarity: {sentic.polarity} (-1.0 = highly negative, +1.0 = highly positive)
- Dominant Emotion: {sentic.dominant_emotion} (Intensity: {sentic.emotion_intensity})
- Hourglass Dimensions: Pleasantness={sentic.pleasantness}, Temper/Rage={sentic.aptitude}, Fear={sentic.sensitivity}
- Extracted Affective Concepts: {sentic.matched_concepts}

[Retrieved Statutory & Platform Safety Policies (Channel A)]
{policy_context}

[Retrieved Historical Human-Adjudicated Precedents (Channel B)]
{precedent_context}

Execute step-by-step reasoning:
Step 1 (Subjectivity): Determine if the post expresses objective factual content, subjective opinion, or targeted animus.
Step 2 (Sarcasm): Detect whether the post employs irony, passive-aggressive sarcasm, or counter-intuitive sentiment.
Step 3 (Policy Grounding): Determine if the post violates any codified clause in Channel A, supported by precedent decisions in Channel B.
Step 4 (Detoxification): Suggest a non-toxic, constructive rephrasing if toxic; otherwise preserve original text.

Respond strictly in valid JSON format:
{{
  "subjectivity": "Subjective" | "Objective",
  "is_sarcastic": true | false,
  "category": "Hate Speech" | "Harassment-Offensive" | "Safe",
  "cited_clause": "<Exact ID from Channel A or null>",
  "trigger_words": ["<words or phrases driving toxicity>"],
  "explanation": "<2-3 sentence grounded justification>",
  "detox_rephrase": "<constructive alternative or original text>"
}}
"""
        return prompt

    def reason(
        self,
        raw_text: str,
        use_sentic: bool = True,
        use_critic: bool = True,
        use_rag: bool = True,
        policies: Optional[List[SearchResult]] = None,
        precedents: Optional[List[SearchResult]] = None
    ) -> ModerationOutput:
        """
        Executes end-to-end classification and explanation with ablation switches.
        """
        # Step 1: Normalization
        clean_text = normalize_text(raw_text)

        # Step 2: SenticNet Features
        sentic_prof = extract_sentic_features(clean_text) if use_sentic else SenticProfile(
            polarity=0.0, dominant_emotion="Neutral", emotion_intensity=0.0,
            pleasantness=0.0, attention=0.0, sensitivity=0.0, aptitude=0.0,
            matched_concepts=[], summary_text="SenticNet disabled"
        )

        # Step 3: Dual-Channel Retrieval
        if use_rag:
            if policies is None:
                policies = self.retriever.retrieve(clean_text, channel="policy", config=self.cfg_policy)
            if precedents is None:
                precedents = self.retriever.retrieve(clean_text, channel="precedent", config=self.cfg_prec)
        else:
            policies = []
            precedents = []

        retrieved_clause_ids = [p.id for p in policies]
        precedent_labels = [p.category_or_section for p in precedents if p.category_or_section]

        # Step 4: Reasoning Decision
        # Sarcasm detection heuristics
        sarcasm_indicators = ["oh wow", "oh look", "truly inspiring", "absolute genius", "brilliant", "thanks a lot", "great job", "slow clap"]
        is_sarcastic = any(ind in clean_text.lower() for ind in sarcasm_indicators)

        # Subjectivity detection heuristics
        opinion_markers = ["i think", "in my opinion", "believe", "feel", "hate", "disgusting", "terrible", "worst", "idiot", "loser", "ugly"]
        is_subjective = any(m in clean_text.lower() for m in opinion_markers) or sentic_prof.polarity != 0.0 or is_sarcastic
        subjectivity_label = "Subjective" if is_subjective else "Objective"

        # Category adjudication
        # 1. Precedent consensus
        prec_counts = {}
        for l in precedent_labels:
            prec_counts[l] = prec_counts.get(l, 0) + 1
        majority_prec = max(prec_counts, key=prec_counts.get) if prec_counts else "Safe"

        # 2. SenticNet affective signals
        high_rage_or_disgust = sentic_prof.aptitude >= 0.7 or sentic_prof.dominant_emotion in ["Rage", "Disgust"]

        # Identify candidate trigger words from Sentic concepts and toxic vocabulary
        trigger_words = [c for c in sentic_prof.matched_concepts]
        if not trigger_words and majority_prec != "Safe":
            # Extract high-salience terms
            words = [w for w in clean_text.lower().split() if len(w) > 4 and w not in ["about", "there", "their", "would", "could", "should"]]
            trigger_words = words[:2]

        if use_rag and precedents:
            # RAG-grounded decision
            pred_category = majority_prec
            if high_rage_or_disgust and pred_category == "Safe":
                pred_category = "Harassment-Offensive"

            # Select best cited policy clause
            if pred_category == "Hate Speech":
                hate_clauses = [p.id for p in policies if "hate" in p.id or "conduct" in p.id]
                cited_clause = hate_clauses[0] if hate_clauses else (retrieved_clause_ids[0] if retrieved_clause_ids else None)
            elif pred_category in ["Harassment-Offensive", "offensive"]:
                harass_clauses = [p.id for p in policies if "harass" in p.id or "bully" in p.id or "abusive" in p.id or "cyberbullying" in p.id]
                cited_clause = harass_clauses[0] if harass_clauses else (retrieved_clause_ids[0] if retrieved_clause_ids else None)
            else:
                cited_clause = None
        else:
            # Non-RAG baseline fallback
            if high_rage_or_disgust or sentic_prof.polarity <= -0.6:
                pred_category = "Hate Speech"
            elif sentic_prof.polarity < -0.2:
                pred_category = "Harassment-Offensive"
            else:
                pred_category = "Safe"
            cited_clause = None

        # Build Explanation
        if pred_category == "Safe":
            explanation = "The content does not violate safety policies. It expresses benign discourse or neutral commentary without targeted hostility."
            detox_rephrase = clean_text
        else:
            top_pol = policies[0].category_or_section if policies else "platform standards"
            explanation = (
                f"Content flagged as {pred_category}. "
                f"It conflicts with {top_pol} ({cited_clause or 'general policy'}) "
                f"and aligns with {prec_counts.get(pred_category, 0)} retrieved human-adjudicated precedents. "
                f"Affective analysis detected {sentic_prof.dominant_emotion} (polarity: {sentic_prof.polarity})."
            )
            detox_rephrase = "Please express your perspective constructively without personal attacks, slurs, or discriminatory generalizations."

        # Step 5: Critic Verification
        if use_critic:
            critic_dec = self.critic.evaluate(
                predicted_category=pred_category,
                cited_clause_id=cited_clause,
                retrieved_clause_ids=retrieved_clause_ids,
                precedent_labels=precedent_labels,
                trigger_words=trigger_words,
                post_text=clean_text,
                sentic_polarity=sentic_prof.polarity
            )
        else:
            critic_dec = CriticDecision(
                is_grounded=True,
                citation_valid=True,
                support_score=1.0,
                requires_human_review=False,
                abstention_reason=None,
                final_verdict=pred_category
            )

        return ModerationOutput(
            category=pred_category,
            is_sarcastic=is_sarcastic,
            subjectivity=subjectivity_label,
            cited_clause=cited_clause,
            trigger_words=trigger_words,
            explanation=explanation,
            detox_rephrase=detox_rephrase,
            critic_decision=critic_dec.to_dict(),
            sentic_profile=sentic_prof.to_dict(),
            retrieved_policies=[asdict(p) for p in policies],
            retrieved_precedents=[asdict(p) for p in precedents]
        )


if __name__ == "__main__":
    reasoner = ExplainableSafetyReasoner()

    test_samples = [
        "go back to your own country, nobody wants you here",
        "Oh wow, another DEI hire pretending to understand engineering, truly inspiring.",
        "Can someone recommend a good beginner tutorial for Python and machine learning?",
    ]

    for sample in test_samples:
        print("=" * 60)
        print("Input:", sample)
        out = reasoner.reason(sample)
        print(f"Verdict: {out.category} | Final (Critic): {out.critic_decision['final_verdict']}")
        print(f"Subjectivity: {out.subjectivity} | Sarcasm: {out.is_sarcastic}")
        print(f"Cited Clause: {out.cited_clause}")
        print(f"Explanation: {out.explanation}")
        print(f"Detox Rephrase: {out.detox_rephrase}\n")
