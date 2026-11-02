#!/usr/bin/env python3
"""
src/generator.py - LLM Judge (Suitcase Reasoner), HyDE Query Expansion & End-to-End Pipeline
Course: NTU AI6127 (2026) — Questions 4 & 5.

The generator is an open-weight LLM (Qwen3-30B-A3B-Instruct via Nebius, temperature 0, JSON mode)
conditioned on the retrieved evidence:
- Channel A: top-3 policy clauses (query = HyDE policy-topic description of the post)
- Channel B: top-5 HateXplain precedents with their human labels and targets
- Optional SenticNet 9 affective features (ablation switch `use_sentic`)
- Optional suitcase chain-of-thought (ablation switch `use_cot`):
  subjectivity -> polarity -> sarcasm -> target -> policy match -> verdict
- Optional evidence-grounding critic with abstention (ablation switch `use_critic`, see src/critic.py)

Output JSON: subjectivity, polarity, is_sarcastic, category, target_group, violation_claim,
cited_policy_id, cited_precedent_ids, trigger_tokens, explanation, detox_rephrase.
"""

import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.critic import CriticDecision, SafetyCritic
from src.llm import GENERATOR_MODEL, get_client
from src.normalize import normalize_text
from src.sentic import SenticProfile, extract_sentic_features

CATEGORIES = ["Safe", "Hate Speech", "Harassment-Offensive"]

# Same definitions the human annotators used (data/annotation/ANNOTATION_GUIDELINES.md)
TASK_DEFINITIONS = """Categories (choose exactly one):
- "Hate Speech": attacks, dehumanises, excludes or calls for harm against people BECAUSE OF a protected attribute
  (race, ethnicity, nationality, immigration status, religion, caste, sex, gender, gender identity, sexual orientation,
  disability, serious disease, age). Includes slurs used as attacks on such a group and stereotypes presented as fact.
  Sexist content aimed at women/men as a group is Hate Speech.
- "Harassment-Offensive": insulting, abusive, vulgar, threatening or degrading content WITHOUT targeting a protected
  attribute (e.g. insults at an individual, profanity aimed at someone, threats, bullying). A gendered insult aimed at one
  person is Harassment-Offensive.
- "Safe": neither; includes civil opinions, criticism of ideas or institutions, casual profanity not aimed at anyone,
  and slurs quoted to condemn them.
Tie-breakers (same as the human annotation guidelines):
  1. A slur quoted to condemn or report it, or reclaimed in-group use with no attack (e.g. "my nigga, love you bro"),
     is NOT Hate Speech: Safe, or Harassment-Offensive if someone is insulted.
  2. Criticism of a government, party, politician, ideology, organisation or of extremists (e.g. "white supremacists",
     "Islamists") is not an attack on a protected group.
  3. Merely mentioning a group, describing events or discussing racism/sexism is not Hate Speech.
  4. When unsure between two labels, choose the less severe one.
Subjectivity: "Objective" (reports facts, neutral question) or "Subjective" (opinion, emotion, judgement, insult).
Polarity: "Positive" or "Negative" attitude of the author if Subjective; "None" if Objective.
Sarcasm: true only if the literal meaning is the opposite of the intended meaning."""

JSON_SCHEMA_PLAIN = """{
  "subjectivity": "Objective" | "Subjective",
  "polarity": "Positive" | "Negative" | "None",
  "is_sarcastic": true | false,
  "category": "Safe" | "Hate Speech" | "Harassment-Offensive",
  "target_group": "<protected group or person attacked, or 'none'>",
  "violation_claim": "<if not Safe: ONE short sentence (max 15 words) starting 'The post ...' that restates only what the post explicitly says, e.g. 'The post calls someone a dumb bitch.' — no interpretation, motives or implications; else null>",
  "cited_policy_id": "<if not Safe: REQUIRED ID of the retrieved policy clause that the post violates; null only if Safe>",
  "cited_precedent_ids": ["<IDs of retrieved precedents that support your verdict>"],
  "trigger_tokens": ["<words copied exactly from the post that drive the verdict; [] if Safe>"],
  "explanation": "<2-3 sentences grounded in the cited clause and precedents>",
  "detox_rephrase": "<a respectful rewrite that keeps any legitimate point, or null if Safe>"
}"""

JSON_SCHEMA_COT = """{
  "reasoning": {
    "step1_subjectivity": "<is it opinion/emotion or neutral fact?>",
    "step2_polarity": "<positive or negative attitude, if subjective>",
    "step3_sarcasm": "<does the literal meaning invert the intended meaning?>",
    "step4_target": "<who is targeted, and is it because of a protected attribute?>",
    "step5_policy_match": "<which retrieved clause (if any) applies and why; how do the precedents compare?>"
  },
""" + JSON_SCHEMA_PLAIN.lstrip("{")

JSON_SCHEMA_NO_RAG = JSON_SCHEMA_PLAIN.replace(
    '  "cited_policy_id": "<ID of the retrieved policy clause that the post violates, or null if Safe>",\n'
    '  "cited_precedent_ids": ["<IDs of retrieved precedents that support your verdict>"],\n', ""
).replace("grounded in the cited clause and precedents", "justifying the verdict")

HYDE_PROMPT = """You help search a rulebook of social-media content policies (Meta, X, Reddit, YouTube, Singapore IMDA).
Write 2-3 neutral sentences in the style of a content policy describing which policy topics the post below could
relate to (e.g. hateful conduct against a protected group, harassment of an individual, violent threats, self-harm,
dangerous organisations). If the post seems harmless, say which topic it is closest to and that it may not violate any rule.
Do not repeat slurs. Output only the sentences.

Post: "{post}"
"""


@dataclass
class ModerationConfig:
    use_rag: bool = True
    use_sentic: bool = True
    use_cot: bool = True
    use_critic: bool = True
    use_hyde: bool = True
    use_nli: bool = True

    def name(self) -> str:
        if not self.use_rag:
            return "zero-shot"
        parts = ["RAG"] + [n for n, f in (("SenticNet", self.use_sentic), ("CoT", self.use_cot), ("Critic", self.use_critic)) if f]
        return " + ".join(parts)


@dataclass
class ModerationOutput:
    category: str
    final_verdict: str
    subjectivity: str
    polarity: str
    is_sarcastic: bool
    target_group: str
    violation_claim: Optional[str]
    cited_clause: Optional[str]
    cited_precedents: List[str]
    trigger_words: List[str]
    explanation: str
    detox_rephrase: Optional[str]
    reasoning: Dict[str, Any]
    critic_decision: Dict[str, Any]
    sentic_profile: Dict[str, Any]
    retrieved_policies: List[Dict[str, Any]]
    retrieved_precedents: List[Dict[str, Any]]
    precedent_consensus: Dict[str, Any]
    hyde_query: Optional[str]
    parse_error: bool = False
    latency_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_hyde_query(post: str) -> str:
    """HyDE (Gao et al., 2023): a hypothetical policy-style passage used as the Channel A query."""
    client = get_client(GENERATOR_MODEL)
    text = client.chat([{"role": "user", "content": HYDE_PROMPT.format(post=post)}], max_tokens=160)
    return text.strip() or post


def precedent_consensus(precedents: List[Any]) -> Dict[str, Any]:
    labels = [p.category_or_section for p in precedents if p.category_or_section]
    if not labels:
        return {"label": None, "confidence": 0.0, "counts": {}}
    counts = Counter(labels)
    label, n = counts.most_common(1)[0]
    return {"label": label, "confidence": round(n / len(labels), 2), "counts": dict(counts)}


def build_messages(post: str, cfg: ModerationConfig, sentic: Optional[SenticProfile],
                   policies: List[Any], precedents: List[Any]) -> List[Dict[str, str]]:
    system = ("You are XAI-Guard, an explainable content-moderation assistant. "
              "You classify one social-media post and justify the decision.\n\n" + TASK_DEFINITIONS)
    parts = [f'Post to moderate: "{post}"']

    if cfg.use_rag:
        system += ("\n\nUse the retrieved evidence: policy clauses say what is prohibited; precedents show how human "
                   "annotators labelled similar posts (they can be wrong or only loosely similar). Cite only IDs that "
                   "appear in the evidence. Copy trigger tokens exactly from the post.")
        pol_lines = [f"[{p.id}] {p.source} — {p.title_or_targets} — {p.category_or_section}\n"
                     f"    {' '.join(p.text.split()[:170])}" for p in policies]
        prec_lines = [f"[{p.id}] human label: {p.category_or_section}; targets: "
                      f"{', '.join(p.title_or_targets) if p.title_or_targets else 'none'}\n"
                      f"    \"{' '.join(p.text.split()[:60])}\"" for p in precedents]
        parts.append("Retrieved policy clauses (Channel A):\n" + ("\n".join(pol_lines) or "none"))
        parts.append("Retrieved precedents with human labels (Channel B):\n" + ("\n".join(prec_lines) or "none"))

    if cfg.use_sentic and sentic is not None:
        parts.append("SenticNet 9 affective features (lexicon-based, may be noisy):\n" + sentic.summary_text)

    if cfg.use_cot:
        parts.append("Reason step by step inside the 'reasoning' object first, then give the verdict. "
                     "Respond with JSON only, in this format:\n" + JSON_SCHEMA_COT)
    elif cfg.use_rag:
        parts.append("Answer directly, without step-by-step reasoning. Respond with JSON only, in this format:\n" + JSON_SCHEMA_PLAIN)
    else:
        parts.append("Answer directly, without step-by-step reasoning. Respond with JSON only, in this format:\n" + JSON_SCHEMA_NO_RAG)
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


def _pick(value: Any, choices: List[str], default: str) -> str:
    v = str(value or "").strip().lower()
    for c in choices:
        if v == c.lower():
            return c
    for c in choices:
        if c.lower().split()[0] in v:
            return c
    return default


class ExplainableSafetyReasoner:
    """End-to-end pipeline: normalise -> SenticNet -> dual-channel retrieval -> LLM judge -> critic."""

    def __init__(self, retriever=None):
        self._retriever = retriever
        self.client = get_client(GENERATOR_MODEL)
        self.critic = SafetyCritic()

    @property
    def retriever(self):
        if self._retriever is None:
            from src.retriever import DualChannelRetriever
            self._retriever = DualChannelRetriever()
        return self._retriever

    def retrieve_evidence(self, clean_text: str, cfg: ModerationConfig):
        from src.retriever import RetrievalConfig
        timings = {}
        t0 = time.perf_counter()
        hyde = generate_hyde_query(clean_text) if cfg.use_hyde else None
        timings["hyde_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        policies = self.retriever.retrieve(hyde or clean_text, channel="policy",
                                           config=RetrievalConfig(top_k=3, use_rerank=True, normalize=hyde is None))
        timings["policy_retrieval_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        precedents = self.retriever.retrieve(clean_text, channel="precedent", config=RetrievalConfig(top_k=5, use_rerank=False))
        timings["precedent_retrieval_ms"] = (time.perf_counter() - t0) * 1000
        return hyde, policies, precedents, timings

    def reason(
        self,
        raw_text: str,
        cfg: Optional[ModerationConfig] = None,
        policies: Optional[List[Any]] = None,
        precedents: Optional[List[Any]] = None,
        hyde_query: Optional[str] = None,
    ) -> ModerationOutput:
        cfg = cfg or ModerationConfig()
        timings: Dict[str, float] = {}
        t_start = time.perf_counter()
        clean_text = normalize_text(raw_text)

        t0 = time.perf_counter()
        sentic = extract_sentic_features(clean_text)
        timings["senticnet_ms"] = (time.perf_counter() - t0) * 1000

        if cfg.use_rag and (policies is None or precedents is None):
            hyde_query, policies, precedents, t = self.retrieve_evidence(clean_text, cfg)
            timings.update(t)
        if not cfg.use_rag:
            policies, precedents, hyde_query = [], [], None

        t0 = time.perf_counter()
        messages = build_messages(clean_text, cfg, sentic if cfg.use_sentic else None, policies, precedents)
        raw = self.client.chat_json(messages, max_tokens=900 if cfg.use_cot else 600)
        timings["llm_ms"] = (time.perf_counter() - t0) * 1000

        consensus = precedent_consensus(precedents)
        parse_error = not raw or "category" not in raw
        category = _pick(raw.get("category"), CATEGORIES, consensus["label"] or "Safe")
        subjectivity = _pick(raw.get("subjectivity"), ["Objective", "Subjective"], "Subjective")
        polarity = _pick(raw.get("polarity"), ["Positive", "Negative", "None"], "None")
        if subjectivity == "Objective":
            polarity = "None"
        sarcastic = raw.get("is_sarcastic")
        is_sarcastic = sarcastic is True or str(sarcastic).lower() in ("true", "yes")
        cited_clause = raw.get("cited_policy_id") or None
        if isinstance(cited_clause, str) and cited_clause.lower() in ("null", "none", ""):
            cited_clause = None
        cited_precedents = [str(x) for x in (raw.get("cited_precedent_ids") or []) if x]
        triggers = [str(t) for t in (raw.get("trigger_tokens") or []) if str(t).lower() in clean_text.lower()]

        t0 = time.perf_counter()
        if cfg.use_critic and cfg.use_rag:
            critic_dec = self.critic.evaluate(
                predicted_category=category,
                post_text=clean_text,
                cited_clause_id=cited_clause,
                cited_precedent_ids=cited_precedents,
                violation_claim=raw.get("violation_claim"),
                retrieved_policies={p.id: p.text for p in policies},
                precedent_labels={p.id: p.category_or_section for p in precedents},
            ) if self.critic.use_nli == cfg.use_nli else SafetyCritic(use_nli=cfg.use_nli).evaluate(
                category, clean_text, cited_clause, cited_precedents, raw.get("violation_claim"),
                {p.id: p.text for p in policies}, {p.id: p.category_or_section for p in precedents})
        else:
            critic_dec = CriticDecision(True, True, consensus["confidence"], False, None, category)
        timings["critic_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000

        return ModerationOutput(
            category=category,
            final_verdict=critic_dec.final_verdict,
            subjectivity=subjectivity,
            polarity=polarity,
            is_sarcastic=is_sarcastic,
            target_group=str(raw.get("target_group") or "none"),
            violation_claim=raw.get("violation_claim"),
            cited_clause=cited_clause,
            cited_precedents=cited_precedents,
            trigger_words=triggers,
            explanation=str(raw.get("explanation") or ""),
            detox_rephrase=raw.get("detox_rephrase"),
            reasoning=raw.get("reasoning") or {},
            critic_decision=critic_dec.to_dict(),
            sentic_profile=sentic.to_dict(),
            retrieved_policies=[p.to_dict() for p in policies],
            retrieved_precedents=[p.to_dict() for p in precedents],
            precedent_consensus=consensus,
            hyde_query=hyde_query,
            parse_error=parse_error,
            latency_ms={k: round(v, 1) for k, v in timings.items()},
        )


if __name__ == "__main__":
    reasoner = ExplainableSafetyReasoner()
    test_samples = [
        "go back to your own country, nobody wants you here",
        "Oh wow, another DEI hire pretending to understand engineering, truly inspiring.",
        "Can someone recommend a good beginner tutorial for Python and machine learning?",
    ]
    for sample in test_samples:
        out = reasoner.reason(sample)
        print("=" * 70)
        print("Input:", sample)
        print(f"Verdict: {out.category} -> final: {out.final_verdict} | subj: {out.subjectivity} | "
              f"polarity: {out.polarity} | sarcasm: {out.is_sarcastic}")
        print(f"Cited: {out.cited_clause} | precedents: {out.cited_precedents} | triggers: {out.trigger_words}")
        print(f"Claim: {out.violation_claim}")
        print(f"Explanation: {out.explanation}")
        print(f"Critic: {out.critic_decision['abstention_reason']} | NLI {out.critic_decision['nli_scores']}")
        print(f"Latency: {out.latency_ms}")
