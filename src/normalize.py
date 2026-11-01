"""
src/normalize.py
Microtext Normalization Pipeline for Online Safety Moderation (NTU AI6127).

Applies consistent preprocessing to:
1. URLs: Replaced with '[URL]'
2. User Mentions: Unified to '[USER]'
3. Emojis: Demojized to text representation (e.g., :face_vomiting:)
4. Character Elongation: Capped at 2 repetitions (e.g., 'sooooo' -> 'soo')
5. Hashtag Segmentation: Splits camelCase or underscored hashtags (e.g., '#SendThemBack' -> 'send them back')
6. Slang & Abbreviations: Dictionary expansion (e.g., 'smh' -> 'shaking my head')
7. Leetspeak / Obfuscation: De-obfuscates evasion tricks (e.g., 'k1ll' -> 'kill', 'f*ck' -> 'fuck')

CRITICAL RULE:
Slurs and explicit offensive terms are deliberately PRESERVED because they represent
the primary evidence required for content safety classification and policy grounding.
"""

import re
from typing import Dict, List, Tuple

# Common social media slang and abbreviation expansion dictionary
SLANG_DICT: Dict[str, str] = {
    "smh": "shaking my head",
    "afaik": "as far as i know",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "idk": "i do not know",
    "idc": "i do not care",
    "fwiw": "for what it is worth",
    "gtfo": "get the fuck out",
    "stfu": "shut the fuck up",
    "rt": "",
    "dm": "direct message",
    "fyi": "for your information",
    "irl": "in real life",
    "omg": "oh my god",
    "wth": "what the hell",
    "wtf": "what the fuck",
    "bs": "bullshit",
    "pos": "piece of shit",
    "sob": "son of a bitch",
    "bc": "because",
    "b/c": "because",
    "w/": "with",
    "w/o": "without",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ur": "your",
    "u": "you",
    "r": "are",
}

# Targeted de-obfuscation patterns for common adversarial evasion tactics
# This ensures that lexical search (BM25) and dense embeddings map evasive spelling to true lexical intent.
OBFUSCATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bk1ll\b", re.IGNORECASE), "kill"),
    (re.compile(r"\bk!ll\b", re.IGNORECASE), "kill"),
    (re.compile(r"\bd1e\b", re.IGNORECASE), "die"),
    (re.compile(r"\bf\*ck\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bf\*\*k\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bfuk\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bfck\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bb!tch\b", re.IGNORECASE), "bitch"),
    (re.compile(r"\bb\*tch\b", re.IGNORECASE), "bitch"),
    (re.compile(r"\bsh!t\b", re.IGNORECASE), "shit"),
    (re.compile(r"\bsh\*t\b", re.IGNORECASE), "shit"),
    (re.compile(r"\ba\$\$\b", re.IGNORECASE), "ass"),
    (re.compile(r"\ba\*\*hole\b", re.IGNORECASE), "asshole"),
    (re.compile(r"\bn1gg[a-z]*\b", re.IGNORECASE), "nigger"),  # De-obfuscate slur evasion for detection
    (re.compile(r"\bn\*gg[a-z]*\b", re.IGNORECASE), "nigger"),
    (re.compile(r"\bf\*g\b", re.IGNORECASE), "fag"),
    (re.compile(r"\br\*tard\b", re.IGNORECASE), "retard"),
]

# URL regex
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# User mention regex (handles @user, @username, and HateXplain's <user> token)
USER_MENTION_PATTERN = re.compile(r"@[\w_]+|<user>", re.IGNORECASE)

# Repeated character regex: matches 3 or more repeated characters (e.g., 'sooooo' -> 'soo')
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}", re.IGNORECASE)

# Hashtag pattern
HASHTAG_PATTERN = re.compile(r"#([a-zA-Z0-9_]+)")


def demojize_text(text: str) -> str:
    """
    Converts Unicode emojis to text tokens using the `emoji` package if installed,
    or a robust fallback for common sentiment/toxicity emojis.
    """
    try:
        import emoji
        return emoji.demojize(text, delimiters=(" :", ": "))
    except ImportError:
        # Fallback dictionary for common affective/toxic emojis
        fallback_emojis = {
            "🤮": " :face_vomiting: ",
            "🤢": " :nauseated_face: ",
            "🤡": " :clown_face: ",
            "💩": " :pile_of_poo: ",
            "💀": " :skull: ",
            "🤬": " :face_with_symbols_on_mouth: ",
            "😡": " :enraged_face: ",
            "😠": " :angry_face: ",
            "🖕": " :middle_finger: ",
            "👎": " :thumbs_down: ",
            "🔥": " :fire: ",
            "❤️": " :red_heart: ",
            "😂": " :face_with_tears_of_joy: ",
            "🤣": " :rolling_on_the_floor_laughing: ",
        }
        for em, repl in fallback_emojis.items():
            text = text.replace(em, repl)
        return text


def segment_hashtag(tag: str) -> str:
    """
    Splits hashtags based on camelCase or underscores.
    Example: 'SendThemBack' -> 'send them back'
    """
    # Replace underscores with spaces
    tag = tag.replace("_", " ")
    # Insert space before capital letters in camelCase
    split_words = re.sub(r"([a-z])([A-Z])", r"\1 \2", tag)
    return split_words.strip()


def replace_hashtags(text: str) -> str:
    """Finds all hashtags and expands them into natural words."""
    def _repl(match: re.Match) -> str:
        raw_tag = match.group(1)
        return segment_hashtag(raw_tag)

    return HASHTAG_PATTERN.sub(_repl, text)


def normalize_text(text: str) -> str:
    """
    Main entry point for microtext normalization.
    Takes raw social media text and applies the deterministic 7-stage cleaning pipeline.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Normalize URLs
    cleaned = URL_PATTERN.sub("[URL]", text)

    # 2. Normalize User Mentions
    cleaned = USER_MENTION_PATTERN.sub("[USER]", cleaned)

    # 3. Demojize emojis to retain affective signal as matchable text
    cleaned = demojize_text(cleaned)

    # 4. De-obfuscate adversarial leetspeak / evasions
    for pattern, replacement in OBFUSCATION_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)

    # 5. Segment hashtags to restore word boundaries
    cleaned = replace_hashtags(cleaned)

    # 6. Reduce character elongation (cap at 2 identical consecutive characters)
    # e.g., 'sooooo' -> 'soo', 'waaaay' -> 'waay'
    cleaned = ELONGATION_PATTERN.sub(r"\1\1", cleaned)

    # 7. Token-level normalization: abbreviations, slang, and whitespace
    tokens = cleaned.split()
    normalized_tokens = []
    for tok in tokens:
        lower_tok = tok.lower().strip(".,!?;:\"'()[]{}")
        if lower_tok in SLANG_DICT:
            expansion = SLANG_DICT[lower_tok]
            if expansion:
                normalized_tokens.append(expansion)
        else:
            normalized_tokens.append(tok)

    result = " ".join(normalized_tokens)

    # Clean redundant whitespace
    result = re.sub(r"\s+", " ", result).strip()
    return result


if __name__ == "__main__":
    # Demonstration of the 10 representative test cases required for the report
    demo_cases = [
        ("Go visit https://t.co/xyz123 right now!", "URL replacement"),
        ("@realUser123 and <user> are talking nonsense", "Mention normalization"),
        ("I hate this policy so much 🤮🤮🤮", "Emoji demojization"),
        ("This is sooooooo ridiculousssss", "Elongation reduction"),
        ("#StopAsianHate and #SendThemBack now", "Hashtag segmentation"),
        ("smh idk why people think that tbh", "Slang and abbreviation expansion"),
        ("I will k1ll you if you show up again", "Leetspeak de-obfuscation (kill)"),
        ("What the f*ck is wrong with this b!tch", "Profanity de-obfuscation (fuck, bitch)"),
        ("You are an a$$hole who deserves to d1e", "Obfuscation de-obfuscation (ass, die)"),
        ("Clean polite message with no slang or tricks.", "Unchanged baseline preservation"),
    ]

    print("=" * 80)
    print("XAI-Guard Microtext Normalization Test Suite")
    print("=" * 80)
    for i, (raw, rule) in enumerate(demo_cases, 1):
        processed = normalize_text(raw)
        print(f"[{i}] Rule: {rule}")
        print(f"    Raw:        {raw}")
        print(f"    Normalized: {processed}\n")
