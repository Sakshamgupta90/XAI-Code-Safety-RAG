"""
src/collect_policies.py
Policy Corpus Collection (Channel A Knowledge Base).
Course: NTU AI6127 (2026) — Question 1 Deliverable.

Actions:
1. Fetches the official, publicly accessible content-policy pages of Meta, X, Reddit and YouTube,
   plus the Singapore IMDA Code of Practice for Online Safety (PDF).
2. JavaScript-rendered pages (Meta, X) are rendered with a headless Chromium browser (Playwright);
   server-rendered pages are fetched with requests. PDFs are parsed with pypdf.
3. Main content is extracted with Trafilatura as Markdown so section headings are preserved.
4. Saves one Markdown file per policy page to data/raw/policies/ with a provenance header
   (source URL, real UTC retrieval timestamp, terms-of-use note) and the raw HTML/PDF next to it.

Only public pages are fetched, once, with a polite delay. The text is used for non-commercial
academic research; each platform retains copyright of its policy text.
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import trafilatura

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = PROJECT_ROOT / "data" / "raw" / "policies"
RAW_DIR = POLICY_DIR / "raw"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
)

TERMS_NOTE = (
    "Publicly available policy text collected once for non-commercial academic research "
    "(NTU AI6127). Copyright remains with the publisher; see the publisher's terms of use."
)

# (doc_id, source, title, url, fetch_mode) — fetch_mode: "browser" | "http" | "pdf"
POLICY_SOURCES: List[Dict[str, str]] = [
    # Meta Community Standards (JS-rendered)
    {"doc_id": "meta_hateful_conduct", "source": "Meta", "title": "Meta Community Standards: Hateful Conduct",
     "url": "https://transparency.meta.com/policies/community-standards/hateful-conduct/", "mode": "browser"},
    {"doc_id": "meta_bullying_harassment", "source": "Meta", "title": "Meta Community Standards: Bullying and Harassment",
     "url": "https://transparency.meta.com/policies/community-standards/bullying-harassment/", "mode": "browser"},
    {"doc_id": "meta_violence_incitement", "source": "Meta", "title": "Meta Community Standards: Violence and Incitement",
     "url": "https://transparency.meta.com/policies/community-standards/violence-incitement/", "mode": "browser"},
    {"doc_id": "meta_dangerous_orgs", "source": "Meta", "title": "Meta Community Standards: Dangerous Organizations and Individuals",
     "url": "https://transparency.meta.com/policies/community-standards/dangerous-individuals-organizations/", "mode": "browser"},
    {"doc_id": "meta_suicide_self_injury", "source": "Meta", "title": "Meta Community Standards: Suicide, Self-Injury, and Eating Disorders",
     "url": "https://transparency.meta.com/policies/community-standards/suicide-self-injury/", "mode": "browser"},
    # X Rules (JS-rendered, blocks plain HTTP clients)
    {"doc_id": "x_rules", "source": "X", "title": "The X Rules",
     "url": "https://help.x.com/en/rules-and-policies/x-rules", "mode": "browser"},
    {"doc_id": "x_hateful_conduct", "source": "X", "title": "X Hateful Conduct Policy",
     "url": "https://help.x.com/en/rules-and-policies/hateful-conduct-policy", "mode": "browser"},
    {"doc_id": "x_abuse_harassment", "source": "X", "title": "X Abuse and Harassment Policy",
     "url": "https://help.x.com/en/rules-and-policies/abusive-behavior", "mode": "browser"},
    {"doc_id": "x_violent_speech", "source": "X", "title": "X Violent Speech Policy",
     "url": "https://help.x.com/en/rules-and-policies/violent-speech", "mode": "browser"},
    {"doc_id": "x_violent_entities", "source": "X", "title": "X Violent and Hateful Entities Policy",
     "url": "https://help.x.com/en/rules-and-policies/violent-entities", "mode": "browser"},
    {"doc_id": "x_suicide_self_harm", "source": "X", "title": "X Suicide and Self-Harm Policy",
     "url": "https://help.x.com/en/rules-and-policies/glorifying-self-harm", "mode": "browser"},
    # Reddit
    {"doc_id": "reddit_rules", "source": "Reddit", "title": "Reddit Rules",
     "url": "https://redditinc.com/policies/reddit-rules", "mode": "http"},
    {"doc_id": "reddit_hate", "source": "Reddit", "title": "Reddit Rule 1: Promoting Hate Based on Identity or Vulnerability",
     "url": "https://support.reddithelp.com/hc/en-us/articles/360045715951", "mode": "browser"},
    {"doc_id": "reddit_harassment", "source": "Reddit", "title": "Reddit Rule 1: Harassment",
     "url": "https://support.reddithelp.com/hc/en-us/articles/360043071072", "mode": "browser"},
    {"doc_id": "reddit_violence", "source": "Reddit", "title": "Reddit Rule 1: Violent Content",
     "url": "https://support.reddithelp.com/hc/en-us/articles/360043513151", "mode": "browser"},
    # YouTube Community Guidelines (server-rendered)
    {"doc_id": "youtube_hate_speech", "source": "YouTube", "title": "YouTube Hate Speech Policy",
     "url": "https://support.google.com/youtube/answer/2801939?hl=en", "mode": "http"},
    {"doc_id": "youtube_harassment", "source": "YouTube", "title": "YouTube Harassment and Cyberbullying Policies",
     "url": "https://support.google.com/youtube/answer/2802268?hl=en", "mode": "http"},
    {"doc_id": "youtube_violent_extremism", "source": "YouTube", "title": "YouTube Violent Extremist or Criminal Organizations Policy",
     "url": "https://support.google.com/youtube/answer/9229472?hl=en", "mode": "http"},
    {"doc_id": "youtube_violent_graphic", "source": "YouTube", "title": "YouTube Violent or Graphic Content Policies",
     "url": "https://support.google.com/youtube/answer/2802008?hl=en", "mode": "http"},
    # Singapore IMDA (statutory code under the Broadcasting Act 1994)
    {"doc_id": "imda_online_safety_code", "source": "Singapore IMDA",
     "title": "IMDA Code of Practice for Online Safety (Social Media Services)",
     "url": "https://www.imda.gov.sg/assets/5d01a679-c85e-4362-820c-c99d741aeaf8.pdf", "mode": "pdf"},
]


def fetch_http(url: str) -> bytes:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    return resp.content


def fetch_browser(urls: List[str]) -> Dict[str, str]:
    """Renders JavaScript-heavy pages in headless Chromium and returns their final HTML."""
    from playwright.sync_api import sync_playwright

    pages: Dict[str, str] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=USER_AGENT)
        for url in urls:
            try:
                page.goto(url, wait_until="networkidle", timeout=90000)
            except Exception:
                # Some pages keep long-polling; the DOM is usually complete anyway
                pass
            time.sleep(2.0)
            pages[url] = page.content()
            print(f"[OK] Rendered {url}")
        browser.close()
    return pages


def html_to_markdown(html: str) -> str:
    md = trafilatura.extract(
        html, output_format="markdown", include_formatting=True,
        include_links=False, include_tables=True, favor_recall=True,
    ) or ""
    return md.strip()


def pdf_to_markdown(data: bytes) -> str:
    """Extracts PDF text and turns numbered clause lines into Markdown headings."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    lines: List[str] = []
    for page in reader.pages:
        for line in (page.extract_text() or "").splitlines():
            line = line.strip()
            if not line or re.fullmatch(r"\d+", line):
                continue
            lines.append(line)

    out: List[str] = []
    for line in lines:
        # "PART 2", "SECTION B: ..." or short all-caps titles become headings
        if re.match(r"^(PART|SECTION)\s+[A-Z0-9]+\b", line) or (line.isupper() and 3 <= len(line.split()) <= 12):
            out.append(f"\n## {line.title()}\n")
        else:
            out.append(line)
    text = "\n".join(out)
    # Re-join hard-wrapped lines inside paragraphs
    text = re.sub(r"(?<![.:;\n])\n(?!\n|## |\(|\d+\.)", " ", text)
    return text.strip()


def write_policy(spec: Dict[str, str], markdown: str, retrieved_at: str):
    header = (
        f"# {spec['title']}\n"
        f"Source: {spec['source']}\n"
        f"Source URL: {spec['url']}\n"
        f"Retrieved: {retrieved_at}\n"
        f"Terms: {TERMS_NOTE}\n\n"
    )
    out = POLICY_DIR / f"{spec['doc_id']}.md"
    out.write_text(header + markdown + "\n", encoding="utf-8")
    print(f"[SAVED] {out.name}: {len(markdown.split()):,} words")


def collect_policies(only: Optional[List[str]] = None):
    print("=" * 80)
    print("XAI-Guard: Policy Corpus Collection (Channel A)")
    print("=" * 80)
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    specs = [s for s in POLICY_SOURCES if not only or s["doc_id"] in only]
    browser_specs = [s for s in specs if s["mode"] == "browser"]
    rendered = fetch_browser([s["url"] for s in browser_specs]) if browser_specs else {}

    manifest = []
    for spec in specs:
        retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        try:
            if spec["mode"] == "browser":
                html = rendered.get(spec["url"], "")
                (RAW_DIR / f"{spec['doc_id']}.html").write_text(html, encoding="utf-8")
                markdown = html_to_markdown(html)
            elif spec["mode"] == "pdf":
                data = fetch_http(spec["url"])
                (RAW_DIR / f"{spec['doc_id']}.pdf").write_bytes(data)
                markdown = pdf_to_markdown(data)
            else:
                data = fetch_http(spec["url"])
                (RAW_DIR / f"{spec['doc_id']}.html").write_bytes(data)
                markdown = html_to_markdown(data.decode("utf-8", errors="ignore"))
                time.sleep(1.0)
        except Exception as e:
            print(f"[FAIL] {spec['doc_id']}: {e}")
            continue

        if len(markdown.split()) < 80:
            print(f"[WARN] {spec['doc_id']}: only {len(markdown.split())} words extracted — check the page")
        write_policy(spec, markdown, retrieved_at)
        manifest.append({**spec, "retrieved_at": retrieved_at, "words": len(markdown.split())})

    manifest_file = POLICY_DIR / "provenance.json"
    old = json.loads(manifest_file.read_text()) if manifest_file.exists() and only else []
    merged = {m["doc_id"]: m for m in old}
    merged.update({m["doc_id"]: m for m in manifest})
    manifest_file.write_text(json.dumps(list(merged.values()), indent=2), encoding="utf-8")
    print(f"[SAVED] Provenance manifest -> {manifest_file}")


if __name__ == "__main__":
    collect_policies(only=sys.argv[1:] or None)
