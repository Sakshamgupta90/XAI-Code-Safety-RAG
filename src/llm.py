"""
src/llm.py
Thin OpenAI-compatible chat client for Nebius Token Factory (open-weight models).
Course: NTU AI6127 (2026) — Questions 3, 4 & 5.

- Reads NEBIUS_API_KEY from the environment or the project's .env file (never hard-coded).
- Temperature 0 by default, JSON mode, retries with exponential back-off.
- Every response is cached on disk (results/cache/llm_cache.sqlite) keyed by model + messages + params,
  so evaluation runs are reproducible and re-running an ablation costs nothing.
- Records token usage so cost per 1,000 posts can be reported (Question 4).
"""

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = PROJECT_ROOT / "results" / "cache" / "llm_cache.sqlite"

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
# Generator (judge, HyDE): open-weight Qwen3 MoE instruct model, fast and cheap
GENERATOR_MODEL = os.environ.get("XAI_GENERATOR_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# Evaluator (RAGAS judge, relevance assessor): a different, larger model to avoid self-grading bias
EVALUATOR_MODEL = os.environ.get("XAI_EVALUATOR_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")

# USD per 1M tokens (input, output) on Nebius Token Factory, catalogue prices as of Oct 2026
PRICES_PER_M = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (0.10, 0.30),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.20, 0.60),
}


def usage_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICES_PER_M.get(model, (0.0, 0.0))
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6


def _load_api_key() -> str:
    key = os.environ.get("NEBIUS_API_KEY")
    env_file = PROJECT_ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("NEBIUS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("NEBIUS_API_KEY not set. Add it to .env in the project root.")
    return key


class LLMClient:
    """Cached, thread-safe chat-completions client."""

    def __init__(self, model: str = GENERATOR_MODEL, base_url: str = BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._key: Optional[str] = None
        self._lock = threading.Lock()
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(CACHE_FILE), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.read_cache = True  # set False to force fresh API calls (latency benchmarks); results are still cached
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def _cache_key(self, payload: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_mode: bool = False,
        seed: int = 0,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "seed": seed,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        key = self._cache_key(payload)

        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone() if self.read_cache else None
        if row:
            cached = json.loads(row[0])
            with self._lock:
                self.usage["cached"] += 1
                self.usage["cached_prompt_tokens"] = self.usage.get("cached_prompt_tokens", 0) + cached.get("usage", {}).get("prompt_tokens", 0)
                self.usage["cached_completion_tokens"] = self.usage.get("cached_completion_tokens", 0) + cached.get("usage", {}).get("completion_tokens", 0)
            return cached["content"]

        if self._key is None:
            self._key = _load_api_key()
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

        last_err: Optional[Exception] = None
        for attempt in range(6):
            t0 = time.perf_counter()
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=180)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage", {})
                with self._lock:
                    self.usage["calls"] += 1
                    self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    self.usage["seconds"] += time.perf_counter() - t0
                    self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)",
                                     (key, json.dumps({"content": content, "usage": usage})))
                    self._db.commit()
                return content
            except Exception as e:
                last_err = e
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def chat_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Chat in JSON mode and parse the reply; tolerates code fences and leading text."""
        text = self.chat(messages, json_mode=True, **kwargs)
        return parse_json_reply(text)


def parse_json_reply(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def parallel_map(fn: Callable[[Any], Any], items: List[Any], workers: int = 8, desc: str = "") -> List[Any]:
    """Runs fn over items with a thread pool, preserving order and printing progress."""
    results: List[Any] = [None] * len(items)
    done = 0
    t0 = time.perf_counter()

    def run(i):
        results[i] = fn(items[i])
        return i

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(run, range(len(items))):
            done += 1
            if done % 100 == 0 or done == len(items):
                print(f"  [{desc}] {done}/{len(items)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return results


_CLIENTS: Dict[str, LLMClient] = {}


def get_client(model: str = GENERATOR_MODEL) -> LLMClient:
    if model not in _CLIENTS:
        _CLIENTS[model] = LLMClient(model=model)
    return _CLIENTS[model]


if __name__ == "__main__":
    for m in (GENERATOR_MODEL, EVALUATOR_MODEL):
        c = get_client(m)
        out = c.chat_json([{"role": "user", "content": 'Reply with JSON {"ok": true, "model": "<your name>"}'}], max_tokens=50)
        print(m, "->", out, c.usage)
