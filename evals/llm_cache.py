"""A tiny file-backed cache for the LLM calls that build eval material.

Generating questions or synthetic documents with an LLM is not reproducible on its
own (sampling, server upgrades), so every response is stored in a JSONL file keyed by
a hash of (model, prompt, parameters). Re-running a builder replays the cached
responses and produces byte-identical output; only new prompts reach the server.

The cache file is committed next to the material it produced.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Optional


def cache_key(model: str, prompt: str, max_tokens: int, temperature: float,
              reasoning_effort: Optional[str]) -> str:
    raw = json.dumps([model, prompt, max_tokens, temperature, reasoning_effort],
                     ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CachedLLM:
    """OpenAI-compatible chat completion with a persistent JSONL response cache."""

    def __init__(self, cache_path: Path, base_url: str, model: str, api_key: str,
                 offline: bool = False):
        self.cache_path = Path(cache_path)
        self.base_url, self.model, self.api_key = base_url, model, api_key
        self.offline = offline
        self._client = None
        self._cache: dict = {}
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        if self.cache_path.is_file():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._cache[row["key"]] = row["response"]

    def _client_once(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  timeout=180, max_retries=2)
        return self._client

    def complete(self, prompt: str, *, tag: str, max_tokens: int = 2048,
                 temperature: float = 0.0, reasoning_effort: Optional[str] = "low") -> str:
        key = cache_key(self.model, prompt, max_tokens, temperature, reasoning_effort)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        if self.offline:
            raise RuntimeError(f"LLM cache miss in offline mode (tag={tag})")
        self.misses += 1
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        r = self._client_once().chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, seed=42, extra_body=extra)
        text = (r.choices[0].message.content or "").strip()
        with self._lock:  # builders call this from a thread pool
            self._cache[key] = text
            os.makedirs(self.cache_path.parent, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "tag": tag, "response": text},
                                   ensure_ascii=False) + "\n")
        return text


def parse_json_block(text: str):
    """The first JSON object or array in an LLM reply (tolerates code fences), else None."""
    import re
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None
