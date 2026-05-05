"""A tiny persistent cache for API responses.

LLM screening is expensive because the same prompt may be sent repeatedly while
debugging parsers, prompts, or metrics. This cache stores the exact provider,
model, and prompt hash, so rerunning the same command reuses prior responses.
"""

# Reading guide for R users:
# - This is a small SQLite-backed memoization layer.
# - The pipeline checks the cache before making an API call, which keeps reruns
#   reproducible and cheaper.

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class ResponseCache:
    """SQLite-backed key/value cache for raw model responses."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                response_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def build_key(provider: str, model: str, prompt: str) -> str:
        """Create a stable cache key without storing the prompt as the key."""

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return f"{provider}:{model}:{prompt_hash}"

    def get(self, provider: str, model: str, prompt: str) -> str | None:
        """Return a cached response if this provider/model/prompt was seen."""

        cache_key = self.build_key(provider, model, prompt)
        row = self.connection.execute(
            "SELECT response_text FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def set(self, provider: str, model: str, prompt: str, response_text: str) -> None:
        """Store a raw response for future reruns."""

        cache_key = self.build_key(provider, model, prompt)
        prompt_hash = cache_key.rsplit(":", 1)[-1]
        self.connection.execute(
            """
            INSERT OR REPLACE INTO llm_cache
                (cache_key, provider, model, prompt_hash, response_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cache_key, provider, model, prompt_hash, response_text),
        )
        self.connection.commit()

    def close(self) -> None:
        """Close the SQLite connection."""

        self.connection.close()
