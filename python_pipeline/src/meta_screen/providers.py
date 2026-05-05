"""Provider clients for OpenAI-compatible APIs and Gemini.

Only two transport patterns are needed here:

1. OpenAI and xAI are called through a chat-completions style endpoint.
2. Gemini is called through the Google Generative Language REST endpoint.

The caller gives each provider a complete prompt and expects a JSON string back.
Parsing and voting happen elsewhere.
"""

# Reading guide for R users:
# - This file is the network layer only. It does not decide include/exclude.
# - `LLMProvider.complete()` is the main public method: give it one prompt and
#   it returns raw JSON text from one model.
# - `ProviderRateLimiter` is just bookkeeping to avoid sending requests too fast.
# - If you are tracing the pipeline, read this file after `screener.py` asks a
#   provider for a completion.

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import deque
from dataclasses import dataclass

import httpx

from meta_screen.cache import ResponseCache
from meta_screen.config import ProviderConfig


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a usable response."""


@dataclass
class ModelResponse:
    """A raw model response plus metadata needed for audit tables."""

    provider: str
    model: str
    text: str
    from_cache: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProviderRateLimiter:
    """Simple rolling-minute limiter for requests and tokens."""

    def __init__(
        self,
        rpm_limit: int,
        utilization: float,
        input_tpm_limit: int | None = None,
        output_tpm_limit: int | None = None,
    ) -> None:
        self.rpm_cap = max(1, int(rpm_limit * utilization))
        self.input_tpm_cap = (
            max(1, int(input_tpm_limit * utilization))
            if input_tpm_limit is not None
            else None
        )
        self.output_tpm_cap = (
            max(1, int(output_tpm_limit * utilization))
            if output_tpm_limit is not None
            else None
        )
        self._request_times: deque[float] = deque()
        self._input_events: deque[tuple[float, int]] = deque()
        self._output_events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    def _trim(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._input_events and self._input_events[0][0] <= cutoff:
            self._input_events.popleft()
        while self._output_events and self._output_events[0][0] <= cutoff:
            self._output_events.popleft()

    async def acquire(
        self,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
    ) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._trim(now)

                current_input = sum(tokens for _, tokens in self._input_events)
                current_output = sum(tokens for _, tokens in self._output_events)

                requests_ok = len(self._request_times) < self.rpm_cap
                input_ok = (
                    self.input_tpm_cap is None
                    or (current_input + estimated_input_tokens) <= self.input_tpm_cap
                )
                output_ok = (
                    self.output_tpm_cap is None
                    or (current_output + estimated_output_tokens) <= self.output_tpm_cap
                )

                if requests_ok and input_ok and output_ok:
                    self._request_times.append(now)
                    return

                waits: list[float] = []
                if not requests_ok and self._request_times:
                    waits.append(max(0.05, 60.0 - (now - self._request_times[0])))
                if (
                    self.input_tpm_cap is not None
                    and not input_ok
                    and self._input_events
                ):
                    waits.append(max(0.05, 60.0 - (now - self._input_events[0][0])))
                if (
                    self.output_tpm_cap is not None
                    and not output_ok
                    and self._output_events
                ):
                    waits.append(max(0.05, 60.0 - (now - self._output_events[0][0])))

            await asyncio.sleep(min(waits) if waits else 0.25)

    async def record_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        now = time.monotonic()
        async with self._lock:
            self._trim(now)
            if prompt_tokens:
                self._input_events.append((now, int(prompt_tokens)))
            if completion_tokens:
                self._output_events.append((now, int(completion_tokens)))


class LLMProvider:
    """One configured provider/model pair."""

    retryable_status_codes = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        config: ProviderConfig,
        cache: ResponseCache,
        phase: str,
        max_retries: int,
        retry_base_seconds: float,
        retry_max_seconds: float,
        rate_limiter: ProviderRateLimiter | None = None,
        estimated_prompt_tokens: int = 2000,
        estimated_completion_tokens: int = 300,
    ) -> None:
        self.config = config
        self.cache = cache
        self.phase = phase
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.rate_limiter = rate_limiter
        self.estimated_prompt_tokens = max(1, estimated_prompt_tokens)
        self.estimated_completion_tokens = max(1, estimated_completion_tokens)
        self.responses: list[ModelResponse] = []

    async def complete(self, prompt: str) -> ModelResponse:
        """Return raw model text, using the persistent cache when possible."""

        cached = self.cache.get(self.config.name, self.config.model, prompt)
        if cached is not None:
            response = ModelResponse(
                provider=self.config.name,
                model=self.config.model,
                text=cached,
                from_cache=True,
            )
            self.responses.append(response)
            return response

        if not self.config.api_key:
            raise ProviderError(
                f"Missing API key for {self.config.name}. Add it to .env."
            )

        if self.config.name == "gemini":
            text, prompt_tokens, completion_tokens = await self._call_gemini(prompt)
        elif self.config.name == "anthropic":
            text, prompt_tokens, completion_tokens = await self._call_anthropic(prompt)
        else:
            text, prompt_tokens, completion_tokens = await self._call_openai_compatible(prompt)

        self.cache.set(self.config.name, self.config.model, prompt, text)
        response = ModelResponse(
            provider=self.config.name,
            model=self.config.model,
            text=text,
            from_cache=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if self.rate_limiter is not None:
            await self.rate_limiter.record_usage(
                response.prompt_tokens,
                response.completion_tokens,
            )
        self.responses.append(response)
        return response

    async def _call_openai_compatible(
        self, prompt: str
    ) -> tuple[str, int | None, int | None]:
        """Call OpenAI or xAI through a chat-completions compatible API."""

        assert self.config.base_url is not None
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful systematic-review screening "
                        "assistant. Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.config.name == "openai" and self.config.reasoning_effort:
            # xAI reasoning models encode reasoning behavior in the model name
            # and reject this parameter. OpenAI reasoning models accept it.
            payload["reasoning_effort"] = self.config.reasoning_effort

        async with httpx.AsyncClient(timeout=None) as client:
            response = await self._post_json_with_retries(
                client=client,
                provider_label=f"{self.config.name} {self.config.model}",
                url=(
                    f"{self.config.base_url.rstrip('/')}/chat/completions"
                ),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                payload=payload,
            )

        if response.status_code >= 400:
            raise ProviderError(
                f"{self.config.name} {self.config.model} failed: "
                f"{response.status_code} {response.text[:500]}"
            )

        data = response.json()
        text = str(data["choices"][0]["message"]["content"])
        usage = data.get("usage", {})
        return text, usage.get("prompt_tokens"), usage.get("completion_tokens")

    async def _post_json_with_retries(
        self,
        client: httpx.AsyncClient,
        provider_label: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        """POST JSON with exponential backoff for rate limits and outages."""

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                if self.rate_limiter is not None:
                    await self.rate_limiter.acquire(
                        estimated_input_tokens=self.estimated_prompt_tokens,
                        estimated_output_tokens=self.estimated_completion_tokens,
                    )
                response = await client.post(
                    url,
                    params=params,
                    headers=headers,
                    json=payload,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc or repr(exc)}"
                if attempt >= self.max_retries:
                    raise ProviderError(
                        f"{provider_label} failed after {attempt + 1} attempts: "
                        f"{last_error}"
                    ) from exc
                await asyncio.sleep(self._retry_delay(attempt))
                continue

            if response.status_code not in self.retryable_status_codes:
                return response
            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:500] or response.reason_phrase}"
            )
            if attempt >= self.max_retries:
                raise ProviderError(
                    f"{provider_label} failed after {attempt + 1} attempts: "
                    f"{last_error}"
                )

            await asyncio.sleep(self._retry_delay(attempt, response))

        raise ProviderError(f"{provider_label} failed after retries: {last_error}")

    def _retry_delay(
        self,
        attempt: int,
        response: httpx.Response | None = None,
    ) -> float:
        """Return the next retry delay, honoring `Retry-After` when present."""

        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(float(retry_after), self.retry_max_seconds)
                except ValueError:
                    pass

        exponential = self.retry_base_seconds * (2**attempt)
        jitter = random.uniform(0, self.retry_base_seconds)
        return min(exponential + jitter, self.retry_max_seconds)

    async def _call_gemini(
        self, prompt: str
    ) -> tuple[str, int | None, int | None]:
        """Call Gemini through Google's REST API."""

        generation_config: dict[str, object] = {
            "temperature": 0,
            "responseMimeType": "application/json",
        }
        if self.config.reasoning_effort:
            effort_to_budget = {"low": 512, "medium": 2048, "high": 8192}
            budget = effort_to_budget.get(self.config.reasoning_effort, 2048)
            generation_config["thinkingConfig"] = {"thinkingBudget": budget}

        payload: dict[str, object] = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are a careful systematic-review screening "
                            "assistant. Return valid JSON only."
                        )
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
        }

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent"
        )
        async with httpx.AsyncClient(timeout=None) as client:
            response = await self._post_json_with_retries(
                client=client,
                provider_label=f"gemini {self.config.model}",
                url=url,
                params={"key": self.config.api_key or ""},
                headers={"Content-Type": "application/json"},
                payload=payload,
            )

        if response.status_code >= 400:
            raise ProviderError(
                f"gemini {self.config.model} failed: "
                f"{response.status_code} {response.text[:500]}"
            )

        data = response.json()
        try:
            text = str(data["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Gemini response did not contain text: {json.dumps(data)[:500]}"
            ) from exc
        meta = data.get("usageMetadata", {})
        return (
            text,
            meta.get("promptTokenCount"),
            meta.get("candidatesTokenCount"),
        )

    async def _call_anthropic(
        self, prompt: str
    ) -> tuple[str, int | None, int | None]:
        """Call Anthropic Messages API."""

        assert self.config.base_url is not None
        payload: dict[str, object] = {
            "model": self.config.model,
            "max_tokens": 4096,
            "system": (
                "You are a careful systematic-review screening assistant. "
                "Return valid JSON only."
            ),
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=None) as client:
            response = await self._post_json_with_retries(
                client=client,
                provider_label=f"anthropic {self.config.model}",
                url=f"{self.config.base_url.rstrip('/')}/v1/messages",
                headers={
                    "x-api-key": self.config.api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                payload=payload,
            )

        if response.status_code >= 400:
            raise ProviderError(
                f"anthropic {self.config.model} failed: "
                f"{response.status_code} {response.text[:500]}"
            )

        data = response.json()
        blocks = data.get("content", [])
        text_parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        text = "".join(text_parts).strip()
        usage = data.get("usage", {})
        return text, usage.get("input_tokens"), usage.get("output_tokens")
