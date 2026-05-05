"""Configuration helpers.

This module is deliberately small and explicit. Most researchers who use R are
used to editing a few variables at the top of a script; this file plays the same
role for the Python command-line tools while still reading secrets from `.env`.
"""

# Reading guide for R users:
# - This file is equivalent to a small block of user-editable parameters at the
#   top of an R script.
# - `load_config()` reads environment variables and returns one `AppConfig`
#   object containing model names, API keys, cache location, and rate limits.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class ProviderConfig:
    """Everything needed to call one model from one API provider."""

    name: str
    model: str
    api_key: str | None
    base_url: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings for the screening pipeline."""

    max_retries: int
    retry_base_seconds: float
    retry_max_seconds: float
    cache_path: Path
    max_in_flight_records: int
    rate_limit_utilization: float
    openai_rpm_limit: int
    anthropic_rpm_limit: int
    gemini_rpm_limit: int
    xai_rpm_limit: int
    anthropic_input_tpm_limit: int
    anthropic_output_tpm_limit: int
    gemini_tpm_limit: int
    xai_tpm_limit: int
    primary_providers: tuple[ProviderConfig, ProviderConfig, ProviderConfig]
    escalation_providers: tuple[ProviderConfig, ProviderConfig, ProviderConfig]


def load_config(env_file: str | Path = ".env") -> AppConfig:
    """Read provider keys, model names, and cache settings from `.env`.

    The function does not require all API keys to be present. Missing keys are
    reported when the pipeline tries to call that provider, which makes it
    possible to prepare and inspect validation data without credentials.
    """

    load_dotenv(env_file)

    max_retries = max(0, int(os.getenv("MAX_RETRIES", "8")))
    retry_base_seconds = float(os.getenv("RETRY_BASE_SECONDS", "1"))
    retry_max_seconds = float(os.getenv("RETRY_MAX_SECONDS", "60"))
    cache_path = Path(os.getenv("CACHE_PATH", ".cache/meta_screen.sqlite3"))
    max_in_flight_records = max(1, int(os.getenv("MAX_IN_FLIGHT_RECORDS", "32")))
    rate_limit_utilization = float(os.getenv("RATE_LIMIT_UTILIZATION", "0.8"))

    openai_rpm_limit = max(1, int(os.getenv("OPENAI_RPM_LIMIT", "1000")))
    anthropic_rpm_limit = max(1, int(os.getenv("ANTHROPIC_RPM_LIMIT", "1000")))
    gemini_rpm_limit = max(1, int(os.getenv("GEMINI_RPM_LIMIT", "1000")))
    xai_rpm_limit = max(1, int(os.getenv("XAI_RPM_LIMIT", "1800")))
    anthropic_input_tpm_limit = max(
        1, int(os.getenv("ANTHROPIC_INPUT_TPM_LIMIT", "450000"))
    )
    anthropic_output_tpm_limit = max(
        1, int(os.getenv("ANTHROPIC_OUTPUT_TPM_LIMIT", "90000"))
    )
    gemini_tpm_limit = max(1, int(os.getenv("GEMINI_TPM_LIMIT", "2000000")))
    xai_tpm_limit = max(1, int(os.getenv("XAI_TPM_LIMIT", "10000000")))

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    primary = (
        ProviderConfig(
            name="openai",
            model=os.getenv("PRIMARY_OPENAI_MODEL", "gpt-5.4-mini"),
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            reasoning_effort=os.getenv("PRIMARY_OPENAI_REASONING", "low"),
        ),
        ProviderConfig(
            name="anthropic",
            model=os.getenv("PRIMARY_ANTHROPIC_MODEL", "claude-haiku-4-5"),
            api_key=anthropic_key,
            base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        ),
        ProviderConfig(
            name="gemini",
            model=os.getenv("PRIMARY_GEMINI_MODEL", "gemini-3-flash-preview"),
            api_key=gemini_key,
            reasoning_effort=os.getenv("PRIMARY_GEMINI_REASONING", "low"),
        ),
    )

    xai_key = os.getenv("XAI_API_KEY")
    escalation = (
        ProviderConfig(
            name="openai",
            model=os.getenv("ESCALATION_OPENAI_MODEL", "gpt-5.4"),
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            reasoning_effort=os.getenv("ESCALATION_OPENAI_REASONING", "medium"),
        ),
        ProviderConfig(
            name="xai",
            model=os.getenv("ESCALATION_XAI_MODEL", "grok-4.20-0309-reasoning"),
            api_key=xai_key,
            base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        ),
        ProviderConfig(
            name="anthropic",
            model=os.getenv("ESCALATION_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            api_key=anthropic_key,
            base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        ),
    )

    return AppConfig(
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        cache_path=cache_path,
        max_in_flight_records=max_in_flight_records,
        rate_limit_utilization=rate_limit_utilization,
        openai_rpm_limit=openai_rpm_limit,
        anthropic_rpm_limit=anthropic_rpm_limit,
        gemini_rpm_limit=gemini_rpm_limit,
        xai_rpm_limit=xai_rpm_limit,
        anthropic_input_tpm_limit=anthropic_input_tpm_limit,
        anthropic_output_tpm_limit=anthropic_output_tpm_limit,
        gemini_tpm_limit=gemini_tpm_limit,
        xai_tpm_limit=xai_tpm_limit,
        primary_providers=primary,
        escalation_providers=escalation,
    )
