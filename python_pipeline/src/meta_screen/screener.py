"""L1/L2 screening logic: batch, vote, escalate, and export results."""

# Reading guide for R users:
# - Think of this file as the main "analysis script" for the LLM pipeline.
# - The dataclasses near the top (`Article`, `ParsedDecision`) play the same
#   role as a tidy tibble schema written down explicitly.
# - Helper functions clean records, build prompts, and parse model JSON.
# - The main workflow is:
#     1. read normalized CSV rows into `Article` objects
#     2. send each record through the primary model tier
#     3. escalate uncertain records to the stronger model tier
#     4. take the final vote and write one predictions CSV
# - The command-line entry point is the `main()` function at the bottom.

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from meta_screen.cache import ResponseCache
from meta_screen.config import AppConfig, load_config
from meta_screen.providers import LLMProvider, ModelResponse, ProviderError
from meta_screen.providers import ProviderRateLimiter


VALID_DECISIONS = {"include", "exclude"}
PICO_DOMAINS = ("population", "intervention", "comparator", "outcomes", "study_design")
PICO_STATUSES = {"include", "exclude", "unclear"}
MIN_USABLE_FULL_TEXT_CHARS = 3000
MIN_USABLE_L1_ABSTRACT_CHARS = 20
L1_ABSTRACT_PLACEHOLDERS = {
    "n a",
    "na",
    "no abstract",
    "no abstract available",
    "not available",
}
L1_SECONDARY_REVIEW_PATTERNS = (
    "systematic review",
    "meta-analysis",
    "meta analysis",
)
FIBER_SOURCE_REVIEW = (
    "Safety of using enteral nutrition formulations containing "
    "dietary fiber in hospitalized critical care patients"
)
EXCLUDED_DUPLICATE_RECORD_IDS = {"40"}


@dataclass(frozen=True)
class Article:
    """The fields sent to an LLM for article screening.

    L1 uses title and abstract. L2 uses raw text extracted from the full-text PDF
    when the `full_text` column is available.
    """

    record_id: str
    title: str
    abstract: str
    year: str = ""
    journal: str = ""
    doi: str = ""
    full_text: str = ""
    source_review: str = ""
    full_text_char_count: int | None = None


@dataclass(frozen=True)
class ParsedDecision:
    """One model's decision for one article."""

    record_id: str
    decision: str
    rationale: str
    confidence: float | None
    provider: str
    model: str
    from_cache: bool
    pico_check: dict[str, str] = field(default_factory=dict)


def _clean_cell(value: object) -> str:
    """Return an empty string for missing dataframe cells, otherwise stripped text."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _optional_int(value: object) -> int | None:
    """Convert a dataframe value to int when possible."""

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def records_from_frame(frame: pd.DataFrame) -> list[Article]:
    """Convert a normalized CSV into Article objects."""

    articles: list[Article] = []
    for index, row in frame.iterrows():
        record_id = str(row.get("record_index", index))
        full_text = _clean_cell(row.get("full_text", ""))
        articles.append(
            Article(
                record_id=record_id,
                title=_clean_cell(row.get("title", "")),
                abstract=_clean_cell(row.get("abstract", "")),
                year=_clean_cell(row.get("year", "")),
                journal=_clean_cell(row.get("journal", "")),
                doi=_clean_cell(row.get("doi", "")),
                full_text=full_text,
                source_review=_clean_cell(row.get("source_review", "")),
                full_text_char_count=_optional_int(row.get("full_text_char_count")),
            )
        )
    return articles


def has_usable_full_text(
    article: Article,
    min_chars: int = MIN_USABLE_FULL_TEXT_CHARS,
) -> bool:
    """Return whether an L2 record has enough extracted full text to screen."""

    text = article.full_text.strip()
    if not text:
        return False
    char_count = article.full_text_char_count
    if char_count is None:
        char_count = len(text)
    return char_count >= min_chars


def has_usable_l1_abstract(
    article: Article,
    min_chars: int = MIN_USABLE_L1_ABSTRACT_CHARS,
) -> bool:
    """Return whether an L1 record has enough abstract text to screen."""

    text = article.abstract.strip()
    if not text:
        return False

    normalized = re.sub(r"[\[\]\.\s]+", " ", text).strip().lower()
    if normalized in L1_ABSTRACT_PLACEHOLDERS:
        return False

    alnum_count = sum(char.isalnum() for char in text)
    return alnum_count >= min_chars


def is_l1_excluded_secondary_review(article: Article) -> bool:
    """Return whether an L1 record is a systematic review or meta-analysis."""

    text = f"{article.title} {article.abstract}".lower()
    return any(pattern in text for pattern in L1_SECONDARY_REVIEW_PATTERNS)


def is_excluded_duplicate_record(article: Article) -> bool:
    """Return whether a record is a known duplicate that should be dropped."""

    return (
        article.source_review == FIBER_SOURCE_REVIEW
        and article.record_id in EXCLUDED_DUPLICATE_RECORD_IDS
    )


def _review_objective(articles: list[Article]) -> str:
    """Use the source review title when the dataset provides one."""

    for article in articles:
        if article.source_review:
            return article.source_review
    return "Apply the user-supplied eligibility criteria below."


def _record_rows(articles: list[Article], stage: str) -> list[dict[str, object]]:
    """Build compact JSON rows for the prompt."""

    rows: list[dict[str, object]] = []
    for article in articles:
        row: dict[str, object] = {
            "record_id": article.record_id,
            "title": article.title,
            "abstract": article.abstract,
            "year": article.year,
            "journal": article.journal,
            "doi": article.doi,
        }
        if stage == "l2":
            row["full_text"] = article.full_text
        rows.append(row)
    return rows


def _stage_policy(stage: str) -> str:
    """Return stage-specific screening policy text."""

    if stage == "l2":
        return (
            "Stage: Level 2 full-text final inclusion.\n"
            "- Use full_text as the primary evidence source; use title and abstract only as context.\n"
            "- This stage is strict: include only when the full text supports every required criterion.\n"
            "- If any required domain is exclude or remains unclear after reading the full text, return exclude.\n"
            "- Review articles, protocols, editorials, letters, conference-only records, and case reports are excluded when the criteria require primary clinical studies.\n"
            "- Records with missing or functionally missing full_text should already be skipped before prompting; if one appears here, return exclude.\n"
        )
    return (
        "Stage: Level 1 title/abstract screening.\n"
        "- Use the reviewer-supplied criteria as the source of truth. If the criteria text gives stage-specific guidance, follow it over any default systematic-review heuristic.\n"
        "- L1 is intentionally sensitive because a false negative permanently removes evidence from the review.\n"
        "- Include when the title/abstract leaves genuine uncertainty about whether a required criterion is met.\n"
        "- Do not let the inclusive L1 rule override a clear exclusion: if the population, intervention/exposure, outcome, or study design is clearly outside scope, return exclude.\n"
        "- Do not infer an eligible intervention/exposure from topic-adjacent language alone unless the title/abstract explicitly supports an intervention or exposure allowed by the criteria.\n"
        "- Do not infer an eligible population, setting, or disease state from nearby but non-equivalent language unless the title/abstract explicitly supports it or the criteria explicitly allow that broader interpretation.\n"
        "- Still exclude records whose title/abstract clearly identifies an excluded design under the criteria, such as protocols, editorials, letters, conference-only records without data, animal/in vitro studies, or case reports when case reports are excluded.\n"
        "- Never exclude at L1 because of suspected poor quality; quality appraisal happens later.\n"
    )


def _criterion_instructions(stage: str) -> str:
    """Return structured PICO-style decision instructions."""

    if stage == "l2":
        decision_rule = (
            "Decision rule for L2: decision must be include only if every required "
            "pico_check domain is include. If any required domain is exclude or unclear, "
            "decision must be exclude."
        )
    else:
        decision_rule = (
            "Decision rule for L1: decision must be exclude only when at least one "
            "required pico_check domain is clearly exclude. If all required domains are "
            "include, or if one or more required domains are unclear but none are clearly "
            "exclude, decision must be include."
        )

    return (
        "# Instructions\n"
        "Evaluate each record against the eligibility criteria domain by domain. "
        "Use these PICO-style domains even when the criteria text is written informally:\n"
        "1. population\n"
        "2. intervention\n"
        "3. comparator\n"
        "4. outcomes\n"
        "5. study_design\n\n"
        "For each domain, assign exactly one status: include, exclude, or unclear. "
        "Use include for domains with no restriction in the criteria, such as 'Any comparator'. "
        "Use unclear when the available text does not settle the domain at this screening stage. "
        "Use the reviewer-supplied criteria rather than generic assumptions about what systematic reviews usually include. "
        "Do not treat topic adjacency as evidence: related topic language is not enough by itself to satisfy a restricted intervention, population, comparator, outcome, or design requirement.\n"
        "Then apply the stage-specific decision rule exactly.\n\n"
        f"{decision_rule}\n\n"
        "Keep the rationale short and tied to the criterion that determines the decision. "
        "Do not add fields outside the JSON schema.\n"
    )


def build_prompt(criteria: str, articles: list[Article], stage: str = "l1") -> str:
    """Build the JSON-only screening prompt for one batch."""

    rows = _record_rows(articles, stage)
    review_objective = _review_objective(articles)
    policy = _stage_policy(stage)
    instructions = _criterion_instructions(stage)

    n = len(articles)
    required_ids = ", ".join(f'"{a.record_id}"' for a in articles)
    records_json = json.dumps(rows, ensure_ascii=False)
    criteria_block = (
        "Systematic/scoping review objective:\n"
        f"{review_objective}\n\n"
        "Eligibility criteria supplied by the reviewer:\n"
        f"{criteria.strip()}\n\n"
        f"{policy}\n"
        f"{instructions}"
    )
    output_schema = (
        f"REQUIRED: return exactly {n} decision(s), one per record_id.\n"
        f"Required record_ids (do not omit any): [{required_ids}]\n\n"
        "Return one JSON object whose decisions array contains exactly the required records:\n"
        "{\n"
        '  "decisions": [\n'
        "    {\n"
        '      "record_id": "same id as input",\n'
        '      "pico_check": {\n'
        '        "population": "include|exclude|unclear",\n'
        '        "intervention": "include|exclude|unclear",\n'
        '        "comparator": "include|exclude|unclear",\n'
        '        "outcomes": "include|exclude|unclear",\n'
        '        "study_design": "include|exclude|unclear"\n'
        "      },\n"
        '      "decision": "include|exclude",\n'
        '      "confidence": 0.0,\n'
        '      "rationale": "one short reason tied to the decisive criterion"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )
    final_reminder = (
        "Final reminder before deciding:\n"
        "- Use the reviewer-supplied criteria above as the source of truth.\n"
        "- Base the decision only on the provided record text.\n"
    )

    if stage == "l2":
        # ISO-style structure: place criteria/instructions before and after the
        # long full text so models do not lose the screening task in context.
        return (
            f"{criteria_block}\n"
            "Records to screen, as JSON:\n"
            f"{records_json}\n\n"
            "Repeat of eligibility criteria and instructions for final decision:\n"
            f"{criteria_block}\n"
            f"{final_reminder}"
            f"{output_schema}"
        )

    return (
        f"{criteria_block}\n"
        "Records to screen, as JSON:\n"
        f"{records_json}\n\n"
        f"{final_reminder}"
        f"{output_schema}"
    )


def _normalize_pico_check(value: object) -> dict[str, str]:
    """Normalize model PICO status output into domain -> status."""

    if not isinstance(value, dict):
        return {}

    normalized: dict[str, str] = {}
    for domain in PICO_DOMAINS:
        raw_status = value.get(domain, "")
        if isinstance(raw_status, dict):
            raw_status = raw_status.get("status", "")
        status = str(raw_status).strip().lower()
        if status in PICO_STATUSES:
            normalized[domain] = status
    return normalized


def parse_decisions(
    response: ModelResponse,
    expected_ids: set[str],
) -> list[ParsedDecision]:
    """Parse and validate the JSON returned by a model."""

    try:
        data = _parse_json_object(response.text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{response.provider} {response.model} returned invalid JSON."
        ) from exc

    if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
        raise ValueError(
            f"{response.provider} {response.model} JSON lacks a decisions list."
        )

    parsed: list[ParsedDecision] = []
    for item in data["decisions"]:
        if not isinstance(item, dict):
            continue
        record_id = str(item.get("record_id", ""))
        if record_id not in expected_ids:
            continue
        decision = str(item.get("decision", "")).strip().lower()
        if decision not in VALID_DECISIONS:
            continue
        confidence = item.get("confidence")
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        parsed.append(
            ParsedDecision(
                record_id=record_id,
                decision=decision,
                rationale=str(item.get("rationale", ""))[:1000],
                confidence=confidence_value,
                provider=response.provider,
                model=response.model,
                from_cache=response.from_cache,
                pico_check=_normalize_pico_check(item.get("pico_check")),
            )
        )

    seen_ids = {item.record_id for item in parsed}
    missing = expected_ids - seen_ids
    if missing:
        raise ValueError(
            f"{response.provider} {response.model} omitted decisions for: "
            f"{sorted(missing)}"
        )
    return parsed


def _parse_json_object(raw_text: str) -> dict:
    """Parse provider output into a JSON object, tolerating markdown wrappers."""

    text = raw_text.strip()
    if not text:
        raise json.JSONDecodeError("empty response", text, 0)

    # Fast path for strict JSON output.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Common provider behavior: wrap JSON inside fenced markdown blocks.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data

    # Fallback: extract the first balanced JSON object in the response.
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("no JSON object found", text, 0)

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
                break

    raise json.JSONDecodeError("could not parse JSON object", text, start)


async def call_provider(
    provider: LLMProvider,
    criteria: str,
    articles: list[Article],
    stage: str,
) -> list[ParsedDecision]:
    """Call one model on one batch and parse the result."""

    prompt = build_prompt(criteria, articles, stage=stage)
    response = await provider.complete(prompt)
    expected_ids = {article.record_id for article in articles}
    return parse_decisions(response, expected_ids)


def unanimous_or_escalate(decisions: list[ParsedDecision]) -> tuple[str | None, bool]:
    """Return a final include only when all three primary models unanimously agree to include.

    At L1 a false negative (lost article) is unrecoverable, so the asymmetric rule is:
      - Unanimous include (3/3) → include, no escalation needed.
      - Everything else (split OR unanimous exclude) → escalate to reasoning models.
    This means an article can only be excluded after the reasoning-model tier confirms it.
    """

    labels = [item.decision for item in decisions]
    if len(labels) == 3 and labels.count("include") == 3:
        return "include", False
    return None, True


def majority(decisions: list[ParsedDecision]) -> str | None:
    """Return the majority label, or None if too few valid model responses."""

    include_votes = sum(item.decision == "include" for item in decisions)
    exclude_votes = sum(item.decision == "exclude" for item in decisions)
    if include_votes > exclude_votes:
        return "include"
    if exclude_votes > include_votes:
        return "exclude"
    return None


def decisions_to_columns(prefix: str, decisions: list[ParsedDecision]) -> dict[str, object]:
    """Flatten provider decisions into CSV-friendly columns."""

    row: dict[str, object] = {}
    for item in decisions:
        provider_key = f"{prefix}_{item.provider}"
        row[f"{provider_key}_model"] = item.model
        row[f"{provider_key}_decision"] = item.decision
        row[f"{provider_key}_confidence"] = item.confidence
        if item.pico_check:
            row[f"{provider_key}_pico_check"] = json.dumps(
                item.pico_check,
                sort_keys=True,
            )
            for domain in PICO_DOMAINS:
                row[f"{provider_key}_{domain}"] = item.pico_check.get(domain, "")
        row[f"{provider_key}_rationale"] = item.rationale
        row[f"{provider_key}_from_cache"] = item.from_cache
    return row


async def _escalate_one_article(
    article: Article,
    criteria: str,
    escalation_providers: list[LLMProvider],
    stage: str,
) -> tuple[str, list[ParsedDecision], list[str]]:
    """Run the three escalation (reasoning) models for one split record."""

    print(
        f"  [escalate {article.record_id}] Sending to {len(escalation_providers)} reasoning models",
        flush=True,
    )
    calls = [
        call_provider(provider, criteria, [article], stage)
        for provider in escalation_providers
    ]
    outputs = await asyncio.gather(*calls, return_exceptions=True)

    decisions: list[ParsedDecision] = []
    errors: list[str] = []
    for provider, output in zip(escalation_providers, outputs):
        label = f"{provider.config.name}/{provider.config.model}"
        if isinstance(output, Exception):
            errors.append(str(output))
            print(f"  [escalate {article.record_id}] {label}: FAILED — {output}", flush=True)
        else:
            decisions.extend(output)
            votes = [d.decision for d in output]
            print(f"  [escalate {article.record_id}] {label}: {votes}", flush=True)

    return article.record_id, decisions, errors


async def _process_one_batch(
    batch_number: int,
    total_batches: int,
    batch_articles: list[Article],
    criteria: str,
    primary_providers: list[LLMProvider],
    escalation_providers: list[LLMProvider],
    stage: str,
    in_flight: asyncio.Semaphore | None = None,
) -> list[dict[str, object]]:
    """Process one batch: primary votes, then escalation for any splits."""

    if in_flight is not None:
        async with in_flight:
            return await _process_one_batch(
                batch_number=batch_number,
                total_batches=total_batches,
                batch_articles=batch_articles,
                criteria=criteria,
                primary_providers=primary_providers,
                escalation_providers=escalation_providers,
                stage=stage,
                in_flight=None,
            )

    provider_names = ", ".join(p.config.name for p in primary_providers)
    print(
        f"[batch {batch_number}/{total_batches}] Starting: {len(batch_articles)} records "
        f"({provider_names})",
        flush=True,
    )

    primary_calls = [
        call_provider(provider, criteria, batch_articles, stage)
        for provider in primary_providers
    ]
    primary_outputs = await asyncio.gather(*primary_calls, return_exceptions=True)

    primary_by_id: dict[str, list[ParsedDecision]] = {
        article.record_id: [] for article in batch_articles
    }
    primary_errors: list[str] = []

    for provider, output in zip(primary_providers, primary_outputs):
        label = f"{provider.config.name}/{provider.config.model}"
        if isinstance(output, Exception):
            primary_errors.append(str(output))
            print(f"[batch {batch_number}] {label}: FAILED — {output}", flush=True)
        else:
            cache_hits = sum(d.from_cache for d in output)
            print(
                f"[batch {batch_number}] {label}: OK "
                f"({len(output)} decisions, {cache_hits} cached)",
                flush=True,
            )
            for decision in output:
                primary_by_id[decision.record_id].append(decision)

    batch_rows: dict[str, dict[str, object]] = {}
    escalation_articles: list[Article] = []

    for article in batch_articles:
        row: dict[str, object] = {
            "record_id": article.record_id,
            "title": article.title,
            "abstract": article.abstract,
            "primary_errors": " | ".join(primary_errors),
        }
        primary_decisions = primary_by_id[article.record_id]
        row.update(decisions_to_columns("primary", primary_decisions))

        final_decision, needs_escalation = unanimous_or_escalate(primary_decisions)
        row["escalated"] = needs_escalation

        if needs_escalation:
            escalation_articles.append(article)
        else:
            row["escalation_errors"] = ""
            row["final_decision"] = final_decision or "unresolved"

        batch_rows[article.record_id] = row

    n_decided = len(batch_articles) - len(escalation_articles)
    print(
        f"[batch {batch_number}/{total_batches}] Primary done: "
        f"{n_decided} unanimous, {len(escalation_articles)} escalating",
        flush=True,
    )

    if escalation_articles:
        escl_ids = [a.record_id for a in escalation_articles]
        print(f"[batch {batch_number}] Escalating record IDs: {escl_ids}", flush=True)

        escl_tasks = [
            _escalate_one_article(article, criteria, escalation_providers, stage)
            for article in escalation_articles
        ]
        escl_results = await asyncio.gather(*escl_tasks, return_exceptions=True)

        for article, result in zip(escalation_articles, escl_results):
            row = batch_rows[article.record_id]
            if isinstance(result, Exception):
                row["escalation_errors"] = str(result)
                row["final_decision"] = "unresolved"
                print(f"  [escalate {article.record_id}] FAILED: {result}", flush=True)
            else:
                record_id, decisions, errors = result
                row["escalation_errors"] = " | ".join(errors)
                row.update(decisions_to_columns("escalation", decisions))
                row["final_decision"] = majority(decisions) or "unresolved"
                inc = sum(d.decision == "include" for d in decisions)
                exc = sum(d.decision == "exclude" for d in decisions)
                print(
                    f"  [escalate {record_id}] Final: {row['final_decision']} "
                    f"({inc} include / {exc} exclude)",
                    flush=True,
                )

    n_include = sum(1 for r in batch_rows.values() if r.get("final_decision") == "include")
    n_exclude = sum(1 for r in batch_rows.values() if r.get("final_decision") == "exclude")
    n_unresolved = sum(1 for r in batch_rows.values() if r.get("final_decision") == "unresolved")
    print(
        f"[batch {batch_number}/{total_batches}] Done: "
        f"{n_include} include, {n_exclude} exclude, {n_unresolved} unresolved",
        flush=True,
    )

    return [batch_rows[article.record_id] for article in batch_articles]


def _build_cost_report(
    all_providers: list[LLMProvider],
    records: int,
    batches: int,
    escalated: int,
    stage: str,
    run_id: str,
    skipped_records: int = 0,
) -> dict:
    """Aggregate per-provider token usage into a cost report dict."""

    by_key: dict[str, dict] = {}
    for provider in all_providers:
        key = f"{provider.phase}/{provider.config.name}/{provider.config.model}"
        if key not in by_key:
            by_key[key] = {
                "phase": provider.phase,
                "provider": provider.config.name,
                "model": provider.config.model,
                "api_calls": 0,
                "cached_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        for r in provider.responses:
            if r.from_cache:
                by_key[key]["cached_calls"] += 1
            else:
                by_key[key]["api_calls"] += 1
                by_key[key]["prompt_tokens"] += r.prompt_tokens or 0
                by_key[key]["completion_tokens"] += r.completion_tokens or 0

    provider_rows = list(by_key.values())
    total_api = sum(v["api_calls"] for v in provider_rows)
    total_cached = sum(v["cached_calls"] for v in provider_rows)
    total_pt = sum(v["prompt_tokens"] for v in provider_rows)
    total_ct = sum(v["completion_tokens"] for v in provider_rows)

    return {
        "run_id": run_id,
        "stage": stage,
        "input_records": records + skipped_records,
        "records": records,
        "skipped_records": skipped_records,
        "batches": batches,
        "escalated": escalated,
        "providers": provider_rows,
        "totals": {
            "api_calls": total_api,
            "cached_calls": total_cached,
            "prompt_tokens": total_pt,
            "completion_tokens": total_ct,
            "total_tokens": total_pt + total_ct,
        },
    }


async def screen_articles(
    articles: list[Article],
    criteria: str,
    config: AppConfig,
    stage: str = "l1",
) -> tuple[pd.DataFrame, dict]:
    """Run the full screening cascade.

    Returns a (predictions DataFrame, cost report dict) tuple.
    """

    skipped_records = 0
    if stage == "l1":
        original_count = len(articles)
        articles = [
            article
            for article in articles
            if has_usable_l1_abstract(article)
            and not is_l1_excluded_secondary_review(article)
            and not is_excluded_duplicate_record(article)
        ]
        skipped_records = original_count - len(articles)
        if skipped_records:
            print(
                f"\nSkipping {skipped_records} L1 record(s) "
                f"because the abstract is missing/functionally missing or the "
                f"record is a systematic review/meta-analysis.\n",
                flush=True,
            )
    if stage == "l2":
        original_count = len(articles)
        articles = [
            article
            for article in articles
            if has_usable_full_text(article)
            and not is_excluded_duplicate_record(article)
        ]
        skipped_records = original_count - len(articles)
        if skipped_records:
            print(
                f"\nSkipping {skipped_records} L2 record(s) with missing or "
                f"functionally missing full_text "
                f"(<{MIN_USABLE_FULL_TEXT_CHARS} extracted characters).\n",
                flush=True,
            )

    cache = ResponseCache(config.cache_path)
    provider_limiters = {
        "openai": ProviderRateLimiter(
            rpm_limit=config.openai_rpm_limit,
            utilization=config.rate_limit_utilization,
        ),
        "anthropic": ProviderRateLimiter(
            rpm_limit=config.anthropic_rpm_limit,
            utilization=config.rate_limit_utilization,
            input_tpm_limit=config.anthropic_input_tpm_limit,
            output_tpm_limit=config.anthropic_output_tpm_limit,
        ),
        "gemini": ProviderRateLimiter(
            rpm_limit=config.gemini_rpm_limit,
            utilization=config.rate_limit_utilization,
            input_tpm_limit=config.gemini_tpm_limit,
        ),
        "xai": ProviderRateLimiter(
            rpm_limit=config.xai_rpm_limit,
            utilization=config.rate_limit_utilization,
            input_tpm_limit=config.xai_tpm_limit,
        ),
    }

    def _estimate_prompt_tokens(provider_name: str, stage_name: str) -> int:
        # L2 prompts include full extracted PDF text; keep estimate conservative.
        if stage_name == "l2":
            return 14000 if provider_name in {"anthropic", "gemini", "xai"} else 12000
        return 2200

    def _estimate_completion_tokens(stage_name: str) -> int:
        return 350 if stage_name == "l2" else 450

    def _make_providers(provider_configs: tuple, phase: str) -> list[LLMProvider]:
        return [
            LLMProvider(
                item,
                cache,
                phase,
                config.max_retries,
                config.retry_base_seconds,
                config.retry_max_seconds,
                rate_limiter=provider_limiters[item.name],
                estimated_prompt_tokens=_estimate_prompt_tokens(item.name, stage),
                estimated_completion_tokens=_estimate_completion_tokens(stage),
            )
            for item in provider_configs
        ]

    primary_providers = _make_providers(config.primary_providers, "primary")
    escalation_providers = _make_providers(config.escalation_providers, "escalation")

    # Batch processing is intentionally disabled. We always screen one article
    # per primary-model call to simplify behavior and auditability.
    batch_size = 1
    batches = [[article] for article in articles]
    total_batches = len(batches)

    print(
        f"\nScreening {len(articles)} articles in {total_batches} batches "
        f"(batch_size={batch_size}, stage={stage}, max_in_flight={config.max_in_flight_records}, "
        f"rate_utilization={config.rate_limit_utilization})\n",
        flush=True,
    )

    results: list[dict[str, object]] = []
    try:
        in_flight = asyncio.Semaphore(config.max_in_flight_records)
        batch_tasks = [
            _process_one_batch(
                batch_num,
                total_batches,
                batch_arts,
                criteria,
                primary_providers,
                escalation_providers,
                stage,
                in_flight=in_flight,
            )
            for batch_num, batch_arts in enumerate(batches, start=1)
        ]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for batch_num, (batch_arts, result) in enumerate(
            zip(batches, batch_results), start=1
        ):
            if isinstance(result, Exception):
                print(f"[batch {batch_num}] FAILED with exception: {result}", flush=True)
                for article in batch_arts:
                    results.append(
                        {
                            "record_id": article.record_id,
                            "title": article.title,
                            "abstract": article.abstract,
                            "final_decision": "unresolved",
                            "primary_errors": str(result),
                            "escalated": False,
                            "escalation_errors": "",
                        }
                    )
            else:
                results.extend(result)
    finally:
        cache.close()

    n_include = sum(1 for r in results if r.get("final_decision") == "include")
    n_exclude = sum(1 for r in results if r.get("final_decision") == "exclude")
    n_unresolved = sum(1 for r in results if r.get("final_decision") == "unresolved")
    n_escalated = sum(1 for r in results if r.get("escalated"))
    print(
        f"\nScreening complete: {len(results)} records — "
        f"{n_include} include, {n_exclude} exclude, {n_unresolved} unresolved "
        f"({n_escalated} escalated to reasoning models)\n",
        flush=True,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cost = _build_cost_report(
        all_providers=primary_providers + escalation_providers,
        records=len(articles),
        batches=total_batches,
        escalated=n_escalated,
        stage=stage,
        run_id=run_id,
        skipped_records=skipped_records,
    )
    return pd.DataFrame(results), cost


def read_criteria(path: str | Path) -> str:
    """Read inclusion/exclusion criteria from a plain text file."""

    return Path(path).read_text(encoding="utf-8")


def default_input_for_stage(stage: str) -> Path:
    """Return the default input CSV for a screening stage."""

    if stage == "l2":
        return Path("data/processed/l2/cara_2021_fiber.csv")
    return Path("data/processed/l1/cara_2021_fiber.csv")


def default_criteria_path() -> Path:
    """Return the default criteria file path."""

    return Path("data/criteria/cara_2021_fiber.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI screening.")
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "CSV with title and abstract columns. If omitted, defaults to "
            "Fiber L1 or Fiber L2 dataset based on --stage."
        ),
    )
    parser.add_argument(
        "--criteria",
        default=None,
        help=(
            "Plain-text screening criteria. If omitted, defaults to "
            "data/criteria/cara_2021_fiber.txt."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Base path for the run directory. Each run creates "
            "<output>_<YYYYMMDD_HHMMSS>/ containing predictions.csv and cost.json."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["l1", "l2"],
        default="l1",
        help="Screening stage. L1 uses title/abstract; L2 uses raw extracted PDF text.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing.")
    parser.add_argument("--env-file", default=".env", help="Path to dotenv file.")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else default_input_for_stage(args.stage)
    criteria_path = Path(args.criteria) if args.criteria else default_criteria_path()

    frame = pd.read_csv(input_path)
    if args.limit is not None:
        frame = frame.head(args.limit)

    config = load_config(args.env_file)
    criteria = read_criteria(criteria_path)
    articles = records_from_frame(frame)

    predictions, cost = asyncio.run(
        screen_articles(articles, criteria, config, stage=args.stage)
    )

    run_dir = Path(f"{args.output}_{cost['run_id']}")
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    cost_path = run_dir / "cost.json"
    cost_path.write_text(json.dumps(cost, indent=2))

    total_tokens = cost["totals"]["total_tokens"]
    api_calls = cost["totals"]["api_calls"]
    cached_calls = cost["totals"]["cached_calls"]
    print(
        f"Run folder:  {run_dir}\n"
        f"Predictions: {predictions_path} ({len(predictions)} rows)\n"
        f"Cost:        {cost_path}\n"
        f"             {api_calls} API calls, {cached_calls} cached, "
        f"{total_tokens:,} total tokens"
    )


if __name__ == "__main__":
    main()
