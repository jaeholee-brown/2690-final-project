"""Simple reproducible baselines for L1/L2 screening experiments."""

# Reading guide for R users:
# - This file contains the non-LLM comparison rules used in the paper.
# - Each baseline is a small deterministic function that takes one record and
#   returns an include/exclude decision plus some diagnostic flags.
# - Conceptually, this is like a hand-written rule-based classifier in R.

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from meta_screen.metrics import confusion_counts, filter_validation_frame, rates


ICU_PATTERN = re.compile(r"\b(icu|intensive care|critical(?:ly)? ill|critical care)\b")
FIBER_PATTERN = re.compile(
    r"\b(fiber|fibre|guar|pectin|prebiotic|probiotic|synbiotic|banana flakes|polysaccharide)\b"
)
ENTERAL_PATTERN = re.compile(
    r"\b(enteral|tube fed|tube-fed|feeding|formula|nutrition)\b"
)
COMPARATOR_PATTERN = re.compile(
    r"\b(compared|comparison|versus|vs\.?|control|placebo|randomi[sz]ed|trial)\b"
)
PUBLICATION_EXCLUSION_PATTERN = re.compile(
    r"\b(meta-analysis|meta analysis|systematic review|protocol|conference|abstract)\b"
)


@dataclass(frozen=True)
class BaselineResult:
    """One baseline decision for one record."""

    baseline_name: str
    record_id: str
    score: float
    final_decision: str
    matched_icu: bool
    matched_fiber: bool
    matched_enteral: bool
    matched_comparator: bool
    matched_publication_exclusion: bool


def _record_text(row: pd.Series, stage: str) -> str:
    parts = [str(row.get("title", "")), str(row.get("abstract", ""))]
    if stage == "l2":
        parts.append(str(row.get("full_text", "")))
    return " ".join(parts).lower()


def _component_hits(text: str) -> dict[str, bool]:
    return {
        "matched_icu": bool(ICU_PATTERN.search(text)),
        "matched_fiber": bool(FIBER_PATTERN.search(text)),
        "matched_enteral": bool(ENTERAL_PATTERN.search(text)),
        "matched_comparator": bool(COMPARATOR_PATTERN.search(text)),
        "matched_publication_exclusion": bool(
            PUBLICATION_EXCLUSION_PATTERN.search(text)
        ),
    }


def _always_exclude(record_id: str, hits: dict[str, bool]) -> BaselineResult:
    return BaselineResult(
        baseline_name="always_exclude",
        record_id=record_id,
        score=0.0,
        final_decision="exclude",
        **hits,
    )


def _keyword_sensitive(record_id: str, hits: dict[str, bool]) -> BaselineResult:
    include = hits["matched_icu"] and (
        hits["matched_fiber"] or hits["matched_enteral"]
    )
    matched_count = (
        int(hits["matched_icu"])
        + int(hits["matched_fiber"])
        + int(hits["matched_enteral"])
    )
    return BaselineResult(
        baseline_name="keyword_sensitive",
        record_id=record_id,
        score=matched_count / 3.0,
        final_decision="include" if include else "exclude",
        **hits,
    )


def _keyword_strict(record_id: str, hits: dict[str, bool]) -> BaselineResult:
    include = (
        hits["matched_icu"]
        and hits["matched_fiber"]
        and hits["matched_enteral"]
        and hits["matched_comparator"]
        and not hits["matched_publication_exclusion"]
    )
    matched_count = (
        int(hits["matched_icu"])
        + int(hits["matched_fiber"])
        + int(hits["matched_enteral"])
        + int(hits["matched_comparator"])
    )
    return BaselineResult(
        baseline_name="keyword_strict",
        record_id=record_id,
        score=matched_count / 4.0,
        final_decision="include" if include else "exclude",
        **hits,
    )


BASELINES = {
    "always_exclude": _always_exclude,
    "keyword_sensitive": _keyword_sensitive,
    "keyword_strict": _keyword_strict,
}


def run_baselines(
    validation_csv: str | Path,
    stage: str,
    baseline_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-record baseline predictions and summary metrics."""

    validation = pd.read_csv(validation_csv)
    filtered = filter_validation_frame(validation, stage)
    gold_column = "gold_l2_label" if stage == "l2" else "gold_l1_label"

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for baseline_name in baseline_names:
        baseline = BASELINES[baseline_name]
        results: list[BaselineResult] = []
        for _, row in filtered.iterrows():
            text = _record_text(row, stage)
            hits = _component_hits(text)
            results.append(baseline(str(row["record_index"]), hits))

        predictions = pd.DataFrame([result.__dict__ for result in results])
        gold = filtered[gold_column].astype(bool).reset_index(drop=True)
        predicted = predictions["final_decision"].eq("include")
        counts = confusion_counts(gold, predicted)
        metric_rows.append(
            {
                "baseline_name": baseline_name,
                **counts,
                **rates(counts),
                "n": len(predictions),
                "stage": stage,
            }
        )

        detail = filtered[
            [
                "record_index",
                "title",
                "abstract",
                gold_column,
                "human_decision",
            ]
        ].copy()
        if stage == "l2":
            detail["full_text_char_count"] = filtered["full_text_char_count"].values
        detail = detail.reset_index(drop=True)
        detail["record_id"] = predictions["record_id"]
        detail["baseline_name"] = baseline_name
        for column in predictions.columns:
            if column not in {"record_id", "baseline_name"}:
                detail[column] = predictions[column]
        prediction_rows.append(detail)

    all_predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values("baseline_name").reset_index(
        drop=True
    )
    return all_predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple screening baselines.")
    parser.add_argument("--validation", required=True, help="Validation CSV.")
    parser.add_argument("--stage", choices=["l1", "l2"], required=True)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(BASELINES),
        choices=sorted(BASELINES),
        help="Which baselines to run.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for predictions.csv and metrics.csv.",
    )
    args = parser.parse_args()

    predictions, metrics = run_baselines(args.validation, args.stage, args.baselines)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
