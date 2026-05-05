"""Metrics for comparing AI screening decisions against human labels."""

# Reading guide for R users:
# - This file corresponds to the "evaluation" part of the project.
# - It merges predictions with the gold labels, applies the benchmark filters,
#   and computes confusion-matrix summaries such as sensitivity and specificity.
# - The paper-analysis scripts import these helpers rather than reimplementing
#   the same calculations in several places.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from meta_screen.screener import (
    has_usable_l1_abstract,
    is_excluded_duplicate_record,
    is_l1_excluded_secondary_review,
    records_from_frame,
)


def confusion_counts(gold: pd.Series, predicted: pd.Series) -> dict[str, int]:
    """Compute binary confusion-matrix counts.

    In this project, True means "include / relevant / advance to next stage" and
    False means "exclude / irrelevant."
    """

    gold_bool = gold.astype(bool)
    pred_bool = predicted.astype(bool)
    return {
        "true_positive": int((gold_bool & pred_bool).sum()),
        "false_positive": int((~gold_bool & pred_bool).sum()),
        "true_negative": int((~gold_bool & ~pred_bool).sum()),
        "false_negative": int((gold_bool & ~pred_bool).sum()),
    }


def rates(counts: dict[str, int]) -> dict[str, float]:
    """Compute sensitivity, specificity, and balanced accuracy."""

    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    balanced_accuracy = (sensitivity + specificity) / 2
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
    }


def filter_validation_frame(
    validation: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    """Apply stage-specific record filtering used by the benchmark."""

    filtered = validation.copy()
    if stage == "l1":
        validation_articles = records_from_frame(filtered)
        usable_record_ids = {
            article.record_id
            for article in validation_articles
            if has_usable_l1_abstract(article)
            and not is_l1_excluded_secondary_review(article)
            and not is_excluded_duplicate_record(article)
        }
        filtered = filtered[
            filtered["record_index"].astype(str).isin(usable_record_ids)
        ].copy()
    elif stage == "l2":
        validation_articles = records_from_frame(filtered)
        usable_record_ids = {
            article.record_id
            for article in validation_articles
            if not is_excluded_duplicate_record(article)
        }
        filtered = filtered[
            filtered["record_index"].astype(str).isin(usable_record_ids)
        ].copy()
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    return filtered


def gold_column_for_stage(stage: str) -> str:
    """Return the gold-label column for a screening stage."""

    if stage == "l2":
        return "gold_l2_label"
    if stage == "l1":
        return "gold_l1_label"
    raise ValueError(f"Unsupported stage: {stage}")


def merge_with_predictions(
    validation_csv: str | Path,
    predictions_csv: str | Path,
    stage: str,
    decision_column: str = "final_decision",
    prediction_id_column: str = "record_id",
) -> pd.DataFrame:
    """Return validation rows merged to prediction rows on record id."""

    validation = pd.read_csv(validation_csv)
    predictions = pd.read_csv(predictions_csv)
    validation = filter_validation_frame(validation, stage)
    validation["record_index_join"] = validation["record_index"].astype(str)
    predictions = predictions.copy()
    predictions["record_id_join"] = predictions[prediction_id_column].astype(str)

    gold_column = gold_column_for_stage(stage)
    if gold_column not in validation.columns:
        raise ValueError(f"{validation_csv} does not contain {gold_column}.")
    if decision_column not in predictions.columns:
        raise ValueError(f"{predictions_csv} does not contain {decision_column}.")

    merged = validation.merge(
        predictions,
        left_on="record_index_join",
        right_on="record_id_join",
        how="inner",
    )
    merged["gold_include"] = merged[gold_column].astype(bool)
    merged["predicted_include"] = (
        merged[decision_column].astype(str).str.lower().eq("include")
    )
    return merged


def evaluate(
    validation_csv: str | Path,
    predictions_csv: str | Path,
    stage: str,
) -> pd.DataFrame:
    """Join validation labels to predictions and return a one-row metrics table."""

    merged = merge_with_predictions(validation_csv, predictions_csv, stage)
    counts = confusion_counts(merged["gold_include"], merged["predicted_include"])
    metric_values = rates(counts)
    return pd.DataFrame([{**counts, **metric_values, "n": len(merged), "stage": stage}])


def evaluate_l1(validation_csv: str | Path, predictions_csv: str | Path) -> pd.DataFrame:
    """Backward-compatible L1 metrics wrapper."""

    return evaluate(validation_csv, predictions_csv, stage="l1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate L1 screening predictions.")
    parser.add_argument("--validation", required=True, help="Normalized validation CSV.")
    parser.add_argument("--predictions", required=True, help="AI prediction CSV.")
    parser.add_argument("--output", required=True, help="Metrics CSV to write.")
    parser.add_argument(
        "--stage",
        choices=["l1", "l2"],
        default="l1",
        help="Which gold label to use.",
    )
    args = parser.parse_args()

    metrics = evaluate(args.validation, args.predictions, args.stage)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
