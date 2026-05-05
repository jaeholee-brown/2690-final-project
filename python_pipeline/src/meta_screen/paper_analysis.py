"""Generate tables and plots for a paper-style evaluation of one screening run."""

# Reading guide for R users:
# - This file is the main manuscript-analysis script on the Python side.
# - It takes one predictions file plus one validation file and writes:
#     - merged prediction tables
#     - summary metrics
#     - error tables
#     - confusion matrices, ROC curves, and score histograms
#     - cost summaries based on token usage
# - If you want to regenerate most of the paper's quantitative artifacts without
#   calling the APIs again, this is the script to run.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import pandas as pd

from meta_screen.metrics import confusion_counts, merge_with_predictions, rates

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PRICE_MAP: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "gemini-3-flash-preview": (0.50, 3.00),
    "grok-4.20-0309-reasoning": (1.25, 2.50),
}


def _safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _provider_decision_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if column.endswith("_decision") and column != "final_decision"
    )


def _vote_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    decision_columns = _provider_decision_columns(frame)
    for _, row in frame.iterrows():
        include_votes = 0
        exclude_votes = 0
        primary_include_votes = 0
        primary_exclude_votes = 0
        escalation_include_votes = 0
        escalation_exclude_votes = 0
        include_probabilities: list[float] = []
        signed_confidences: list[float] = []

        for decision_column in decision_columns:
            decision = str(row.get(decision_column, "")).strip().lower()
            if decision not in {"include", "exclude"}:
                continue

            confidence_column = decision_column[: -len("_decision")] + "_confidence"
            confidence = _safe_float(row.get(confidence_column))
            if confidence is None:
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            if decision == "include":
                include_votes += 1
                include_probabilities.append(confidence)
                signed_confidences.append(confidence)
                if decision_column.startswith("primary_"):
                    primary_include_votes += 1
                elif decision_column.startswith("escalation_"):
                    escalation_include_votes += 1
            else:
                exclude_votes += 1
                include_probabilities.append(1.0 - confidence)
                signed_confidences.append(-confidence)
                if decision_column.startswith("primary_"):
                    primary_exclude_votes += 1
                elif decision_column.startswith("escalation_"):
                    escalation_exclude_votes += 1

        total_votes = include_votes + exclude_votes
        rows.append(
            {
                "include_votes": include_votes,
                "exclude_votes": exclude_votes,
                "total_votes": total_votes,
                "primary_include_votes": primary_include_votes,
                "primary_exclude_votes": primary_exclude_votes,
                "escalation_include_votes": escalation_include_votes,
                "escalation_exclude_votes": escalation_exclude_votes,
                "vote_margin": include_votes - exclude_votes,
                "mean_include_probability": (
                    sum(include_probabilities) / len(include_probabilities)
                    if include_probabilities
                    else float("nan")
                ),
                "mean_signed_confidence": (
                    sum(signed_confidences) / len(signed_confidences)
                    if signed_confidences
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + (z**2 / total)
    center = (p + (z**2 / (2 * total))) / denom
    margin = (
        z
        * math.sqrt((p * (1 - p) / total) + (z**2 / (4 * (total**2))))
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _summary_metrics(merged: pd.DataFrame, stage: str, label: str) -> pd.DataFrame:
    counts = confusion_counts(merged["gold_include"], merged["predicted_include"])
    metric_values = rates(counts)
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]
    sens_lo, sens_hi = _wilson_interval(tp, tp + fn)
    spec_lo, spec_hi = _wilson_interval(tn, tn + fp)
    prec_lo, prec_hi = _wilson_interval(tp, tp + fp)
    return pd.DataFrame(
        [
            {
                "label": label,
                "stage": stage,
                **counts,
                **metric_values,
                "sensitivity_ci_low": sens_lo,
                "sensitivity_ci_high": sens_hi,
                "specificity_ci_low": spec_lo,
                "specificity_ci_high": spec_hi,
                "precision_ci_low": prec_lo,
                "precision_ci_high": prec_hi,
                "n": len(merged),
            }
        ]
    )


def _confusion_table(merged: pd.DataFrame) -> pd.DataFrame:
    table = (
        merged.assign(
            gold_label=merged["gold_include"].map({True: "include", False: "exclude"}),
            predicted_label=merged["predicted_include"].map(
                {True: "include", False: "exclude"}
            ),
        )
        .groupby(["predicted_label", "gold_label"])
        .size()
        .reset_index(name="count")
    )
    return table


def _label_error_type(row: pd.Series) -> str:
    if row["gold_include"] and row["predicted_include"]:
        return "true_positive"
    if (not row["gold_include"]) and row["predicted_include"]:
        return "false_positive"
    if (not row["gold_include"]) and (not row["predicted_include"]):
        return "true_negative"
    return "false_negative"


def _threshold_metrics(
    scores: pd.Series,
    gold: pd.Series,
) -> tuple[pd.DataFrame, float]:
    data = pd.DataFrame({"score": scores.astype(float), "gold": gold.astype(bool)}).dropna()
    if data.empty:
        return pd.DataFrame(), float("nan")

    thresholds = sorted(data["score"].unique(), reverse=True)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        predicted = data["score"] >= threshold
        counts = confusion_counts(data["gold"], predicted)
        metric_values = rates(counts)
        tp = counts["true_positive"]
        fp = counts["false_positive"]
        tn = counts["true_negative"]
        fn = counts["false_negative"]
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        rows.append(
            {
                "threshold": threshold,
                **counts,
                **metric_values,
                "recall": metric_values["sensitivity"],
                "fpr": fpr,
            }
        )

    curve = pd.DataFrame(rows).sort_values("fpr")
    auc = 0.0
    last_fpr = 0.0
    last_tpr = 0.0
    for _, row in curve.iterrows():
        fpr = float(row["fpr"])
        tpr = float(row["recall"])
        auc += (fpr - last_fpr) * (tpr + last_tpr) / 2
        last_fpr = fpr
        last_tpr = tpr
    auc += (1.0 - last_fpr) * (1.0 + last_tpr) / 2
    return curve.reset_index(drop=True), auc


def _cost_table(
    cost_path: str | Path,
    label: str,
    price_map: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    data = json.loads(Path(cost_path).read_text())
    rows: list[dict[str, object]] = []
    for provider_row in data.get("providers", []):
        model = provider_row["model"]
        input_price, output_price = price_map.get(model, (float("nan"), float("nan")))
        prompt_tokens = int(provider_row.get("prompt_tokens", 0))
        completion_tokens = int(provider_row.get("completion_tokens", 0))
        if math.isnan(input_price) or math.isnan(output_price):
            estimated_cost = float("nan")
        else:
            estimated_cost = (
                (prompt_tokens / 1_000_000) * input_price
                + (completion_tokens / 1_000_000) * output_price
            )
        rows.append(
            {
                "label": label,
                "run_id": data.get("run_id", ""),
                "stage": data.get("stage", ""),
                "phase": provider_row.get("phase", ""),
                "provider": provider_row.get("provider", ""),
                "model": model,
                "api_calls": int(provider_row.get("api_calls", 0)),
                "cached_calls": int(provider_row.get("cached_calls", 0)),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_input_price_per_million": input_price,
                "estimated_output_price_per_million": output_price,
                "estimated_cost_usd": estimated_cost,
            }
        )
    cost_frame = pd.DataFrame(rows)
    if not cost_frame.empty:
        total_row = {
            "label": label,
            "run_id": data.get("run_id", ""),
            "stage": data.get("stage", ""),
            "phase": "total",
            "provider": "all",
            "model": "all",
            "api_calls": int(cost_frame["api_calls"].sum()),
            "cached_calls": int(cost_frame["cached_calls"].sum()),
            "prompt_tokens": int(cost_frame["prompt_tokens"].sum()),
            "completion_tokens": int(cost_frame["completion_tokens"].sum()),
            "estimated_input_price_per_million": float("nan"),
            "estimated_output_price_per_million": float("nan"),
            "estimated_cost_usd": cost_frame["estimated_cost_usd"].sum(min_count=1),
        }
        cost_frame = pd.concat([cost_frame, pd.DataFrame([total_row])], ignore_index=True)
    return cost_frame


def _save_confusion_plot(confusion_table: pd.DataFrame, output_path: Path, title: str) -> None:
    pivot = (
        confusion_table.pivot(
            index="predicted_label", columns="gold_label", values="count"
        )
        .reindex(index=["include", "exclude"], columns=["include", "exclude"])
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(pivot.values, cmap="Blues")
    ax.set_xticks([0, 1], labels=pivot.columns)
    ax.set_yticks([0, 1], labels=pivot.index)
    ax.set_xlabel("Gold label")
    ax.set_ylabel("Predicted label")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_score_histogram(merged: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for gold_value, label, color in [
        (True, "Gold include", "#1b9e77"),
        (False, "Gold exclude", "#d95f02"),
    ]:
        subset = merged.loc[merged["gold_include"] == gold_value, "mean_include_probability"]
        ax.hist(subset.dropna(), bins=12, alpha=0.6, label=label, color=color)
    ax.set_xlabel("Mean include probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_roc_plot(
    threshold_frame: pd.DataFrame,
    auc: float,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(threshold_frame["fpr"], threshold_frame["recall"], color="#1f78b4")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title} (AUC={auc:.3f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_vote_margin_plot(merged: pd.DataFrame, output_path: Path, title: str) -> None:
    counts = (
        merged.groupby(["error_type", "vote_margin"])
        .size()
        .reset_index(name="count")
        .pivot(index="vote_margin", columns="error_type", values="count")
        .fillna(0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Vote margin (include votes - exclude votes)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def analyze_run(
    validation_csv: str | Path,
    predictions_csv: str | Path,
    stage: str,
    output_dir: str | Path,
    label: str,
    cost_path: str | Path | None = None,
    price_map: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Generate paper-ready outputs for one benchmark run."""

    output_path = Path(output_dir)
    plots_path = output_path / "plots"
    output_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    merged = merge_with_predictions(validation_csv, predictions_csv, stage)
    for canonical, preferred, fallback in [
        ("title", "title_x", "title_y"),
        ("abstract", "abstract_x", "abstract_y"),
    ]:
        if canonical not in merged.columns:
            if preferred in merged.columns:
                merged[canonical] = merged[preferred]
            elif fallback in merged.columns:
                merged[canonical] = merged[fallback]
    vote_frame = _vote_features(merged)
    merged = pd.concat([merged.reset_index(drop=True), vote_frame], axis=1)
    merged["error_type"] = merged.apply(_label_error_type, axis=1)

    summary = _summary_metrics(merged, stage, label)
    confusion = _confusion_table(merged)
    errors = merged[merged["error_type"].isin(["false_positive", "false_negative"])].copy()
    threshold_frame, auc = _threshold_metrics(
        merged["mean_include_probability"], merged["gold_include"]
    )

    summary["mean_primary_include_votes"] = merged["primary_include_votes"].mean()
    summary["mean_escalation_include_votes"] = merged["escalation_include_votes"].mean()
    summary["mean_total_votes"] = merged["total_votes"].mean()
    summary["mean_include_probability_auc"] = auc

    summary.to_csv(output_path / "metrics_summary.csv", index=False)
    confusion.to_csv(output_path / "confusion_matrix.csv", index=False)
    merged.to_csv(output_path / "merged_predictions.csv", index=False)
    errors.to_csv(output_path / "errors.csv", index=False)
    threshold_frame.to_csv(output_path / "confidence_thresholds.csv", index=False)

    provider_summary = (
        merged.groupby("error_type")[
            [
                "include_votes",
                "exclude_votes",
                "primary_include_votes",
                "primary_exclude_votes",
                "escalation_include_votes",
                "escalation_exclude_votes",
                "mean_include_probability",
                "mean_signed_confidence",
            ]
        ]
        .mean()
        .reset_index()
    )
    provider_summary.to_csv(output_path / "vote_summary.csv", index=False)

    if cost_path is not None:
        cost_frame = _cost_table(cost_path, label, price_map or DEFAULT_PRICE_MAP)
        cost_frame.to_csv(output_path / "cost_summary.csv", index=False)

    _save_confusion_plot(confusion, plots_path / "confusion_matrix.png", f"{label} confusion")
    _save_score_histogram(
        merged,
        plots_path / "score_histogram.png",
        f"{label} score distribution",
    )
    if not threshold_frame.empty:
        _save_roc_plot(threshold_frame, auc, plots_path / "roc_curve.png", f"{label} ROC")
    _save_vote_margin_plot(
        merged,
        plots_path / "vote_margin.png",
        f"{label} vote margin by outcome",
    )


def _parse_price_overrides(values: list[str]) -> dict[str, tuple[float, float]]:
    overrides: dict[str, tuple[float, float]] = {}
    for value in values:
        model, input_price, output_price = value.split(":")
        overrides[model] = (float(input_price), float(output_price))
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-style tables and plots for one screening run."
    )
    parser.add_argument("--validation", required=True, help="Validation CSV.")
    parser.add_argument("--predictions", required=True, help="Predictions CSV.")
    parser.add_argument("--stage", choices=["l1", "l2"], required=True)
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--label", required=True, help="Run label for tables/plots.")
    parser.add_argument("--cost", help="Optional cost.json path.")
    parser.add_argument(
        "--price-override",
        action="append",
        default=[],
        help="Override price as model:input_per_million:output_per_million.",
    )
    args = parser.parse_args()

    price_map = DEFAULT_PRICE_MAP.copy()
    price_map.update(_parse_price_overrides(args.price_override))
    analyze_run(
        validation_csv=args.validation,
        predictions_csv=args.predictions,
        stage=args.stage,
        output_dir=args.output_dir,
        label=args.label,
        cost_path=args.cost,
        price_map=price_map,
    )


if __name__ == "__main__":
    main()
