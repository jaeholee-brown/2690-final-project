"""Paired significance tests for benchmark comparisons."""

# Reading guide for R users:
# - This file contains the paired-comparison statistics used in the paper.
# - The main result is an exact McNemar test comparing whether two classifiers
#   are correct on the same records.
# - It reads previously generated prediction outputs and writes one comparison
#   table; no API calls happen here.

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def _mcnemar_exact_p_value(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via the Binomial(n, 0.5) distribution."""

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cumulative = 0.0
    for i in range(k + 1):
        cumulative += math.comb(n, i) * (0.5**n)
    return min(1.0, 2.0 * cumulative)


def _compare_correctness(name_a: str, pred_a: pd.Series, name_b: str, pred_b: pd.Series, gold: pd.Series) -> dict[str, object]:
    a_correct = pred_a.astype(bool) == gold.astype(bool)
    b_correct = pred_b.astype(bool) == gold.astype(bool)
    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())
    total_discordant = b + c
    p_value = _mcnemar_exact_p_value(b, c)
    return {
        "model_a": name_a,
        "model_b": name_b,
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "discordant_total": total_discordant,
        "exact_mcnemar_p_value": p_value,
    }


def _stage_rows(stage: str) -> pd.DataFrame:
    merged = pd.read_csv(Path("outputs/paper_analysis") / f"fiber_{stage}" / "merged_predictions.csv")
    gold = merged["gold_include"].astype(bool)
    primary_decision_columns = [c for c in merged.columns if c.startswith("primary_") and c.endswith("_decision")]
    primary_include_votes = primary_decision_columns and merged[primary_decision_columns].apply(
        lambda row: sum(value == "include" for value in row), axis=1
    )

    baselines = pd.read_csv(Path("outputs/paper_baselines") / f"fiber_{stage}" / "predictions.csv")
    baseline_pivot = baselines.pivot(index="record_id", columns="baseline_name", values="final_decision").reset_index()
    merged["record_id"] = merged["record_id"].astype(str)
    baseline_pivot["record_id"] = baseline_pivot["record_id"].astype(str)
    merged = merged.merge(baseline_pivot, on="record_id", how="left")

    variants: dict[str, pd.Series] = {
        "final_pipeline": merged["predicted_include"],
        "keyword_sensitive": merged["keyword_sensitive"].eq("include"),
        "keyword_strict": merged["keyword_strict"].eq("include"),
        "always_exclude": merged["always_exclude"].eq("include"),
    }
    if isinstance(primary_include_votes, pd.Series):
        variants["primary_unanimous_include"] = primary_include_votes.eq(3)
        variants["primary_majority_include"] = primary_include_votes.ge(2)

    comparisons = [
        ("final_pipeline", "keyword_sensitive"),
        ("final_pipeline", "keyword_strict"),
        ("final_pipeline", "always_exclude"),
        ("final_pipeline", "primary_majority_include"),
        ("final_pipeline", "primary_unanimous_include"),
    ]

    rows = []
    for left, right in comparisons:
        if left not in variants or right not in variants:
            continue
        row = _compare_correctness(left, variants[left], right, variants[right], gold)
        row["stage"] = stage
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    frames = [_stage_rows("l1"), _stage_rows("l2")]
    output = pd.concat(frames, ignore_index=True)
    output.to_csv("outputs/paper_analysis/fiber_mcnemar_tests.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
