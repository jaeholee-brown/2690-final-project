"""Build appendix-ready qualitative error tables for the Fiber paper."""

# Reading guide for R users:
# - This file turns manually tagged error cases into presentation-ready appendix
#   tables.
# - It is not part of the screening pipeline itself; it is a reporting step for
#   the paper.

from __future__ import annotations

from pathlib import Path

import pandas as pd


EXAMPLE_CASES = [
    {
        "stage": "l1",
        "record_index": 58,
        "bucket": "clearly_wrong",
        "bucket_label": "Clearly Wrong",
        "short_reason": "Neonatal/NICU population was outside the intended adult critical-care target.",
    },
    {
        "stage": "l1",
        "record_index": 59,
        "bucket": "clearly_wrong",
        "bucket_label": "Clearly Wrong",
        "short_reason": "Very-low-birth-weight infant study was treated as on-target ICU evidence.",
    },
    {
        "stage": "l1",
        "record_index": 61,
        "bucket": "clearly_wrong",
        "bucket_label": "Clearly Wrong",
        "short_reason": "Pediatric ICU synbiotic trial was advanced despite the review targeting adult critical-care patients.",
    },
    {
        "stage": "l1",
        "record_index": 167,
        "bucket": "clearly_wrong",
        "bucket_label": "Clearly Wrong",
        "short_reason": "Head-injured tube-fed patients were treated as ICU/critical-care despite no explicit ICU signal.",
    },
    {
        "stage": "l1",
        "record_index": 179,
        "bucket": "clearly_wrong",
        "bucket_label": "Clearly Wrong",
        "short_reason": "Immobile tube-fed patients were advanced on topic similarity alone, not population match.",
    },
    {
        "stage": "l1",
        "record_index": 214,
        "bucket": "clearly_wrong",
        "bucket_label": "Clearly Wrong",
        "short_reason": "Postoperative enteral-nutrition trial lacked the ICU/critical-ill population required by the review.",
    },
    {
        "stage": "l1",
        "record_index": 23,
        "bucket": "ambiguous_or_reasonable",
        "bucket_label": "Arguably Reasonable",
        "short_reason": "Review-style ICU diarrhea paper had no explicit fiber intervention, so exclusion was defensible.",
    },
    {
        "stage": "l1",
        "record_index": 25,
        "bucket": "ambiguous_or_reasonable",
        "bucket_label": "Arguably Reasonable",
        "short_reason": "Another generic ICU diarrhea review that humans sent to full text but models saw as clearly out of scope.",
    },
    {
        "stage": "l2",
        "record_index": 17,
        "bucket": "ambiguous_or_reasonable",
        "bucket_label": "Arguably Reasonable",
        "short_reason": "The full text described a mixed ICU and medical-surgical sample, so strict exclusion was understandable.",
    },
    {
        "stage": "l2",
        "record_index": 20,
        "bucket": "ambiguous_or_reasonable",
        "bucket_label": "Arguably Reasonable",
        "short_reason": "Comparator eligibility in the synbiotic trauma trial appears genuinely contestable from the criteria wording.",
    },
    {
        "stage": "l2",
        "record_index": 22,
        "bucket": "ambiguous_or_reasonable",
        "bucket_label": "Arguably Reasonable",
        "short_reason": "The placebo was non-fiber, but the formula context made the comparator logic debatable.",
    },
    {
        "stage": "l2",
        "record_index": 28,
        "bucket": "ambiguous_or_reasonable",
        "bucket_label": "Arguably Reasonable",
        "short_reason": "The comparator looked fiber-based to one reading and non-fiber enough to another.",
    },
]


def _stage_paths(stage: str) -> tuple[Path, Path]:
    base = Path("outputs/paper_analysis") / f"fiber_{stage}"
    return base / "errors.csv", base / "error_tags.csv"


def _title_case_theme(theme: str) -> str:
    return theme.replace("_", " ").capitalize()


def _choose_rationale(row: pd.Series) -> str:
    for column in [
        "escalation_openai_rationale",
        "primary_openai_rationale",
        "escalation_anthropic_rationale",
        "primary_anthropic_rationale",
    ]:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_appendix_tables(output_dir: str | Path = "outputs/paper_analysis") -> None:
    """Write appendix-ready CSV and Markdown tables for selected examples."""

    output_path = Path(output_dir)
    rows: list[dict[str, object]] = []

    for stage in ["l1", "l2"]:
        errors_path, tags_path = _stage_paths(stage)
        errors = pd.read_csv(errors_path)
        tags = pd.read_csv(tags_path)
        merged = errors.merge(tags, on=["record_index", "error_type"], how="left")
        merged["stage"] = stage.upper()

        for case in [item for item in EXAMPLE_CASES if item["stage"] == stage]:
            match = merged[merged["record_index"] == case["record_index"]]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                {
                    "bucket": case["bucket"],
                    "bucket_label": case["bucket_label"],
                    "stage": row["stage"],
                    "record_index": int(row["record_index"]),
                    "title": row["title"],
                    "human_decision": row["human_decision"],
                    "model_decision": row["final_decision"],
                    "vote_split": f"{int(row['include_votes'])} include / {int(row['exclude_votes'])} exclude",
                    "mean_include_probability": round(float(row["mean_include_probability"]), 3),
                    "theme_label": _title_case_theme(str(row.get("theme", ""))),
                    "paper_note": case["short_reason"],
                    "representative_rationale": _choose_rationale(row),
                }
            )

    appendix = pd.DataFrame(rows)
    appendix = appendix.sort_values(
        ["bucket_label", "stage", "record_index"]
    ).reset_index(drop=True)
    appendix.to_csv(output_path / "fiber_appendix_error_examples.csv", index=False)

    lines: list[str] = []
    for bucket, section_title in [
        ("clearly_wrong", "Table A1. Representative Errors Where the LLM Pipeline Was Clearly Wrong"),
        (
            "ambiguous_or_reasonable",
            "Table A2. Representative Errors That Were Arguably Reasonable Under Ambiguous Criteria",
        ),
    ]:
        section = appendix[appendix["bucket"] == bucket].copy()
        lines.append(f"## {section_title}")
        lines.append("")
        lines.append(
            "| Stage | Record | Title | Human | Model | Vote Split | Mean Include Prob. | Why This Case Matters |"
        )
        lines.append(
            "|---|---:|---|---|---|---|---:|---|"
        )
        for _, row in section.iterrows():
            title = str(row["title"]).replace("|", "\\|")
            note = str(row["paper_note"]).replace("|", "\\|")
            lines.append(
                f"| {row['stage']} | {row['record_index']} | {title} | {row['human_decision']} | "
                f"{row['model_decision']} | {row['vote_split']} | {row['mean_include_probability']:.3f} | {note} |"
            )
        lines.append("")

    (output_path / "fiber_appendix_error_examples.md").write_text("\n".join(lines))

    latex_lines: list[str] = []
    for bucket, caption, label in [
        (
            "clearly_wrong",
            "Representative errors where the LLM pipeline was clearly wrong.",
            "tab:fiber-clearly-wrong-errors",
        ),
        (
            "ambiguous_or_reasonable",
            "Representative errors that were arguably reasonable under ambiguous criteria.",
            "tab:fiber-ambiguous-errors",
        ),
    ]:
        section = appendix[appendix["bucket"] == bucket].copy()
        latex_lines.append("\\begin{table}[ht]")
        latex_lines.append("\\centering")
        latex_lines.append(f"\\caption{{{caption}}}")
        latex_lines.append(f"\\label{{{label}}}")
        latex_lines.append("\\begin{tabular}{p{0.06\\linewidth} p{0.06\\linewidth} p{0.30\\linewidth} p{0.13\\linewidth} p{0.11\\linewidth} p{0.24\\linewidth}}")
        latex_lines.append("\\hline")
        latex_lines.append("Stage & Record & Title & Human & Model & Why this case matters \\\\")
        latex_lines.append("\\hline")
        for _, row in section.iterrows():
            title = str(row["title"]).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            note = str(row["paper_note"]).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            human = str(row["human_decision"]).replace("_", "\\_")
            model = str(row["model_decision"]).replace("_", "\\_")
            latex_lines.append(
                f"{row['stage']} & {row['record_index']} & {title} & {human} & {model} & {note} \\\\"
            )
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        latex_lines.append("")

    (output_path / "fiber_appendix_error_examples.tex").write_text("\n".join(latex_lines))


if __name__ == "__main__":
    build_appendix_tables()
