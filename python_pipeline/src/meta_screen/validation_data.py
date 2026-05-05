"""Download and normalize validation datasets.

The raw ReviewCopilot files use different column names across reviews. This
module turns them into the same shape so the screening pipeline and metrics can
use one set of column names.
"""

# Reading guide for R users:
# - This file is the data-ingestion and cleaning script for benchmark datasets.
# - The main idea is to convert several differently formatted review datasets
#   into one standard rectangular schema that the rest of the code can assume.
# - If you want to know where the normalized `cara_2021_fiber.csv` files come
#   from, start here.

from __future__ import annotations

import io
import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


SOURCE_BASE_URL = "https://github.com/jamesjiadazhan/ReviewCopilot/raw/main"
MIN_USABLE_FULL_TEXT_CHARS = 3000


@dataclass(frozen=True)
class DatasetSpec:
    """Location and citation metadata for one source dataset."""

    dataset_id: str
    display_name: str
    archive_name: str
    l1_csv: str
    l2_csv: str | None
    l2_pdf_prefixes: tuple[str, ...]
    l2_pdf_title_csv: str | None
    expected_l1_rows_from_archive: int
    expected_l1_includes_from_archive: int
    source_review: str


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="cara_2021_fiber",
        display_name="Cara et al. 2021 (Fiber)",
        archive_name="Fiber.zip",
        l1_csv="Fiber/title_abstract_screening/Fiber_human_final.csv",
        l2_csv="Fiber/full_text_screening/Fiber_human_final_full_text.csv",
        l2_pdf_prefixes=(
            "Fiber/full_text_screening/05_PDFs for Included Papers/",
            "Fiber/full_text_screening/Excluded papers/",
        ),
        l2_pdf_title_csv=(
            "Fiber/full_text_screening/Review_Copilot_results_run1/"
            "Fiber_full_text_human_AI_run1.csv"
        ),
        expected_l1_rows_from_archive=482,
        expected_l1_includes_from_archive=53,
        source_review=(
            "Safety of using enteral nutrition formulations containing "
            "dietary fiber in hospitalized critical care patients"
        ),
    ),
    DatasetSpec(
        dataset_id="cara_2021_juice",
        display_name="Cara et al. 2021 (Juice)",
        archive_name="Juice.zip",
        l1_csv="Juice/title_abstract_screening/Juice_human_final.csv",
        l2_csv="Juice/full_text_screening/Juice_human_final_full_text.csv",
        l2_pdf_prefixes=("Juice/full_text_screening/All/",),
        l2_pdf_title_csv=(
            "Juice/full_text_screening/Review_Copilot_results_run1/"
            "Juice_full_text_PICOS_DR_run1.csv"
        ),
        expected_l1_rows_from_archive=1183,
        expected_l1_includes_from_archive=85,
        source_review=(
            "Effects of 100% orange juice on markers of inflammation and "
            "oxidation in healthy and at-risk adult populations"
        ),
    ),
    DatasetSpec(
        dataset_id="galaviz_2022_prediabetes",
        display_name="Galaviz et al. 2022 (Prediabetes)",
        archive_name="prediabetes.zip",
        l1_csv="prediabetes/results/prediabetes_human_final.csv",
        l2_csv=None,
        l2_pdf_prefixes=(),
        l2_pdf_title_csv=None,
        expected_l1_rows_from_archive=3506,
        expected_l1_includes_from_archive=45,
        source_review=(
            "Interventions for reversing prediabetes in adults"
        ),
    ),
    DatasetSpec(
        dataset_id="meijboom_2022_biosimilars",
        display_name="Meijboom et al. 2022 (Biosimilars)",
        archive_name="Meijboom_2022.zip",
        l1_csv="Meijboom_2022/results/Meijboom_2022_human_final.csv",
        l2_csv=None,
        l2_pdf_prefixes=(),
        l2_pdf_title_csv=None,
        expected_l1_rows_from_archive=882,
        expected_l1_includes_from_archive=37,
        source_review=(
            "Outcomes after transitioning between cancer drugs and biosimilars"
        ),
    ),
)


def download_archives(source_dir: str | Path = "data/source_archives") -> None:
    """Download the raw ReviewCopilot zip files if they are absent.

    The function does not overwrite existing archives. This matters because the
    GitHub repository could change later; once downloaded, the local copy stays
    stable for reproducible project results.
    """

    source_path = Path(source_dir)
    source_path.mkdir(parents=True, exist_ok=True)

    for spec in DATASETS:
        archive_path = source_path / spec.archive_name
        if archive_path.exists():
            continue
        urlretrieve(f"{SOURCE_BASE_URL}/{spec.archive_name}", archive_path)


def read_csv_from_archive(archive_path: str | Path, member_name: str) -> pd.DataFrame:
    """Read one CSV file from a zip archive into a pandas DataFrame."""

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as file_handle:
            # Read bytes first so pandas can infer encoding from a normal buffer.
            raw_bytes = file_handle.read()
    return pd.read_csv(io.BytesIO(raw_bytes))


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column that exists in a DataFrame."""

    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _clean_text(series: pd.Series) -> pd.Series:
    """Return text that is safe to write as one CSV row per article."""

    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"[\r\n\t]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _clean_one_text(value: object) -> str:
    """Clean one value using the same rules as `_clean_text`."""

    return re.sub(r"\s+", " ", re.sub(r"[\r\n\t]+", " ", str(value or ""))).strip()


def normalize_l1(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Standardize title/abstract screening rows.

    The L1 target is "send this record to full-text screening." Therefore both
    `Full_text_include` and `Full_text_exclude` count as L1 positives, because a
    human reviewer judged them worth reading at full text.
    """

    # Reset the index so pandas does not align filtered rows by their old row
    # numbers when we assign Series into the normalized DataFrame.
    frame = frame.reset_index(drop=True)

    title_col = _pick_column(frame, ("title", "Title"))
    abstract_col = _pick_column(frame, ("abstract", "Abstract"))
    id_col = _pick_column(frame, ("key", "id", "record_id", "openalex_id"))
    year_col = _pick_column(frame, ("year", "Published Year", "publication_year"))
    authors_col = _pick_column(frame, ("authors", "Authors"))
    journal_col = _pick_column(frame, ("journal", "Journal"))
    doi_col = _pick_column(frame, ("doi", "DOI"))
    url_col = _pick_column(frame, ("url", "openalex_id"))

    if title_col is None or abstract_col is None:
        raise ValueError(f"{spec.dataset_id} is missing title or abstract columns.")

    normalized = pd.DataFrame(index=frame.index)
    normalized["dataset_id"] = spec.dataset_id
    normalized["dataset_name"] = spec.display_name
    normalized["record_index"] = range(len(frame))
    normalized["source_record_id"] = (
        _clean_text(frame[id_col]) if id_col else normalized["record_index"].astype(str)
    )
    normalized["title"] = _clean_text(frame[title_col])
    normalized["abstract"] = _clean_text(frame[abstract_col])
    normalized["year"] = frame[year_col] if year_col else pd.NA
    normalized["authors"] = _clean_text(frame[authors_col]) if authors_col else pd.NA
    normalized["journal"] = _clean_text(frame[journal_col]) if journal_col else pd.NA
    normalized["doi"] = _clean_text(frame[doi_col]) if doi_col else pd.NA
    normalized["url"] = _clean_text(frame[url_col]) if url_col else pd.NA
    normalized["human_decision"] = frame["Human_Decision"].astype(str)
    normalized["gold_l1_label"] = normalized["human_decision"].ne("Title_abstract_exclude")
    normalized["gold_final_include"] = normalized["human_decision"].eq("Full_text_include")
    normalized["source_review"] = spec.source_review

    return normalized.reset_index(drop=True)


def _normalized_tokens(value: object) -> list[str]:
    """Tokenize strings for fuzzy PDF-to-human-row matching."""

    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    stopwords = {
        "a",
        "an",
        "and",
        "in",
        "of",
        "on",
        "or",
        "pdf",
        "the",
        "to",
        "with",
    }
    return [token for token in text.split() if token and token not in stopwords]


def _match_score(left: object, right: object) -> float:
    """Score how likely two citation strings refer to the same article."""

    left_tokens = _normalized_tokens(left)
    right_tokens = _normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0

    left_set = set(left_tokens)
    right_set = set(right_tokens)
    overlap = len(left_set & right_set)
    dice = 2 * overlap / (len(left_set) + len(right_set))
    containment = overlap / min(len(left_set), len(right_set))
    left_text = " ".join(left_tokens)
    right_text = " ".join(right_tokens)

    # Difflib is in the standard library and good enough here; the token scores
    # make truncated PDF filenames match their full human-reviewed titles.
    import difflib

    sequence = difflib.SequenceMatcher(None, left_text, right_text).ratio()
    return max(dice, containment * 0.95, sequence)


def _extract_text_with_pdftotext(
    pdf_bytes: bytes,
    pdf_name: str,
    min_chars: int = MIN_USABLE_FULL_TEXT_CHARS,
) -> tuple[str, str]:
    """Extract raw PDF text using Poppler's `pdftotext` command."""

    if shutil.which("pdftotext") is None:
        raise RuntimeError(
            "Raw PDF L2 preparation requires `pdftotext` from Poppler. "
            "Install it with `brew install poppler` on macOS."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "article.pdf"
        text_path = Path(tmp_dir) / "article.txt"
        pdf_path.write_bytes(pdf_bytes)
        result = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(pdf_path), str(text_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return "", f"pdftotext_failed: {result.stderr[:300]}"
        text = text_path.read_text(encoding="utf-8", errors="ignore")

    text = _clean_one_text(text.replace("\f", " "))
    if len(text) < min_chars:
        return text, f"functionally_missing_text_extracted_from_{pdf_name}"
    return text, "ok"


def _pdf_records_from_archive(
    archive_path: Path,
    spec: DatasetSpec,
) -> list[dict[str, object]]:
    """Extract all usable PDF texts for one L2 dataset."""

    title_by_file: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        if spec.l2_pdf_title_csv:
            with archive.open(spec.l2_pdf_title_csv) as file_handle:
                title_lookup = pd.read_csv(file_handle)
            if "file_name" in title_lookup.columns and "title" in title_lookup.columns:
                # Some source archives include a ReviewCopilot output CSV that
                # maps PDF filenames to article titles. We use only those two
                # citation columns to match PDFs back to the human decisions;
                # no generated PICO or summary screening fields are used.
                title_by_file = {
                    Path(str(row["file_name"])).name: str(row["title"])
                    for _, row in title_lookup.iterrows()
                    if pd.notna(row.get("file_name"))
                }

        pdf_members = [
            member
            for member in archive.namelist()
            if member.lower().endswith(".pdf")
            and "__MACOSX" not in member
            and any(member.startswith(prefix) for prefix in spec.l2_pdf_prefixes)
        ]

        records: list[dict[str, object]] = []
        for member in pdf_members:
            pdf_name = Path(member).name
            pdf_bytes = archive.read(member)
            full_text, status = _extract_text_with_pdftotext(pdf_bytes, pdf_name)
            records.append(
                {
                    "pdf_member": member,
                    "pdf_file": pdf_name,
                    "pdf_reference_title": _clean_one_text(title_by_file.get(pdf_name, "")),
                    "pdf_stem": _clean_one_text(Path(pdf_name).stem),
                    "full_text": full_text,
                    "pdf_text_extraction_status": status,
                    "full_text_char_count": len(full_text),
                    "full_text_token_estimate": math.ceil(len(full_text) / 4),
                }
            )

    return records


def normalize_l2_raw_pdf(
    human_frame: pd.DataFrame,
    archive_path: Path,
    spec: DatasetSpec,
    min_match_score: float = 0.55,
) -> pd.DataFrame:
    """Build an L2 dataset from raw extracted PDF text and human decisions."""

    if not spec.l2_pdf_prefixes:
        raise ValueError(f"{spec.dataset_id} does not define L2 PDF folders.")

    human_l2 = human_frame[
        human_frame["Human_Decision"].isin(["Full_text_include", "Full_text_exclude"])
    ].copy()
    human_l2 = human_l2.reset_index(drop=True)
    normalized_human = normalize_l1(human_l2, spec)

    pdf_records = _pdf_records_from_archive(archive_path, spec)
    usable_pdf_records = [
        record
        for record in pdf_records
        if record["pdf_text_extraction_status"] == "ok"
    ]

    candidate_pairs: list[tuple[float, int, int]] = []
    for human_index, human_row in normalized_human.iterrows():
        for pdf_index, pdf_record in enumerate(usable_pdf_records):
            score = max(
                _match_score(human_row["title"], pdf_record["pdf_reference_title"]),
                _match_score(human_row["title"], pdf_record["pdf_stem"]),
            )
            if score >= min_match_score:
                candidate_pairs.append((score, int(human_index), pdf_index))

    # Assign matches globally from strongest to weakest. This prevents a weak
    # early match from consuming a PDF that should have been an exact match for a
    # later human-reviewed title.
    candidate_pairs.sort(reverse=True)
    matched_human_indexes: set[int] = set()
    used_pdf_indexes: set[int] = set()
    assignments: list[tuple[int, int, float]] = []
    for score, human_index, pdf_index in candidate_pairs:
        if human_index in matched_human_indexes or pdf_index in used_pdf_indexes:
            continue
        matched_human_indexes.add(human_index)
        used_pdf_indexes.add(pdf_index)
        assignments.append((human_index, pdf_index, score))

    matches: list[dict[str, object]] = []
    for human_index, pdf_index, score in sorted(assignments):
        human_row = normalized_human.loc[human_index]
        pdf_record = usable_pdf_records[pdf_index]
        row = human_row.to_dict()
        row.update(pdf_record)
        row["pdf_match_score"] = round(score, 4)
        row["gold_l2_label"] = row["human_decision"] == "Full_text_include"
        matches.append(row)

    output = pd.DataFrame(matches)
    return output.reset_index(drop=True)


def prepare_all(
    source_dir: str | Path = "data/source_archives",
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Create normalized L1 and L2 CSV files and return a summary table."""

    source_path = Path(source_dir)
    output_path = Path(output_dir)
    (output_path / "l1").mkdir(parents=True, exist_ok=True)
    (output_path / "l2").mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []

    for spec in DATASETS:
        archive_path = source_path / spec.archive_name
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Missing {archive_path}. Run download_archives() or "
                "python -m meta_screen.prepare_validation_data --download."
            )

        raw_l1 = read_csv_from_archive(archive_path, spec.l1_csv)
        l1 = normalize_l1(raw_l1, spec)
        l1.to_csv(output_path / "l1" / f"{spec.dataset_id}.csv", index=False)

        summaries.append(
            {
                "dataset_id": spec.dataset_id,
                "dataset_name": spec.display_name,
                "stage": "L1_title_abstract",
                "rows": len(l1),
                "human_l1_includes": int(l1["gold_l1_label"].sum()),
                "human_final_includes": int(l1["gold_final_include"].sum()),
                "title_abstract_excludes": int((~l1["gold_l1_label"]).sum()),
                "archive_expected_rows": spec.expected_l1_rows_from_archive,
                "archive_expected_l1_includes": spec.expected_l1_includes_from_archive,
            }
        )

        if spec.l2_csv:
            raw_l2 = read_csv_from_archive(archive_path, spec.l2_csv)
            l2 = normalize_l2_raw_pdf(raw_l2, archive_path, spec)
            l2.to_csv(output_path / "l2" / f"{spec.dataset_id}.csv", index=False)
            summaries.append(
                {
                    "dataset_id": spec.dataset_id,
                    "dataset_name": spec.display_name,
                    "stage": "L2_raw_pdf_full_text",
                    "rows": len(l2),
                    "human_l1_includes": pd.NA,
                    "human_final_includes": int(l2["gold_l2_label"].sum()),
                    "title_abstract_excludes": pd.NA,
                    "archive_expected_rows": pd.NA,
                    "archive_expected_l1_includes": pd.NA,
                }
            )

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_path / "validation_summary.csv", index=False)
    return summary
