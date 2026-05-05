# Submission Code

This directory is a trimmed, self-contained copy of the project code intended for course submission.

The original repository is unchanged. This folder is a curated bundle containing:

- `python_pipeline/`
  - Python code for the screening pipeline and the paper analysis.
- `r_analysis/`
  - R notebooks/scripts for the additional diagnostics and the meta-analysis audit.
- `data/`
  - Fiber-specific benchmark inputs copied into the bundle so the code can run without downloading anything.
- `outputs/`
  - The Fiber prediction files and the main derived analysis outputs needed by the manuscript workflow.

## Folder map

### `python_pipeline/src/meta_screen/`

- `screener.py`
  - Main L1/L2 screening pipeline.
- `providers.py`
  - API clients and retry/rate-limit logic.
- `config.py`
  - Reads API keys, model names, and runtime settings from `.env`.
- `cache.py`
  - SQLite response cache.
- `validation_data.py`
  - ReviewCopilot dataset download and normalization.
- `prepare_validation_data.py`
  - CLI wrapper for preparing normalized benchmark CSVs.
- `metrics.py`
  - Benchmark filtering and confusion-matrix evaluation.
- `baselines.py`
  - Simple non-LLM baselines used in the paper.
- `paper_analysis.py`
  - Main paper tables, plots, merged prediction tables, and cost summaries.
- `paper_appendix.py`
  - Appendix-ready representative error tables.
- `paper_significance.py`
  - Exact McNemar tests.

### `r_analysis/`

- `fiber_additional_diagnostics.Rmd`
  - Calibration and agreement diagnostics explored for the paper.
- `fiber_additional_diagnostics.R`
  - Script version of the same diagnostics.
- `fiber_meta_analysis_audit.Rmd`
  - Audit of which published Cara et al. pooled outcomes change under the LLM-selected full-text set.

### `data/`

Included Fiber-specific files:

- `data/criteria/cara_2021_fiber.txt`
- `data/processed/l1/cara_2021_fiber.csv`
- `data/processed/l2/cara_2021_fiber.csv`
- `data/source_archives/Fiber.zip`

### `outputs/`

Included Fiber prediction files:

- `outputs/fiber_l1_full_20260504_120716/predictions.csv`
- `outputs/fiber_l2_full_20260504_120923/predictions.csv`

Also included are the main paper-analysis outputs regenerated inside this bundle, so the R notebooks have their expected inputs.

## Quick start

Run these commands from the `submission_code/` directory.

### 1. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python_pipeline/requirements.txt
```

### 2. Install R packages

In R:

```r
install.packages(c("dplyr", "tidyr", "readr", "ggplot2", "pROC", "metafor", "rmarkdown"))
```

## What works without API keys

Because the Fiber benchmark data and the saved Fiber prediction files are already included, you can rerun the main paper analyses without calling any model APIs.

### Recreate baselines

```bash
PYTHONPATH=python_pipeline/src python3 -m meta_screen.baselines \
  --validation data/processed/l1/cara_2021_fiber.csv \
  --stage l1 \
  --output-dir outputs/paper_baselines/fiber_l1

PYTHONPATH=python_pipeline/src python3 -m meta_screen.baselines \
  --validation data/processed/l2/cara_2021_fiber.csv \
  --stage l2 \
  --output-dir outputs/paper_baselines/fiber_l2
```

### Recreate the main paper-analysis outputs

```bash
PYTHONPATH=python_pipeline/src python3 -m meta_screen.paper_analysis \
  --validation data/processed/l1/cara_2021_fiber.csv \
  --predictions outputs/fiber_l1_full_20260504_120716/predictions.csv \
  --stage l1 \
  --output-dir outputs/paper_analysis/fiber_l1 \
  --label final_pipeline

PYTHONPATH=python_pipeline/src python3 -m meta_screen.paper_analysis \
  --validation data/processed/l2/cara_2021_fiber.csv \
  --predictions outputs/fiber_l2_full_20260504_120923/predictions.csv \
  --stage l2 \
  --output-dir outputs/paper_analysis/fiber_l2 \
  --label final_pipeline
```

### Recreate appendix and significance outputs

```bash
PYTHONPATH=python_pipeline/src python3 -m meta_screen.paper_appendix
PYTHONPATH=python_pipeline/src python3 -m meta_screen.paper_significance
```

### Render the R notebooks

The notebooks assume they are rendered from inside `r_analysis/`.

```bash
cd r_analysis
Rscript -e "rmarkdown::render('fiber_meta_analysis_audit.Rmd')"
Rscript -e "rmarkdown::render('fiber_additional_diagnostics.Rmd')"
cd ..
```

## Optional: rerun the LLM screening pipeline

The full screening pipeline requires API keys.

1. Copy `python_pipeline/.env.example` to `.env`.
2. Fill in the API keys you have available.

Example commands:

```bash
cp python_pipeline/.env.example .env
```

```bash
PYTHONPATH=python_pipeline/src python3 -m meta_screen.screener \
  --input data/processed/l1/cara_2021_fiber.csv \
  --criteria data/criteria/cara_2021_fiber.txt \
  --stage l1 \
  --output outputs/fiber_l1_full

PYTHONPATH=python_pipeline/src python3 -m meta_screen.screener \
  --input data/processed/l2/cara_2021_fiber.csv \
  --criteria data/criteria/cara_2021_fiber.txt \
  --stage l2 \
  --output outputs/fiber_l2_full
```

Each run creates a timestamped output directory containing at least:

- `predictions.csv`
- `cost.json`

You can then point `paper_analysis.py` at the new `predictions.csv`.

## Notes for readers who mainly use R

- The Python code is organized much like a small package of reusable analysis scripts.
- The bottom of each file contains the command-line entry point when that file is runnable as a script.
- High-level comments were added near the top of the Python files to explain how each module fits into the overall workflow.
- The most important files to read in order are:
  1. `screener.py`
  2. `metrics.py`
  3. `paper_analysis.py`
  4. `paper_significance.py`
  5. `r_analysis/fiber_meta_analysis_audit.Rmd`

## What was intentionally excluded

This bundle excludes repo material that is not needed to understand or rerun the Fiber manuscript workflow, for example:

- manuscript source files
- unrelated datasets
- unrelated exploratory files
- editor settings
- Python cache files
