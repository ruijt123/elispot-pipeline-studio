# ELISpot Pipeline Studio

A Streamlit and command-line implementation of a six-stage,
structure-constrained workflow for recovering cell-level experimental data from
scientific heatmaps and linking it to supplementary reference tables.

## What changed in v1.1

- A generic heatmap proposal ensemble combines grid, colour-region,
  edge-density, and full-figure multimodal localization.
- OpenCV fixes the grid structure before Qwen reads cell content. The tool
  assigns coordinates, validates shape and index coverage, and records raw
  validity separately from post-normalization usability.
- Every Qwen task records a prompt version, full prompt and schema, model,
  `temperature=0`, `top_p=0.01`, retry count, raw response, parsed JSON,
  validation errors, and manual-review state.
- Parse or schema failures retry at most once. Remaining failures enter manual
  review.
- Qualitative confidence is stored as `model_self_reported_confidence`.
  Numeric confidence thresholds are not used.
- Colour-only heatmaps are detected structurally, but numeric values are left
  missing unless publisher source data provide them.
- Stage 1-4 can run without a supplement. Stage 5-6 runs only when a compatible
  supplementary table is available.
- A separate cell-level analysis command creates a manual validation template
  when ground truth is absent and computes error boundaries only from supplied,
  source-backed ground truth.

## Evaluation data

The synchronized [`v1.1` validation directory](validation/v1.1/README.md)
contains the revised development benchmark and audit artifacts:

- 7 papers, 10 panels, and 1,876 heatmap cells;
- exact structure for 10/10 panels;
- 398/399 printed integers after the documented disagreement reread;
- 339/399 printed integers on the automatic route before local rereading;
- 4 colour-only panels and 977 numeric abstentions;
- Stage 5-6 integration for 4 papers, 7 panels, and 980 cells, with explicit
  exclusion reasons for the remaining papers.

The 10-panel collection was used during development and is not presented as a
held-out test set. Original articles, supplements, and publisher figures are
not redistributed.

## Local Windows launch

```powershell
.\start.ps1
```

Alternatively:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Enter the DashScope key in the password field. The key remains in process
memory and is not written to the run directory or audit files.

## Command line

Run heatmap detection and extraction through Stage 4 without a supplement:

```powershell
python pipeline.py --paper main.pdf --stop-after stage4
```

Run all six stages with a supplement:

```powershell
python pipeline.py --paper main.pdf --supplement supplement.xlsx
```

The revised notebook is
[`notebooks/module1_generic_heatmap.ipynb`](notebooks/module1_generic_heatmap.ipynb).
Reusable deterministic structure detection is available in
[`elispot_pipeline/heatmap_structure.py`](elispot_pipeline/heatmap_structure.py).

## Cell-level error analysis

Without ground truth, the command writes a prefilled manual validation template
and does not calculate accuracy:

```powershell
python tools/cell_level_error_analysis.py `
  --predictions pipeline_runs/article/stage4/Stage4_Cell_Records.xlsx `
  --output reviewer_comment_7_outputs
```

After manual validation, pass the completed source-backed CSV:

```powershell
python tools/cell_level_error_analysis.py `
  --predictions pipeline_runs/article/stage4/Stage4_Cell_Records.xlsx `
  --ground-truth manual_cell_validation.csv `
  --output reviewer_comment_7_outputs
```

The analysis reports exact match, absolute error, error type, adjacent-column
patterns, cell-size and density groups, text-height and visual-contrast proxies,
value quantiles, valid timepoint transitions, and failure-boundary summaries.
It never fabricates ground truth or converts heatmap colour to numeric values.

## Rebuild the public validation package

```powershell
python tools/build_release_package.py `
  --project-root C:\path\to\local-derived-outputs
```

This command copies only derived tables, audit responses, and plots, sanitizes
local paths and key-like strings, creates checksums, and writes
`dist/v1.1-revision-data.zip`.

## Security

Do not commit API keys. The app and notebook read `DASHSCOPE_API_KEY` at runtime.
The repository is intended for local or trusted single-user deployment.
