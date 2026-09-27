"""Build the public v1.1 validation package from local derived outputs.

The package contains tabular results, audit metadata, and derived plots. It
deliberately excludes article PDFs, supplementary files, publisher figures,
cell-crop atlases, and API credentials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


VERSION = "v1.1-revision-data"
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".tex", ".txt"}

BENCHMARK_FILES = [
    "benchmark_cell_level_validation.csv",
    "benchmark_cell_visual_features.csv",
    "benchmark_error_by_cell_size.csv",
    "benchmark_error_by_panel.csv",
    "benchmark_error_by_text_height_proxy.csv",
    "benchmark_error_by_type.csv",
    "benchmark_error_by_value_range.csv",
    "benchmark_error_by_visual_contrast.csv",
    "benchmark_extraction_summary.csv",
    "benchmark_failure_boundary_summary.csv",
    "benchmark_failure_cases.csv",
    "benchmark_manifest.csv",
    "benchmark_manifest.json",
    "benchmark_method_notes.md",
    "benchmark_numeric_ground_truth.csv",
    "benchmark_overall_results.csv",
    "benchmark_overall_results.json",
    "benchmark_panel_results.csv",
    "benchmark_predictions.csv",
    "benchmark_run_metadata.json",
    "benchmark_stage_accuracy.csv",
    "disagreement_refinement_audit.csv",
    "structure_detection_results.csv",
]

REVIEWER3_FILES = [
    "ablation_panel_results.csv",
    "ablation_protocol.json",
    "ablation_summary.csv",
    "structural_fault_injection.csv",
]

REVIEWER7_FILES = [
    "adjacent_column_error_analysis.csv",
    "cell_level_validation.csv",
    "error_by_cell_size.csv",
    "error_by_panel.csv",
    "error_by_text_height_proxy.csv",
    "error_by_transition.csv",
    "error_by_type.csv",
    "error_by_value_range.csv",
    "error_by_visual_contrast.csv",
    "failure_boundary_summary.csv",
    "failure_cases.csv",
    "ground_truth_status.json",
    "image_proxy_metrics.csv",
    "manual_cell_validation_from_paper.csv",
    "manual_cell_validation_template.csv",
    "pdf_ground_truth_comparison_summary.json",
    "source_figure_ground_truth.csv",
    "unique_error_diagnostic.csv",
    "unique_error_diagnostic.json",
    "error_5_to_6_glyph_comparison.png",
    "error_rate_by_cell_density.png",
    "error_rate_by_cell_size.png",
    "error_rate_by_value_range.png",
    "error_rate_by_visual_contrast.png",
    "error_type_distribution.png",
    "transition_error_rate.png",
]

STAGE56_FILES = [
    "stage5_reference_manifest.csv",
    "stage56_eligibility.csv",
    "stage56_exclusions.csv",
    "stage56_failure_cases.csv",
    "stage56_run_metadata.json",
    "stage56_summary.csv",
    "stage6_integrated_cells.csv",
    "stage6_panel_audit.csv",
]

RESULT_FIGURES = [
    "results_a_integer_agreement.png",
    "results_b_evaluation_coverage.png",
    "results_c_coordinate_error.png",
]


def _safe_destination(destination: Path, repository_root: Path) -> Path:
    destination = destination.resolve()
    repository_root = repository_root.resolve()
    if destination == repository_root or repository_root not in destination.parents:
        raise ValueError(f"Release destination must be inside repository: {destination}")
    return destination


def _sanitize_text(text: str, project_root: Path) -> str:
    variants = {
        str(project_root),
        str(project_root).replace("\\", "\\\\"),
        str(project_root).replace("\\", "/"),
    }
    for value in sorted(variants, key=len, reverse=True):
        text = text.replace(value, "<PROJECT_ROOT>")
    text = re.sub(r"sk-[A-Za-z0-9_-]{10,}", "<REDACTED_API_KEY>", text)
    text = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,\"']+", r"\1<REDACTED>", text)
    return text


def _copy(source: Path, destination: Path, project_root: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8-sig")
        destination.write_text(_sanitize_text(text, project_root), encoding="utf-8")
    else:
        shutil.copy2(source, destination)


def _copy_list(project_root: Path, source_dir: Path, names: list[str], destination: Path) -> None:
    for name in names:
        _copy(source_dir / name, destination / name, project_root)


def _write_readme(destination: Path, overall: dict, stage56: dict, sources: list[dict]) -> None:
    source_lines = "\n".join(
        f"- `{row['paper_id']}` — DOI [`{row['doi']}`](https://doi.org/{row['doi']}); "
        + (row["source_url"] if row["source_url"].startswith("http") else f"https://doi.org/{row['doi']}")
        for row in sources
    )
    text = f"""# ELISpot Validation Dataset {VERSION}

This revision package matches the generalized structure-constrained heatmap
pipeline and the revised evaluation. It contains derived tables, validation
records, prompt and inference audit files, and plots. It does not redistribute
article PDFs, supplementary files, publisher figure images, or API keys.

## Reported evaluation

- 7 papers, 10 heatmap panels, and {overall['total_cells']:,} cells in the Stage 4 development benchmark.
- Structure recovered exactly for {overall['structure_exact_panels']}/10 panels.
- {overall['printed_integer_exact_matches']}/{overall['printed_integer_ground_truth_cells']} printed integers matched after the documented disagreement reread.
- The automatic route before the local reread matched {overall['pre_local_reread_printed_integer_exact_matches']}/{overall['printed_integer_ground_truth_cells']} printed integers.
- Four colour-only panels ({overall['color_only_cells']} cells) were correctly treated as numeric abstentions.
- Stage 5-6 was applicable to {stage56['papers_stage56_included']} papers and {stage56['panels_stage56_included']} panels, producing {stage56['integrated_cells']} integrated cells; three papers were excluded with reasons recorded in `stage56/stage56_exclusions.csv`.
- Numeric confidence thresholds were not used. `high/medium/low` values remain model self-reports only.

The 10-panel set was used during pipeline development and is labeled a
development benchmark, not a held-out test set. Accuracy is calculated only
where source-backed ground truth is available. Colour-only cells are excluded
from numeric accuracy and reported as abstentions.

## Layout

- `benchmark/`: multi-paper Stage 4 predictions, ground truth, structural results, error tables, and failure cases.
- `reviewer3_ablation/`: internal ablation of unconstrained and structure-constrained variants.
- `reviewer7_error_analysis/`: the 311-cell manual validation set, cell proxies, grouped errors, and plots.
- `stage56/`: cross-paper eligibility, normalized reference tables, integrated records, audits, and exclusions.
- `ai_audit/`: prompt manifest plus raw and parsed Qwen responses used by the benchmark.
- `figures/`: derived result figures used in the revised manuscript.

## Source articles

{source_lines}

The `source_url` and `doi` columns provide provenance. Obtain original articles,
supplements, and figures from their publishers or repositories under the
applicable terms.
"""
    (destination / "README.md").write_text(text, encoding="utf-8")


def _manifest(destination: Path) -> None:
    rows = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.csv":
            rows.append({
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    with (destination / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def build(project_root: Path, repository_root: Path, destination: Path) -> Path:
    project_root = project_root.resolve()
    repository_root = repository_root.resolve()
    destination = _safe_destination(destination, repository_root)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    benchmark = project_root / "generic_heatmap_benchmark_10"
    _copy_list(project_root, benchmark, BENCHMARK_FILES, destination / "benchmark")
    _copy_list(project_root, project_root / "reviewer3_internal_ablation", REVIEWER3_FILES, destination / "reviewer3_ablation")
    _copy_list(project_root, project_root / "reviewer_comment_7_outputs", REVIEWER7_FILES, destination / "reviewer7_error_analysis")
    _copy_list(project_root, project_root / "stage56_cross_paper", STAGE56_FILES, destination / "stage56")
    _copy_list(project_root, project_root, RESULT_FIGURES, destination / "figures")

    for source in sorted((project_root / "stage56_cross_paper" / "stage5").rglob("*.csv")):
        relative = source.relative_to(project_root / "stage56_cross_paper" / "stage5")
        _copy(source, destination / "stage56" / "normalized_stage5" / relative, project_root)

    audit_source = benchmark / "ai_audit"
    _copy(audit_source / "prompt_manifest.json", destination / "ai_audit" / "prompt_manifest.json", project_root)
    _copy(audit_source / "ai_inference_log.jsonl", destination / "ai_audit" / "ai_inference_log.jsonl", project_root)
    for folder in ["raw_responses", "parsed_json"]:
        for source in sorted((audit_source / folder).glob("*")):
            if source.is_file():
                _copy(source, destination / "ai_audit" / folder / source.name, project_root)

    overall = json.loads((benchmark / "benchmark_overall_results.json").read_text(encoding="utf-8"))
    stage56 = json.loads((project_root / "stage56_cross_paper" / "stage56_run_metadata.json").read_text(encoding="utf-8"))
    manifest_rows = []
    with (benchmark / "benchmark_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            manifest_rows.append({key: row.get(key, "") for key in ["paper_id", "doi", "source_url"]})
    unique_sources = list({row["paper_id"]: row for row in manifest_rows}.values())
    _write_readme(destination, overall, stage56, unique_sources)

    dictionary = {
        "version": VERSION,
        "ground_truth_policy": "Only source-backed or manually transcribed values are used; no ground truth is fabricated.",
        "confidence_policy": "No numeric confidence threshold is used; qualitative confidence is model_self_reported_confidence.",
        "colour_policy": "Colour-only heatmap cells are not converted to numeric values.",
        "proxy_fields": {
            "text_height_proxy_px": "Connected-component estimate in pixels; not a true font size.",
            "visual_contrast_proxy": "90th minus 10th grayscale percentile inside a border-trimmed cell.",
            "cell_area_px2": "Tool-detected cell width multiplied by height in pixels squared.",
        },
    }
    (destination / "DATA_DICTIONARY.json").write_text(json.dumps(dictionary, indent=2), encoding="utf-8")
    _manifest(destination)

    archive_base = repository_root / "dist" / VERSION
    archive_base.parent.mkdir(exist_ok=True)
    archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=destination.parent, base_dir=destination.name))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    destination = args.destination or repository_root / "validation" / "v1.1"
    print(build(args.project_root, repository_root, destination))


if __name__ == "__main__":
    main()
