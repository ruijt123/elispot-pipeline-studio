"""Cross-paper Stage 5/6 reconstruction using cached Stage 4 benchmark cells.

This module deliberately does not call Qwen or infer numbers from colour.  It
parses publisher-supplied supplementary tables, validates explicit joins, and
keeps image predictions separate from supplementary source values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path.cwd()


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _unique_join(values: Iterable[Any]) -> str:
    seen: list[str] = []
    for value in values:
        item = _text(value)
        if item and item not in seen:
            seen.append(item)
    return " | ".join(seen)


def _safe_number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _base_stage6_row(pred: pd.Series, paper_id: str) -> dict[str, Any]:
    predicted = _safe_number(pred.get("predicted_value"))
    return {
        "benchmark_id": pred["benchmark_id"],
        "paper_id": paper_id,
        "figure": pred["figure"],
        "panel": pred["panel"],
        "row_index": int(pred["row_index"]),
        "column_index": int(pred["column_index"]),
        "stage4_predicted_value": predicted,
        "stage4_value_text": _text(pred.get("value_text")),
        "stage4_extraction_status": _text(pred.get("extraction_status")),
        "stage4_numeric_from_image": predicted is not None,
        "stage4_color_only_abstention": _text(pred.get("extraction_status")) == "abstained_color_only",
        "supplementary_source_value": None,
        "supplementary_source_value_role": "",
        "supplementary_source_units": "",
        "supplementary_source_file": "",
        "x_label": "",
        "y_label": "",
        "axis_role_x": "",
        "axis_role_y": "",
        "join_key": "",
        "join_strategy": "",
        "join_scope": "",
        "stage5_link_status": "",
        "stage6_join_status": "",
        "join_candidate_count": 0,
        "manual_review": False,
        "manual_review_reason": "",
        "source_display_rounded_value": None,
        "source_expected_printed_value": None,
        "image_source_display_exact_match": None,
        "patient_id": "",
        "condition": "",
        "timepoint": "",
        "epitope_id": "",
        "peptide_id": "",
        "peptide_pool": "",
        "peptide_sequence": "",
        "gene": "",
        "mutation_or_substitution": "",
        "mhc_or_hla": "",
        "treatment": "",
        "source_normalized_value": None,
        "supplementary_pool_mean": None,
        "supplementary_pool_n": None,
        "supplementary_pool_er_positive": None,
        "supplementary_pool_dfr2x_positive": None,
        "supplementary_pool_selected": "",
        "reference_record_count": 0,
        "reference_note": "",
    }


def _assert_stage4_shape(predictions: pd.DataFrame, manifest: pd.DataFrame) -> None:
    duplicate_count = int(
        predictions.duplicated(["benchmark_id", "row_index", "column_index"]).sum()
    )
    if duplicate_count:
        raise ValueError(f"Stage 4 cache contains {duplicate_count} duplicate coordinates")

    for item in manifest.itertuples(index=False):
        subset = predictions[predictions["benchmark_id"] == item.benchmark_id]
        expected = int(item.expected_rows) * int(item.expected_columns)
        if len(subset) != expected:
            raise ValueError(
                f"{item.benchmark_id}: expected {expected} cached cells, found {len(subset)}"
            )
        expected_coords = {
            (r, c)
            for r in range(int(item.expected_rows))
            for c in range(int(item.expected_columns))
        }
        observed_coords = set(
            zip(subset["row_index"].astype(int), subset["column_index"].astype(int))
        )
        if expected_coords != observed_coords:
            raise ValueError(f"{item.benchmark_id}: cached coordinate coverage is incomplete")


def _build_fan(
    project_root: Path, predictions: pd.DataFrame, output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = (
        project_root
        / "stage5_final_universal_tool_table_extraction"
        / "Epitope_Reference_Table_FINAL.csv"
    )
    ref = pd.read_csv(source)
    ref["Epitope_ID"] = pd.to_numeric(ref["Epitope_ID"], errors="raise").astype(int)
    ref_out = output_root / "stage5" / "fan_2024" / "epitope_reference_records.csv"
    _write_csv(ref, ref_out)

    ct26 = ref[ref["Vaccine_or_Antigen_Set"] == "LPP-CT26"].copy()
    by_epitope = {int(row.Epitope_ID): row for row in ct26.itertuples(index=False)}
    if sorted(by_epitope) != list(range(1, 21)):
        raise ValueError("Fan Stage 5 reference table must contain unique CT26 epitopes 1-20")

    subset13 = [1, 3, 4, 5, 7, 8, 10, 11, 13, 15, 16, 18, 19]
    panel_config = {
        "B01": {
            "rows": ["0", "6", "24", "48", "72", "96", "168"],
            "cols": list(range(1, 20)),
            "x_role": "epitope",
            "y_role": "timepoint_hours",
        },
        "B02": {
            "rows": ["s.c.", "i.m.", "i.d."],
            "cols": subset13,
            "x_role": "epitope",
            "y_role": "administration_route",
        },
        "B03": {
            "rows": ["PBS", "Single site", "Multisite"],
            "cols": subset13,
            "x_role": "epitope",
            "y_role": "vaccination_site_condition",
        },
        "B04": {
            "rows": ["PBS", "1 μg", "3 μg", "10 μg", "30 μg"],
            "cols": list(range(1, 21)),
            "x_role": "epitope",
            "y_role": "vaccine_dose",
        },
    }

    integrated: list[dict[str, Any]] = []
    for benchmark_id, cfg in panel_config.items():
        panel = predictions[predictions["benchmark_id"] == benchmark_id]
        for _, pred in panel.iterrows():
            row_index = int(pred["row_index"])
            col_index = int(pred["column_index"])
            epitope_id = int(cfg["cols"][col_index])
            label = cfg["rows"][row_index]
            annotation = by_epitope[epitope_id]
            out = _base_stage6_row(pred, "fan_2024")
            out.update(
                {
                    "x_label": f"Epitope {epitope_id}",
                    "y_label": label,
                    "axis_role_x": cfg["x_role"],
                    "axis_role_y": cfg["y_role"],
                    "join_key": f"LPP-CT26__Epitope_{epitope_id}",
                    "join_strategy": "explicit_epitope_id_from_figure_axis",
                    "join_scope": "cell_semantic_annotation",
                    "stage5_link_status": "matched_unique",
                    "stage6_join_status": "complete",
                    "join_candidate_count": 1,
                    "condition": label if benchmark_id != "B01" else "",
                    "timepoint": label if benchmark_id == "B01" else "",
                    "epitope_id": epitope_id,
                    "peptide_sequence": _text(annotation.Sequence),
                    "gene": _text(annotation.Gene),
                    "mutation_or_substitution": _text(annotation.Mutation_or_Substitution),
                    "mhc_or_hla": _text(annotation.MHC_or_HLA),
                    "reference_record_count": 1,
                    "reference_note": (
                        "Supplement supplies epitope annotations; the cell value remains the "
                        "cached Stage 4 image reading."
                    ),
                    "supplementary_source_file": source.name,
                }
            )
            integrated.append(out)

    reference_manifest = [
        {
            "paper_id": "fan_2024",
            "reference_type": "epitope_annotation",
            "source_file": str(source),
            "output_file": str(ref_out),
            "record_count": len(ref),
            "join_keys": "Vaccine_or_Antigen_Set + Epitope_ID",
            "numeric_cell_values_available": False,
        }
    ]
    return integrated, reference_manifest


def _week_number(name: Any) -> int:
    match = re.search(r"(\d+)", _text(name))
    if not match:
        raise ValueError(f"Cannot parse study week from {name!r}")
    return int(match.group(1))


def _build_braun(
    predictions: pd.DataFrame, source_dir: Path, output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_fig = source_dir / "braun_MOESM10.xlsx"
    source_table = source_dir / "braun_MOESM4.xlsx"
    normalized = pd.read_excel(source_fig, sheet_name="2a", header=1)
    normalized = normalized[normalized["ID"].notna()].copy()
    week_columns = [column for column in normalized.columns if "week" in _text(column).lower()]
    if len(normalized) != 36 or len(week_columns) != 7:
        raise ValueError("Braun source Fig. 2a must contain 36 rows and 7 week columns")

    raw = pd.read_excel(source_table, sheet_name="Ex Vivo")
    raw = raw[raw["Patient ID"].notna()].copy()
    raw_week_columns = [column for column in raw.columns if "week" in _text(column).lower()]
    if len(raw) != 36 or len(raw_week_columns) != 7:
        raise ValueError("Braun ex vivo table must contain 36 rows and 7 week columns")

    norm_keys = list(
        zip(normalized["ID"].astype(int), normalized["Peptide Pool"].map(_text))
    )
    raw_keys = list(zip(raw["Patient ID"].astype(int), raw["Vaccine Pool"].map(_text)))
    if norm_keys != raw_keys:
        raise ValueError("Braun Figure 2a and Ex Vivo row order do not agree")
    if [_week_number(c) for c in week_columns] != [_week_number(c) for c in raw_week_columns]:
        raise ValueError("Braun Figure 2a and Ex Vivo week order do not agree")

    peptide_ref = pd.read_excel(source_table, sheet_name="In Vitro")
    peptide_ref = peptide_ref[peptide_ref["Patient_ID"].notna()].copy()
    peptide_ref["Patient_ID"] = peptide_ref["Patient_ID"].astype(int)
    peptide_out = output_root / "stage5" / "braun_2025" / "peptide_reference_records.csv"
    _write_csv(peptide_ref, peptide_out)

    peptide_groups: dict[tuple[int, str], dict[str, Any]] = {}
    for (patient, pool), group in peptide_ref.groupby(["Patient_ID", "Pool"], dropna=False):
        peptide_groups[(int(patient), f"Pool {_text(pool).replace('Pool ', '')}")] = {
            "peptide_count": len(group),
            "peptide_ids": _unique_join(group["Peptide_ID"]),
            "peptide_sequences": _unique_join(group["Vaccine_Peptide"]),
            "genes": _unique_join(group["Gene_and_Protein_Change"]),
            "hla": _unique_join(group["HLA_of_best_short_epitope"]),
        }

    matrix_records: list[dict[str, Any]] = []
    for row_index, (norm_row, raw_row) in enumerate(
        zip(normalized.itertuples(index=False), raw.itertuples(index=False))
    ):
        patient = int(norm_row[0])
        pool = _text(norm_row[1])
        treatment = _text(raw_row[2])
        annotations = peptide_groups.get((patient, pool), {})
        for column_index, (norm_column, raw_column) in enumerate(
            zip(week_columns, raw_week_columns)
        ):
            normalized_value = _safe_number(normalized.iloc[row_index][norm_column])
            raw_value = _safe_number(raw.iloc[row_index][raw_column])
            matrix_records.append(
                {
                    "row_index": row_index,
                    "column_index": column_index,
                    "patient_id": patient,
                    "peptide_pool": pool,
                    "treatment": treatment,
                    "week": _week_number(norm_column),
                    "raw_ex_vivo_mean_background_subtracted": raw_value,
                    "figure_row_normalized_value": normalized_value,
                    **annotations,
                }
            )
    matrix_ref = pd.DataFrame(matrix_records)
    matrix_out = output_root / "stage5" / "braun_2025" / "matrix_source_records.csv"
    _write_csv(matrix_ref, matrix_out)

    lookup = {
        (int(row.row_index), int(row.column_index)): row
        for row in matrix_ref.itertuples(index=False)
    }
    panel = predictions[predictions["benchmark_id"] == "B05"]
    integrated: list[dict[str, Any]] = []
    for _, pred in panel.iterrows():
        key = (int(pred["row_index"]), int(pred["column_index"]))
        ref_row = lookup[key]
        predicted = _safe_number(pred.get("predicted_value"))
        rounded_raw = (
            float(np.floor(ref_row.raw_ex_vivo_mean_background_subtracted + 0.5))
            if ref_row.raw_ex_vivo_mean_background_subtracted is not None
            and not pd.isna(ref_row.raw_ex_vivo_mean_background_subtracted)
            else None
        )
        is_expected_printed_cell = (
            ref_row.figure_row_normalized_value is not None
            and not pd.isna(ref_row.figure_row_normalized_value)
            and bool(np.isclose(float(ref_row.figure_row_normalized_value), 1.0))
        )
        expected_printed_value = rounded_raw if is_expected_printed_cell else None
        if expected_printed_value is not None or predicted is not None:
            display_match = bool(
                expected_printed_value is not None
                and predicted is not None
                and predicted == expected_printed_value
            )
        else:
            display_match = None
        source_mismatch = display_match is False
        out = _base_stage6_row(pred, "braun_2025")
        out.update(
            {
                "x_label": f"Week {int(ref_row.week)}",
                "y_label": f"Patient {int(ref_row.patient_id)} / {ref_row.peptide_pool}",
                "axis_role_x": "study_week",
                "axis_role_y": "patient_and_vaccine_pool",
                "join_key": (
                    f"patient={int(ref_row.patient_id)}|pool={ref_row.peptide_pool}|"
                    f"week={int(ref_row.week)}"
                ),
                "join_strategy": "shape_checked_patient_pool_week_order",
                "join_scope": "cell_numeric_source_and_semantic_annotation",
                "stage5_link_status": "matched_unique",
                "stage6_join_status": (
                    "complete_with_source_mismatch_review" if source_mismatch else "complete"
                ),
                "join_candidate_count": 1,
                "manual_review": source_mismatch,
                "manual_review_reason": (
                    "Printed Stage 4 token does not match the rounded source value at the "
                    "assigned coordinate; inspect row assignment against the labelled figure."
                    if source_mismatch
                    else ""
                ),
                "patient_id": int(ref_row.patient_id),
                "condition": ref_row.peptide_pool,
                "timepoint": int(ref_row.week),
                "peptide_pool": ref_row.peptide_pool,
                "peptide_id": _text(ref_row.peptide_ids),
                "peptide_sequence": _text(ref_row.peptide_sequences),
                "gene": _text(ref_row.genes),
                "mhc_or_hla": _text(ref_row.hla),
                "treatment": _text(ref_row.treatment),
                "supplementary_source_value": ref_row.raw_ex_vivo_mean_background_subtracted,
                "supplementary_source_value_role": "ex_vivo_mean_background_subtracted",
                "supplementary_source_units": "reported ELISpot response",
                "supplementary_source_file": source_table.name,
                "source_normalized_value": ref_row.figure_row_normalized_value,
                "source_display_rounded_value": expected_printed_value,
                "source_expected_printed_value": expected_printed_value,
                "image_source_display_exact_match": display_match,
                "reference_record_count": int(ref_row.peptide_count),
                "reference_note": (
                    "Image colour encodes the patient-normalized value; one cell per patient "
                    "prints the rounded raw patient maximum."
                ),
            }
        )
        integrated.append(out)

    reference_manifest = [
        {
            "paper_id": "braun_2025",
            "reference_type": "figure_2a_cell_source",
            "source_file": str(source_fig),
            "output_file": str(matrix_out),
            "record_count": len(matrix_ref),
            "join_keys": "Patient ID + Peptide Pool + Week",
            "numeric_cell_values_available": True,
        },
        {
            "paper_id": "braun_2025",
            "reference_type": "peptide_annotation",
            "source_file": str(source_table),
            "output_file": str(peptide_out),
            "record_count": len(peptide_ref),
            "join_keys": "Patient_ID + Pool",
            "numeric_cell_values_available": False,
        },
    ]
    return integrated, reference_manifest


def _build_zhang(
    predictions: pd.DataFrame, source_dir: Path, output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = source_dir / "zhang_mmc7.xlsx"
    binding = pd.read_excel(source, sheet_name="Q1_Replacement_Percentage")
    elispot = pd.read_excel(source, sheet_name="ELISPOT_Results")
    if len(binding) != 38 or len(elispot) != 38:
        raise ValueError("Zhang Table S6 must contain 38 peptides")
    if not (
        binding[["Gene", "Peptide"]].astype(str).values
        == elispot[["Gene", "Peptide"]].astype(str).values
    ).all():
        raise ValueError("Zhang binding and ELISpot peptide order do not agree")

    donor_columns = [f"Donor{i}_value" for i in range(1, 5)]
    long_records: list[dict[str, Any]] = []
    for column_index, row in elispot.iterrows():
        q1 = _safe_number(binding.iloc[column_index]["Q1_Replacement_Percentage"])
        for row_index, donor_column in enumerate(donor_columns):
            long_records.append(
                {
                    "row_index": row_index,
                    "column_index": column_index,
                    "donor": f"Donor{row_index + 1}",
                    "gene": _text(row["Gene"]),
                    "peptide": _text(row["Peptide"]),
                    "q1_replacement_percentage": q1,
                    "elispot_sfu_per_100000_cells": _safe_number(row[donor_column]),
                }
            )
    source_records = pd.DataFrame(long_records)
    source_out = output_root / "stage5" / "zhang_2024" / "table_s6_cell_source_records.csv"
    _write_csv(source_records, source_out)

    lookup = {
        (int(row.row_index), int(row.column_index)): row
        for row in source_records.itertuples(index=False)
    }
    panel = predictions[predictions["benchmark_id"] == "B08"]
    integrated: list[dict[str, Any]] = []
    for _, pred in panel.iterrows():
        key = (int(pred["row_index"]), int(pred["column_index"]))
        ref_row = lookup[key]
        out = _base_stage6_row(pred, "zhang_2024")
        out.update(
            {
                "x_label": ref_row.peptide,
                "y_label": ref_row.donor,
                "axis_role_x": "peptide",
                "axis_role_y": "donor",
                "join_key": f"donor={ref_row.donor}|peptide={ref_row.peptide}",
                "join_strategy": "exact_4_by_38_source_table_order_with_full_figure_labels",
                "join_scope": "cell_numeric_source_and_semantic_annotation",
                "stage5_link_status": "matched_unique",
                "stage6_join_status": "complete",
                "join_candidate_count": 1,
                "condition": ref_row.donor,
                "peptide_id": f"PEP{int(ref_row.column_index) + 1}",
                "peptide_sequence": ref_row.peptide,
                "gene": ref_row.gene,
                "supplementary_source_value": ref_row.elispot_sfu_per_100000_cells,
                "supplementary_source_value_role": "table_s6_elispot_value",
                "supplementary_source_units": "SFU per 100,000 cells",
                "supplementary_source_file": source.name,
                "reference_record_count": 1,
                "reference_note": (
                    "Stage 4 correctly abstained from converting heatmap colour into a number; "
                    "Table S6 is retained as a separate source-data field."
                ),
            }
        )
        integrated.append(out)

    reference_manifest = [
        {
            "paper_id": "zhang_2024",
            "reference_type": "table_s6_cell_source",
            "source_file": str(source),
            "output_file": str(source_out),
            "record_count": len(source_records),
            "join_keys": "Donor order + peptide order",
            "numeric_cell_values_available": True,
        }
    ]
    return integrated, reference_manifest


def _read_hhv6_pool_table(source: Path) -> pd.DataFrame:
    frame = pd.read_excel(source, sheet_name="S2 Table", header=3)
    frame = frame[pd.to_numeric(frame["Pool#"], errors="coerce").notna()].copy()
    frame["Pool#"] = pd.to_numeric(frame["Pool#"], errors="raise").astype(int)
    if sorted(frame["Pool#"].tolist()) != list(range(1, 54)):
        raise ValueError("HHV6 S2 Table must contain unique pools 1-53")
    return frame


def _build_hhv6(
    predictions: pd.DataFrame, source_dir: Path, output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_pool = source_dir / "hhv6_s004.xlsx"
    source_all_peptides = source_dir / "hhv6_s003.xlsx"
    source_positive_peptides = source_dir / "hhv6_s005.xlsx"
    pool = _read_hhv6_pool_table(source_pool)

    all_peptides = pd.read_excel(source_all_peptides, sheet_name="S1 Table", header=2)
    all_peptides = all_peptides[pd.to_numeric(all_peptides["Pool#8"], errors="coerce").notna()].copy()
    all_peptides["Pool#8"] = pd.to_numeric(all_peptides["Pool#8"], errors="raise").astype(int)

    positive_raw = pd.read_excel(source_positive_peptides, sheet_name="S3 Table", header=None)
    positive = positive_raw.iloc[4:, :8].copy()
    positive.columns = [
        "Protein",
        "Pool#",
        "Peptide Id",
        "Sequence",
        "Mean",
        "n",
        "ER",
        "DFR2x",
    ]
    positive = positive[pd.to_numeric(positive["Pool#"], errors="coerce").notna()].copy()
    positive["Pool#"] = pd.to_numeric(positive["Pool#"], errors="raise").astype(int)

    all_groups = {
        int(pool_id): {
            "all_peptide_ids": _unique_join(group["Peptide Name6"]),
            "all_peptide_sequences": _unique_join(group["Peptide sequence5"]),
            "all_peptide_count": len(group),
        }
        for pool_id, group in all_peptides.groupby("Pool#8")
    }
    positive_groups = {
        int(pool_id): {
            "positive_peptide_ids": _unique_join(group["Peptide Id"]),
            "positive_peptide_sequences": _unique_join(group["Sequence"]),
            "positive_peptide_count": len(group),
        }
        for pool_id, group in positive.groupby("Pool#")
    }

    # Figure 3B explicitly sorts proteins by overall response and pools within
    # each protein by average response.  The order below is transcribed from
    # the published axis, not inferred from heatmap colour.
    figure_pool_order = [
        28, 27, 26, 29, 24, 25,
        17, 19, 20, 16, 18,
        7, 5, 4, 6,
        21, 23, 22,
        8, 10, 9,
        11, 13, 12, 15, 14,
        1, 2, 3,
        30, 38, 32, 34, 37, 31, 33, 39, 36, 35,
        46, 47, 48, 42, 45, 44, 43, 40,
        51, 49, 41, 53, 50, 52,
    ]
    if len(figure_pool_order) != 53 or len(set(figure_pool_order)) != 53:
        raise ValueError("HHV6 Figure 3B pool order must cover pools 1-53 exactly once")
    donors = ["037", "118", "130", "131", "132"]

    pool_by_id = {int(row["Pool#"]): row for _, row in pool.iterrows()}
    pool_records: list[dict[str, Any]] = []
    for row_index, pool_id in enumerate(figure_pool_order):
        base = pool_by_id[pool_id]
        pool_records.append(
            {
                "figure_row_index": row_index,
                "pool_id": pool_id,
                "protein": _text(base.get("Protein")),
                "pool_mean_sfu_per_1e6_cells": _safe_number(base.get("Mean")),
                "pool_n": _safe_number(base.get("n")),
                "pool_er_positive": _safe_number(base.get("ER")),
                "pool_dfr2x_positive": _safe_number(base.get("DFR2x")),
                "selected_for_further_analysis": _text(base.get("Selected3")),
                **all_groups.get(pool_id, {}),
                **positive_groups.get(pool_id, {}),
            }
        )
    pool_ref = pd.DataFrame(pool_records)
    pool_out = output_root / "stage5" / "hhv6_2015" / "pool_reference_records.csv"
    _write_csv(pool_ref, pool_out)

    pool_lookup = {int(row.figure_row_index): row for row in pool_ref.itertuples(index=False)}
    panel = predictions[predictions["benchmark_id"] == "B10"]
    integrated: list[dict[str, Any]] = []
    for _, pred in panel.iterrows():
        row_index = int(pred["row_index"])
        column_index = int(pred["column_index"])
        ref_row = pool_lookup[row_index]
        donor = donors[column_index]
        out = _base_stage6_row(pred, "hhv6_2015")
        out.update(
            {
                "x_label": f"Donor {donor}",
                "y_label": f"{ref_row.protein} / Pool {int(ref_row.pool_id)}",
                "axis_role_x": "donor",
                "axis_role_y": "protein_and_peptide_pool",
                "join_key": f"pool={int(ref_row.pool_id)}|donor={donor}",
                "join_strategy": "published_figure_axis_order_plus_pool_number",
                "join_scope": "row_level_source_summary_and_cell_semantic_annotation",
                "stage5_link_status": "matched_unique_pool_record",
                "stage6_join_status": "complete_row_level_only",
                "join_candidate_count": 1,
                "patient_id": donor,
                "condition": f"Donor {donor}",
                "peptide_pool": int(ref_row.pool_id),
                "peptide_id": _text(getattr(ref_row, "all_peptide_ids", "")),
                "peptide_sequence": _text(getattr(ref_row, "all_peptide_sequences", "")),
                "gene": ref_row.protein,
                "supplementary_source_value": None,
                "supplementary_source_value_role": "donor_level_cell_value_not_available",
                "supplementary_source_units": "",
                "supplementary_source_file": source_pool.name,
                "supplementary_pool_mean": ref_row.pool_mean_sfu_per_1e6_cells,
                "supplementary_pool_n": ref_row.pool_n,
                "supplementary_pool_er_positive": ref_row.pool_er_positive,
                "supplementary_pool_dfr2x_positive": ref_row.pool_dfr2x_positive,
                "supplementary_pool_selected": ref_row.selected_for_further_analysis,
                "reference_record_count": int(getattr(ref_row, "all_peptide_count", 0) or 0),
                "reference_note": (
                    "S2 supplies pool-level summary statistics, not the five donor-level "
                    "cell values; no numeric cell accuracy is claimed."
                ),
            }
        )
        integrated.append(out)

    reference_manifest = [
        {
            "paper_id": "hhv6_2015",
            "reference_type": "pool_summary_and_peptide_annotation",
            "source_file": f"{source_pool}; {source_all_peptides}; {source_positive_peptides}",
            "output_file": str(pool_out),
            "record_count": len(pool_ref),
            "join_keys": "published Figure 3B row order + Pool#",
            "numeric_cell_values_available": False,
        }
    ]
    return integrated, reference_manifest


def _eligibility_rows(project_root: Path, output_root: Path) -> list[dict[str, Any]]:
    source_dir = output_root / "supplementary_sources"
    return [
        {
            "paper_id": "fan_2024",
            "benchmark_panels": "B01; B02; B03; B04",
            "supplement_status": "available_local_pdf_and_parsed_table",
            "relevant_structured_reference": True,
            "stage5_status": "complete",
            "stage6_status": "complete",
            "stage6_scope": "cell semantic annotation for 311 cells",
            "exclusion_reason": "",
            "evidence": str(project_root / "sciadv.adn9961_sm.pdf"),
        },
        {
            "paper_id": "braun_2025",
            "benchmark_panels": "B05",
            "supplement_status": "available_official_xlsx",
            "relevant_structured_reference": True,
            "stage5_status": "complete",
            "stage6_status": "complete",
            "stage6_scope": "cell numeric source plus peptide annotation for 252 cells",
            "exclusion_reason": "",
            "evidence": f"{source_dir / 'braun_MOESM4.xlsx'}; {source_dir / 'braun_MOESM10.xlsx'}",
        },
        {
            "paper_id": "lindner_2025",
            "benchmark_panels": "B06",
            "supplement_status": "available_but_not_linkable_to_panel_matrix",
            "relevant_structured_reference": False,
            "stage5_status": "excluded",
            "stage6_status": "excluded",
            "stage6_scope": "",
            "exclusion_reason": (
                "The supplementary tables describe patients, assay inclusion, antibody panels, "
                "and TCR records, but do not provide the Figure 1E 20×20 cell matrix or an "
                "unambiguous peptide-to-grid lookup."
            ),
            "evidence": str(source_dir / "PMC12458743_fullText.xml"),
        },
        {
            "paper_id": "haars_2021",
            "benchmark_panels": "B07",
            "supplement_status": "available_pdf_but_unrelated_to_clinical_matrix",
            "relevant_structured_reference": False,
            "stage5_status": "excluded",
            "stage6_status": "excluded",
            "stage6_scope": "",
            "exclusion_reason": (
                "Supplementary Tables S1-S2 report assay validation and algorithm agreement; "
                "they do not contain the Figure 5A patient-by-week clinical matrix."
            ),
            "evidence": str(source_dir / "haars_mmc1.pdf"),
        },
        {
            "paper_id": "zhang_2024",
            "benchmark_panels": "B08",
            "supplement_status": "available_official_xlsx",
            "relevant_structured_reference": True,
            "stage5_status": "complete",
            "stage6_status": "complete",
            "stage6_scope": "cell numeric source plus peptide annotation for 152 cells",
            "exclusion_reason": "",
            "evidence": str(source_dir / "zhang_mmc7.xlsx"),
        },
        {
            "paper_id": "geiger_2020",
            "benchmark_panels": "B09",
            "supplement_status": "images_only",
            "relevant_structured_reference": False,
            "stage5_status": "excluded",
            "stage6_status": "excluded",
            "stage6_scope": "",
            "exclusion_reason": (
                "The publisher supplementary package contains only three TIFF images and no "
                "structured reference table or Figure 2A cell source data."
            ),
            "evidence": str(source_dir / "PMC7076177_fullText.xml"),
        },
        {
            "paper_id": "hhv6_2015",
            "benchmark_panels": "B10",
            "supplement_status": "available_official_xlsx",
            "relevant_structured_reference": True,
            "stage5_status": "complete",
            "stage6_status": "complete_row_level_only",
            "stage6_scope": "pool-level summary and peptide annotation for 265 cells",
            "exclusion_reason": "",
            "evidence": (
                f"{source_dir / 'hhv6_s003.xlsx'}; {source_dir / 'hhv6_s004.xlsx'}; "
                f"{source_dir / 'hhv6_s005.xlsx'}"
            ),
        },
    ]


def _build_panel_audit(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    integrated: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    paper_status = eligibility.set_index("paper_id")["stage6_status"].to_dict()
    rows: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        stage4 = predictions[predictions["benchmark_id"] == item.benchmark_id]
        stage6 = integrated[integrated["benchmark_id"] == item.benchmark_id]
        expected_printed = stage6[stage6["source_expected_printed_value"].notna()]
        false_positive_coordinates = stage6[
            stage6["stage4_predicted_value"].notna()
            & stage6["source_expected_printed_value"].isna()
        ]
        rows.append(
            {
                "benchmark_id": item.benchmark_id,
                "paper_id": item.paper_id,
                "figure": item.figure,
                "panel": item.panel,
                "expected_cells": int(item.expected_rows) * int(item.expected_columns),
                "stage4_cached_cells": len(stage4),
                "stage4_image_numeric_cells": int(stage4["predicted_value"].notna().sum()),
                "stage6_cells": len(stage6),
                "stage6_linked_cells": int(
                    stage6["stage6_join_status"].astype(str).str.startswith("complete").sum()
                ),
                "supplementary_numeric_source_cells": int(
                    stage6["supplementary_source_value"].notna().sum()
                ),
                "printed_source_crosschecks": len(expected_printed),
                "printed_source_exact_matches": int(
                    (expected_printed["image_source_display_exact_match"] == True).sum()  # noqa: E712
                ),
                "printed_source_false_positive_coordinates": len(false_positive_coordinates),
                "stage6_status": paper_status[item.paper_id],
                "excluded": paper_status[item.paper_id] == "excluded",
            }
        )
    return pd.DataFrame(rows)


def _results_text(
    eligibility: pd.DataFrame, integrated: pd.DataFrame, panel_audit: pd.DataFrame
) -> tuple[str, str]:
    included = eligibility[eligibility["stage6_status"] != "excluded"]
    excluded = eligibility[eligibility["stage6_status"] == "excluded"]
    complete_cells = len(integrated)
    complete_panels = int((panel_audit["stage6_cells"] > 0).sum())
    zhang_source_numeric = int(
        integrated.loc[integrated["paper_id"] == "zhang_2024", "supplementary_source_value"]
        .notna()
        .sum()
    )
    braun_raw_numeric = int(
        integrated.loc[integrated["paper_id"] == "braun_2025", "supplementary_source_value"]
        .notna()
        .sum()
    )
    braun_checks = panel_audit[panel_audit["benchmark_id"] == "B05"].iloc[0]

    markdown = f"""# Cross-paper Stage 5–6 results draft

Stage 5–6 reused the cached Stage 4 outputs and made no new Qwen calls. Relevant structured supplementary records were available for {len(included)} of 7 papers, covering {complete_panels} of 10 panels and {complete_cells} cells. The remaining {len(excluded)} papers were excluded before merging because their supplementary packages did not contain records that could be linked unambiguously to the benchmark panel.

- Fan et al.: all 311 cells across four panels were linked uniquely to the CT26 epitope reference table. Supplementary data supplied gene, mutation, peptide sequence, and MHC annotations; image-read ELISpot values remained the Stage 4 values.
- Braun et al.: all 252 cells were linked by patient, vaccine pool, and study week to the row-normalized Figure 2a source data; {braun_raw_numeric} cells also had raw background-subtracted values and 20 unavailable raw measurements remained NA. {int(braun_checks.printed_source_exact_matches)}/{int(braun_checks.printed_source_crosschecks)} visible printed maxima agreed with the rounded raw source values. The remaining value (570) exposed a Stage 4 row-coordinate assignment mismatch and was flagged for manual review rather than silently moved.
- Zhang et al.: all 152 cells were linked to the 38-peptide by four-donor Table S6 matrix, yielding {zhang_source_numeric} source ELISpot values. Because Panel 6C is color-only, these values are stored as supplementary source data and are not counted as image-recognition predictions.
- HHV6 study: all 265 cells were linked to donor labels and the correct pool-level annotation. The supplement provides pool-level means and peptide annotations but no five-donor cell matrix, so donor-level numeric accuracy is not reported.
- Lindner, Haars, and Geiger were excluded with explicit reasons in `stage56_exclusions.csv`; no values or links were fabricated.

This analysis reports supplementary linkage coverage. It does not convert heatmap colour to a numeric prediction.
"""

    latex = rf"""\subsection{{Cross-paper Stage 5--6 reconstruction}}
Stage 5--6 reused cached Stage 4 outputs without additional multimodal-model calls. Relevant structured supplementary records were available for {len(included)} of seven papers, covering {complete_panels} of ten panels and {complete_cells} cells. Three papers were excluded before merging because their supplementary packages did not contain records that could be linked unambiguously to the benchmark panel.

For Fan et al., all 311 cells across four panels were linked uniquely to CT26 epitope annotations. For Braun et al., all 252 cells were linked by patient, vaccine pool, and study week to the row-normalized Figure 2a source data; {braun_raw_numeric} cells also had raw background-subtracted values and 20 unavailable raw measurements remained missing. {int(braun_checks.printed_source_exact_matches)} of {int(braun_checks.printed_source_crosschecks)} visible printed maxima agreed with the rounded raw source values. The remaining value (570) exposed a Stage 4 row-coordinate assignment mismatch and was retained for manual review. For Zhang et al., all 152 cells were linked to the 38-peptide by four-donor Table S6 matrix. The source values were retained separately because the benchmark panel is colour-only and Stage 4 abstained from numeric colour conversion. For the HHV6 panel, all 265 cells were linked to donor labels and pool-level annotations; donor-level numeric accuracy was not reported because the supplement provides pool-level summaries rather than the five-donor cell matrix. Lindner, Haars, and Geiger were excluded with explicit, source-specific reasons. No missing values or links were fabricated.
"""
    return markdown, latex


def run_stage56_cross_paper(project_root: str | Path = DEFAULT_PROJECT_ROOT) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    benchmark_root = project_root / "generic_heatmap_benchmark_10"
    output_root = project_root / "stage56_cross_paper"
    source_dir = output_root / "supplementary_sources"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(benchmark_root / "benchmark_manifest.csv")
    predictions = pd.read_csv(benchmark_root / "benchmark_predictions.csv")
    _assert_stage4_shape(predictions, manifest)

    integrated_rows: list[dict[str, Any]] = []
    reference_manifest: list[dict[str, Any]] = []
    for builder, args in [
        (_build_fan, (project_root, predictions, output_root)),
        (_build_braun, (predictions, source_dir, output_root)),
        (_build_zhang, (predictions, source_dir, output_root)),
        (_build_hhv6, (predictions, source_dir, output_root)),
    ]:
        records, references = builder(*args)
        integrated_rows.extend(records)
        reference_manifest.extend(references)

    integrated = pd.DataFrame(integrated_rows).sort_values(
        ["benchmark_id", "row_index", "column_index"]
    )
    if integrated.duplicated(["benchmark_id", "row_index", "column_index"]).any():
        raise ValueError("Stage 6 produced duplicate benchmark coordinates")
    if len(integrated) != 980:
        raise ValueError(f"Expected 980 eligible Stage 6 cells, found {len(integrated)}")

    eligibility = pd.DataFrame(_eligibility_rows(project_root, output_root))
    exclusions = eligibility[eligibility["stage6_status"] == "excluded"].copy()
    panel_audit = _build_panel_audit(manifest, predictions, integrated, eligibility)
    ref_manifest = pd.DataFrame(reference_manifest)

    _write_csv(eligibility, output_root / "stage56_eligibility.csv")
    _write_csv(exclusions, output_root / "stage56_exclusions.csv")
    _write_csv(ref_manifest, output_root / "stage5_reference_manifest.csv")
    _write_csv(integrated, output_root / "stage6_integrated_cells.csv")
    _write_csv(panel_audit, output_root / "stage6_panel_audit.csv")
    _write_csv(
        integrated[integrated["manual_review"] == True].copy(),  # noqa: E712
        output_root / "stage56_failure_cases.csv",
    )

    paper_summary = (
        integrated.groupby("paper_id", as_index=False)
        .agg(
            panels=("benchmark_id", "nunique"),
            integrated_cells=("benchmark_id", "size"),
            linked_cells=("stage6_join_status", lambda s: int(s.astype(str).str.startswith("complete").sum())),
            image_numeric_cells=("stage4_numeric_from_image", "sum"),
            supplementary_numeric_source_cells=("supplementary_source_value", "count"),
            manual_review_cells=("manual_review", "sum"),
        )
        .merge(
            eligibility[["paper_id", "stage5_status", "stage6_status", "stage6_scope"]],
            on="paper_id",
            how="right",
        )
        .fillna(
            {
                "panels": 0,
                "integrated_cells": 0,
                "linked_cells": 0,
                "image_numeric_cells": 0,
                "supplementary_numeric_source_cells": 0,
                "manual_review_cells": 0,
            }
        )
    )
    _write_csv(paper_summary, output_root / "stage56_summary.csv")

    metadata = {
        "uses_cached_stage1_to_stage4_outputs": True,
        "reran_stage1_to_stage4": False,
        "qwen_api_calls": 0,
        "allows_numeric_inference_from_colour": False,
        "uses_numeric_confidence_threshold": False,
        "papers_total": 7,
        "papers_stage56_included": int((eligibility["stage6_status"] != "excluded").sum()),
        "papers_stage56_excluded": int((eligibility["stage6_status"] == "excluded").sum()),
        "panels_total": 10,
        "panels_stage56_included": int((panel_audit["stage6_cells"] > 0).sum()),
        "integrated_cells": len(integrated),
        "supplementary_numeric_source_cells": int(integrated["supplementary_source_value"].notna().sum()),
        "source_normalized_numeric_cells": int(integrated["source_normalized_value"].notna().sum()),
        "manual_review_cells": int(integrated["manual_review"].sum()),
        "printed_source_crosschecks": int(panel_audit["printed_source_crosschecks"].sum()),
        "printed_source_exact_matches": int(panel_audit["printed_source_exact_matches"].sum()),
    }
    _write_json(metadata, output_root / "stage56_run_metadata.json")

    markdown, latex = _results_text(eligibility, integrated, panel_audit)
    (output_root / "stage56_results_draft.md").write_text(markdown, encoding="utf-8")
    (output_root / "stage56_results_draft.tex").write_text(latex, encoding="utf-8")

    return {
        **metadata,
        "output_root": str(output_root),
        "included_papers": eligibility.loc[
            eligibility["stage6_status"] != "excluded", "paper_id"
        ].tolist(),
        "excluded_papers": eligibility.loc[
            eligibility["stage6_status"] == "excluded", "paper_id"
        ].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    args = parser.parse_args()
    result = run_stage56_cross_paper(args.project_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
