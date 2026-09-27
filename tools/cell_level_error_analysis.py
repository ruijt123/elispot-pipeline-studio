"""Generate source-backed cell-level validation and failure-boundary reports.

No model call is made and ground truth is never inferred. If a ground-truth
file is not supplied, the command writes a prefilled manual validation template.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path.cwd()
RUN_ROOT = PROJECT_ROOT
PAPER_CONFIG = {"paper_id": "runtime_input", "main_pdf": ""}
ACTIVE_PAPER = "runtime_input"
COMMENT7_OUT_DIR = PROJECT_ROOT / "reviewer_comment_7_outputs"
MANUAL_GT_TEMPLATE_PATH = COMMENT7_OUT_DIR / "manual_cell_validation_template.csv"
GROUND_TRUTH_STATUS_PATH = COMMENT7_OUT_DIR / "ground_truth_status.json"
IMAGE_PROXY_METRICS_PATH = COMMENT7_OUT_DIR / "image_proxy_metrics.csv"
MIN_GROUP_N = 5


def locate_stage4_cell_records():
    candidates = [
        Path(RUN_ROOT) / "stage4_final_generic_outputs_after_stage3b3c" / "Stage4_Cell_Records.xlsx",
        Path(RUN_ROOT) / "stage4_final_generic_outputs_after_stage3b3c" / "Stage4_Cell_Records.csv",
        PROJECT_ROOT / "stage4_final_generic_outputs_after_stage3b3c" / "Stage4_Cell_Records.xlsx",
        PROJECT_ROOT / "stage4_final_generic_outputs_after_stage3b3c" / "Stage4_Cell_Records.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No cached Stage4_Cell_Records.xlsx/csv was found.")

def load_stage4_cell_records(path=None):
    path = Path(path or locate_stage4_cell_records())
    return pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)

def _first_existing_column(df, names, default=None):
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)

def _canonical_prediction_table(records_df):
    out = pd.DataFrame(index=records_df.index)
    out["paper_id"] = PAPER_CONFIG.get("paper_id", ACTIVE_PAPER)
    out["source_document"] = _first_existing_column(
        records_df, ["Source_Document", "source_document", "source_doc"], PAPER_CONFIG.get("main_pdf", "")
    )
    out["figure_unit_id"] = _first_existing_column(records_df, ["Figure_Unit_ID", "figure_unit_id"])
    out["figure"] = _first_existing_column(records_df, ["Figure_Number", "figure_number", "Figure_Unit_ID"])
    out["panel_id"] = _first_existing_column(records_df, ["Panel_ID", "panel_id"], "unknown")
    out["row_index"] = pd.to_numeric(_first_existing_column(records_df, ["Row_Index", "row_index"]), errors="coerce")
    out["column_index"] = pd.to_numeric(_first_existing_column(records_df, ["Col_Index", "Column_Index", "column_index"]), errors="coerce")
    out["predicted_value"] = _first_existing_column(records_df, ["ELISpot_Value", "Predicted_Value", "predicted_value"])
    out["x_value"] = _first_existing_column(records_df, ["X_Value", "x_value"])
    out["y_value"] = _first_existing_column(records_df, ["Y_Value", "y_value"])
    out["x_axis_role"] = _first_existing_column(records_df, ["X_Axis_Role", "x_axis_role"])
    out["y_axis_role"] = _first_existing_column(records_df, ["Y_Axis_Role", "y_axis_role"])
    out["core_image_path"] = _first_existing_column(records_df, ["Core_Image_Path", "core_image_path"])
    out["tool_row_count"] = pd.to_numeric(_first_existing_column(records_df, ["Tool_Row_Count", "tool_row_count"]), errors="coerce")
    out["tool_column_count"] = pd.to_numeric(_first_existing_column(records_df, ["Tool_Column_Count", "tool_column_count"]), errors="coerce")
    return out

def _detect_grid_lines_for_validation(image_path, expected_rows, expected_cols):
    """Use existing Stage 4 detector when available; otherwise return unavailable."""
    try:
        info = detect_core_grid_shape(image_path, out_dir=None, safe_name="comment7", scale=4, debug=False)
        h = list(info.get("horizontal_lines") or [])
        v = list(info.get("vertical_lines") or [])
        scale = float(info.get("scale") or 4)
        if len(h) != int(expected_rows) + 1 or len(v) != int(expected_cols) + 1:
            return None, "grid_line_count_does_not_match_tool_fixed_shape"
        return {"horizontal": h, "vertical": v, "scale": scale}, "available"
    except Exception as exc:
        return None, f"unavailable: {type(exc).__name__}: {exc}"

def _cell_image_proxies(gray, x0, y0, x1, y1):
    x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
    width, height = max(0, x1 - x0), max(0, y1 - y0)
    inset = max(1, int(round(min(width, height) * 0.08)))
    inner = gray[y0 + inset:y1 - inset, x0 + inset:x1 - inset]
    if inner.size == 0:
        return np.nan, "unavailable", np.nan, "unavailable"
    contrast = float(np.percentile(inner, 90) - np.percentile(inner, 10))
    contrast_status = "available"
    try:
        blur = cv2.GaussianBlur(inner, (3, 3), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        heights = []
        for idx in range(1, n_labels):
            w, h, area = int(stats[idx, cv2.CC_STAT_WIDTH]), int(stats[idx, cv2.CC_STAT_HEIGHT]), int(stats[idx, cv2.CC_STAT_AREA])
            if 2 <= area <= inner.size * 0.25 and 1 <= h <= inner.shape[0] * 0.8 and 1 <= w <= inner.shape[1] * 0.8:
                heights.append(h)
        if heights:
            text_proxy, text_status = float(np.median(heights)), "available"
        else:
            text_proxy, text_status = np.nan, "unavailable_no_reliable_components"
    except Exception as exc:
        text_proxy, text_status = np.nan, f"unavailable: {type(exc).__name__}"
    return text_proxy, text_status, contrast, contrast_status

def compute_cell_image_proxy_metrics(prediction_df):
    rows = []
    key_cols = ["paper_id", "figure_unit_id", "panel_id", "row_index", "column_index"]
    group_cols = ["paper_id", "figure_unit_id", "panel_id", "core_image_path", "tool_row_count", "tool_column_count"]
    for keys, group in prediction_df.groupby(group_cols, dropna=False):
        paper, figure_unit, panel, image_path, n_rows, n_cols = keys
        base = {"paper_id": paper, "figure_unit_id": figure_unit, "panel_id": panel}
        try:
            n_rows, n_cols = int(n_rows), int(n_cols)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(str(image_path))
            grid, grid_status = _detect_grid_lines_for_validation(image_path, n_rows, n_cols)
            if grid is None:
                for _, pred in group.iterrows():
                    rows.append({**{k: pred[k] for k in key_cols}, "cell_width_px": np.nan, "cell_height_px": np.nan,
                                 "cell_area_px2": np.nan, "mean_cell_area_px2": np.nan, "matrix_cell_density": np.nan,
                                 "text_height_proxy_px": np.nan, "text_height_proxy_status": "unavailable",
                                 "visual_contrast_proxy": np.nan, "visual_contrast_proxy_status": "unavailable",
                                 "grid_metric_status": grid_status})
                continue
            h_lines = np.array(grid["horizontal"], dtype=float) / grid["scale"]
            v_lines = np.array(grid["vertical"], dtype=float) / grid["scale"]
            cell_areas = [max(0, v_lines[c+1]-v_lines[c]) * max(0, h_lines[r+1]-h_lines[r])
                          for r in range(n_rows) for c in range(n_cols)]
            mean_area = float(np.mean(cell_areas)) if cell_areas else np.nan
            matrix_area = max(0, v_lines[-1]-v_lines[0]) * max(0, h_lines[-1]-h_lines[0])
            density = float((n_rows*n_cols) / matrix_area) if matrix_area > 0 else np.nan
            for _, pred in group.iterrows():
                r, c = int(pred["row_index"]), int(pred["column_index"])
                if not (0 <= r < n_rows and 0 <= c < n_cols):
                    continue
                x0, x1, y0, y1 = v_lines[c], v_lines[c+1], h_lines[r], h_lines[r+1]
                text_h, text_status, contrast, contrast_status = _cell_image_proxies(image, x0, y0, x1, y1)
                rows.append({**{k: pred[k] for k in key_cols},
                             "cell_width_px": float(x1-x0), "cell_height_px": float(y1-y0),
                             "cell_area_px2": float((x1-x0)*(y1-y0)), "mean_cell_area_px2": mean_area,
                             "matrix_cell_density": density, "text_height_proxy_px": text_h,
                             "text_height_proxy_status": text_status, "visual_contrast_proxy": contrast,
                             "visual_contrast_proxy_status": contrast_status, "grid_metric_status": "available"})
        except Exception as exc:
            for _, pred in group.iterrows():
                rows.append({**{k: pred[k] for k in key_cols}, "cell_width_px": np.nan, "cell_height_px": np.nan,
                             "cell_area_px2": np.nan, "mean_cell_area_px2": np.nan, "matrix_cell_density": np.nan,
                             "text_height_proxy_px": np.nan, "text_height_proxy_status": "unavailable",
                             "visual_contrast_proxy": np.nan, "visual_contrast_proxy_status": "unavailable",
                             "grid_metric_status": f"unavailable: {type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)

def create_manual_cell_validation_template(prediction_df, proxy_df, path=MANUAL_GT_TEMPLATE_PATH):
    keys = ["paper_id", "figure_unit_id", "panel_id", "row_index", "column_index"]
    template = prediction_df.merge(proxy_df, on=keys, how="left") if len(proxy_df) else prediction_df.copy()
    template["ground_truth_value"] = ""  # intentionally blank; must be entered by a human
    template["ground_truth_source"] = ""
    template["validated_by"] = ""
    template["validation_notes"] = ""
    template.to_csv(path, index=False, encoding="utf-8-sig")
    return template

def find_verified_ground_truth(search_root=PROJECT_ROOT):
    """Accept only explicit ground_truth_value columns with at least one nonblank value."""
    candidates = [MANUAL_GT_TEMPLATE_PATH]
    candidates += [p for p in Path(search_root).rglob("*.csv") if p != MANUAL_GT_TEMPLATE_PATH]
    candidates += list(Path(search_root).rglob("*.xlsx"))
    for path in candidates:
        try:
            header = pd.read_excel(path, nrows=5) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path, nrows=5)
        except Exception:
            continue
        gt_col = next((c for c in header.columns if str(c).strip().lower() in {"ground_truth_value", "ground_truth"}), None)
        if not gt_col:
            continue
        try:
            frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
        except Exception:
            continue
        numeric_gt = pd.to_numeric(frame[gt_col], errors="coerce")
        if numeric_gt.notna().any():
            frame = frame.rename(columns={gt_col: "ground_truth_value"})
            return path, frame
    return None, None

def _safe_qcut(series, requested=4):
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.nunique() < 2:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    try:
        codes = pd.qcut(numeric, q=min(requested, valid.nunique()), labels=False, duplicates="drop")
    except Exception:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    actual = int(codes.dropna().max()) + 1 if codes.notna().any() else 0
    names = ["low", "mid-low", "mid-high", "high"]
    if actual == 2:
        labels = ["low", "high"]
    elif actual == 3:
        labels = ["low", "mid", "high"]
    else:
        labels = names[:actual]
    return codes.map({i: labels[i] for i in range(actual)}).astype("object")

def _error_type(pred, gt):
    if pd.isna(pred): return "prediction_missing"
    if pd.isna(gt): return "ground_truth_missing"
    return "exact" if float(pred) == float(gt) else "numeric_mismatch"

def _group_error_table(df, group_col):
    table = df.groupby(group_col, dropna=False).agg(n=("exact_match", "size"), errors=("exact_match", lambda s: int((~s).sum()))).reset_index()
    table["error_rate"] = table["errors"] / table["n"]
    table["insufficient_sample"] = table["n"] < MIN_GROUP_N
    return table

def _parse_timepoint(value):
    text = str(value or "").strip().lower()
    match = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|d|day|days|w|wk|week|weeks|m|month|months)?\s*", text)
    if not match:
        return np.nan
    number = float(match.group(1)); unit = match.group(2) or ""
    factor = {"": 1, "h": 1, "hr": 1, "hrs": 1, "hour": 1, "hours": 1,
              "d": 24, "day": 24, "days": 24, "w": 168, "wk": 168, "week": 168, "weeks": 168,
              "m": 24*30, "month": 24*30, "months": 24*30}.get(unit)
    return number * factor if factor is not None else np.nan

def add_transition_analysis(df):
    out = df.copy()
    out["delta_value"] = np.nan
    out["transition"] = pd.NA
    applicable_any = False
    for (figure, panel), panel_df in out.groupby(["figure_unit_id", "panel_id"], dropna=False):
        x_role = str(panel_df["x_axis_role"].dropna().iloc[0]).lower() if panel_df["x_axis_role"].notna().any() else ""
        y_role = str(panel_df["y_axis_role"].dropna().iloc[0]).lower() if panel_df["y_axis_role"].notna().any() else ""
        if x_role == "timepoint":
            time_index, time_label, other_index = "column_index", "x_value", "row_index"
        elif y_role == "timepoint":
            time_index, time_label, other_index = "row_index", "y_value", "column_index"
        else:
            continue
        times = panel_df[[time_index, time_label]].drop_duplicates(time_index).copy()
        times["parsed_time"] = times[time_label].map(_parse_timepoint)
        if times["parsed_time"].isna().any() or times["parsed_time"].duplicated().any():
            continue
        applicable_any = True
        order = times.sort_values("parsed_time")[time_index].tolist()
        for _, sequence_df in panel_df.groupby(other_index):
            idx_by_time = {int(out.loc[i, time_index]): i for i in sequence_df.index}
            prev_gt = None
            for time_position in order:
                idx = idx_by_time.get(int(time_position))
                if idx is None: continue
                gt = out.loc[idx, "ground_truth_value"]
                if prev_gt is not None and pd.notna(gt) and pd.notna(prev_gt):
                    delta = float(gt) - float(prev_gt)
                    out.loc[idx, "delta_value"] = delta
                    out.loc[idx, "transition"] = "rising" if delta > 0 else ("falling" if delta < 0 else "unchanged")
                prev_gt = gt
    return out, applicable_any

def adjacent_column_error_analysis(df):
    rows = []
    for (figure, panel), panel_df in df.groupby(["figure_unit_id", "panel_id"], dropna=False):
        counts = {}
        gt_lookup = {(int(r.row_index), int(r.column_index)): r.ground_truth_value for r in panel_df.itertuples()}
        for offset in [0, -1, 1]:
            matched = 0; compared = 0
            for rec in panel_df.itertuples():
                gt = gt_lookup.get((int(rec.row_index), int(rec.column_index) + offset))
                if pd.notna(rec.predicted_value) and pd.notna(gt):
                    compared += 1
                    matched += int(float(rec.predicted_value) == float(gt))
            counts[offset] = (matched, compared)
        best = max(counts, key=lambda k: counts[k][0])
        shift = "adjacent_column_shift_left" if best == -1 and counts[-1][0] > counts[0][0] else (
                "adjacent_column_shift_right" if best == 1 and counts[1][0] > counts[0][0] else "other")
        for offset in [0, -1, 1]:
            rows.append({"figure_unit_id": figure, "panel_id": panel, "offset": offset,
                         "matches": counts[offset][0], "comparisons": counts[offset][1], "shift_classification": shift})
    return pd.DataFrame(rows)

def _plot_error_rate(table, x_col, title, out_path):
    if table.empty: return
    ax = table.plot(kind="bar", x=x_col, y="error_rate", legend=False, figsize=(7, 4), color="#4472C4")
    ax.set_title(title); ax.set_ylabel("Error rate"); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()

def run_cell_level_error_analysis(ground_truth_df, prediction_df, proxy_df, out_dir=COMMENT7_OUT_DIR):
    keys = ["paper_id", "figure_unit_id", "panel_id", "row_index", "column_index"]
    gt = ground_truth_df.copy()
    for key in keys:
        if key not in gt.columns:
            raise ValueError(f"Ground-truth file is missing required key: {key}")
    gt["ground_truth_value"] = pd.to_numeric(gt["ground_truth_value"], errors="coerce")
    gt = gt[gt["ground_truth_value"].notna()].copy()
    if gt.empty:
        raise ValueError("No real numeric ground_truth_value entries are available.")
    base = prediction_df.merge(proxy_df, on=keys, how="left") if len(proxy_df) else prediction_df.copy()
    keep = keys + ["ground_truth_value"]
    merged = base.merge(gt[keep].drop_duplicates(keys), on=keys, how="inner")
    merged["predicted_value"] = pd.to_numeric(merged["predicted_value"], errors="coerce")
    merged["exact_match"] = merged["predicted_value"].eq(merged["ground_truth_value"])
    merged["absolute_error"] = (merged["predicted_value"] - merged["ground_truth_value"]).abs()
    merged["error_type"] = [_error_type(p, g) for p, g in zip(merged["predicted_value"], merged["ground_truth_value"])]
    merged["value_range"] = _safe_qcut(merged["ground_truth_value"])
    merged["cell_size_bin"] = _safe_qcut(merged["cell_area_px2"])
    merged["density_bin"] = _safe_qcut(merged["matrix_cell_density"])
    merged["text_height_proxy_bin"] = _safe_qcut(merged["text_height_proxy_px"])
    merged["visual_contrast_proxy_bin"] = _safe_qcut(merged["visual_contrast_proxy"])
    merged, transition_applicable = add_transition_analysis(merged)
    merged.to_csv(Path(out_dir) / "cell_level_validation.csv", index=False, encoding="utf-8-sig")

    outputs = {
        "error_by_panel.csv": _group_error_table(merged, "panel_id"),
        "error_by_value_range.csv": _group_error_table(merged, "value_range"),
        "error_by_cell_size.csv": pd.concat([
            _group_error_table(merged, "cell_size_bin").assign(metric="cell_area_px2"),
            _group_error_table(merged, "density_bin").assign(metric="matrix_cell_density")
        ], ignore_index=True),
        "error_by_text_height_proxy.csv": _group_error_table(merged, "text_height_proxy_bin"),
        "error_by_visual_contrast.csv": _group_error_table(merged, "visual_contrast_proxy_bin"),
        "error_by_type.csv": _group_error_table(merged, "error_type"),
    }
    if transition_applicable and merged["transition"].notna().any():
        outputs["error_by_transition.csv"] = _group_error_table(merged[merged["transition"].notna()], "transition")
    failures = merged[~merged["exact_match"]].copy()
    failures.to_csv(Path(out_dir) / "failure_cases.csv", index=False, encoding="utf-8-sig")
    adjacent_column_error_analysis(merged).to_csv(Path(out_dir) / "adjacent_column_error_analysis.csv", index=False, encoding="utf-8-sig")
    boundary_parts = []
    for metric in ["cell_size_bin", "density_bin", "text_height_proxy_bin", "visual_contrast_proxy_bin",
                   "value_range", "transition", "panel_id", "row_index", "column_index"]:
        subset = merged if metric != "transition" else merged[merged["transition"].notna()]
        if subset.empty: continue
        part = _group_error_table(subset, metric).rename(columns={metric: "group"})
        part.insert(0, "metric", metric)
        boundary_parts.append(part)
    pd.concat(boundary_parts, ignore_index=True).to_csv(Path(out_dir) / "failure_boundary_summary.csv", index=False, encoding="utf-8-sig")
    for name, table in outputs.items():
        table.to_csv(Path(out_dir) / name, index=False, encoding="utf-8-sig")

    _plot_error_rate(outputs["error_by_value_range.csv"], "value_range", "Error rate by ground-truth value range", Path(out_dir) / "error_rate_by_value_range.png")
    size_plot = outputs["error_by_cell_size.csv"]
    _plot_error_rate(size_plot[size_plot["metric"] == "cell_area_px2"], "cell_size_bin", "Error rate by cell size", Path(out_dir) / "error_rate_by_cell_size.png")
    _plot_error_rate(size_plot[size_plot["metric"] == "matrix_cell_density"], "density_bin", "Error rate by matrix cell density", Path(out_dir) / "error_rate_by_cell_density.png")
    _plot_error_rate(outputs["error_by_visual_contrast.csv"], "visual_contrast_proxy_bin", "Error rate by visual contrast proxy", Path(out_dir) / "error_rate_by_visual_contrast.png")
    type_counts = outputs["error_by_type.csv"]
    ax = type_counts.plot(kind="bar", x="error_type", y="n", legend=False, figsize=(7, 4), color="#ED7D31")
    ax.set_title("Error type distribution"); ax.set_ylabel("Count")
    plt.tight_layout(); plt.savefig(Path(out_dir) / "error_type_distribution.png", dpi=180); plt.close()
    if "error_by_transition.csv" in outputs:
        _plot_error_rate(outputs["error_by_transition.csv"], "transition", "Transition error rate", Path(out_dir) / "transition_error_rate.png")
    return merged, outputs



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Stage4 cell-record CSV or XLSX")
    parser.add_argument("--ground-truth", type=Path, help="Manually verified CSV with ground_truth_value")
    parser.add_argument("--output", type=Path, default=Path("reviewer_comment_7_outputs"))
    parser.add_argument("--paper-id", default="runtime_input")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Base directory for relative core-image paths")
    args = parser.parse_args()

    global COMMENT7_OUT_DIR, MANUAL_GT_TEMPLATE_PATH, GROUND_TRUTH_STATUS_PATH
    global IMAGE_PROXY_METRICS_PATH, PAPER_CONFIG, ACTIVE_PAPER
    COMMENT7_OUT_DIR = args.output.resolve()
    COMMENT7_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_GT_TEMPLATE_PATH = COMMENT7_OUT_DIR / "manual_cell_validation_template.csv"
    GROUND_TRUTH_STATUS_PATH = COMMENT7_OUT_DIR / "ground_truth_status.json"
    IMAGE_PROXY_METRICS_PATH = COMMENT7_OUT_DIR / "image_proxy_metrics.csv"
    ACTIVE_PAPER = args.paper_id
    PAPER_CONFIG = {"paper_id": args.paper_id, "main_pdf": str(args.predictions)}

    records = load_stage4_cell_records(args.predictions)
    predictions = _canonical_prediction_table(records)
    project_root = args.project_root.resolve()
    predictions["core_image_path"] = predictions["core_image_path"].map(
        lambda value: str((project_root / Path(str(value))).resolve())
        if str(value).strip() and not Path(str(value)).is_absolute()
        else str(value)
    )
    proxies = compute_cell_image_proxy_metrics(predictions)
    proxies.to_csv(IMAGE_PROXY_METRICS_PATH, index=False, encoding="utf-8-sig")
    if args.ground_truth is None:
        create_manual_cell_validation_template(predictions, proxies)
        status = {
            "ground_truth_available": False,
            "reason": "No ground-truth file supplied; predictions were not treated as ground truth.",
            "manual_validation_template": str(MANUAL_GT_TEMPLATE_PATH),
            "error_analysis_generated": False,
            "uses_fabricated_ground_truth": False,
        }
    else:
        ground_truth = pd.read_csv(args.ground_truth)
        if "ground_truth_value" not in ground_truth or ground_truth["ground_truth_value"].notna().sum() == 0:
            raise ValueError("Ground-truth CSV has no nonblank ground_truth_value entries")
        merged, _ = run_cell_level_error_analysis(ground_truth, predictions, proxies, out_dir=COMMENT7_OUT_DIR)
        status = {
            "ground_truth_available": True,
            "ground_truth_path": str(args.ground_truth.resolve()),
            "validated_cell_count": int(len(merged)),
            "error_analysis_generated": True,
            "uses_fabricated_ground_truth": False,
        }
    GROUND_TRUTH_STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
