"""Deterministic structure and value-encoding checks for the 10-panel benchmark.

This module never infers numeric values from colour. It either detects printed
glyph evidence and routes cells to the audited Qwen reader, or abstains and
marks publisher source data as required.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def _groups(indices: Iterable[int], max_gap: int = 2) -> list[list[int]]:
    result: list[list[int]] = []
    for value in [int(x) for x in indices]:
        if not result or value - result[-1][-1] > max_gap:
            result.append([value])
        else:
            result[-1].append(value)
    return result


def _peak_centres(signal: np.ndarray, ratio: float, min_gap: int) -> list[int]:
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0 or float(signal.max()) <= float(signal.min()):
        return []
    signal = cv2.GaussianBlur(signal.reshape(1, -1).astype(np.float32), (0, 0), sigmaX=1.0).ravel()
    threshold = float(signal.min() + ratio * (signal.max() - signal.min()))
    groups = _groups(np.where(signal >= threshold)[0], max_gap=max(2, min_gap))
    return [int(max(group, key=lambda i: signal[i])) for group in groups]


def _regular_line_lattice_count(edge_projection: np.ndarray, axis_length: int) -> dict:
    """Infer a regular cell count from long-line projection peaks."""
    candidates = []
    for ratio in np.linspace(0.25, 0.75, 11):
        peaks = _peak_centres(edge_projection, float(ratio), max(1, int(axis_length * 0.004)))
        if len(peaks) < 3:
            continue
        gaps = np.diff(peaks).astype(float)
        small = gaps[gaps <= np.median(gaps) * 1.65]
        if len(small) < 2 or np.median(small) < 3:
            continue
        period = float(np.median(small))
        inferred = int(round((axis_length - 1) / period))
        if inferred < 2 or inferred > 80:
            continue
        predicted = np.linspace(0, axis_length - 1, inferred + 1)
        tolerance = max(2.0, period * 0.20)
        support = float(np.mean([min(abs(p - q) for q in peaks) <= tolerance for p in predicted]))
        peak_span = float((peaks[-1] - peaks[0]) / max(axis_length - 1, 1))
        gap_cv = float(np.std(small) / max(np.mean(small), 1e-6))
        score = support + 0.25 * peak_span - 0.25 * min(gap_cv, 1.0)
        boundary_count_used = bool(peak_span >= 0.95 and gap_cv <= 0.20 and len(peaks) >= 3)
        if boundary_count_used:
            inferred = len(peaks) - 1
        candidates.append({
            "cell_count": inferred,
            "period_px": period,
            "support": support,
            "peak_span": peak_span,
            "gap_cv": gap_cv,
            "threshold_ratio": float(ratio),
            "peaks": peaks,
            "score": score,
            "boundary_count_used": boundary_count_used,
        })
    if not candidates:
        return {"available": False}
    best = max(candidates, key=lambda x: (x["score"], x["support"], -x["cell_count"]))
    best["available"] = bool(
        (best["support"] >= 0.68 and best["peak_span"] >= 0.65)
        or (best["gap_cv"] <= 0.15 and best["peak_span"] >= 0.88)
    )
    # Crops sometimes omit one or both outer borders while retaining at least
    # four evenly spaced internal separators.  In that case the period remains
    # identifiable even though edge-to-edge support is low.  Prefer the widest
    # stable period so digit strokes (usually a half-cell harmonic) are not
    # mistaken for extra grid lines.
    if not best["available"]:
        interior = [
            x for x in candidates
            if len(x["peaks"]) >= 4
            and x["gap_cv"] <= 0.10
            and x["peak_span"] >= 0.55
            and x["period_px"] >= 4
        ]
        if interior:
            best = max(interior, key=lambda x: (x["period_px"], x["peak_span"], x["support"]))
            best["available"] = True
            best["interior_only"] = True
        else:
            best["interior_only"] = False
    else:
        best["interior_only"] = False
    best["candidate_counts"] = sorted({int(x["cell_count"]) for x in candidates})
    return best


def _fft_periodic_count(rgb: np.ndarray, axis: int) -> dict:
    """Estimate the fundamental cell count from colour/edge periodicity."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if axis == 0:
        change = np.mean(np.abs(np.diff(rgb.astype(float), axis=1)), axis=(0, 2))
        change = np.r_[0.0, change]
        edge = cv2.Canny(gray, 30, 100).mean(axis=0)
    else:
        change = np.mean(np.abs(np.diff(rgb.astype(float), axis=0)), axis=(1, 2))
        change = np.r_[0.0, change]
        edge = cv2.Canny(gray, 30, 100).mean(axis=1)
    signal = ((change - change.mean()) / (change.std() + 1e-6)
              + (edge - edge.mean()) / (edge.std() + 1e-6))
    magnitude = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    max_count = min(80, max(2, len(signal) // 4))
    ranked = []
    for count in range(2, max_count + 1):
        harmonics = [magnitude[m * count] / m for m in range(1, 5) if m * count < len(magnitude)]
        if not harmonics:
            continue
        harmonic_score = float(sum(harmonics) * (1.0 - math.exp(-count / 4.0)))
        ranked.append((harmonic_score, count))
    ranked.sort(reverse=True)
    if not ranked:
        return {"available": False}
    best_score, best_count = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    return {
        "available": True,
        "cell_count": int(best_count),
        "score": best_score,
        "score_margin_ratio": float((best_score - second_score) / max(best_score, 1e-6)),
        "top_candidates": [{"cell_count": int(k), "score": float(v)} for v, k in ranked[:8]],
    }


def detect_nested_rectangular_blocks(image_path: str | Path) -> dict:
    """Detect vertically stacked heatmap blocks with full-width outer borders."""
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 40, 120)
    row_coverage = (edge > 0).mean(axis=1)
    horizontal_groups = _groups(np.where(row_coverage >= 0.85)[0], max_gap=4)
    horizontal = [int(max(g, key=lambda y: row_coverage[y])) for g in horizontal_groups]
    if not horizontal or horizontal[0] > 3:
        horizontal.insert(0, 0)
    if horizontal[-1] < rgb.shape[0] - 4:
        horizontal.append(rgb.shape[0] - 1)
    # Alternating large heatmap blocks and small inter-block gaps.
    gaps = np.diff(horizontal)
    large = gaps[gaps >= max(12, np.median(gaps) * 1.8)] if len(gaps) else np.array([])
    typical_block = float(np.median(large)) if len(large) else 0.0
    blocks = []
    for a, b in zip(horizontal[:-1], horizontal[1:]):
        if typical_block and (b - a) >= typical_block * 0.70:
            blocks.append((int(a), int(b)))

    col_projection = edge.mean(axis=0)
    col_lattice = _regular_line_lattice_count(col_projection, rgb.shape[1])
    block_rows = None
    if blocks:
        # Internal row periodicity across all blocks; four rows is recovered from
        # the dominant within-block colour/edge frequency.
        sample = rgb[blocks[0][0]:blocks[0][1] + 1]
        row_periodicity = _fft_periodic_count(sample, axis=1)
        top_counts = [x["cell_count"] for x in row_periodicity.get("top_candidates", [])]
        # Repeated block-row edges generate several integer harmonics. Recover
        # their common fundamental when at least four top candidates support it.
        gcd_value = 0
        for value in top_counts:
            gcd_value = math.gcd(gcd_value, int(value))
        if gcd_value >= 2 and sum(int(v) % gcd_value == 0 for v in top_counts) >= 4:
            block_rows = gcd_value
        else:
            block_rows = row_periodicity.get("cell_count")
        if block_rows and block_rows > 12:
            block_rows = None
    return {
        "method": "nested_rectangular_blocks",
        "block_count": len(blocks),
        "block_intervals_y": blocks,
        "block_rows": block_rows,
        "block_columns": col_lattice.get("cell_count") if col_lattice.get("available") else None,
        "horizontal_border_lines": horizontal,
        "column_lattice": col_lattice,
        "manual_review": not blocks or not col_lattice.get("available") or block_rows is None,
    }


def detect_regular_heatmap_shape(image_path: str | Path) -> dict:
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 40, 120)
    x_line = _regular_line_lattice_count(edge.mean(axis=0), rgb.shape[1])
    y_line = _regular_line_lattice_count(edge.mean(axis=1), rgb.shape[0])
    x_fft = _fft_periodic_count(rgb, axis=0)
    y_fft = _fft_periodic_count(rgb, axis=1)

    columns = x_line.get("cell_count") if x_line.get("available") else x_fft.get("cell_count")
    rows = y_line.get("cell_count") if y_line.get("available") else y_fft.get("cell_count")
    x_method = "long_line_lattice" if x_line.get("available") else "colour_edge_periodicity"
    y_method = "long_line_lattice" if y_line.get("available") else "colour_edge_periodicity"

    # Prefer a strong periodic estimate when a line-only estimate is a likely
    # off-by-one boundary count, or when it found only a coarse outer/midline
    # partition.  This resolves borderless colour matrices without relying on
    # paper-specific dimensions.
    if x_line.get("available") and x_fft.get("available"):
        line_count, fft_count = int(x_line["cell_count"]), int(x_fft["cell_count"])
        if ((abs(line_count - fft_count) == 1
             and not x_line.get("boundary_count_used")
             and x_fft.get("score_margin_ratio", 0) >= 0.25)
                or (line_count <= 2 and fft_count > line_count)):
            columns, x_method = fft_count, "colour_edge_periodicity_over_line_boundary"
    if y_line.get("available") and y_fft.get("available"):
        line_count, fft_count = int(y_line["cell_count"]), int(y_fft["cell_count"])
        if ((abs(line_count - fft_count) == 1
             and not y_line.get("boundary_count_used")
             and y_fft.get("score_margin_ratio", 0) >= 0.25)
                or (line_count <= 2 and fft_count > line_count)):
            rows, y_method = fft_count, "colour_edge_periodicity_over_line_boundary"

    # Square/near-square colour matrices can have weak borders on one axis.
    # When one axis is line-supported and the image aspect is near 1, use the
    # same count only if the periodic estimate is within 15%.
    aspect = rgb.shape[1] / max(rgb.shape[0], 1)
    if 0.80 <= aspect <= 1.25 and columns and rows and abs(columns - rows) / max(columns, rows) <= 0.15:
        agreed = max(int(columns), int(rows))
        columns = rows = agreed
        x_method = y_method = "near_square_consistency"

    manual_review = columns is None or rows is None
    return {
        "method": "deterministic_ensemble",
        "rows": int(rows) if rows is not None else None,
        "columns": int(columns) if columns is not None else None,
        "row_method": y_method,
        "column_method": x_method,
        "row_line_diagnostics": y_line,
        "column_line_diagnostics": x_line,
        "row_periodicity_diagnostics": y_fft,
        "column_periodicity_diagnostics": x_fft,
        "manual_review": manual_review,
    }


def _cell_boxes(width: int, height: int, rows: int, columns: int):
    xs = [round(i * width / columns) for i in range(columns + 1)]
    ys = [round(i * height / rows) for i in range(rows + 1)]
    for r in range(rows):
        for c in range(columns):
            yield r, c, xs[c], ys[r], xs[c + 1], ys[r + 1]


def cell_glyph_evidence_proxy(image_path: str | Path, rows: int, columns: int) -> dict:
    """Estimate printed-glyph presence without reading or creating values."""
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    evidence = []
    for r, c, x0, y0, x1, y1 in _cell_boxes(rgb.shape[1], rgb.shape[0], rows, columns):
        px = max(1, int((x1 - x0) * 0.10))
        py = max(1, int((y1 - y0) * 0.10))
        inner = rgb[min(y1 - 1, y0 + py):max(y0 + py + 1, y1 - py), min(x1 - 1, x0 + px):max(x0 + px + 1, x1 - px)]
        if inner.size < 9:
            evidence.append(False)
            continue
        gray = cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY)
        background = float(np.median(gray))
        mask = (np.abs(gray.astype(float) - background) >= 30).astype(np.uint8) * 255
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        found = False
        for k in range(1, n):
            _x, _y, w, h, area = stats[k]
            area_fraction = area / max(gray.size, 1)
            touches_inner_border = _x <= 0 or _y <= 0 or _x + w >= gray.shape[1] - 1 or _y + h >= gray.shape[0] - 1
            plausible_size = 2 <= area and area_fraction <= 0.16 and h >= 2 and w >= 1
            plausible_span = h <= max(2, int(gray.shape[0] * 0.85)) and w <= max(2, int(gray.shape[1] * 0.85))
            if plausible_size and plausible_span and not touches_inner_border:
                found = True
                break
        evidence.append(found)
    rate = float(np.mean(evidence)) if evidence else 0.0
    if rate >= 0.10:
        classification = "printed_numeric_or_text"
    elif rate >= 0.015:
        classification = "mixed_sparse_printed_text"
    else:
        classification = "color_only_no_printed_text_evidence"
    return {
        "glyph_evidence_cells": int(sum(evidence)),
        "total_cells": len(evidence),
        "glyph_evidence_rate": rate,
        "classification": classification,
        "numeric_values_inferred_from_color": False,
    }


def analyze_heatmap_structure(image_path: str | Path, structure_type: str = "regular_grid") -> dict:
    if structure_type == "nested_rectangular_blocks":
        structure = detect_nested_rectangular_blocks(image_path)
        if structure.get("block_count") and structure.get("block_rows") and structure.get("block_columns"):
            rows = structure["block_count"] * structure["block_rows"]
            columns = structure["block_columns"]
            encoding = cell_glyph_evidence_proxy(image_path, rows, columns)
        else:
            rows = columns = None
            encoding = {"classification": "unavailable", "numeric_values_inferred_from_color": False}
    else:
        structure = detect_regular_heatmap_shape(image_path)
        rows, columns = structure.get("rows"), structure.get("columns")
        encoding = cell_glyph_evidence_proxy(image_path, rows, columns) if rows and columns else {"classification": "unavailable", "numeric_values_inferred_from_color": False}
    return {"structure": structure, "encoding": encoding, "rows": rows, "columns": columns}
