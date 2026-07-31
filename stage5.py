"""Stage 5: normalize epitope reference tables from spreadsheets or text PDFs.

The module is deliberately independent of Stages 1-4.  It preserves every
accepted source data row, records skipped rows in an audit table, and emits the
six stable core fields plus provenance and JSON metadata.  Legacy aliases used
by the current Stage 6 are retained for backward compatibility.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import fitz
import pandas as pd


NOT_SPECIFIED = "Not specified"
CORE_COLUMNS = [
    "Epitope_ID",
    "Epitope_Gene",
    "Epitope_Mutation",
    "Epitope_Sequence",
    "MHC_Class",
    "Vaccine_or_Antigen_Set",
]

PROVENANCE_COLUMNS = [
    "Source_File",
    "Source_Sheet",
    "Source_Page",
    "Source_Row",
    "Header_Row",
    "Extraction_Method",
    "Raw_Record_JSON",
    "Additional_Metadata_JSON",
    "Manual_Review",
    "Review_Reason",
]

LEGACY_COLUMNS = [
    "Gene",
    "Mutation_or_Substitution",
    "Sequence",
    "MHC_or_HLA",
    "MHC_type",
    "Patient_ID",
    "Tumor_Model",
    "Source_Document",
    "Source_Table",
    "Source_Page",
    "Row_Index_On_Page",
    "Extraction_Method",
    "Manual_Review",
    "Review_Reason",
    "Vaccine_Assignment_Method",
    "Vaccine_Assignment_Confidence",
    "_group_id",
    "raw_row_text",
    "Stage6_Merge_Key",
]

FIELD_ALIASES = {
    "Epitope_ID": [
        "epitope id", "ept peptide id", "neoepitope id", "neoantigen id",
        "immunizing peptide id", "peptide id", "antigen id", "record id",
        "number", "no", "id", "index",
    ],
    "Epitope_Gene": [
        "gene", "gene name", "gene symbol", "genesymbol", "protein gene",
    ],
    "Epitope_Mutation": [
        "protein change", "amino acid change", "aa change", "mutation",
        "substitution", "variant", "mutant", "protein alteration",
    ],
    "Epitope_Sequence": [
        "mutated peptide sequence", "binding peptide", "epitope sequence",
        "peptide sequence", "immunizing peptide", "27aa sequence used for immunization",
        "27aa sequence", "mutant peptide", "sequence",
    ],
    "MHC_Class": [
        "mhc class", "mhc type", "mhc", "hla allele", "hla", "allele",
        "binding allele",
    ],
    "Vaccine_or_Antigen_Set": [
        "vaccine or antigen set", "vaccine group", "antigen set", "antigen group",
        "immunizing pool", "vaccine", "pool", "group", "construct",
    ],
}

HEADER_TERMS = {
    token
    for aliases in FIELD_ALIASES.values()
    for alias in aliases
    for token in alias.split()
} | {
    "patient", "affinity", "reactivity", "prediction", "percentile", "score",
    "wild", "type", "length", "method", "expression", "elispot",
}

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{8,80}$", re.I)
MUTATION_RE = re.compile(
    r"(?:p\.)?[A-Z][A-Za-z]{0,2}\d+[A-Z*][A-Za-z]{0,2}|"
    r"\d+\.[A-Z]/[A-ZX]{1,3}|\d+-\d+\.-/[A-Z]|"
    r"c\.\d+[ACGT]>[ACGT]",
    re.I,
)
MHC_RE = re.compile(r"MHC[-\s]?(?:I{1,2}|1|2)|class\s+I{1,2}|HLA[-\s]?[^\s,;]+", re.I)
PDF_ROW_RE = re.compile(
    r"(?ms)(?:^|\n)\s*(\d{1,3})\s+"
    r"([A-Za-z0-9_.\-\s]+?)\s+"
    r"((?:p\.)?[A-Z][A-Za-z]{0,2}\d+[A-Z*][A-Za-z]{0,2}|"
    r"\d+\.[A-Z]/[A-ZX]{1,3}|\d+-\d+\.-/[A-Z])\s+"
    r"([ACDEFGHIKLMNPQRSTVWY]{8,80})\s+"
    r"(MHC[-\s]?(?:I{1,2}|1|2))"
)


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\u00ad", "").replace("\ufeff", "")
    text = text.replace("鈭?, "-").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _value(value: Any) -> str:
    text = _clean(value)
    return text if text and text.lower() not in {"nan", "none", "null"} else NOT_SPECIFIED


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean(value).lower()).strip()


def _json_safe(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _dump_json(record: dict[str, Any]) -> str:
    return json.dumps({str(k): _json_safe(v) for k, v in record.items()}, ensure_ascii=False, sort_keys=True)


def normalize_mhc(value: Any, context: str = "") -> str:
    text = _value(value)
    combined = f"{text} {context}".lower()
    if re.search(r"\b(?:mhc|class)[-\s_]*ii\b", combined) or "class ii" in combined:
        return "MHC-II"
    if re.search(r"\b(?:mhc|class)[-\s_]*i\b", combined) or "class i" in combined:
        return "MHC-I"
    if text != NOT_SPECIFIED and "hla" in text.lower():
        return text
    return text


def _header_score(row: Iterable[Any]) -> float:
    values = [_norm(v) for v in row if _clean(v)]
    if not values:
        return -1.0
    joined = " | ".join(values)
    alias_hits = sum(any(alias in cell for alias in sum(FIELD_ALIASES.values(), [])) for cell in values)
    term_hits = sum(token in HEADER_TERMS for token in re.findall(r"[a-z0-9]+", joined))
    return alias_hits * 4 + term_hits * 0.25 + min(len(values), 12) * 0.05


def detect_header_rows(raw: pd.DataFrame, scan_rows: int = 40) -> tuple[int, int]:
    """Return zero-based (start, end-exclusive) header row indices."""
    if raw.empty:
        raise ValueError("The table is empty.")
    limit = min(scan_rows, len(raw))
    scores = [(_header_score(raw.iloc[i].tolist()), i) for i in range(limit)]
    score, start = max(scores)
    if score < 2.0:
        raise ValueError("Could not identify a plausible table header.")

    end = start + 1
    if end < len(raw):
        next_values = [_clean(v) for v in raw.iloc[end].tolist()]
        nonempty = [v for v in next_values if v]
        numeric = sum(bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", v)) for v in nonempty)
        header_words = sum(any(t in _norm(v).split() for t in HEADER_TERMS) for v in nonempty)
        biological = any(AA_RE.fullmatch(v) or MUTATION_RE.search(v) or MHC_RE.search(v) for v in nonempty)
        if nonempty and not biological and numeric / len(nonempty) < 0.35 and (header_words >= 1 or len(nonempty) <= 5):
            end += 1
    return start, end


def build_headers(raw: pd.DataFrame, start: int, end: int) -> list[str]:
    top = [_clean(v) for v in raw.iloc[start].tolist()]
    propagated: list[str] = []
    current = ""
    for value in top:
        if value:
            current = value
        propagated.append(current)

    headers: list[str] = []
    for col in range(raw.shape[1]):
        parts = []
        parent = top[col] or propagated[col]
        if parent:
            parts.append(parent)
        for row in range(start + 1, end):
            child = _clean(raw.iat[row, col])
            if child and _norm(child) != _norm(parent):
                parts.append(child)
        header = " | ".join(parts) if parts else f"Unnamed_Column_{col + 1}"
        base = header
        n = 2
        while header in headers:
            header = f"{base} [{n}]"
            n += 1
        headers.append(header)
    return headers


def _column_similarity(column: str, alias: str) -> float:
    c, a = _norm(column), _norm(alias)
    if not c or not a:
        return 0.0
    if c == a:
        return 100.0
    c_tokens, a_tokens = set(c.split()), set(a.split())
    if a in c:
        return 80.0 + min(len(a), 20) / 20
    overlap = len(c_tokens & a_tokens)
    return 60.0 * overlap / max(len(a_tokens), 1) + 20.0 * overlap / max(len(c_tokens), 1)


def map_columns(columns: Iterable[str]) -> dict[str, str | None]:
    columns = list(columns)
    mapping: dict[str, str | None] = {}
    used: set[str] = set()
    for field, aliases in FIELD_ALIASES.items():
        candidates = []
        for col in columns:
            if col in used:
                continue
            score = max(_column_similarity(col, alias) for alias in aliases)
            # Prefer biologically specific sequence columns over generic long peptides.
            if field == "Epitope_Sequence":
                cn = _norm(col)
                if "mutated peptide" in cn or "binding peptide" in cn:
                    score += 18
                if "wild type" in cn:
                    score -= 40
            if field == "Epitope_ID" and ("ept peptide id" in _norm(col) or "neoepitope" in _norm(col)):
                score += 8
            candidates.append((score, col))
        score, selected = max(candidates, default=(0.0, None))
        mapping[field] = selected if selected is not None and score >= 45 else None
        if mapping[field]:
            used.add(mapping[field])
    return mapping


def _row_is_data(record: dict[str, Any], mapping: dict[str, str | None]) -> bool:
    values = [_clean(v) for v in record.values()]
    nonempty = [v for v in values if v]
    if not nonempty:
        return False
    mapped_values = [_clean(record.get(col)) for col in mapping.values() if col]
    evidence = sum(bool(v) for v in mapped_values)
    biological = any(AA_RE.fullmatch(v) or MUTATION_RE.search(v) or MHC_RE.search(v) for v in nonempty)
    return evidence >= 2 or (evidence >= 1 and biological)


def _make_record(
    source: dict[str, Any], mapping: dict[str, str | None], *, source_file: Path,
    source_sheet: str, source_row: int, header_row: str, method: str, context: str,
) -> dict[str, Any]:
    core = {field: _value(source.get(col)) if col else NOT_SPECIFIED for field, col in mapping.items()}
    raw_mhc = core["MHC_Class"]
    core["MHC_Class"] = normalize_mhc(raw_mhc, context)
    mapped_columns = {col for col in mapping.values() if col}
    additional = {k: v for k, v in source.items() if k not in mapped_columns and _clean(v)}
    missing = [field for field in CORE_COLUMNS if core[field] == NOT_SPECIFIED]
    row = {
        **core,
        "Source_File": str(source_file.resolve()),
        "Source_Sheet": source_sheet,
        "Source_Page": NOT_SPECIFIED,
        "Source_Row": source_row,
        "Header_Row": header_row,
        "Extraction_Method": method,
        "Raw_Record_JSON": _dump_json(source),
        "Additional_Metadata_JSON": _dump_json(additional),
        "Manual_Review": bool(missing),
        "Review_Reason": f"Missing source fields: {', '.join(missing)}" if missing else "",
    }
    patient_col = next((c for c in source if "patient" in _norm(c) and "id" in _norm(c)), None)
    row["Patient_ID"] = _value(source.get(patient_col)) if patient_col else NOT_SPECIFIED
    row["MHC_or_HLA"] = raw_mhc
    return _add_legacy_aliases(row)


def _add_legacy_aliases(row: dict[str, Any]) -> dict[str, Any]:
    row["Gene"] = row["Epitope_Gene"]
    row["Mutation_or_Substitution"] = row["Epitope_Mutation"]
    row["Sequence"] = row["Epitope_Sequence"]
    row["MHC_type"] = row["MHC_Class"]
    row.setdefault("MHC_or_HLA", row["MHC_Class"])
    row.setdefault("Patient_ID", NOT_SPECIFIED)
    row.setdefault("Tumor_Model", NOT_SPECIFIED)
    row["Source_Document"] = row["Source_File"]
    row["Source_Table"] = row["Source_Sheet"]
    row["Row_Index_On_Page"] = row["Source_Row"]
    row.setdefault("Vaccine_Assignment_Method", "source_column" if row["Vaccine_or_Antigen_Set"] != NOT_SPECIFIED else "not_assigned")
    row.setdefault("Vaccine_Assignment_Confidence", "high" if row["Vaccine_or_Antigen_Set"] != NOT_SPECIFIED else "low")
    row.setdefault("_group_id", NOT_SPECIFIED)
    row.setdefault("raw_row_text", row["Raw_Record_JSON"])
    row["Stage6_Merge_Key"] = f"{row['Vaccine_or_Antigen_Set']}__Epitope_{row['Epitope_ID']}"
    return row


def _read_delimited(path: Path) -> pd.DataFrame:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "gb18030"):
        try:
            return pd.read_csv(path, header=None, dtype=object, sep=delimiter, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unable to decode {path}")


def extract_spreadsheet(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sheets = {path.stem: _read_delimited(path)}
    elif suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    else:
        raise ValueError(f"Unsupported spreadsheet type: {suffix}")

    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    sheet_summaries = []
    for sheet_name, raw in sheets.items():
        raw = raw.dropna(axis=1, how="all")
        try:
            header_start, header_end = detect_header_rows(raw)
        except ValueError as exc:
            audit.append({"Source_File": str(path.resolve()), "Source_Sheet": sheet_name, "Status": "sheet_skipped", "Reason": str(exc)})
            sheet_summaries.append({"sheet": sheet_name, "status": "skipped", "reason": str(exc), "rows": len(raw)})
            continue
        headers = build_headers(raw, header_start, header_end)
        mapping = map_columns(headers)
        context = " ".join(_clean(v) for v in raw.iloc[:header_start].to_numpy().ravel() if _clean(v))
        accepted = 0
        skipped = 0
        for idx in range(header_end, len(raw)):
            source = {headers[col]: raw.iat[idx, col] for col in range(len(headers))}
            if _row_is_data(source, mapping):
                records.append(_make_record(
                    source, mapping, source_file=path, source_sheet=sheet_name,
                    source_row=idx + 1, header_row=f"{header_start + 1}:{header_end}",
                    method="spreadsheet_auto_header_alias_mapping", context=context,
                ))
                status, reason, accepted = "accepted", "", accepted + 1
            else:
                status, reason, skipped = "skipped", "blank/note/non-data row (insufficient mapped biological evidence)", skipped + 1
            audit.append({
                "Source_File": str(path.resolve()), "Source_Sheet": sheet_name,
                "Source_Row": idx + 1, "Status": status, "Reason": reason,
                "Raw_Record_JSON": _dump_json(source),
            })
        sheet_summaries.append({
            "sheet": sheet_name, "status": "processed", "source_rows": len(raw),
            "header_rows": [header_start + 1, header_end], "mapping": mapping,
            "accepted_rows": accepted, "skipped_rows": skipped,
        })
    return _finalize(records), pd.DataFrame(audit), {"file": str(path.resolve()), "sheets": sheet_summaries}


def extract_pdf(path: str | Path, pages: Iterable[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    doc = fitz.open(str(path))
    page_numbers = list(pages) if pages is not None else list(range(1, len(doc) + 1))
    page_texts = {page_no: doc[page_no - 1].get_text("text") for page_no in page_numbers}
    doc.close()
    page_matches = {page_no: list(PDF_ROW_RE.finditer(text)) for page_no, text in page_texts.items()}
    data_pages = [page_no for page_no, matches in page_matches.items() if matches]
    # Group labels elsewhere in a supplement must not contaminate the table.
    # When pages are auto-scanned, derive labels only from pages that contain
    # actual reference-table records.
    label_page_texts = [page_texts[p] for p in data_pages] if data_pages else list(page_texts.values())
    combined = "\n".join(label_page_texts)
    labels = []
    for label in re.findall(r"\b(?:LPP|vaccine|pool|antigen)[-_ ][A-Za-z0-9]+\b", combined, re.I):
        clean = _clean(label)
        if clean.lower() not in {x.lower() for x in labels}:
            labels.append(clean)

    parsed = []
    audit = []
    for page_no, text in page_texts.items():
        matches = page_matches[page_no]
        for index, match in enumerate(matches, 1):
            parsed.append({
                "Epitope_ID": str(int(match.group(1))),
                "Epitope_Gene": _clean(match.group(2)),
                "Epitope_Mutation": _clean(match.group(3)),
                "Epitope_Sequence": _clean(match.group(4)),
                "MHC_Class": normalize_mhc(match.group(5)),
                "Source_Page": page_no,
                "Source_Row": index,
                "raw_text": _clean(match.group(0)),
            })
        audit.append({
            "Source_File": str(path.resolve()), "Source_Sheet": "PDF text table",
            "Source_Page": page_no, "Status": "processed",
            "Reason": f"parsed {len(matches)} record rows",
        })

    group_id = -1
    previous_id = None
    records = []
    for item in parsed:
        eid = int(item["Epitope_ID"])
        if previous_id is None or eid <= previous_id:
            group_id += 1
        previous_id = eid
        label = labels[group_id] if group_id < len(labels) else NOT_SPECIFIED
        raw = {
            "Epitope ID": item["Epitope_ID"], "Gene": item["Epitope_Gene"],
            "Substitution": item["Epitope_Mutation"], "Sequence": item["Epitope_Sequence"],
            "MHC type": item["MHC_Class"], "Vaccine": label,
        }
        row = {
            **{field: raw_value for field, raw_value in zip(CORE_COLUMNS, [
                item["Epitope_ID"], item["Epitope_Gene"], item["Epitope_Mutation"],
                item["Epitope_Sequence"], item["MHC_Class"], label,
            ])},
            "Source_File": str(path.resolve()), "Source_Sheet": "PDF text table",
            "Source_Page": item["Source_Page"], "Source_Row": item["Source_Row"],
            "Header_Row": NOT_SPECIFIED, "Extraction_Method": "pdf_text_record_regex_and_group_propagation",
            "Raw_Record_JSON": _dump_json(raw), "Additional_Metadata_JSON": "{}",
            "Manual_Review": label == NOT_SPECIFIED,
            "Review_Reason": "Vaccine/antigen group label unavailable" if label == NOT_SPECIFIED else "",
            "MHC_or_HLA": item["MHC_Class"], "Patient_ID": NOT_SPECIFIED,
            "_group_id": group_id,
            "Vaccine_Assignment_Method": "group_label_order_after_epitope_id_reset",
            "Vaccine_Assignment_Confidence": "high" if label != NOT_SPECIFIED else "low",
            "raw_row_text": item["raw_text"],
        }
        records.append(_add_legacy_aliases(row))
    summary = {
        "file": str(path.resolve()), "pages_checked": page_numbers,
        "pages_with_reference_records": data_pages,
        "detected_group_labels": labels, "parsed_rows": len(records),
        "groups": group_id + 1 if records else 0,
    }
    return _finalize(records), pd.DataFrame(audit), summary


def _finalize(records: list[dict[str, Any]]) -> pd.DataFrame:
    columns = CORE_COLUMNS + PROVENANCE_COLUMNS + [c for c in LEGACY_COLUMNS if c not in PROVENANCE_COLUMNS]
    if not records:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(records)
    for column in columns:
        if column not in df:
            df[column] = NOT_SPECIFIED
    return df[columns + [c for c in df.columns if c not in columns]]


def extract_table(path: str | Path, pages: Iterable[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, pages=pages)
    if suffix in {".xlsx", ".xls", ".csv", ".tsv"}:
        return extract_spreadsheet(path)
    raise ValueError(f"Unsupported Stage 5 input type: {suffix}")


def run_stage5(
    inputs: str | Path | Iterable[str | Path], *, out_root: str | Path = "stage5_final_universal_tool_table_extraction",
    pdf_pages: dict[str, Iterable[int]] | None = None,
) -> dict[str, Any]:
    paths = [Path(inputs)] if isinstance(inputs, (str, Path)) else [Path(p) for p in inputs]
    output = Path(out_root)
    output.mkdir(parents=True, exist_ok=True)
    all_records, all_audits, input_summaries = [], [], []
    for path in paths:
        pages = None
        if pdf_pages:
            pages = pdf_pages.get(str(path)) or pdf_pages.get(path.name)
        records, audit, summary = extract_table(path, pages=pages)
        all_records.append(records)
        all_audits.append(audit)
        input_summaries.append(summary)
    final = pd.concat(all_records, ignore_index=True) if all_records else _finalize([])
    audit = pd.concat(all_audits, ignore_index=True) if all_audits else pd.DataFrame()

    final_csv = output / "Epitope_Reference_Table_FINAL.csv"
    final_xlsx = output / "Epitope_Reference_Table_FINAL.xlsx"
    audit_csv = output / "Stage5_Row_Audit.csv"
    qc_json = output / "stage5_final_qc.json"
    final.to_csv(final_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    final.to_excel(final_xlsx, index=False)
    audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    qc = {
        "inputs": input_summaries,
        "final_reference_row_count": int(len(final)),
        "manual_review_count": int(final["Manual_Review"].astype(bool).sum()) if len(final) else 0,
        "core_schema": CORE_COLUMNS,
        "output_columns": list(final.columns),
        "outputs": {"final_csv": str(final_csv), "final_xlsx": str(final_xlsx), "row_audit_csv": str(audit_csv)},
    }
    qc_json.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"reference_df": final, "audit_df": audit, "qc": qc, "final_csv": final_csv, "final_xlsx": final_xlsx, "qc_json": qc_json}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out-root", default="stage5_final_universal_tool_table_extraction")
    parser.add_argument("--pdf-pages", nargs="*", type=int)
    args = parser.parse_args()
    page_map = {Path(p).name: args.pdf_pages for p in args.inputs if Path(p).suffix.lower() == ".pdf"} if args.pdf_pages else None
    result = run_stage5(args.inputs, out_root=args.out_root, pdf_pages=page_map)
    print(json.dumps(result["qc"], ensure_ascii=False, indent=2))
