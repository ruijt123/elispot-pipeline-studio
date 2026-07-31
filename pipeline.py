"""One-command ELISpot reconstruction pipeline (Stages 1-6).

Example:
    python pipeline.py --paper main.pdf --supplement supplement.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legacy_pipeline_loader import DEFAULT_NOTEBOOK, load_stage
from stage5 import run_stage5


AI_STAGES = {"stage3", "stage3bc", "stage4"}
STAGE_ORDER = ["stage1", "stage2", "stage3", "stage3bc", "stage4", "stage5", "stage6"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(path: Path) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_.")
    return text or "article"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_ai_credentials() -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Stage 3, Stage 3B/3C and Stage 4 require "
            "a valid DashScope key. Set it in the environment and rerun; completed "
            "local stages will be reused."
        )


def _figure_ids_for_assay(stage2_summary: Path, assay: str) -> list[str]:
    summary = _read_json(stage2_summary)
    units = summary.get("all_figure_units") or summary.get("figure_units") or []
    assay_norm = assay.lower()
    aliases = [assay_norm]
    if assay_norm == "elispot":
        aliases += ["eli spot", "immunospot", "ifn-纬", "ifn-g", "ifn 纬"]
    selected = []
    for unit in units:
        legend = str(unit.get("full_legend") or unit.get("legend_text") or "").lower()
        if any(alias in legend for alias in aliases):
            selected.append(str(unit["figure_unit_id"]))
    return selected


def preflight(paper: Path, supplement: Path, notebook: Path, require_ai: bool = True) -> dict[str, Any]:
    _require_file(paper, "Main paper")
    _require_file(supplement, "Supplement")
    _require_file(notebook, "Legacy pipeline notebook")
    if paper.suffix.lower() != ".pdf":
        raise ValueError("The current Stage 1 requires the main paper as a PDF.")
    if supplement.suffix.lower() not in {".pdf", ".xlsx", ".xls", ".csv", ".tsv"}:
        raise ValueError("Supplement must be PDF, XLSX, XLS, CSV, or TSV.")
    dependencies = {}
    for module in ["fitz", "cv2", "numpy", "pandas", "PIL", "openai"]:
        try:
            __import__(module)
            dependencies[module] = "ok"
        except Exception as exc:
            dependencies[module] = f"missing: {exc}"
    missing = [name for name, status in dependencies.items() if status != "ok"]
    if missing:
        raise RuntimeError(f"Missing Python dependencies: {', '.join(missing)}")
    if require_ai:
        _require_ai_credentials()
    return {
        "paper": str(paper.resolve()), "supplement": str(supplement.resolve()),
        "notebook": str(notebook.resolve()), "dependencies": dependencies,
        "dashscope_key": "set" if os.getenv("DASHSCOPE_API_KEY") else "missing",
    }


class ELISpotPipeline:
    def __init__(
        self, paper: str | Path, supplement: str | Path, *,
        output_root: str | Path = "pipeline_runs", notebook: str | Path = DEFAULT_NOTEBOOK,
        assay: str = "ELISpot", stage3_model: str = "qwen3-vl-plus",
        legend_model: str = "qwen3.7-plus", stage4_model: str = "qwen3-vl-plus",
        resume: bool = True, pages: list[int] | None = None,
    ):
        self.paper = Path(paper).resolve()
        self.supplement = Path(supplement).resolve()
        self.notebook = Path(notebook).resolve()
        self.assay = assay
        self.stage3_model = stage3_model
        self.legend_model = legend_model
        self.stage4_model = stage4_model
        self.resume = resume
        self.pages = sorted(set(int(p) for p in pages)) if pages else None
        run_name = _slug(self.paper)
        if self.pages:
            run_name += "_pages_" + "_".join(str(p) for p in self.pages)
        self.run_root = Path(output_root).resolve() / run_name
        self.manifest_path = self.run_root / "pipeline_manifest.json"
        fresh_manifest = {
            "pipeline": "ELISpot Stage 1-6", "created_at": _now(),
            "paper": str(self.paper), "supplement": str(self.supplement),
            "assay": assay, "main_paper_pages": self.pages or "all",
            "run_root": str(self.run_root), "stages": {},
        }
        if resume and self.manifest_path.exists():
            try:
                existing = _read_json(self.manifest_path)
                same_inputs = (
                    existing.get("paper") == str(self.paper)
                    and existing.get("supplement") == str(self.supplement)
                )
                self.manifest = existing if same_inputs else fresh_manifest
            except (OSError, ValueError, TypeError):
                self.manifest = fresh_manifest
        else:
            self.manifest = fresh_manifest

    def _stage_dir(self, stage: str) -> Path:
        return self.run_root / stage

    def _record(self, stage: str, status: str, **details: Any) -> None:
        self.manifest["stages"][stage] = {"status": status, "updated_at": _now(), **details}
        _write_json(self.manifest_path, self.manifest)

    def _reuse(self, stage: str, required: list[Path]) -> bool:
        if not self.resume or not all(path.exists() for path in required):
            return False
        self._record(stage, "reused", outputs=[str(p) for p in required])
        return True

    def run(self, stop_after: str | None = None) -> dict[str, Any]:
        if stop_after not in STAGE_ORDER + [None]:
            raise ValueError(f"stop_after must be one of {STAGE_ORDER}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        preflight_info = preflight(self.paper, self.supplement, self.notebook, require_ai=False)
        self.manifest["preflight"] = preflight_info
        _write_json(self.manifest_path, self.manifest)
        try:
            self._run_stage1()
            if stop_after == "stage1": return self.manifest
            self._run_stage2()
            if stop_after == "stage2": return self.manifest
            _require_ai_credentials()
            self._run_stage3()
            if stop_after == "stage3": return self.manifest
            self._run_stage3bc()
            if stop_after == "stage3bc": return self.manifest
            self._run_stage4()
            if stop_after == "stage4": return self.manifest
            self._run_stage5()
            if stop_after == "stage5": return self.manifest
            self._run_stage6()
            self.manifest["status"] = "complete"
            self.manifest["completed_at"] = _now()
            _write_json(self.manifest_path, self.manifest)
            return self.manifest
        except Exception as exc:
            self.manifest["status"] = "failed"
            self.manifest["error"] = str(exc)
            self.manifest["traceback"] = traceback.format_exc()
            _write_json(self.manifest_path, self.manifest)
            raise

    def _run_stage1(self) -> None:
        out = self._stage_dir("stage1")
        summary = out / "stage1_summary.json"
        if self._reuse("stage1", [summary]): return
        lib = load_stage("stage1", self.notebook)
        result = lib.process_pdf_stage1(
            pdf_path=str(self.paper), pages=self.pages, out_root=str(out),
            source_doc=self.paper.name, source_section="main_article",
        )
        self._record("stage1", "complete", summary=str(summary), qc=result.get("stage1_qc", {}))

    def _run_stage2(self) -> None:
        out = self._stage_dir("stage2")
        summary = out / "stage2_summary.json"
        if self._reuse("stage2", [summary]): return
        lib = load_stage("stage2", self.notebook)
        result = lib.process_stage2_from_stage1(
            stage1_summary_path=str(self._stage_dir("stage1") / "stage1_summary.json"),
            out_root=str(out), use_filter=False,
        )
        self._record("stage2", "complete", summary=str(summary), qc=result.get("stage2_qc", {}))

    def _run_stage3(self) -> None:
        out = self._stage_dir("stage3")
        manifest = out / "stage3_selected_heatmap_context_manifest.xlsx"
        if self._reuse("stage3", [manifest]): return
        stage2_summary = self._stage_dir("stage2") / "stage2_summary.json"
        figure_ids = _figure_ids_for_assay(stage2_summary, self.assay)
        if not figure_ids:
            raise RuntimeError(f"Stage 2 found no figure legend containing assay '{self.assay}'.")
        lib = load_stage("stage3", self.notebook)
        result = lib.process_stage3_heatmaps_from_stage2(
            stage2_summary_path=str(stage2_summary), out_root=str(out),
            model_name=self.stage3_model, target_assay=self.assay,
            figure_unit_filter=figure_ids,
        )
        if int(result.get("stage3_qc", {}).get("total_selected_heatmap_contexts", 0)) == 0:
            raise RuntimeError("Stage 3 completed but selected no ELISpot heatmap contexts.")
        self._record("stage3", "complete", manifest=str(manifest), figure_unit_filter=figure_ids, qc=result.get("stage3_qc", {}))

    def _run_stage3bc(self) -> None:
        out = self._stage_dir("stage3bc")
        handoff = out / "Stage3C_Clean_Handoff_Manifest.xlsx"
        if self._reuse("stage3bc", [handoff]): return
        lib = load_stage("stage3bc", self.notebook)
        result = lib.run_stage3b3c_panel_recovery_and_legend_extraction(
            stage3_selected_manifest_path=str(self._stage_dir("stage3") / "stage3_selected_heatmap_context_manifest.xlsx"),
            stage2_summary_path=str(self._stage_dir("stage2") / "stage2_summary.json"),
            out_root=str(out), model_name=self.legend_model,
        )
        self._record("stage3bc", "complete", handoff=str(handoff), qc=result.get("qc", {}))

    def _run_stage4(self) -> None:
        out = self._stage_dir("stage4")
        records = out / "Stage4_Cell_Records.xlsx"
        if self._reuse("stage4", [records]): return
        lib = load_stage("stage4", self.notebook)
        result = lib.run_stage4_generic_all_selected(
            stage3_selected_manifest_path=str(self._stage_dir("stage3bc") / "Stage3C_Clean_Handoff_Manifest.xlsx"),
            out_root=str(out), model_name=self.stage4_model, debug_display=False,
        )
        if not records.exists():
            raise RuntimeError("Stage 4 did not create Stage4_Cell_Records.xlsx")
        self._record("stage4", "complete", records=str(records), qc=result.get("qc", {}))

    def _run_stage5(self) -> None:
        out = self._stage_dir("stage5")
        reference = out / "Epitope_Reference_Table_FINAL.xlsx"
        if self._reuse("stage5", [reference]): return
        result = run_stage5(self.supplement, out_root=out)
        if len(result["reference_df"]) == 0:
            raise RuntimeError("Stage 5 found no epitope reference records in the supplement.")
        self._record("stage5", "complete", reference=str(reference), qc=result["qc"])

    def _run_stage6(self) -> None:
        out = self._stage_dir("stage6")
        final = out / "Stage6_Final_ELISpot_Output.xlsx"
        if self._reuse("stage6", [final]): return
        lib = load_stage("stage6", self.notebook)
        result = lib.run_stage6_final_elispot_merge(
            stage4_cell_records_paths=[str(self._stage_dir("stage4") / "Stage4_Cell_Records.xlsx")],
            reference_table_path=str(self._stage_dir("stage5") / "Epitope_Reference_Table_FINAL.xlsx"),
            out_root=str(out),
        )
        self._record("stage6", "complete", final=str(final), qc=result.get("qc", {}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", required=True, help="Main-paper PDF")
    parser.add_argument("--supplement", required=True, help="Supplement PDF/XLSX/XLS/CSV/TSV")
    parser.add_argument("--output-root", default="pipeline_runs")
    parser.add_argument("--assay", default="ELISpot")
    parser.add_argument("--pages", nargs="+", type=int, help="Only process these 1-based main-paper pages, e.g. --pages 3 4")
    parser.add_argument("--notebook", default=str(DEFAULT_NOTEBOOK))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after", choices=STAGE_ORDER)
    parser.add_argument("--preflight", action="store_true", help="Validate inputs/dependencies only")
    args = parser.parse_args(argv)
    paper, supplement, notebook = Path(args.paper), Path(args.supplement), Path(args.notebook)
    if args.preflight:
        print(json.dumps(preflight(paper, supplement, notebook, require_ai=True), ensure_ascii=False, indent=2))
        return 0
    pipeline = ELISpotPipeline(
        paper, supplement, output_root=args.output_root, notebook=notebook,
        assay=args.assay, resume=not args.no_resume, pages=args.pages,
    )
    manifest = pipeline.run(stop_after=args.stop_after)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise
