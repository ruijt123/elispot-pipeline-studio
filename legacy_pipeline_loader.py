"""Load reusable stage libraries from the research notebook.

Only definition cells are executed. Installation, credential, configuration,
example-run, benchmark, and paper-specific cells are excluded. The loader
keeps the notebook as the provenance source while allowing the command-line
and Streamlit applications to use the revised generic heatmap implementation.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DEFAULT_NOTEBOOK = Path(__file__).resolve().parent / "notebooks" / "module1_generic_heatmap.ipynb"

# Cell indexes in module1_generic_heatmap.ipynb. Each list contains library
# definitions only. Stage 3/4 include Reviewer 6 audit wrappers; Stage 4 also
# includes the deterministic structure constraint and indexed-row reader.
STAGE_CELLS = {
    "stage1": [11],
    "stage2": [13],
    "stage3": [3, 16, 9, 10, 17, 18],
    "stage3bc": [3, 20, 9, 21],
    "stage4": [3, 23, 9, 10, 24, 25, 26, 27, 29],
    "stage6": [35],
}

STOP_MARKERS = {
    # Cell 35 ends with a paper-specific example invocation. The exported
    # run_stage6_final_elispot_merge function is defined before this marker.
    "stage6": "if stage4_source_data_result is not None",
}


def _quiet_display(*args: Any, **kwargs: Any) -> None:
    return None


def _python_source(source: str) -> str:
    """Remove Jupyter line magics from an otherwise ordinary Python cell."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )


def load_stage(
    stage: str,
    notebook_path: str | Path = DEFAULT_NOTEBOOK,
    *,
    run_root: str | Path | None = None,
    paper_id: str | None = None,
) -> SimpleNamespace:
    if stage not in STAGE_CELLS:
        raise KeyError(f"Unknown stage library: {stage}")
    notebook_path = Path(notebook_path).resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    effective_run_root = Path(run_root).resolve() if run_root else notebook_path.parent / "_runtime"
    effective_run_root.mkdir(parents=True, exist_ok=True)
    module_name = f"notebook_{stage}_{abs(hash((str(notebook_path), str(effective_run_root))))}"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    namespace: dict[str, Any] = module.__dict__
    namespace.update({
        "__name__": module_name,
        "__file__": str(notebook_path),
        "display": _quiet_display,
        "PROJECT_ROOT": notebook_path.parent.parent,
        "RUN_ROOT": effective_run_root,
        "PAPER_CONFIG": {"paper_id": paper_id or "runtime_input"},
    })
    for cell_index in STAGE_CELLS[stage]:
        try:
            source = "".join(notebook["cells"][cell_index]["source"])
        except (IndexError, KeyError) as exc:
            raise RuntimeError(
                f"Notebook {notebook_path} does not contain required {stage} cell {cell_index}"
            ) from exc
        marker = STOP_MARKERS.get(stage)
        if marker and marker in source:
            source = source.split(marker, 1)[0]
        source = _python_source(source)
        exec(compile(source, f"{notebook_path}#cell-{cell_index}", "exec"), namespace)
    return SimpleNamespace(**namespace)


def load_all(
    notebook_path: str | Path = DEFAULT_NOTEBOOK,
    *,
    run_root: str | Path | None = None,
) -> dict[str, SimpleNamespace]:
    return {
        stage: load_stage(stage, notebook_path, run_root=run_root)
        for stage in STAGE_CELLS
    }
