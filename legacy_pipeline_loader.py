"""Load the stable Stage 1-4 and Stage 6 function libraries from the legacy notebook.

The original research notebook remains the provenance source.  Only definition
cells are executed: install cells, embedded credentials, and example run cells
are never loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DEFAULT_NOTEBOOK = Path(__file__).resolve().parent / "notebooks" / "module1_test-Copy1 (1) (1).ipynb"

STAGE_CELLS = {
    "stage1": (8, None),
    "stage2": (12, None),
    "stage3": (20, None),
    "stage3bc": (25, None),
    "stage4": (32, "STAGE4_INPUT_PATH ="),
    "stage6": (46, "STAGE4_CELL_RECORDS_PATHS ="),
}


def _quiet_display(*args: Any, **kwargs: Any) -> None:
    return None


def load_stage(stage: str, notebook_path: str | Path = DEFAULT_NOTEBOOK) -> SimpleNamespace:
    if stage not in STAGE_CELLS:
        raise KeyError(f"Unknown stage library: {stage}")
    notebook_path = Path(notebook_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell_index, stop_marker = STAGE_CELLS[stage]
    source = "".join(notebook["cells"][cell_index]["source"])
    if stop_marker and stop_marker in source:
        source = source.split(stop_marker, 1)[0]
    namespace: dict[str, Any] = {
        "__name__": f"legacy_{stage}",
        "__file__": str(notebook_path),
        "display": _quiet_display,
    }
    exec(compile(source, f"{notebook_path}#cell-{cell_index}", "exec"), namespace)
    return SimpleNamespace(**namespace)


def load_all(notebook_path: str | Path = DEFAULT_NOTEBOOK) -> dict[str, SimpleNamespace]:
    return {stage: load_stage(stage, notebook_path) for stage in STAGE_CELLS}
