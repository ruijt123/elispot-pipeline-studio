# ELISpot Pipeline Studio

A Streamlit interface for the six-stage ELISpot reconstruction pipeline.

## Features

- Upload a main-paper PDF and a supplement in PDF, Excel, CSV, or TSV format.
- Limit the main-paper run to selected pages for inexpensive testing.
- Enter the DashScope API key in a password field; it is used in memory only.
- Inspect selected heatmap crops, recovered panels, matrix dimensions, and
  cell-level values while the pipeline runs.
- Inspect the normalized Stage 5 reference table and row-level extraction audit.
- Download the final Stage 6 workbook or the complete provenance package.
- Resume completed stages for the same inputs.

## Local Windows launch

```powershell
.\start.ps1
```

Alternatively:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The original research notebook under `notebooks/` is retained as the provenance
source for the current Stage 1-4 and Stage 6 implementations. Stage 5 is the
generalized table extractor in `stage5.py`.

## Security

Do not commit API keys. The app does not write the key to the run directory,
manifest, workbook, or source files. This version is intended for local or
single-user deployment because the legacy model client reads its credential
from a process environment variable during execution.
