# ELISpot Validation Dataset v1.1-revision-data

This revision package matches the generalized structure-constrained heatmap
pipeline and the revised evaluation. It contains derived tables, validation
records, prompt and inference audit files, and plots. It does not redistribute
article PDFs, supplementary files, publisher figure images, or API keys.

## Reported evaluation

- 7 papers, 10 heatmap panels, and 1,876 cells in the Stage 4 development benchmark.
- Structure recovered exactly for 10/10 panels.
- 398/399 printed integers matched after the documented disagreement reread.
- The automatic route before the local reread matched 339/399 printed integers.
- Four colour-only panels (977 cells) were correctly treated as numeric abstentions.
- Stage 5-6 was applicable to 4 papers and 7 panels, producing 980 integrated cells; three papers were excluded with reasons recorded in `stage56/stage56_exclusions.csv`.
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

- `fan_2024` — DOI [`10.1126/sciadv.adn9961`](https://doi.org/10.1126/sciadv.adn9961); https://doi.org/10.1126/sciadv.adn9961
- `braun_2025` — DOI [`10.1038/s41586-024-08507-5`](https://doi.org/10.1038/s41586-024-08507-5); https://www.nature.com/articles/s41586-024-08507-5
- `lindner_2025` — DOI [`10.1136/jitc-2025-012216`](https://doi.org/10.1136/jitc-2025-012216); https://pmc.ncbi.nlm.nih.gov/articles/PMC12458743/
- `haars_2021` — DOI [`10.1002/cpt.2319`](https://doi.org/10.1002/cpt.2319); https://pmc.ncbi.nlm.nih.gov/articles/PMC8399379/
- `zhang_2024` — DOI [`10.1016/j.crmeth.2024.100906`](https://doi.org/10.1016/j.crmeth.2024.100906); https://pmc.ncbi.nlm.nih.gov/articles/PMC11705763/
- `geiger_2020` — DOI [`10.3389/fimmu.2020.00412`](https://doi.org/10.3389/fimmu.2020.00412); https://pmc.ncbi.nlm.nih.gov/articles/PMC7076177/
- `hhv6_2015` — DOI [`10.1371/journal.pone.0142871`](https://doi.org/10.1371/journal.pone.0142871); https://pmc.ncbi.nlm.nih.gov/articles/PMC4658110/

The `source_url` and `doi` columns provide provenance. Obtain original articles,
supplements, and figures from their publishers or repositories under the
applicable terms.
