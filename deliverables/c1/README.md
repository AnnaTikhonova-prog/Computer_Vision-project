# C1 local delivery

This directory contains the nine C1 validation-selected runs (three learning rates x seeds 42, 43, and 44), validation-only evaluator CSVs, and training summaries.

Selected LR: `1e-04`. Re-evaluated validation macro-F1: `0.98556759181619447 +/- 0.0038371311380492102` (mean and sample standard deviation across three seed-level scores, ddof=1; not pooled predictions and not an ensemble).

`runs/` preserves one `best.pt`, effective `config.yaml`, `metadata.json`, `history.csv`, and `runtime.json` for each completed run. The filenames and file contents are unchanged. `evaluation/` keeps each LR separate and contains seed-specific predictions plus per-LR metrics and aggregates. `chosen_lr_manifest.json` references the same three selected-LR checkpoints already under `runs/`; no second checkpoint copies are made.
