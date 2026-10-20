# C3 local delivery

This directory contains the nine C3 validation-selected runs (three learning rates x seeds 42, 43, and 44), validation-only evaluator CSVs, and training summaries.

Selected LR: `1e-05`. Re-evaluated validation macro-F1: `0.97669574682732463 +/- 0.0066161717377039506` (mean and sample standard deviation across three seed-level scores, ddof=1; not pooled predictions and not an ensemble).

`runs/` preserves one `best.pt`, effective `config.yaml`, `metadata.json`, and `history.csv` for each completed run. The filenames and file contents are unchanged. `evaluation/` keeps each LR separate and contains seed-specific predictions plus per-LR metrics and aggregates. `chosen_lr_manifest.json` references the same three selected-LR checkpoints already under `runs/`; no second checkpoint copies are made. `training_summary/` also contains train/validation curves and an overfitting check for the selected LR.
