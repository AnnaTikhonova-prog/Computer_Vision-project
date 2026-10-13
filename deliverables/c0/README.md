# C0 local delivery

This directory contains the nine authoritative C0 validation-selected runs (three learning rates x seeds 42, 43, and 44), validation-only evaluator CSVs, and training summaries.

Selected LR: `1e-03`. Re-evaluated validation macro-F1: `0.97006627447218596 +/- 0.0034077671913688083` (mean and sample standard deviation across three seed-level scores, ddof=1; not pooled predictions and not an ensemble).

`runs/` preserves one `best.pt`, effective `config.yaml`, `metadata.json`, and `history.csv` for each authoritative completed run. The filenames are unchanged. Historical absolute paths inside copied metadata/config/history are preserved rather than rewritten. `evaluation/` keeps each LR separate and contains seed-specific predictions plus per-LR metrics and aggregates. `chosen_lr_manifest.json` references the same three selected-LR checkpoints already under `runs/`; no second checkpoint copies are made.

The interrupted `c0_lr1e-4_seed42` attempt, datasets, caches, logs, and environment files are excluded. Its source remains untouched under `artifacts/c0_cpu_grid`. No test-split evaluation was performed. CPU validation does not resolve the separate Task 1 GPU reproducibility check. Commit and delivery to the Team Lead remain pending user actions.
