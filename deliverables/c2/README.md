# C2 local delivery

This directory contains the nine authoritative C2 validation-selected runs (three learning rates x seeds 42, 43, and 44), validation-only evaluator CSVs, and training summaries. C2 trains ResNet50 layer3, layer4, and the classification head while applying the shared frozen-BatchNorm-statistics policy.

Selected LR: `1e-05`. Re-evaluated validation macro-F1: `0.98007839707344357 +/- 2.3778457343752538e-05` (mean and sample standard deviation across three seed-level scores, ddof=1; not pooled predictions and not an ensemble).

`runs/` preserves one `best.pt`, effective `config.yaml`, `metadata.json`, and `history.csv` for each authoritative completed run. The filenames are unchanged. Historical absolute paths inside copied metadata/config/history are preserved rather than rewritten. `evaluation/` keeps each LR separate and contains seed-specific predictions plus per-LR metrics and aggregates. `chosen_lr_manifest.json` references the same three selected-LR checkpoints already under `runs/`; no second checkpoint copies are made.

Interrupted or incompatible attempts, datasets, caches, logs, and environment files are excluded. Their sources remain untouched under `C:\Users\albiina\Computer_Vision-project\artifacts\c2_cpu_grid_6t_20261019_1420`. Training and validation actually ran on CPU with `torch_num_threads=6` and `num_workers=0`, as recorded in each effective config and metadata file. No test-split evaluation was performed.

The runs record Git commit `f98649d7545db9f02b45e5e1bc9c0a9693002038` with a dirty worktree and the exact training-source SHA-256 in each `metadata.json`. The run metadata marks the source worktree dirty. Its recorded `src/train.py` SHA-256 matches the current committed source file, but the metadata commit hash alone does not identify that exact source snapshot. CPU validation does not resolve the separate Task 1 GPU reproducibility check. Commit and delivery to the Team Lead remain pending user actions.
