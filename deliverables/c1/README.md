# C1 local delivery

This directory contains nine completed C1 runs (three learning rates x seeds 42, 43, and 44), validation-only evaluator CSVs, and training summaries.

Selected LR: `1e-4`. Re-evaluated validation macro-F1: `0.98556759181619447 +/- 0.0038371311380492102` (mean and sample standard deviation across three seed-level scores, ddof=1; not pooled predictions and not an ensemble).

`runs/` preserves `best.pt`, effective `config.yaml`, `metadata.json`, `history.csv`, and `runtime.json` for every run, byte-for-byte. `evaluation/` uses the same CSV layout and columns as the C0 delivery: seed-specific predictions, per-LR metrics and aggregates, learning-rate selection, and report mapping. `chosen_lr_manifest.json` references the three selected-LR checkpoints already under `runs/`; no second checkpoint copies are made and no seed is selected.

All nine checkpoints were evaluated on the 300 frozen `splits/v2/val.csv` images with the shared `scripts/evaluate_c0_validation.py:evaluate_run` implementation. Image content hashes were checked against the manifest. Every re-evaluated macro-F1 equals its saved training score (absolute difference zero; tolerance 1e-12). No test images were evaluated and no test metrics were used for selection.

Training ran on Kaggle Tesla T4; this independent evaluation ran locally on CPU with Python 3.14.0, PyTorch 2.10.0+cpu, and torchvision 0.25.0+cpu, batch size 32, four compute threads, and no loader workers. These evaluation settings do not modify the recorded training configs. Full evaluation provenance is recorded in `manifest.json`.

The source ZIP archives are unchanged. `manifest.json` records source archive/member paths and SHA-256 hashes. Checkpoints are covered by the C1 Git LFS rule. Commit and push have not been performed.
