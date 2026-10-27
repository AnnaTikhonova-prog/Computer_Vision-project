# C0 production grid status — 2026-10-13

## Status

Task 2 is **in progress**, not complete. Three of nine production runs are
complete and validated. A local hidden process is continuing the remaining grid
sequentially. No learning rate or seed has been selected, and no test metrics
have been computed.

- Background launcher PID at handoff: `8396`
- Batch directory: `artifacts/c0_cpu_grid`
- Live stdout: `artifacts/c0_cpu_grid/orchestrator.restart2.stdout.log`
- Live stderr: `artifacts/c0_cpu_grid/orchestrator.restart2.stderr.log`
- Batch provenance: `artifacts/c0_cpu_grid/batch_manifest.json`
- Active cell at handoff: C0, LR `1e-4`, seed 42,
  `runs/c0_lr1e-4_seed42_attempt2`

The earlier `runs/c0_lr1e-4_seed42` attempt was interrupted after six complete
epochs when the attached chat execution was converted to a background process.
It is preserved and is not considered completed. The fresh attempt starts from
the pretrained backbone and freshly seeded head as required.

## Completed results

| LR | Seed | Selected epoch | Selected validation macro-F1 | Epochs completed | Status |
|---:|---:|---:|---:|---:|:---|
| 1e-3 | 42 | 9 | 0.9666666666666668 | 14 | complete |
| 1e-3 | 43 | 9 | 0.9700500138927481 | 14 | complete |
| 1e-3 | 44 | 11 | 0.973482142857143 | 16 | complete |
| 1e-4 | 42 | — | — | — | running fresh attempt |
| 1e-4 | 43 | — | — | — | pending |
| 1e-4 | 44 | — | — | — | pending |
| 1e-5 | 42 | — | — | — | pending |
| 1e-5 | 43 | — | — | — | pending |
| 1e-5 | 44 | — | — | — | pending |

The completed checkpoint hashes are evidence only; these are not a chosen-LR
delivery manifest because the grid is incomplete:

- seed 42: `f6bcabb8c763c3a4d03241da3f93074aef68fc71c8fcd348d95c42db7c55f170`
- seed 43: `0fc7a8cc6211f1af729846681b69013e55627fbec48bd760a72383a3d3b40814`
- seed 44: `c462c91bba77d3ef833f6dac946d2126850134f06ca624551134aa1a09bb0f2b`

## Protocol

- Config: `configs/c0_cpu.yaml`, inheriting the unmodified production C0
  settings and changing only `device: cpu`.
- Full frozen `splits/v2`: 1,399 train, 300 validation, 300 test.
- Test loader is constructed by the shared loader but never iterated by
  `src/train.py`; selection is validation-only.
- ImageNet V2 ResNet50, head-only training, augmentation enabled.
- AdamW, batch size 32, microbatch size 4, no scheduler.
- Maximum 30 epochs, early-stopping patience 5.
- Strict `>` checkpoint replacement; the earliest epoch wins an exact score tie.
- LR tie rule declared before aggregation: smaller LR on an exact full-precision
  mean tie.
- Reported spread after completion: sample standard deviation (`ddof=1`).

## Commands

Monitor the current process and logs:

```powershell
Get-Process -Id 8396
Get-Content artifacts\c0_cpu_grid\orchestrator.restart2.stdout.log -Wait
Get-Content artifacts\c0_cpu_grid\orchestrator.restart2.stderr.log
```

Do **not** run a second orchestrator concurrently. If PID 8396 is no longer
present before the batch completes, restart safely with:

```powershell
.\.venv\Scripts\python.exe scripts\run_c0_grid.py `
  --batch-dir artifacts\c0_cpu_grid --config configs\c0_cpu.yaml
```

The restart validates and reuses compatible completed runs, preserves partial
runs, and creates a fresh `_attemptN` directory for an incomplete cell.

After all nine runs complete, regenerate and validate the summary with:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_c0_grid.py `
  --batch-dir artifacts\c0_cpu_grid --config configs\c0_cpu.yaml
```

Expected final files are `summary/runs.csv`, `summary/learning_rates.csv`,
`summary/delivery_manifest.json`, and `summary/report.md`. The summarizer refuses
missing or duplicate completed cells and does not produce a provisional result.

## Repository delivery conventions

`artifacts/` and `reports/` are currently ignored by `.gitignore`. Git LFS 3.7.1
is installed, but the repository has no `.gitattributes` and no configured LFS
pattern. Before a future user-authorized commit, the team must decide a tracked
artifact location or narrow `.gitignore` exception and configure `*.pt` with Git
LFS. Nothing has been staged, committed, uploaded, or delivered.

Commit status: **PENDING USER ACTION**.
Delivery to Team Lead: **PENDING USER ACTION**.
