# Augmentation Ablation

The final test evaluation of the augmentation-on/off configurations is included
in the consolidated [results report](results.md). This document records the
validation protocol and the limitations that affect comparability.

Validation macro-F1 is reported as the mean ± sample standard deviation across
seeds 42, 43, and 44 (`ddof=1`).

| Condition | Augmentation on | Augmentation off |
|---|---:|---:|
| C0 | 0.9700663 ± 0.0034078 | 0.9597596 ± 0.0057848 |
| C3 | 0.9766957 ± 0.0066162 | 0.9677224 ± 0.0052049 |

## Protocol

- C0 uses the selected LR `1e-3`; C3 uses `1e-5`. No new learning-rate search was performed.
- Training seeds are `42, 43, 44`; `split_seed=42`.
- Each checkpoint was selected by validation macro-F1, with at most 30 epochs and early-stopping patience of 5.
- With augmentation disabled, training uses the deterministic `ResNet50_Weights.IMAGENET1K_V2.transforms()` transform; training shuffling remains enabled.
- The new runs used CPU, `num_workers=0`, one CPU thread, float32, and the current shared trainer.

## Sources

- Full aggregation: `artifacts/ablation_noaug/summary/ablation.csv`.
- C0 with augmentation: `deliverables/c0/evaluation/lr_1e-3/metrics.csv`.
- C3 with augmentation: `deliverables/c3/evaluation/lr_1e-5/metrics.csv`.
- C0/C3 without augmentation: histories, configs, metadata, and checkpoints under `artifacts/ablation_noaug/runs/`.

## Comparability limitations

C0 with augmentation was trained on CPU using Python 3.14.6 and torch 2.14.1
from the historical dirty commit `155bf1b`. C3 with augmentation was trained
on a Tesla T4 using Python 3.13.15 and torch 2.11.0+cu128 from commit
`37f1e66`. Both runs without augmentation were performed locally on CPU using
the current trainer from commit `367b6c2`. The historical same-seed head
initialization hashes differ between C0 and C3, so this is not a perfectly
controlled isolation of augmentation. Split membership, labels, and row order
match; the C3 byte-hash difference is due to CRLF/LF line endings.

## Interpretation

Across these six runs, disabling augmentation changed mean validation
macro-F1 by -0.010307 for C0 and -0.008973 for C3. This descriptive comparison
does not establish statistical significance or causality.
