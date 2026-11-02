# Final and Immutable Results

**Status:** final. All six frozen configurations were evaluated on the fixed
`splits/v2/test.csv` split (300 images). Accuracy and macro-F1 were not rerun
during the finalization work described here.

## Quality metrics

Values are mean ± sample standard deviation (`ddof=1`) across the retained
training-seed checkpoints 42, 43, and 44. Model and learning-rate selection was
validation-only; test results were used only for final reporting and diagnostic
error analysis.

| Configuration | Selected LR | Test accuracy | Test macro-F1 | Exact source |
| --- | ---: | ---: | ---: | --- |
| C0 · augmentation on | `1e-3` | 0.9511 ± 0.0051 | 0.9512 ± 0.0051 | [summary](tables/C0_augment_on_test_summary.csv) |
| C1 · augmentation on | `1e-4` | 0.9656 ± 0.0069 | 0.9655 ± 0.0070 | [summary](tables/C1_augment_on_test_summary.csv) |
| C2 · augmentation on | `1e-5` | 0.9600 ± 0.0033 | 0.9601 ± 0.0033 | [summary](tables/C2_augment_on_test_summary.csv) |
| C3 · augmentation on | `1e-5` | 0.9633 ± 0.0067 | 0.9633 ± 0.0067 | [summary](tables/C3_augment_on_test_summary.csv) |
| C0 · augmentation off | `1e-3` | 0.9489 ± 0.0019 | 0.9485 ± 0.0020 | [summary](tables/C0_augment_off_test_summary.csv) |
| C3 · augmentation off | `1e-5` | 0.9578 ± 0.0107 | 0.9579 ± 0.0107 | [summary](tables/C3_augment_off_test_summary.csv) |

The source CSVs retain full precision. Per-class F1, predictions, and confusion
matrices are linked from the completed [six-configuration report](results_preliminary.md).

## Latency-only rerun

The original latency CSVs did not record their CPU model, and no reliable
existing evidence recovered it. They remain historical artifacts but are not
the canonical latency values below. On 2 November 2026, the reserved rerun
buffer was therefore used for latency only. No dataset was loaded and no
accuracy or F1 value was recomputed.

| Configuration | Batch 1, ms | Batch 32, ms per batch |
| --- | ---: | ---: |
| C0 · augmentation on | 79.40 ± 4.88 | 2031.67 ± 14.41 |
| C1 · augmentation on | 118.97 ± 57.01 | 2421.08 ± 359.52 |
| C2 · augmentation on | 86.72 ± 6.62 | 2162.50 ± 94.36 |
| C3 · augmentation on | 87.58 ± 11.59 | 2045.25 ± 7.39 |
| C0 · augmentation off | 96.01 ± 36.43 | 2403.26 ± 494.30 |
| C3 · augmentation off | 90.74 ± 18.80 | 2100.44 ± 48.81 |

These values are mean ± sample standard deviation of the three checkpoint
means. Each checkpoint mean uses 10 timed forward passes after 3 warm-ups, with
`torch.inference_mode()`, `time.perf_counter`, and a synthetic zero-valued
float32 input of shape N×3×224×224. Measurements used CPU only: 13th Gen
Intel(R) Core(TM) i5-1335U, 12 logical CPUs, 10 PyTorch intra-op and 10 inter-op
threads, Windows 11 build 26300, Python 3.14.6, PyTorch 2.14.1+cpu, and
torchvision 0.29.1+cpu. Full per-checkpoint timings are in the [rerun CSV](tables/latency_rerun_2026-11-02.csv),
and timestamp, checkpoint hashes, Git commit, runtime, and storage provenance
are in the [runtime record](tables/latency_rerun_2026-11-02_runtime.json). The
reproducible latency-only entry point is
[`scripts/benchmark_inference_latency.py`](../scripts/benchmark_inference_latency.py).

## Finality declaration

**Declared 2 November 2026: the accuracy, macro-F1, and dated latency-rerun
metrics in this report are final and immutable.** They must not be replaced or
retuned. Any future replication must be reported separately and must preserve
this report and its source artifacts unchanged.

The reserved buffer is closed as **used for the documented latency-only rerun**.
No retraining or quality-metric rerun was needed, and no further rerun remains
open.
