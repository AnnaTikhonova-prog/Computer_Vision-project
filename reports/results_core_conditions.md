# Final Test Results: Core Conditions (Augmentation On)

**Status:** completed for the four core training conditions, C0-C3. This report
covers the retained augmentation-on checkpoints and does not include the
augmentation-off ablation.

## Evaluation protocol

`src.evaluate` evaluated the frozen `splits/v2/test.csv` split, containing 300
images, for the three retained checkpoints (training seeds 42, 43, and 44) at
each condition's validation-selected learning rate. C0 used `1e-3`, C1 used
`1e-4`, and C2 and C3 used `1e-5`. Accuracy, macro-F1, and latency are reported
as the mean ± sample standard deviation (`ddof=1`) across the three
checkpoints. The three prediction files for each condition contain 900 rows:
300 test predictions per seed.

| Condition | Trainable modules | Test accuracy, mean ± SD | Test macro-F1, mean ± SD | Batch-1 latency, ms | Batch-32 latency, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| C0 | Classifier head | 0.9511 ± 0.0051 | 0.9512 ± 0.0051 | 120.77 ± 20.69 | 3260.32 ± 403.30 |
| C1 | `layer4` + classifier head | **0.9656 ± 0.0069** | **0.9655 ± 0.0070** | 151.39 ± 46.36 | 3102.85 ± 162.79 |
| C2 | `layer3`–`layer4` + classifier head | 0.9600 ± 0.0033 | 0.9601 ± 0.0033 | 154.85 ± 11.62 | 3008.04 ± 62.19 |
| C3 | All parameters | 0.9633 ± 0.0067 | 0.9633 ± 0.0067 | 125.99 ± 9.34 | 3007.90 ± 159.27 |

Latency was measured on the local CPU and is hardware-specific. Batch-32 latency
is total time for 32 images, not time per image. The exact CPU model was not
recorded in the source evaluation artifacts.

## Interpretation

C1 achieved the highest mean accuracy and macro-F1 on this fixed test split.
Compared with the head-only C0 baseline, C1 is higher by 1.44 percentage points
in accuracy and 1.43 points in macro-F1. Unfreezing more of the network did not
produce a monotonic increase: C2 and C3 scored below C1. These results describe
one 300-image test split and do not establish the same ranking on other data.

The test results are reported after validation-based model and learning-rate
selection. **No settings are changed based on test results.** Test inspection is
used only for final reporting and descriptive error analysis; tuning on test is
forbidden.

## Per-condition artifacts

For each condition, the per-class F1 table reports the mean and sample standard
deviation across its three checkpoints. Confusion matrices pool the three
checkpoints' predictions on the fixed test set. Prediction CSVs preserve the
seed and checkpoint for every row.

| Condition | Per-class F1 | Confusion matrix | Predictions | Latency details | Summary |
| --- | --- | --- | --- | --- | --- |
| C0 | [CSV](tables/C0_augment_on_test_per_class_f1.csv) | [PNG](figures/C0_augment_on_test_confusion_matrix.png) | [CSV](tables/C0_augment_on_test_predictions.csv) | [CSV](tables/C0_augment_on_test_latency.csv) | [CSV](tables/C0_augment_on_test_summary.csv) |
| C1 | [CSV](tables/C1_augment_on_test_per_class_f1.csv) | [PNG](figures/C1_augment_on_test_confusion_matrix.png) | [CSV](tables/C1_augment_on_test_predictions.csv) | [CSV](tables/C1_augment_on_test_latency.csv) | [CSV](tables/C1_augment_on_test_summary.csv) |
| C2 | [CSV](tables/C2_augment_on_test_per_class_f1.csv) | [PNG](figures/C2_augment_on_test_confusion_matrix.png) | [CSV](tables/C2_augment_on_test_predictions.csv) | [CSV](tables/C2_augment_on_test_latency.csv) | [CSV](tables/C2_augment_on_test_summary.csv) |
| C3 | [CSV](tables/C3_augment_on_test_per_class_f1.csv) | [PNG](figures/C3_augment_on_test_confusion_matrix.png) | [CSV](tables/C3_augment_on_test_predictions.csv) | [CSV](tables/C3_augment_on_test_latency.csv) | [CSV](tables/C3_augment_on_test_summary.csv) |

Three representative C0 test failures, their probable causes, and Grad-CAM
supporting visualizations are documented in the [failure analysis](failure_analysis.md).
The analysis is diagnostic only and was not used to change any experiment
settings.

## Scope and limitations

- The results apply to the project's custom frozen split, not the official
  Oxford-IIIT Pet benchmark split.
- The four selected conditions were trained in different historical compute
  environments. Differences between conditions are descriptive and should not
  be interpreted as a perfectly isolated causal effect of trainable depth.
- This report is limited to C0-C3 with augmentation on. The C0/C3
  augmentation-off ablation is summarized separately in [ablation.md](ablation.md).
