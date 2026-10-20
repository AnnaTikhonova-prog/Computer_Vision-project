# Second Approach Report: Fine-Tuning ResNet50 Layer4 (C1)

**Project repository:** [AnnaTikhonova-prog/Computer_Vision-project](https://github.com/AnnaTikhonova-prog/Computer_Vision-project)
**First approach:** C0 head-only baseline ([baseline report](initial-baseline-report.md))
**Second approach:** C1, classifier head + ResNet50 `layer4`
**Status:** C1 learning-rate grid completed; validation-based selection and final test evaluation completed.

## New approach and motivation

The baseline C0 replaces the pretrained ResNet50 classifier with a 10-class
head and trains only that head. This is an efficient transfer-learning
baseline, but it cannot adapt the pretrained visual representation to the
project's ten pet breeds. C1 therefore unfreezes the final residual stage,
`layer4`, in addition to the classifier head. The final stage contains the
highest-level ResNet features, so adapting it is expected to improve
breed-specific representations while retaining the generic features in the
earlier stages.

This report focuses on the C1 change and its results. Dataset selection,
split construction, class mapping, and image preprocessing are described in
the [initial baseline report](initial-baseline-report.md) and are shared with
C1.

## Implementation

Both conditions use the same `src/train.py`, shared data loader, and YAML
configuration system; there is no C1-specific training-code fork. The model
remains an ImageNet-1K V2 pretrained ResNet50 with a new 10-class linear head.

| Setting | C0 baseline | C1 second approach |
| --- | --- | --- |
| Trainable modules | Classifier head | Classifier head + `layer4` |
| Frozen feature blocks | Stem, `layer1`–`layer4` | Stem, `layer1`–`layer3` |
| BatchNorm running statistics | Frozen | Frozen; the eval-mode policy is restored after `model.train()` |
| BatchNorm affine parameters | Trainable only in unfrozen blocks | Trainable in `layer4` only; earlier blocks remain frozen |
| Selected learning rate | `1e-3` | `1e-4` |

C1 uses AdamW (weight decay `0.01`, betas `(0.9, 0.999)`, epsilon `1e-8`),
batch size 32 processed as microbatches of 4, augmentation enabled, a maximum
of 30 epochs, and early stopping patience 5. Training seeds are 42, 43, and
44, independent of `split_seed=42`. The candidates `1e-3`, `1e-4`, and `1e-5`
were compared, selecting the LR with the highest mean validation macro-F1
across the three seeds. The best checkpoint within each run is selected by
validation macro-F1; all three selected-LR seed checkpoints are retained.

## Validation results

The model and learning rate were selected using the same frozen 300-image
validation split. Values below are means and sample standard deviations
(`ddof=1`) across the three retained checkpoints, not pooled predictions or an
ensemble score.

| Condition | Selected LR | Validation macro-F1 | Validation accuracy |
| --- | ---: | ---: | ---: |
| C0 baseline | `1e-3` | 0.9701 ± 0.0034 | 0.9700 ± 0.0033 |
| C1 second approach | `1e-4` | **0.9856 ± 0.0038** | **0.9856 ± 0.0038** |

C1's mean validation macro-F1 is **1.55 percentage points** higher than C0's.
The C1 seed-level validation macro-F1 scores are:

| Training seed | Best epoch | Validation macro-F1 | Accuracy | Loss |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 5 | 0.9834 | 0.9833 | 0.0711 |
| 43 | 4 | 0.9833 | 0.9833 | 0.0710 |
| 44 | 7 | 0.9900 | 0.9900 | 0.0484 |

The C1 learning-rate comparison was:

| Learning rate | Validation macro-F1 mean ± sample SD | Selected |
| ---: | ---: | :---: |
| `1e-3` | 0.9667 ± 0.0001 | No |
| `1e-4` | **0.9856 ± 0.0038** | **Yes** |
| `1e-5` | 0.9767 ± 0.0033 | No |

## Final test comparison

After the configurations were frozen, the project evaluator ran all three
retained checkpoints for C0 and C1 on the same 300-image test split. These test
scores are reported for final analysis only; they were not used for model or
learning-rate selection.

| Condition | Test accuracy mean ± sample SD | Test macro-F1 mean ± sample SD |
| --- | ---: | ---: |
| C0 baseline | 0.9511 ± 0.0051 | 0.9512 ± 0.0051 |
| C1 second approach | **0.9656 ± 0.0069** | **0.9655 ± 0.0070** |

C1 is **1.44 percentage points** higher in test accuracy and **1.43 points**
higher in test macro-F1 for this split. As in the validation results, these
are means and sample standard deviations across independent seed-level
checkpoints, not an ensemble.

The test artifacts were produced by `src/evaluate.py`: [C1 per-class F1](../reports/tables/C1_augment_on_test_per_class_f1.csv),
[C1 confusion matrix](../reports/figures/C1_augment_on_test_confusion_matrix.png),
[C1 latency](../reports/tables/C1_augment_on_test_latency.csv), and
[full preliminary comparison and error analysis](../reports/results_preliminary.md).

## Observations and limitations

- C1 reached its selected checkpoints after 4–7 epochs, compared with 9–11
  epochs for the selected C0 runs. The selected C1 training accuracy was
  0.988–0.998 while validation accuracy was 0.983–0.990. This indicates strong
  fit to the training data and some train/validation gap; early stopping and
  validation checkpoint selection limit, but do not eliminate, overfitting.
- The selected learning rate differs between conditions: C0 selected `1e-3`,
  while C1 selected `1e-4`. Each selection followed the same three-LR,
  three-seed validation protocol.
- The recorded C1 training runs used a Tesla T4, Python 3.13.15, PyTorch
  2.11.0+cu128, and torchvision 0.26.0+cu128. The historical C0 baseline was
  recorded on CPU with Python 3.14.6, PyTorch 2.14.1+cpu, and torchvision
  0.29.1+cpu. Although the split and protocol are shared, this environment
  difference means the observed score change should not be treated as a
  perfectly isolated causal estimate of unfreezing `layer4`.
- The three-seed test comparison is descriptive. **No settings are changed
  based on test results**; tuning on the test split is forbidden.

## Reproduction and artifacts

The C1 run reports, histories, resolved configurations, and retained
checkpoints are under [`deliverables/c1/`](../deliverables/c1/). The selected
learning rate and seed-level scores are recorded in
[`deliverables/c1/evaluation/lr_1e-4/aggregate.csv`](../deliverables/c1/evaluation/lr_1e-4/aggregate.csv).
Install, data-download, training-grid, and evaluation commands are documented
in the [README](../README.md).
