# Initial Baseline Report: Fine-Tuning Depth on Oxford-IIIT Pet

**Project repository:** [AnnaTikhonova-prog/Computer_Vision-project](https://github.com/AnnaTikhonova-prog/Computer_Vision-project)  
**Experiment reported:** C0 head-only baseline  
**Status:** C0 learning-rate grid completed; validation results reported below. The test split has not been evaluated.

## Project overview

This project studies how the depth of fine-tuning affects image classification on a fixed 10-breed subset of Oxford-IIIT Pet. The experiment conditions share one training implementation and YAML-based configuration: C0 trains only the classifier head, C1 also trains `layer4`, C2 trains `layer3` and `layer4`, and C3 trains the full network. This report presents the first completed model condition, C0.

## Dataset and preprocessing

The project uses ten verified Oxford-IIIT Pet annotation classes: British Shorthair, Russian Blue, Siamese, Birman, American Bulldog, Pug, Sphynx, Maine Coon, Beagle, and Samoyed. Class labels are fixed in `splits/v2/class_mapping.json`, with integer labels 0–9 in that order.

The frozen `splits/v2` manifests define a custom stratified 70/15/15 split with `split_seed=42` over the union of the official `trainval.txt` and `test.txt` annotations. This is not the official Oxford-IIIT Pet benchmark split. There are **1,999 images**: **1,399 training**, **300 validation**, and **300 test** images. Each class has 30 validation and 30 test images; training has 140 per class except Siamese, which has 139. The one-image deviation follows the reviewed duplicate-group constraints recorded in the split report. Confirmed duplicate groups are kept in a single split, and exact duplicate images are handled during preparation.

The shared `src.dataset.get_loaders` data interface reads the committed CSV manifests and class mapping without modifying them. Training uses a random resized crop to 224×224 (`scale=0.7–1.0`, `ratio=0.75–1.3333`), horizontal flip, ColorJitter, tensor conversion, and ImageNet normalization. Validation and test use the deterministic ResNet50 ImageNet V2 resize/crop and normalization transform. Validation and test loaders do not shuffle or drop their final batch. The raw image files are not stored in Git.

## Baseline model and training setup

The baseline is an ImageNet-1K V2 pretrained ResNet50. Its 1,000-class fully connected layer is replaced with `Linear(2048, 10)`. In C0, only `fc.weight` and `fc.bias` are trainable, totaling **20,490 trainable parameters**. BatchNorm running statistics remain frozen; affine parameters are trainable only when their block is unfrozen.

The C0 grid uses training seeds **42, 43, and 44**, independent of `split_seed=42`, and compares learning rates **1e-3, 1e-4, and 1e-5**. Shared settings are AdamW (`weight_decay=0.01`, `betas=(0.9, 0.999)`, `eps=1e-8`), cross-entropy loss, batch size **32** processed as microbatches of **4**, augmentation enabled, at most **30 epochs**, and early stopping patience **5**. The selected checkpoint is the earliest epoch attaining the best validation macro-F1. Hyperparameter selection uses validation macro-F1 averaged across the three seeds; no seed is discarded.

Recorded run environment: **CPU**, Python **3.14.6**, PyTorch **2.14.1+cpu**, and torchvision **0.29.1+cpu**. The run metadata records Git commit `155bf1b5325c758455de03431f08008ae310d1ff` with a dirty working tree. The current repository snapshot is newer, so this recorded provenance should be considered when reproducing the exact historical run.

## Initial validation results

Every entry below is measured on the same **300-image validation split**. Macro-F1 is the checkpoint-selection metric. Validation loss and accuracy are included for completeness.

| Learning rate | Seed | Selected epoch | Epochs run | Validation macro-F1 | Validation accuracy | Validation loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 42 | 9 | 14 | 0.9667 | 0.9667 | 0.1497 |
| 1e-3 | 43 | 9 | 14 | 0.9701 | 0.9700 | 0.1520 |
| 1e-3 | 44 | 11 | 16 | 0.9735 | 0.9733 | 0.1366 |
| 1e-4 | 42 | 12 | 17 | 0.9499 | 0.9500 | 0.7509 |
| 1e-4 | 43 | 11 | 16 | 0.9436 | 0.9433 | 0.8110 |
| 1e-4 | 44 | 11 | 16 | 0.9534 | 0.9533 | 0.7973 |
| 1e-5 | 42 | 30 | 30 | 0.8813 | 0.8833 | 1.6947 |
| 1e-5 | 43 | 30 | 30 | 0.8137 | 0.8200 | 1.6991 |
| 1e-5 | 44 | 30 | 30 | 0.8696 | 0.8700 | 1.6614 |

### Learning-rate selection

The values below are the mean and **sample standard deviation** (`ddof=1`) across the three independent training seeds. They are not pooled predictions or an ensemble score.

| Learning rate | Mean validation macro-F1 ± SD | Mean validation accuracy ± SD | Mean validation loss ± SD | Selected |
| ---: | ---: | ---: | ---: | :---: |
| 1e-3 | **0.9701 ± 0.0034** | 0.9700 ± 0.0033 | 0.1461 ± 0.0083 | Yes |
| 1e-4 | 0.9490 ± 0.0049 | 0.9489 ± 0.0051 | 0.7864 ± 0.0315 | No |
| 1e-5 | 0.8549 ± 0.0361 | 0.8578 ± 0.0334 | 1.6851 ± 0.0206 | No |

The selected C0 learning rate is **1e-3**. Its three retained checkpoints have validation macro-F1 **0.9667**, **0.9701**, and **0.9735** for seeds 42, 43, and 44, respectively. The larger learning rate outperformed the two smaller candidates in this grid. The 1e-5 runs reached the 30-epoch limit, while the other runs stopped earlier under the patience rule.

### Per-class results at the selected learning rate

For each retained 1e-3 checkpoint, validation support is 30 images per class. The table reports mean and sample standard deviation of class F1 over seeds 42, 43, and 44.

| Class | Support per seed | F1 mean ± sample SD |
| --- | ---: | ---: |
| British Shorthair | 30 | 0.8972 ± 0.0108 |
| Russian Blue | 30 | 0.9257 ± 0.0094 |
| Siamese | 30 | 0.9556 ± 0.0192 |
| Birman | 30 | 0.9556 ± 0.0192 |
| American Bulldog | 30 | 1.0000 ± 0.0000 |
| Pug | 30 | 1.0000 ± 0.0000 |
| Sphynx | 30 | 1.0000 ± 0.0000 |
| Maine Coon | 30 | 0.9667 ± 0.0000 |
| Beagle | 30 | 1.0000 ± 0.0000 |
| Samoyed | 30 | 1.0000 ± 0.0000 |

The following pooled confusion counts sum predictions from the three separate retained checkpoints (**900 predictions total**, 90 true examples per class). They are a count summary, not a combined model or ensemble. Rows are true classes and columns are predicted classes, in the same order as the table above.

| True \\ Predicted | Brit. Shorthair | Russian Blue | Siamese | Birman | Amer. Bulldog | Pug | Sphynx | Maine Coon | Beagle | Samoyed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| British Shorthair | 83 | 4 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| Russian Blue | 9 | 81 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Siamese | 0 | 0 | 86 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| Birman | 0 | 0 | 4 | 86 | 0 | 0 | 0 | 0 | 0 | 0 |
| American Bulldog | 0 | 0 | 0 | 0 | 90 | 0 | 0 | 0 | 0 | 0 |
| Pug | 0 | 0 | 0 | 0 | 0 | 90 | 0 | 0 | 0 | 0 |
| Sphynx | 0 | 0 | 0 | 0 | 0 | 0 | 90 | 0 | 0 | 0 |
| Maine Coon | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 87 | 0 | 0 |
| Beagle | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 90 | 0 |
| Samoyed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 90 |

Most residual errors in the selected-LR predictions occur between British Shorthair and Russian Blue (13 of 900 pooled predictions), followed by Siamese and Birman (8 of 900). These observations describe this validation split and do not establish generalization to the held-out test split.

## Evaluation and artifacts

The project evaluator is `src/evaluate.py`. It accepts one checkpoint or the three retained checkpoints for a condition and split. It writes mean and standard deviation for macro-F1 and accuracy, per-class F1, batch-1 and batch-32 inference latency, and a confusion-matrix image. Output paths are fixed under `reports/tables/` and `reports/figures/`; README documents the command.

The C0 grid results in this report are the committed validation artifacts under `deliverables/c0/evaluation/`, along with the selected checkpoints and run histories under `deliverables/c0/`. They were produced by the grid-validation workflow on the training machine. The raw image dataset is not present in a fresh repository checkout, and the current report-generation environment has no PyTorch installation, so the new general evaluator was not run here to regenerate those artifacts or measure latency.

## Implementation observations and limitations

- The custom split is class-balanced on validation and test (30 images for every class), with 139 rather than 140 Siamese training images due to the split grouping constraints.
- In the selected 1e-3 grid, validation accuracy ranges from 0.9667 to 0.9733 and macro-F1 from 0.9667 to 0.9735 across seeds. The 1e-5 learning rate produced both the lowest mean score and the greatest seed-to-seed macro-F1 variation of the tested rates.
- Training ran on CPU. This provides a completed baseline grid but does not verify GPU history reproducibility.
- The test split was deliberately not iterated during training or learning-rate selection. Consequently, this report contains **no test-set score**.
- No batch-1 or batch-32 latency measurement is included in the saved C0 grid. The evaluator can measure these when run in an environment with the project images, dependencies, and checkpoints.
- The tracked checkpoint metadata records commit `155bf1b` and `dirty: true`; this is a limitation for source-level reproducibility of those historical runs.

## Reproduction

After installing dependencies and downloading the official data as described in the README, aggregate the three selected C0 checkpoints on validation with:

```powershell
py -3 -m src.evaluate --condition C0 --checkpoints deliverables/c0/runs/lr_1e-3/seed_42/best.pt deliverables/c0/runs/lr_1e-3/seed_43/best.pt deliverables/c0/runs/lr_1e-3/seed_44/best.pt --config configs/c0.yaml --split val --device cpu
```

To evaluate the held-out test set, change `--split val` to `--split test` only after the experimental protocol is finalized. This report does not claim a test result.
