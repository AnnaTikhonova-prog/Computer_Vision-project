# Fine-Tuning Depth on Oxford-IIIT Pet
**Computer Vision 2026 | Final technical summary**<br>
**Repository:** [AnnaTikhonova-prog/Computer_Vision-project](https://github.com/AnnaTikhonova-prog/Computer_Vision-project)

## 1. Problem and question

Does progressively unfreezing a pretrained ResNet50 improve ten-breed pet classification on a small, fixed dataset? We also compare augmentation on and off for the head-only baseline (C0) and full fine-tuning condition (C3).

## 2. Dataset

We use 1,999 Oxford-IIIT Pet images from ten breeds: British Shorthair, Russian Blue, Siamese, Birman, American Bulldog, Pug, Sphynx, Maine Coon, Beagle, and Samoyed. The frozen stratified project split contains 1,399 training, 300 validation, and 300 test images (`split_seed=42`); it is not the official benchmark split. Inputs are full RGB photographs without masks or boxes. Deterministic preprocessing resizes the shorter side to 232 pixels, center-crops to 224 x 224, and applies ImageNet normalization. Training augmentation uses random resized crop, horizontal flip, and mild color jitter.

## 3. Method

All conditions use ImageNet-1K V2 pretrained ResNet50 with a new 10-class linear head and the shared training pipeline. C0 trains the head only; C1 trains `layer4` and the head; C2 trains `layer3`, `layer4`, and the head; C3 trains all parameters. The C0/C3 ablation disables augmentation while retaining each condition's selected learning rate.

Training uses AdamW, cross-entropy, batch size 32, up to 30 epochs, and early stopping on validation macro-F1 (patience 5). Training seeds are 42, 43, and 44. Each condition's learning rate is selected from 1e-3, 1e-4, and 1e-5 by mean validation macro-F1 across the three seeds. BatchNorm running statistics remain frozen; affine parameters train only in unfrozen blocks.

## 4. Results

Test metrics are mean +/- sample standard deviation across the three retained checkpoints. No settings were changed based on test results.

| Configuration | Selected LR | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| C0 - augmentation on | 1e-3 | 0.9511 +/- 0.0051 | 0.9512 +/- 0.0051 |
| C1 - augmentation on | 1e-4 | 0.9656 +/- 0.0069 | 0.9655 +/- 0.0070 |
| C2 - augmentation on | 1e-5 | 0.9600 +/- 0.0033 | 0.9601 +/- 0.0033 |
| C3 - augmentation on | 1e-5 | 0.9633 +/- 0.0067 | 0.9633 +/- 0.0067 |
| C0 - augmentation off | 1e-3 | 0.9489 +/- 0.0019 | 0.9485 +/- 0.0020 |
| C3 - augmentation off | 1e-5 | 0.9578 +/- 0.0107 | 0.9579 +/- 0.0107 |

## 5. Interpretation

C1 (layer4 plus head) performed best, reaching 0.9656 +/- 0.0069 accuracy and 0.9655 +/- 0.0070 macro-F1. Against the head-only C0 baseline with augmentation, it improved accuracy by 1.44 percentage points and macro-F1 by 1.43 points. Unfreezing more layers did not produce a monotonic gain. Turning augmentation off lowered the observed C0 and C3 scores, but the historical runs differ in environment and initialization, so the ablation is descriptive rather than a clean causal estimate. Results cover one 300-image test split.

## 6. Failure analysis

Three high-confidence C0 test errors were inspected with Grad-CAM:

- `Siamese_257`: Siamese was classified as Birman (confidence 0.916). The close, soft crop emphasizes a color-point face shared by both breeds; eye glare and little body-shape context make distinguishing cues difficult.
- `Russian_Blue_264`: Russian Blue was classified as British Shorthair (0.908). The oblique pose and lighting obscure fine facial and coat-texture cues.
- `pug_74`: Pug was classified as American Bulldog (0.810). The seated pose hides body proportions, while the short muzzle and broad chest are shared visual cues.

These probable causes are hypotheses, not proof; Grad-CAM shows spatial support and does not establish why a prediction was made. The [detailed failure analysis](failure_analysis.md) contains the overlays, provenance, and limitations.

## 7. Reproducibility

The [Git repository](https://github.com/AnnaTikhonova-prog/Computer_Vision-project) contains the source code, split manifests, configs, evaluation artifacts, and the README with setup and run commands. A clean clone can fetch the 12 core checkpoints and the original six C0/C3 augmentation-off checkpoints through Git LFS; the latter are packaged in `deliverables/ablation_noaug/augmentation_off_checkpoints.zip` and extracted into ignored `artifacts/` paths. Their SHA-256 values are listed in the provenance CSV. A fresh-clone audit using that archive matched all 5,400 saved predictions on 6 November 2026. The original per-run training histories and metadata are not included, and retraining does not reproduce those exact checkpoint binaries. Final inference latency was measured on CPU (13th Gen Intel Core i5-1335U, Windows 11, Python 3.14.6, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu), with 3 warm-ups and 10 timed passes per checkpoint. Batch-1 latency ranged from 79.40 to 118.97 ms; batch-32 latency ranged from 2,031.67 to 2,421.08 ms per batch. Training wall-clock time was not recorded. Full latency data and environment details are in [`reports/tables/latency_rerun_2026-11-02.csv`](tables/latency_rerun_2026-11-02.csv) and [`reports/tables/latency_rerun_2026-11-02_runtime.json`](tables/latency_rerun_2026-11-02_runtime.json).

Presentation flow: research question and split; C0-C3 fine-tuning ladder; the result comparison and C1 finding; one Grad-CAM failure and limitations. Demonstrate the evaluation pipeline on saved examples and outputs. Course timing: 5-minute presentation, 2-minute demo, 3-minute Q&A.

## 8. Members' Contributions

- **Anna Tikhonova - Team Lead:** evaluation pipeline, Grad-CAM failure analysis, final report, repository integration, and reproducibility coordination.
- **Georgij Pyanov - Data Engineer:** dataset download and verification, breed subset and frozen splits, data loader, README and requirements, runtime measurements, and error-image export.
- **Albina Fadeeva - ML Engineer:** shared training implementation and C0/C1 training grids, validation-based learning-rate selection, checkpoints, and histories.
- **Maksim Kirillov - ML Engineer:** Grad-CAM implementation and C2/C3 training grids; augmentation ablation on C0 and C3.
