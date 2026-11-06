# Fine-Tuning Depth on Oxford-IIIT Pet
**Computer Vision 2026 | Final technical summary**

**Repository:** [AnnaTikhonova-prog/Computer_Vision-project](https://github.com/AnnaTikhonova-prog/Computer_Vision-project)

## 1. Problem and question
Does progressively unfreezing a pretrained ResNet50 improve ten-breed pet classification on a small fixed dataset? We also test augmentation on/off for the head-only baseline (C0) and full fine-tuning (C3).

## 2. Dataset
Oxford-IIIT Pet subset: 1,999 RGB images from ten breeds; frozen stratified split, 1,399 train / 300 validation / 300 test (`split_seed=42`). This is not the official benchmark split. No masks or boxes are used. Preprocessing: resize shorter side to 232, center crop to 224 x 224, ImageNet normalization. Training augmentation: random resized crop, horizontal flip, mild color jitter.

## 3. Method
All conditions use ImageNet-1K V2 pretrained ResNet50, a new 10-class head, shared trainer, AdamW, cross-entropy, batch 32, up to 30 epochs, and early stopping on validation macro-F1 (patience 5). C0 trains the head; C1 adds `layer4`; C2 adds `layer3`; C3 trains all parameters. BatchNorm running statistics are frozen. Seeds are 42, 43, 44; LR is selected from 1e-3, 1e-4, 1e-5 by mean validation macro-F1. The C0/C3 ablation disables augmentation at the selected LR.

## 4. Results
Test metrics are mean +/- sample SD over three checkpoints. No settings were changed based on test results.

| Configuration | LR | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| C0, augmentation on | 1e-3 | 0.9511 +/- 0.0051 | 0.9512 +/- 0.0051 |
| C1, augmentation on | 1e-4 | 0.9656 +/- 0.0069 | 0.9655 +/- 0.0070 |
| C2, augmentation on | 1e-5 | 0.9600 +/- 0.0033 | 0.9601 +/- 0.0033 |
| C3, augmentation on | 1e-5 | 0.9633 +/- 0.0067 | 0.9633 +/- 0.0067 |
| C0, augmentation off | 1e-3 | 0.9489 +/- 0.0019 | 0.9485 +/- 0.0020 |
| C3, augmentation off | 1e-5 | 0.9578 +/- 0.0107 | 0.9579 +/- 0.0107 |

## 5. Interpretation
C1 performed best: versus C0 with augmentation, it gained 1.44 percentage points in accuracy and 1.43 in macro-F1. Unfreezing more layers did not improve scores monotonically. Augmentation-off results were lower for C0 and C3, but differing historical environments and initialization limit causal interpretation. Results describe one 300-image test split.

## 6. Failure analysis
Three high-confidence C0 errors were reviewed with Grad-CAM:
- `Siamese_257`: Siamese -> Birman (0.916); soft close crop, eye glare, shared color-point face, little body context.
- `Russian_Blue_264`: Russian Blue -> British Shorthair (0.908); oblique pose and lighting obscure fine face and coat cues.
- `pug_74`: Pug -> American Bulldog (0.810); seated pose hides proportions; short muzzle and broad chest are shared cues.

These are probable explanations, not causal proof. See the [detailed analysis](failure_analysis.md) for overlays and limitations.

## 7. Reproducibility
The repository README documents environment setup, data download/verification, training, and evaluation. Exact six-configuration inference uses the 12 core checkpoints and the six original C0/C3 augmentation-off checkpoints packaged in Git LFS; the dataset is downloaded separately. The 6 November fresh-clone audit matched all 5,400 predictions using all 18 checkpoints. Batch-1 CPU latency was 79.40-118.97 ms; batch-32 was 2,031.67-2,421.08 ms per batch, measured on Intel Core i5-1335U, Python 3.14.6, PyTorch 2.14.1+cpu. Training wall-clock time was not recorded. The [latency data](tables/latency_rerun_2026-11-02.csv) and [runtime record](tables/latency_rerun_2026-11-02_runtime.json) give full details.

## 8. Members' Contributions
- **Anna Tikhonova, Team Lead:** evaluation, Grad-CAM failure analysis, final report, repository integration, reproducibility.
- **Georgij Pyanov, Data Engineer:** data download/verification, subset and splits, loader, README/requirements, runtime, error-image export.
- **Albina Fadeeva, ML Engineer:** shared trainer; C0/C1 grids, LR selection, checkpoints and histories.
- **Maksim Kirillov, ML Engineer:** Grad-CAM; C2/C3 grids and C0/C3 augmentation ablation.
