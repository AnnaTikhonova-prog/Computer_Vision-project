# Project Plan — Computer Vision 2026

**Project:** Fine-tuning depth on a small dataset: where does unfreezing stop paying off?


**Team:** 
+ Tikhonova Anna (Team Lead) 
+ Pyanov Georgii (Data Engineer) 
+ Fadeeva Albina (ML Engineer) 
+ Kirillov Maksim (ML Engineer)

---

## 1. Project scope

**Objective.** Where is the cutoff at which unfreezing more layers of a pretrained ResNet50 stops helping on a small image-classification dataset? We treat fine-tuning depth as a single controlled experimental factor and measure its effect on a 10-breed subset of Oxford-IIIT Pet.

**Exact task:** single-label image classification over 10 pet breeds.

**Experimental settings:** four conditions forming an "unfreezing ladder" 

| Condition | Trainable parameters | Role |
|---|---|---|
| **C0 — Frozen** | classification head only (~20K params) | Baseline |
| **C1 — layer4** | head + layer4 (~15M) | Ladder step |
| **C2 — layer3+4** | head + layer3 + layer4 (~24M) | Ladder step |
| **C3 — Full fine-tuning** | all ~25.5M params | Second compared condition |

**Hypothesis (falsifiable):** on ~1,400 training images, full fine-tuning is not the optimum and an intermediate unfreezing depth gives a better trade-off between domain adaptation and overfitting. Testable implication: validation macro-F1 as a function of unfreezing depth peaks at an intermediate point rather than at either extreme.

**Secondary question:** which visually similar breed pairs remain the dominant source of errors under every strategy (failure analysis).

## 2. Dataset preparation

- **Dataset:** Oxford-IIIT Pet Dataset (https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 cat and dog breeds, ~7,400 images, public, no registration required.
- **Selected subset:** 10 breeds, ~200 images per breed (~2,000 total). Three visually similar pairs are included deliberately, as the expected source of errors:
  - British Shorthair ↔ Russian Blue (solid-grey cats)
  - Siamese ↔ Birman (colorpoint cats)
  - English Bulldog ↔ Pug (brachycephalic dogs)
  - plus four high-contrast breeds: Sphynx, Maine Coon, Beagle, German Shepherd.
- **Preprocessing:** resize to 224×224; ImageNet mean/std normalization; train-time augmentation: RandomResizedCrop, horizontal flip, color jitter (augmentation can be switched off via config for the ablation).
- **Split:** stratified train/validation/test **70/15/15** (~1,400 / ~300 / ~300 images), generated once with seed 42, frozen into `splits/{train,val,test}.csv`, and never modified afterwards. All four conditions use identical split files.
- **Leakage prevention:** the test set is loaded exactly once for the final evaluation of frozen models. All hyperparameter decisions use the validation split only.

## 3. Experimental plan

**Model:** ResNet50 with ImageNet weights (`torchvision`, `IMAGENET1K_V2`); the classifier head is replaced with `Linear(2048 → 10)`.

**Compared approaches:** the four conditions C0–C3 above. They differ in  unfreezing depth. The ladder doubles as the required ablation: C0 to C3 is a systematic sweep over the controlled factor.

**Training strategy (identical across conditions):** AdamW optimizer; batch size 32; up to 30 epochs; early stopping on validation macro-F1 (patience 5); seed 42; cross-entropy loss; one shared `train.py`  conditions differ only via a YAML config.

**Fair-comparison protocol:** for each condition, the learning rate is selected from {1e-3, 1e-4} on the validation split, so no condition loses merely because of an unsuitable LR. This is the only tuned hyperparameter.

**Additional ablation:** the best-performing condition is re-run with augmentation switched off (augment on/off) to estimate the data pipeline's contribution.

**Evaluation metrics:**

| Metric | Role |
|---|---|
| **Macro-F1** (primary) | Comparison across conditions; equal weight per breed, sensitive to similar-pair errors |
| Accuracy | Secondary, interpretability |
| Per-class F1 + confusion matrix | Localization of confused breed pairs per condition |
| Validation loss / training curves | Diagnostics: "frozen underfits" vs "full FT overfits" |
| Training time + inference latency (ms/img) | Practical trade-off; required by the guideline |
| ≥3 failure cases with Grad-CAM | Qualitative analysis with explained causes |

**Significance criterion:** a macro-F1 difference counts as meaningful above ~1 percentage point and when confirmed at the per-class level (with ~300 test images, 1 p.p. ≈ 3 images — stated honestly as a limitation).

## 4. Implementation plan

**Pipeline steps:**
1. Download and verify the dataset; select the 10-breed subset; generate and freeze the stratified splits (CSV manifests).
2. `src/dataset.py` — Dataset/DataLoader with preprocessing and a config-controllable augmentation pipeline.
3. `src/train.py` — shared training engine: per-epoch history (CSV), best checkpoint by val macro-F1, early stopping, seed handling, unfreeze-depth as a config parameter.
4. Train conditions C0–C3 (two per ML engineer, in parallel); LR selection on val.
5. `src/evaluate.py` — single evaluation script producing macro-F1, accuracy, confusion matrix (PNG), per-class F1 (CSV), and latency for any checkpoint. All reported numbers come from this script only.
6. One-time test evaluation of all frozen conditions; comparison table and figures.
7. Augmentation ablation on the best condition.
8. Failure analysis: export misclassified test images grouped by true→predicted breed pair; inspect ≥3 representative cases with Grad-CAM and explain causes.
9. Runtime measurements (training time per condition, inference latency, memory).
10. Two-page technical summary, reproducibility check from a fresh clone, demo preparation.

**Libraries and resources:** PyTorch + torchvision (pretrained ResNet50, `IMAGENET1K_V2`), scikit-learn (metrics), pandas (splits and results), matplotlib/seaborn (figures), Grad-CAM via `pytorch-grad-cam` or manual hooks, PyYAML (configs). 

**Reproducibility:** pinned `requirements.txt`, README with exact install/download/train/evaluate commands, fixed seeds, frozen splits and configs in the repository; the final check is a fresh-clone run reproducing the reported numbers.

## 5. Team responsibilities

| Member          | Role | Responsibility |
|-----------------|---|---|
| Tikhonova Anna  | Team Lead | Single `evaluate.py`; failure analysis with Grad-CAM; two-page summary; demo and slides; repository integration and final reproducibility check |
| Pyanov Georgii  | Data Engineer | Dataset download and verification; breed subset selection; frozen stratified splits; `src/dataset.py`; README/requirements; runtime measurements; error-image export |
| Fadeeva Albina  | ML Engineer | Shared `train.py`; conditions C0 (frozen) and C1 (layer4): training, LR selection on val, checkpoints and histories |
| Kirillov Maksim | ML Engineer | Shared `train.py`; conditions C2 (layer3+4) and C3 (full FT); augmentation ablation |

**Working agreements:** one shared `train.py` with no forks (conditions differ only via config); splits frozen after week 1; metrics are produced only by the lead's `evaluate.py`; the test set is opened exactly once; weekly 30-minute sync before each Tuesday milestone.

