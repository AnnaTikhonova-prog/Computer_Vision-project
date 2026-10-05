# Project Plan — Computer Vision 2026

**Project:** Fine-tuning depth on a small dataset: where does unfreezing stop paying off?


**Team:** 
+ Tikhonova Anna (Team Lead) 
+ Pyanov Georgij (Data Engineer) 
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

**Secondary question:** which breed pairs actually dominate the errors under every strategy — and whether these are the presumed visually similar pairs (failure analysis).

## 2. Dataset preparation

- **Dataset:** Oxford-IIIT Pet Dataset (https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 cat and dog breeds, ~7,400 images, public, no registration required.
- **Selected subset:** 10 breeds, ~200 images per breed (~2,000 total). Three pairs are included as *presumed* hard cases (visually similar breeds); whether they actually dominate the errors will be determined from the results, not assumed in advance:
  - British Shorthair ↔ Russian Blue (solid-grey cats)
  - Siamese ↔ Birman (colorpoint cats)
  - American Bulldog ↔ Pug (stocky, short-muzzled dogs)
  - plus four high-contrast breeds: Sphynx, Maine Coon, Beagle, Samoyed.
- **Preprocessing:** validation and test use the standard preprocessing shipped with the chosen pretrained weights (`IMAGENET1K_V2`: resize 256 → center crop 224 → ImageNet mean/std normalization). Train-time augmentation: RandomResizedCrop, horizontal flip, color jitter; with `augment: false` the train split receives the same deterministic preprocessing as val/test.
- **Split:** stratified train/validation/test **70/15/15** (~1,400 / ~300 / ~300 images), generated once with seed 42 by a dedicated command (`python -m src.prepare_splits`) and frozen into `splits/{train,val,test}.csv`. Training code only reads these CSVs and never recreates or re-splits the data. All four conditions use identical split files.
- **Leakage prevention:** the test set is loaded exactly once for the final evaluation of frozen models. All hyperparameter decisions use the validation split only.

## 3. Experimental plan

**Model:** ResNet50 with ImageNet weights (`torchvision`, `IMAGENET1K_V2`); the classifier head is replaced with `Linear(2048 → 10)`.

**Compared approaches:** the four conditions C0–C3 above. They differ in  unfreezing depth. The ladder doubles as the required ablation: C0 to C3 is a systematic sweep over the controlled factor.

**Training strategy (identical across conditions):** AdamW optimizer; batch size 32; up to 30 epochs; early stopping on validation macro-F1 (patience 5); cross-entropy loss; one shared `train.py` — conditions differ only via a YAML config. Two details fixed in advance: (a) **BatchNorm layers stay frozen** (eval mode, ImageNet running statistics) in every unfrozen condition — with batch 32, updating BN statistics is a known hidden cause of fine-tuning degradation; (b) each condition is trained with **3 seeds {42, 43, 44}** and reported as mean ± std, so that a difference between conditions is never read off a single run.

**Determinism:** seed everything (Python/NumPy/PyTorch), `torch.backends.cudnn.deterministic=True`, `benchmark=False`, seeded DataLoader generator and `worker_init_fn` — a repeated run of the same config must reproduce the same history.

**Fair-comparison protocol:** for each condition, the learning rate is selected from {1e-3, 1e-4} on the validation split, so no condition loses merely because of an unsuitable LR. This is the only tuned hyperparameter; the grid is deliberately coarse because the val split is small (~300 images) and we do not want to overfit it.

**Additional ablation (pre-committed):** conditions **C0 and C3** (the two ends of the ladder) are re-run with augmentation switched off (`augment: false`, deterministic weight-provided preprocessing). The ablation targets are fixed in advance — not chosen after seeing results — to avoid post-hoc selection.

**Evaluation metrics:**

| Metric | Role |
|---|---|
| **Macro-F1** (primary) | Comparison across conditions; equal weight per breed, sensitive to similar-pair errors |
| Accuracy | Secondary, interpretability |
| Per-class F1 + confusion matrix | Localization of confused breed pairs per condition |
| Validation loss / training curves | Diagnostics: "frozen underfits" vs "full FT overfits" |
| Training time + inference latency (ms/img) | Practical trade-off; required by the guideline |
| ≥3 failure cases with Grad-CAM | Qualitative analysis with explained causes |

**Significance criterion:** differences between conditions are read from 3 seeds as mean ± std; a macro-F1 difference counts as meaningful when it exceeds the across-seed spread and is confirmed at the per-class level. With ~300 val/test images, 1 p.p. ≈ 3 images — this is stated honestly as a limitation, and single-seed differences are never interpreted.

## 4. Implementation plan

**Pipeline steps:**
1. Download and verify the dataset; select the 10-breed subset; generate and freeze the stratified splits (CSV manifests).
2. `src/dataset.py` — Dataset/DataLoader with preprocessing and a config-controllable augmentation pipeline.
3. `src/train.py` — shared training engine: per-epoch history (CSV), best checkpoint by val macro-F1, early stopping, full determinism (seeded loaders, cudnn deterministic), unfreeze-depth as a config parameter, BatchNorm kept frozen in all conditions.
4. Train conditions C0–C3 (two per ML engineer, in parallel), each with 3 seeds; LR selection on val.
5. `src/evaluate.py` — single evaluation script producing macro-F1, accuracy, confusion matrix (PNG), per-class F1 (CSV), and latency for any checkpoint. All reported numbers come from this script only.
6. One-time test evaluation of all frozen conditions; comparison table and figures (mean ± std across seeds).
7. Pre-committed augmentation ablation on C0 and C3 (augment on/off).
8. Failure analysis: export misclassified test images grouped by true→predicted breed pair; inspect ≥3 representative cases with Grad-CAM and explain causes.
9. Runtime measurements (training time per condition, inference latency, memory).
10. Two-page technical summary, reproducibility check from a fresh clone, demo preparation.

**Libraries and resources:** PyTorch + torchvision (pretrained ResNet50, `IMAGENET1K_V2`), scikit-learn (metrics), pandas (splits and results), matplotlib/seaborn (figures), Grad-CAM via `pytorch-grad-cam` or manual hooks, PyYAML (configs). 

**Reproducibility:** pinned `requirements.txt`, README with exact install/download/train/evaluate commands, fixed seeds, frozen splits and configs in the repository; the final check is a fresh-clone run reproducing the reported numbers.

## 5. Team responsibilities

| Member          | Role | Responsibility |
|-----------------|---|---|
| Tikhonova Anna  | Team Lead | Single `evaluate.py`; failure analysis with Grad-CAM; two-page summary; demo and slides; repository integration and final reproducibility check |
| Pyanov Georgij  | Data Engineer | Dataset download and verification; breed subset selection; frozen stratified splits; `src/dataset.py`; README/requirements; runtime measurements; error-image export |
| Fadeeva Albina  | ML Engineer | `src/train.py` (sole owner); conditions C0 (frozen) and C1 (layer4), 3 seeds each: training, LR selection on val, checkpoints and histories |
| Kirillov Maksim | ML Engineer | `src/gradcam.py` (sole owner); conditions C2 (layer3+4) and C3 (full FT), 3 seeds each; augmentation ablation on C0 and C3 |

**Working agreements:** one shared `train.py` owned by ML Engineer 1 (no forks — conditions differ only via config); splits frozen after week 1; metrics are produced only by the lead's `evaluate.py`; the test set is opened exactly once; every run logs its config and git commit hash; weekly 30-minute sync before each Tuesday milestone.

