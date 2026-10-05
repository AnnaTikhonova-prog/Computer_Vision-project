# Project Plan — Computer Vision 2026

**Project:** Fine-tuning depth on a small dataset: where does unfreezing stop paying off?

**Team:**

- Tikhonova Anna (Team Lead)
- Pyanov Georgij (Data Engineer)
- Fadeeva Albina (ML Engineer)
- Kirillov Maksim (ML Engineer)

---

## 1. Project scope

**Objective.** The project studies how ResNet50 fine-tuning depth affects a small, fixed image-classification task. The controlled factor is the set of unfrozen ResNet50 blocks; all conditions use the same data version, preprocessing policy, optimizer family, and evaluation protocol.

**Exact task.** Single-label classification over ten Oxford-IIIT Pet breeds:

- British Shorthair, Russian Blue, Siamese, Birman, American Bulldog;
- Pug, Sphynx, Maine Coon, Beagle, Samoyed.

**Experimental settings.** The four conditions form an unfreezing ladder. Parameter values below are approximate; exact trainable counts will be computed programmatically for the replaced 10-class head and the selected BatchNorm policy before experiments.

| Condition | Trainable components | Approximate role |
|---|---|---|
| **C0 — Frozen** | New classification head only | ≈20K; baseline |
| **C1 — layer4** | Head + `layer4` | ≈15M; first unfreezing step |
| **C2 — layer3+4** | Head + `layer3` + `layer4` | ≈22–24M; intermediate step |
| **C3 — Full fine-tuning** | All ResNet50 blocks + head | ≈25M; full adaptation |

**Hypothesis.** On this small fixed dataset, an intermediate fine-tuning depth may offer the best validation trade-off between adaptation and overfitting. Full fine-tuning may also be the best condition; that is an admissible result. Four conditions do not establish a universal fine-tuning cutoff.

**Secondary question.** Which breed pairs dominate observed errors under each trained condition? Presumed visual similarities are hypotheses for later failure analysis, not conclusions made before evaluation.

## 2. Dataset preparation

### Frozen dataset and split

The source is Oxford-IIIT Pet. The active reviewed dataset version is `splits/v2` and contains **1,999** valid annotated RGB images: **1,399 train**, **300 validation**, and **300 test**. British Shorthair, Russian Blue, Birman, American Bulldog, Pug, Sphynx, Maine Coon, Beagle, and Samoyed each have 140/30/30 train/validation/test examples; Siamese has 139/30/30.

The split is a custom stratified 70/15/15 split with `split_seed=42`, made over the union of official `trainval.txt` and `test.txt` annotations. It is therefore **not** an evaluation on the official Oxford-IIIT Pet benchmark split. `splits/class_mapping.json` fixes labels 0–9 for every consumer of the data.

Committed manifests are read as-is; loaders do not recreate a split. The data-preparation command is:

```powershell
py -3 scripts/prepare_oxford_pets.py --dataset-root data/oxford-iiit-pet --output-dir splits --seed 42
```

The command refuses to overwrite frozen manifests. A new split version must be an explicit action in a new output directory, for example `--output-dir splits/v2`.

The preparation workflow records RGB decoding failures, exact-duplicate groups, perceptual-hash candidates, and split deviations. It checks decoded RGB content with image dimensions in the exact hash, uses the fixed 0–9 mapping, and verifies no train/validation/test overlap by image ID, relative path, or `content_hash`.

The 91 perceptual-hash candidates have completed manual review: 84 are `different images`, 3 are `duplicate`, and 4 are `near-duplicate`; none remains `uncertain`. Seven confirmed candidate pairs form six transitive groups. Two confirmed groups crossed v1 splits (`Birman_199`/`Birman_25` and `British_Shorthair_186`/`British_Shorthair_271`), so `splits/v2` was created with all confirmed groups assigned wholly to one split. V1 was retained unchanged and no photographs were removed. V2 passed RGB/hash integrity, ID/path/content-hash disjointness, class-presence, confirmed-group, reproducibility, and DataLoader-contract checks.

### Preprocessing and EDA

All inputs are full RGB photographs. Segmentation masks and head bounding boxes are not used.

Validation and test, and train when `augment=false`, use `ResNet50_Weights.IMAGENET1K_V2.transforms()` as the deterministic preprocessing source: resize the shorter side to 232, center crop to 224, convert to a tensor, then apply ImageNet normalization. Train shuffling remains enabled when `augment=false`.

With `augment=true`, train uses:

1. `RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.3333))` without first stretching the image to a square;
2. `RandomHorizontalFlip(p=0.5)`;
3. `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)`;
4. tensor conversion and ImageNet normalization.

The EDA notebook reads the active split from `configs/data.yaml` and has saved executed outputs for integrity reports, class distributions, split-overlap checks, image dimensions/aspect ratios, and original plus augmented **train** examples for every class. It was executed against `splits/v2` without modifying frozen manifests. EDA and duplicate review are data-integrity activities only and must not be used to choose model settings.

## 3. Experimental plan

### Training and controlled comparison

All C0–C3 runs use ImageNet-pretrained ResNet50 with a replaced `Linear(2048 → 10)` head, AdamW, batch size 32, cross-entropy loss, at most 30 epochs, and early stopping on validation macro-F1 with patience 5.

`split_seed=42` is fixed independently from the run seed. Each condition is run with training seeds `{42, 43, 44}`; the run seed is passed to loaders as `training_seed`. Before every run, Python, NumPy, and PyTorch are seeded. For a given seed, all compared conditions start from identical pretrained weights and identical new-head initialization.

For every C0–C3 condition, the learning-rate candidates `{1e-3, 1e-4, 1e-5}` are evaluated for all three seeds. One learning rate per condition is selected from the **mean validation macro-F1**. The three checkpoints from the chosen learning rate are retained for final evaluation; every run selects its own checkpoint using validation data, and no single “best seed” is selected.

BatchNorm running statistics are fixed through eval mode for all C0–C3 runs. BatchNorm affine `weight` and `bias` parameters are trainable only when their block is unfrozen. The training loop must restore this BatchNorm eval policy after `model.train()`; running statistics are not described as trainable parameters.

The augmentation ablation is pre-specified for C0 and C3. Its `augment=false` runs use the same selected learning rate and the same three seeds as their corresponding augmentation-on condition; no new learning-rate search is performed. This isolates the effect of augmentation while holding other choices fixed.

### Interpretation and uncertainty

Results are reported as mean ± standard deviation across the three retained checkpoints/seeds. This is a descriptive estimate of training variability, not a claim of statistical significance. Conclusions are limited to these ten classes, this fixed custom split, ResNet50, and the evaluated settings. A full fine-tuning win is compatible with the hypothesis; results from four conditions do not establish a general cutoff.

### Test protocol and pipeline order

The test set is not used to select classes, preprocessing, hyperparameters, checkpoints, seeds, or later training. Test-file integrity checks are allowed. Error analysis after final evaluation is allowed but must not change models or training choices.

The required order is:

1. complete data review and freeze the split version;
2. train C0–C3 and select learning rates using validation only;
3. run the C0/C3 augmentation ablation;
4. freeze all configurations and checkpoints;
5. evaluate all planned variants and seeds once on test;
6. perform failure analysis without modifying models.

Preliminary comparisons are validation-only. Final test evaluation includes macro-F1, accuracy, per-class F1, a confusion matrix, and planned failure-analysis material.

## 4. Implementation plan

1. Maintain the reviewed `splits/v2` manifests and repeat the documented integrity checks after any explicitly approved future split version.
2. Use `src/dataset.py` as the shared, read-only Dataset/DataLoader interface; it consumes only the committed mapping and manifests.
3. Implement the shared `train.py` with per-epoch history, checkpoint selection by validation macro-F1, configured unfreezing depth, deterministic run setup, and BatchNorm policy restoration.
4. Run the C0–C3 learning-rate protocol and the fixed augmentation ablation.
5. Implement one `evaluate.py` that reports all final metrics and persists evaluation artifacts.
6. Measure the primary computational trade-offs—training time and peak GPU memory—on the same hardware and under identical measurement settings. Inference latency is a supplementary metric because the architecture is unchanged and freezing is not assumed to accelerate inference.
7. Complete test evaluation, error analysis, the technical summary, and a fresh-clone reproducibility check.

The current `requirements.txt` uses `>=` constraints; it is not a pinned environment. Before final experiments, exact compatible package versions, PyTorch/CUDA details, device information, and runtime environment must be recorded and frozen. Mean ± standard deviation always refers to all three checkpoints/runs selected by the protocol, not a single best-seed checkpoint.

## 5. Team responsibilities

| Member | Role | Responsibility |
|---|---|---|
| Tikhonova Anna | Team Lead | Single `evaluate.py`; failure analysis with Grad-CAM; two-page summary; demo and slides; repository integration and final reproducibility check |
| Pyanov Georgij | Data Engineer | Dataset download and verification; breed subset selection; frozen stratified splits; `src/dataset.py`; README/requirements; runtime measurements; error-image export |
| Fadeeva Albina | ML Engineer | `src/train.py` (sole owner); conditions C0 and C1, three seeds each: training, validation LR selection, checkpoints, and histories |
| Kirillov Maksim | ML Engineer | `src/gradcam.py` (sole owner); conditions C2 and C3, three seeds each; augmentation ablation on C0 and C3 |

**Working agreements.** One shared `train.py` has no condition-specific forks; conditions differ only through configuration. The split version is frozen before training, metrics are produced only by the lead’s `evaluate.py`, every run logs its configuration and Git commit hash, and the team synchronizes before Tuesday milestones.
