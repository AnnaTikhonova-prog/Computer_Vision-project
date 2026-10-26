# Fine-tuning depth on Oxford-IIIT Pet

This repository compares ResNet50 fine-tuning depth on one fixed 10-breed
subset. The data interface here is shared by C0–C3; it does not change any
experiment condition or run model training.

## Fixed subset and labels

`splits/v2/class_mapping.json` is the active class mapping. Its labels are
stable and contiguous from 0 to 9:

| Label | Project name | Official annotation prefix |
| ---: | --- | --- |
| 0 | British Shorthair | `British_Shorthair` |
| 1 | Russian Blue | `Russian_Blue` |
| 2 | Siamese | `Siamese` |
| 3 | Birman | `Birman` |
| 4 | American Bulldog | `american_bulldog` |
| 5 | Pug | `pug` |
| 6 | Sphynx | `Sphynx` |
| 7 | Maine Coon | `Maine_Coon` |
| 8 | Beagle | `beagle` |
| 9 | Samoyed | `samoyed` |

The preparation command checks these prefixes in the actual official
annotations. `English Bulldog` and `German Shepherd` are **not** Oxford-IIIT
Pet annotation classes; they are not silently substituted. In particular,
the earlier planning text must not be interpreted as selecting them.

## Install and prepare data

Use Python 3.13.15 (or Python 3.13.x) and install the matching CPU or CUDA
PyTorch wheels first; the remaining direct dependencies are pinned in
`requirements.txt`. The recorded GPU runs used PyTorch 2.11.0 and torchvision
0.26.0.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pip==25.1.1
# Choose exactly one PyTorch build:
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# CPU-only alternative:
# python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/prepare_oxford_pets.py --download-only
python scripts/verify_data.py --dataset-root data/oxford-iiit-pet --split-dir splits/v2 --check-loaders --check-reproducibility
```

The dataset command downloads and extracts the two official archives from the
official Oxford URL to `data/oxford-iiit-pet/`. It does **not** generate,
replace, or otherwise touch frozen manifests. A fresh clone then uses the
committed active `splits/v2` manifests directly. The CUDA 12.8 command matches
the recorded Tesla T4 runs; use the CPU wheel on a CPU-only host. torchvision
downloads the ImageNet V2 ResNet50 weights on the first training run, so allow
network access. Reproducing the training grid requires a compatible CUDA GPU;
the CPU wheel supports data verification and evaluation, but the frozen
training configs request CUDA. The PyTorch wheel commands follow the [official
installation matrix](https://pytorch.org/get-started/previous-versions/).

For a fresh clone, the complete read-only data workflow after installation is:

```powershell
python scripts/prepare_oxford_pets.py --download-only
python scripts/verify_data.py --dataset-root data/oxford-iiit-pet --split-dir splits/v2 --check-loaders --check-reproducibility
python -m jupyter notebook notebooks/eda_oxford_pets.ipynb
```

The download-only and verification commands do not regenerate or overwrite the
committed active manifests; the notebook reads them through `configs/data.yaml`.

Generating a manifest is an explicit, versioned action. The preparation
command verifies every selected annotated image and writes:

```text
splits/
  class_mapping.json
  train.csv
  val.csv
  test.csv
  integrity_report.json
  exact_duplicates.json
  near_duplicate_candidates.json
  confirmed_groups_applied.json
  split_report.json
```

The images and generated reports are local artifacts. Raw data is ignored by
Git. The CSV paths are relative to `dataset_root`, e.g.
`images/Birman_101.jpg`, never absolute paths.

The result is a **custom** stratified 70/15/15 split made with seed 42 over the
union of official `trainval.txt` and `test.txt`. It is not the official
Oxford-IIIT Pet benchmark split. `split_report.json` records real class counts,
targets, and any small deviations caused by a manually confirmed duplicate
group.

Preparation never overwrites a frozen manifest. To create a future explicit
new version, use a new output directory:

```powershell
python scripts/prepare_oxford_pets.py --dataset-root data/oxford-iiit-pet --output-dir splits/v3 --seed 42
```

The active configuration uses the reviewed `splits/v2` manifests: 1,399 train,
300 validation, and 300 test samples. Every class has 140/30/30 samples except
Siamese, which has 139/30/30.

`near_duplicate_candidates.json` is only a review queue based on an 8×8
grayscale average-hash Hamming distance (default at most 5). It removes
nothing. If review confirms a group, save it as
`{"groups": [["image_id_a", "image_id_b"]]}` and explicitly make a new
version; confirmed groups are kept in one split.

```powershell
python scripts/prepare_oxford_pets.py --dataset-root data/oxford-iiit-pet --output-dir splits/v3 --confirmed-groups confirmed_duplicates.json
```

Exact RGB duplicates are different: preparation hashes decoded RGB bytes plus
width and height, keeps the lexicographically smallest `image_id`, and records
the discarded entries and reasons in `integrity_report.json` and
`exact_duplicates.json`.

## Validate frozen artifacts

Run this after download on a fresh clone, or after preparation of a new split.
It re-decodes each file, recomputes hashes, checks
the mapping and all cross-split ID/path/content-hash intersections. The two
flags additionally check the loader contract and regenerate manifests in a
temporary directory for a bytewise reproducibility comparison.

```powershell
python scripts/verify_data.py --dataset-root data/oxford-iiit-pet --split-dir splits/v2 --check-loaders --check-reproducibility
```

Use the EDA only after manifests are frozen:

```powershell
python -m jupyter notebook notebooks/eda_oxford_pets.ipynb
```

It reads reports/manifests and shows only training examples for augmentation
visualization. No settings are changed based on test results. The test split may be inspected
for final evaluation and diagnostic error analysis; tuning on test results is
forbidden.

## Perceptual-hash review queue

Perceptual hashes are only an integrity-review heuristic. They must not be
used to choose model classes, preprocessing, hyperparameters, or any C0–C3
condition. Build/open the local pair viewer after data preparation:

```powershell
python scripts/review_near_duplicates.py --split-dir splits
Start-Process splits\near_duplicate_review.html
```

The page displays each candidate pair side by side with `image_id`, class,
current split, and Hamming distance. Decisions persist in
`splits/near_duplicate_review.csv`; each row must be one of `duplicate`,
`near-duplicate`, `different images`, or `uncertain`. It is safe to edit this
CSV in a spreadsheet, or record one decision reproducibly:

```powershell
python scripts/review_near_duplicates.py --set-decision "Birman_120__Russian_Blue_226" "different images" --notes "Distinct photos after visual review"
```

To import a completed semicolon-delimited Excel export while preserving its
source file, validate it and write the normalised review plus transitive groups:

```powershell
python scripts/review_near_duplicates.py --split-dir splits --import-completed splits/near_duplicate_review_completed.csv --confirmed-groups-output splits/confirmed_groups_v2.json
```

The completed Excel export was validated against all 91 source pairs and
normalised into `splits/near_duplicate_review.csv`: 84 `different images`, 3
`duplicate`, 4 `near-duplicate`, and no `uncertain` decisions. Its seven
confirmed pairs form six transitive groups in `splits/confirmed_groups_v2.json`.
Two confirmed v1 cross-split groups were resolved in v2; v1 itself was not
edited and the original Excel export is preserved.

## Definition of Done status

The v2 manifests passed RGB/hash integrity, ID/path/content-hash disjointness,
class-presence, confirmed-group, reproducibility, and DataLoader-contract
checks. The EDA notebook reads the active split from `configs/data.yaml`; its
executed outputs are saved in `notebooks/eda_oxford_pets.ipynb`. It includes
the v2 tables, overlap and confirmed-group checks, image-size distributions,
and original plus augmented training examples for all ten classes. It did not
modify the frozen manifests.

## DataLoader interface

`src.dataset.get_loaders(config)` returns a dictionary with exactly `train`,
`val`, and `test` DataLoaders. It is strictly read-only: it loads existing CSVs
and the mapping and cannot regenerate or edit them.

```python
import torch
import yaml
from src.dataset import get_loaders

with open("configs/data.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)

loaders = get_loaders(config)
images, labels = next(iter(loaders["train"]))
assert images.shape[1:] == (3, 224, 224)
assert labels.dtype == torch.int64
```

The config must provide `dataset_root`, `split_dir`, `batch_size`,
`num_workers`, `augment`, and `training_seed` (legacy `seed` is accepted).
The train loader always shuffles. With `augment: true`, it applies
`RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.3333))`, horizontal
flip, ColorJitter, tensor conversion, and ImageNet normalization without first
stretching an image to a square. Validation/test, and train with
`augment: false`, use the deterministic
`ResNet50_Weights.IMAGENET1K_V2.transforms()` transform. Validation/test never
shuffle and never drop their final batch.

## Training C0–C3

All conditions use the same `src/train.py`. Choose the condition through YAML;
each condition inherits shared settings from `configs/training.yaml` and
`configs/data.yaml`.

| Config | What trains |
| --- | --- |
| `configs/c0.yaml` | Classifier head only |
| `configs/c1.yaml` | Head + `layer4` |
| `configs/c2.yaml` | Head + `layer3` + `layer4` |
| `configs/c3.yaml` | Entire ResNet50 |

Run from the repository root using the activated environment. To reproduce
the full four-condition learning-rate grid, run this PowerShell block. It
launches 36 runs (4 conditions × 3 learning rates × 3 seeds) and writes each
run to a new directory under `artifacts/`:

```powershell
$conditions = @("c0", "c1", "c2", "c3")
$learningRates = @("1e-3", "1e-4", "1e-5")
$seeds = @(42, 43, 44)
foreach ($condition in $conditions) {
    foreach ($lr in $learningRates) {
        foreach ($seed in $seeds) {
            $runDir = "artifacts/${condition}_lr${lr}_seed${seed}"
            python -m src.train --config "configs/${condition}.yaml" --training-seed $seed --lr $lr --output-dir $runDir
            if ($LASTEXITCODE -ne 0) { throw "Training failed: $condition LR=$lr seed=$seed" }
        }
    }
}
```

Training seeds are **42, 43, 44**, independent of `split_seed=42`; LR candidates
are **1e-3, 1e-4, 1e-5**. Training uses ImageNet V2 ResNet50, AdamW and batches of
32, processed in microbatches of four. The best checkpoint is selected by
validation macro-F1, with at most 30 epochs and early stopping after five epochs
without improvement. BatchNorm running statistics stay frozen; its weight and
bias train only in unfrozen blocks. The test loader is not used during training.

Each run creates a new output directory containing `best.pt`, `history.csv`,
the resolved `config.yaml`, `metadata.json` (seeds, Git commit and environment)
and `runtime.json`. Existing directories are never overwritten. **`artifacts/`
is ignored by Git**, so back up results and share them separately with the team.

## Training reproducibility

The trainer seeds Python, NumPy, PyTorch and loader workers and enables strict
deterministic GPU operations. For a fixed seed, C0–C3 start with identical heads;
BN running statistics are checked for changes after every epoch.

To repeat a run, use its saved config with a new output directory and compare histories:

```powershell
python -m src.train --config artifacts/c2_seed43_lr1e-4/config.yaml --output-dir artifacts/c2_repeat
fc.exe /b artifacts\c2_seed43_lr1e-4\history.csv artifacts\c2_repeat\history.csv
```

After the training implementation is merged, `src/train.py` is frozen.
Subsequent changes to that file require a new issue and review; this workflow
rule takes effect at merge and is not enforced through file permissions.

## Evaluation

Use `src/evaluate.py` as the only source for final project test metrics. For
three checkpoints it reports the mean and sample standard deviation over seeds
42, 43, and 44. Summary, per-class F1, per-image predictions, and latency CSVs
are saved under `reports/tables/`; pooled confusion matrices are saved under
`reports/figures/`. Filenames include condition and augmentation state so
ablation outputs cannot overwrite C0/C3 outputs.

For example, aggregate the retained C0 checkpoints on validation:

```powershell
python -m src.evaluate --condition C0 --checkpoints deliverables/c0/runs/lr_1e-3/seed_42/best.pt deliverables/c0/runs/lr_1e-3/seed_43/best.pt deliverables/c0/runs/lr_1e-3/seed_44/best.pt --config configs/c0.yaml --split val --device cpu
```

### Final test evaluation and augmentation ablation

The ablation is pre-specified: hold the selected LR and seeds fixed and set
only `augment: false`. C0 selected LR is `1e-3`; C3 selected LR is `1e-5`.
The configs are `configs/c0_noaug.yaml` and `configs/c3_noaug.yaml`. Train the
six ablation runs after C0/C3 settings have been frozen:

```powershell
python -m src.train --config configs/c0_noaug.yaml --training-seed 42 --lr 1e-3 --output-dir artifacts/c0_noaug_lr1e-3_seed42
python -m src.train --config configs/c0_noaug.yaml --training-seed 43 --lr 1e-3 --output-dir artifacts/c0_noaug_lr1e-3_seed43
python -m src.train --config configs/c0_noaug.yaml --training-seed 44 --lr 1e-3 --output-dir artifacts/c0_noaug_lr1e-3_seed44
python -m src.train --config configs/c3_noaug.yaml --training-seed 42 --lr 1e-5 --output-dir artifacts/c3_noaug_lr1e-5_seed42
python -m src.train --config configs/c3_noaug.yaml --training-seed 43 --lr 1e-5 --output-dir artifacts/c3_noaug_lr1e-5_seed43
python -m src.train --config configs/c3_noaug.yaml --training-seed 44 --lr 1e-5 --output-dir artifacts/c3_noaug_lr1e-5_seed44
```

Run all six final evaluations only after every model and setting is frozen.
The first four use the retained selected-LR checkpoints in `deliverables/`;
the last two use the ablation checkpoints produced above.

```powershell
python -m src.evaluate --condition C0 --checkpoints deliverables/c0/runs/lr_1e-3/seed_42/best.pt deliverables/c0/runs/lr_1e-3/seed_43/best.pt deliverables/c0/runs/lr_1e-3/seed_44/best.pt --config configs/c0.yaml --split test --device cpu
python -m src.evaluate --condition C1 --checkpoints deliverables/c1/runs/lr_1e-4/seed_42/best.pt deliverables/c1/runs/lr_1e-4/seed_43/best.pt deliverables/c1/runs/lr_1e-4/seed_44/best.pt --config configs/c1.yaml --split test --device cpu
python -m src.evaluate --condition C2 --checkpoints deliverables/c2/runs/lr_1e-5/seed_42/best.pt deliverables/c2/runs/lr_1e-5/seed_43/best.pt deliverables/c2/runs/lr_1e-5/seed_44/best.pt --config configs/c2.yaml --split test --device cpu
python -m src.evaluate --condition C3 --checkpoints deliverables/c3/runs/lr_1e-5/seed_42/best.pt deliverables/c3/runs/lr_1e-5/seed_43/best.pt deliverables/c3/runs/lr_1e-5/seed_44/best.pt --config configs/c3.yaml --split test --device cpu
python -m src.evaluate --condition C0 --checkpoints artifacts/c0_noaug_lr1e-3_seed42/best.pt artifacts/c0_noaug_lr1e-3_seed43/best.pt artifacts/c0_noaug_lr1e-3_seed44/best.pt --config configs/c0_noaug.yaml --split test --device cpu
python -m src.evaluate --condition C3 --checkpoints artifacts/c3_noaug_lr1e-5_seed42/best.pt artifacts/c3_noaug_lr1e-5_seed43/best.pt artifacts/c3_noaug_lr1e-5_seed44/best.pt --config configs/c3_noaug.yaml --split test --device cpu
```

After evaluation, compile the six summary CSVs into
`reports/results_preliminary.md`, add mean ± sample standard deviation for
accuracy and macro-F1, link all per-class F1 tables and confusion matrices,
and analyze at least three representative errors with Grad-CAM visualizations.
Use a row with `correct=0` from one of the generated `*_predictions.csv` files
to choose a test `image_id`, then generate a supporting overlay:

```powershell
python scripts/generate_gradcam.py --checkpoint deliverables/c3/runs/lr_1e-5/seed_42/best.pt --config configs/c3.yaml --image-id pug_123 --output reports/figures/gradcam_pug_123.png --device cpu
```

Replace the illustrative image ID with an actual misclassified ID from the
CSV. Describe likely visual causes from the source image and overlay; Grad-CAM
is supporting evidence, not a causal explanation.
**No settings are changed based on test results.** Test errors may be inspected
for analysis, but tuning on test results is forbidden.

The current partial comparison and error review are in
[`reports/results_preliminary.md`](reports/results_preliminary.md). It marks
the two augmentation-off results as pending until their checkpoints are
trained and evaluated.

### Team clean-environment review

Please ask every team member to follow the install → download and verify data →
train → evaluate sequence above from a fresh clone. Each person should confirm
the Python/PyTorch versions, successful data verification, and whether the
commands run without undocumented steps before this README is treated as the
reproducibility handoff.
