# Fine-tuning depth on Oxford-IIIT Pet

This repository compares ResNet50 fine-tuning depth on one fixed 10-breed
subset. The data interface here is shared by C0–C3; it does not change any
experiment condition or run model training.

## Fixed subset and labels

`splits/class_mapping.json` is the sole class mapping. Its labels are stable
and contiguous from 0 to 9:

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

Install a PyTorch build suitable for your platform, then install the remaining
packages. `torchvision` is required for both official-data download support and
the ResNet50 preprocessing contract.

```powershell
py -3 -m pip install -r requirements.txt
py -3 scripts/prepare_oxford_pets.py --download
```

The second command downloads the two archives from the official Oxford URL to
`data/oxford-iiit-pet/`, verifies every selected annotated image, and writes:

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

Preparation never overwrites a frozen manifest. To create an explicit new
version, use a new output directory:

```powershell
py -3 scripts/prepare_oxford_pets.py --dataset-root data/oxford-iiit-pet --output-dir splits/v2 --seed 42
```

`near_duplicate_candidates.json` is only a review queue based on an 8×8
grayscale average-hash Hamming distance (default at most 5). It removes
nothing. If review confirms a group, save it as
`{"groups": [["image_id_a", "image_id_b"]]}` and explicitly make a new
version; confirmed groups are kept in one split.

```powershell
py -3 scripts/prepare_oxford_pets.py --dataset-root data/oxford-iiit-pet --output-dir splits/v2 --confirmed-groups confirmed_duplicates.json
```

Exact RGB duplicates are different: preparation hashes decoded RGB bytes plus
width and height, keeps the lexicographically smallest `image_id`, and records
the discarded entries and reasons in `integrity_report.json` and
`exact_duplicates.json`.

## Validate frozen artifacts

Run this after preparation. It re-decodes each file, recomputes hashes, checks
the mapping and all cross-split ID/path/content-hash intersections. The two
flags additionally check the loader contract and regenerate manifests in a
temporary directory for a bytewise reproducibility comparison.

```powershell
py -3 scripts/verify_data.py --check-loaders --check-reproducibility
```

Use the EDA only after manifests are frozen:

```powershell
py -3 -m jupyter notebook notebooks/eda_oxford_pets.ipynb
```

It reads reports/manifests and shows only training examples for augmentation
visualization. It never uses model predictions or test examples to select
classes, preprocessing, or a split.

## Perceptual-hash review queue

Perceptual hashes are only an integrity-review heuristic. They must not be
used to choose model classes, preprocessing, hyperparameters, or any C0–C3
condition. Build/open the local pair viewer after data preparation:

```powershell
py -3 scripts/review_near_duplicates.py --split-dir splits
Start-Process splits\near_duplicate_review.html
```

The page displays each candidate pair side by side with `image_id`, class,
current split, and Hamming distance. Decisions persist in
`splits/near_duplicate_review.csv`; each row must be one of `duplicate`,
`near-duplicate`, `different images`, or `uncertain`. It is safe to edit this
CSV in a spreadsheet, or record one decision reproducibly:

```powershell
py -3 scripts/review_near_duplicates.py --set-decision "Birman_120__Russian_Blue_226" "different images" --notes "Distinct photos after visual review"
```

Re-run the first command to rebuild the HTML and
`near_duplicate_review_status.json`. Both `duplicate` and `near-duplicate`
are treated conservatively as confirmed for leakage review. If either member
is in another split, the status report lists it. **Do not edit frozen CSVs.**
First review and approve the listed groups, then create a separate versioned
split with `prepare_oxford_pets.py --output-dir splits/v2 --confirmed-groups …`.

## Definition of Done status

Automated preparation and integrity checks are complete for the frozen v1
split. The perceptual-hash candidate review is deliberately open until a human
visually assigns each row in `near_duplicate_review.csv`. Current completion
status is machine-readable in `near_duplicate_review_status.json`; it lists
confirmed and unresolved cross-split pairs separately. At this point all 91 pairs are
`uncertain`, so no correction to v1 is proposed and no frozen CSV has changed.

## DataLoader interface

`src.dataset.get_loaders(config)` returns a dictionary with exactly `train`,
`val`, and `test` DataLoaders. It is strictly read-only: it loads existing CSVs
and the mapping and cannot regenerate or edit them.

```python
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
