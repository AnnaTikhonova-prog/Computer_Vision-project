# Misclassified test images — EVAL-1 / issue #15

**Leakage note:** these files are read-only, post-evaluation diagnostic data.
Do not use the test errors to choose or change models, checkpoints, seeds,
hyperparameters, preprocessing, labels, or training data. Exporting does not
run inference, train a model, or modify the frozen split or prediction tables.

## Layout

```text
reports/errors/
  C0_augment_on/
    british_shorthair__as__russian_blue/
      British_Shorthair_*.jpg
    siamese__as__birman/
      Siamese_*.jpg
  C1_augment_on/...
  C2_augment_on/...
  C3_augment_on/...
  C0_augment_off/...
  C3_augment_off/...
  index.csv
  summary.csv
```

Folder names read **true breed `__as__` predicted breed**. Original filenames
and JPEG bytes are preserved. Each image is copied once per condition,
augmentation state, and predicted-breed pair, even when multiple seeds make
the same error. An image can appear in multiple pair folders if different
seeds predict different wrong classes. Images shared by conditions are copied
into each relevant condition folder.

`index.csv` has one row per incorrect prediction, preserving training seed,
checkpoint, true/predicted labels and names, softmax confidence, source table,
source path, exported path, and source-file SHA-256. Paths in the index are
relative to the repository root. `summary.csv` counts errors and unique images
per condition/pair, including per-seed error counts. Counts across seeds are
correlated observations, not independent test samples or ensemble results.

## Current export

Sources are all six saved augmentation-on/off
[`reports/tables/*_test_predictions.csv`](../tables/) tables. Each contains
900 predictions: the frozen 300-image test split for seeds 42, 43, and 44.

| Condition | Incorrect predictions | Distinct error images | Exported image files | Pair folders |
| --- | ---: | ---: | ---: | ---: |
| C0, augmentation on | 44 | 16 | 16 | 9 |
| C1, augmentation on | 31 | 15 | 15 | 7 |
| C2, augmentation on | 36 | 18 | 20 | 10 |
| C3, augmentation on | 33 | 17 | 18 | 10 |
| C0, augmentation off | 46 | 16 | 16 | 7 |
| C3, augmentation off | 38 | 21 | 22 | 10 |
| Total across conditions | 228 | — | 107 | 53 |

Both augmentation-off test tables and their error images are included.
The existing three source images, Grad-CAM panels, and `cases.csv` at this
directory's top level support the [failure analysis](../failure_analysis.md)
and are preserved separately from the complete grouped export.

## Reproduce

After downloading the official dataset, run from the repository root:

```powershell
python scripts/export_test_errors.py
```

The script needs Python and Pillow, not checkpoints, PyTorch, or a GPU. It
discovers all condition-level test prediction tables, validates complete
coverage for the three seeds against `splits/v2/test.csv`, checks label and
correctness consistency, and verifies decoded error-image content against
the frozen manifest before writing. JPEGs are copied without re-encoding.
Rerunning with unchanged inputs preserves existing image files and produces
the same index and summary. A differing existing destination image causes an
error rather than being overwritten; unrelated files are not removed.

Custom dataset paths can be supplied with `--dataset-root`. To export a
subset of tables, use `--predictions` and a separate `--output-dir` so its
index describes only that export. The script's source contains the same
leakage note as this README.
