# Preliminary test-set results

**Status: partial.** The four frozen augmentation-on conditions have been
evaluated. The two augmentation-off ablations do not yet have trained
checkpoints, so the six-configuration comparison is not complete. `N/A` means
the evaluation has not been run; it is not a zero score.

## Comparison

`src/evaluate.py` evaluated the frozen `splits/v2/test.csv` split (300 images)
with the retained checkpoints for training seeds 42, 43, and 44. Values are
mean ± sample standard deviation across those three checkpoints.

| Configuration | Accuracy (mean ± std) | Macro-F1 (mean ± std) | Artifacts |
| --- | ---: | ---: | --- |
| C0 · augmentation on | 0.9511 ± 0.0051 | 0.9512 ± 0.0051 | [Per-class F1](tables/C0_augment_on_test_per_class_f1.csv) · [confusion matrix](figures/C0_augment_on_test_confusion_matrix.png) · [latency](tables/C0_augment_on_test_latency.csv) |
| C1 · augmentation on | 0.9656 ± 0.0069 | 0.9655 ± 0.0070 | [Per-class F1](tables/C1_augment_on_test_per_class_f1.csv) · [confusion matrix](figures/C1_augment_on_test_confusion_matrix.png) · [latency](tables/C1_augment_on_test_latency.csv) |
| C2 · augmentation on | 0.9600 ± 0.0033 | 0.9601 ± 0.0033 | [Per-class F1](tables/C2_augment_on_test_per_class_f1.csv) · [confusion matrix](figures/C2_augment_on_test_confusion_matrix.png) · [latency](tables/C2_augment_on_test_latency.csv) |
| C3 · augmentation on | 0.9633 ± 0.0067 | 0.9633 ± 0.0067 | [Per-class F1](tables/C3_augment_on_test_per_class_f1.csv) · [confusion matrix](figures/C3_augment_on_test_confusion_matrix.png) · [latency](tables/C3_augment_on_test_latency.csv) |
| C0 · augmentation off | N/A · checkpoints pending | N/A · checkpoints pending | Config: [`configs/c0_noaug.yaml`](../configs/c0_noaug.yaml) |
| C3 · augmentation off | N/A · checkpoints pending | N/A · checkpoints pending | Config: [`configs/c3_noaug.yaml`](../configs/c3_noaug.yaml) |

Per-class F1 is the mean and sample standard deviation across the same three
checkpoints. Confusion matrices pool their predictions on the fixed test set.
The latency CSVs contain batch-1 and batch-32 measurements for each checkpoint;
these were measured on the local CPU and are hardware-specific. Full sample
predictions are available in the corresponding `*_predictions.csv` files.

## Test error review

The detailed [failure analysis](failure_analysis.md) includes full source
images, regenerated Grad-CAM overlays, checkpoint/seed provenance, probable
causes, cross-seed observations, and limitations for the three cases below.

The examples below are C0 augmentation-on test errors. Probable causes describe
the visible image ambiguity and model attention; Grad-CAM is supporting
visualization, not proof of a causal explanation.

1. **`Siamese_257`: Siamese → Birman (seed 44, confidence 0.916).** The image
   is a tight, somewhat soft crop of a color-point cat. Grad-CAM concentrates
   around the face, where these two breeds share visible color-point features;
   the crop provides little body-shape context, and red-eye glare further
   obscures the eye detail.

   ![Siamese_257 input and Grad-CAM](figures/gradcam_siamese_257.png)

2. **`Russian_Blue_264`: Russian Blue → British Shorthair (seed 44, confidence
   0.908).** This outdoor image shows a gray cat in a side-facing pose amid
   flowers. The heatmap focuses on the head and shoulder. Both classes have a
   solid blue-gray coat, while the pose, foliage, and lighting make fine facial
   and coat-texture differences harder to read.

   ![Russian_Blue_264 input and Grad-CAM](figures/gradcam_russian_blue_264.png)

3. **`pug_74`: Pug → American Bulldog (seed 42, confidence 0.810).** The dog is
   seated in a chair and photographed front-on. Attention is strongest around
   the face and muzzle. The short muzzle and tan coat are shared visual cues;
   the seated pose and chair obscure body proportions useful for distinguishing
   a small pug from a bulldog.

   ![pug_74 input and Grad-CAM](figures/gradcam_pug_74.png)

These are representative, high-confidence errors selected from the saved
per-image predictions. Test inspection is used only for this final descriptive
analysis.

## Protocol note

**No settings are changed based on test results.** Model selection and
hyperparameter choices were made from validation results before this test
evaluation. Test results are used only for reporting and error analysis; tuning
on them is forbidden.

## Remaining work

Train and retain the three C0 augmentation-off checkpoints at LR `1e-3` and the
three C3 augmentation-off checkpoints at LR `1e-5`, for seeds 42, 43, and 44,
using the commands in the [README](../README.md#final-test-evaluation-and-augmentation-ablation).
Then run the two matching `src.evaluate` commands from that section and replace
the two pending rows above with their metrics and artifact links. The
six-configuration Definition of Done is not met until those runs are present.
