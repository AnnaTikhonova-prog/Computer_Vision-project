# Milestone Status Report: Progress, Results, and Remaining Work

**Project:** Oxford-IIIT Pet, ten-breed classification subset
**Status date:** 27 October 2026
**Repository:** [AnnaTikhonova-prog/Computer_Vision-project](https://github.com/AnnaTikhonova-prog/Computer_Vision-project)

## Executive summary

The shared data pipeline, training implementation, and four progressive
fine-tuning conditions (C0-C3) are in place. The complete three-learning-rate,
three-seed training grids have been run for all four conditions, and the
validation-selected checkpoints are retained in `deliverables/`. The four
augmentation-on conditions have also been evaluated on the frozen test split.
Three representative test failures have a written analysis and Grad-CAM
visualizations.

The main experiment and the pre-specified C0/C3 augmentation ablation are
complete. The ablation's validation comparison is recorded in
`reports/ablation.md`. Separately, the current checkout's final test-results
table still has no augmentation-off test metrics, and the ablation run bundles
are not present in this checkout. Exact training wall-clock time and the exact
CPU model for the saved inference-latency measurements also need to be
recorded. The remaining plan below fits into two weeks, with recovery of the
existing ablation bundles ahead of any decision to rerun them.

## Completed experiments

- Established the frozen, stratified `splits/v2` split: 1,399 training, 300
  validation, and 300 test images across ten classes. The split is custom and
  is not the official Oxford-IIIT Pet benchmark split.
- Implemented one shared trainer and YAML configuration system for all
  conditions, with ImageNet V2 pretrained ResNet50, validation macro-F1
  checkpoint selection, and frozen BatchNorm running statistics.
- Completed the 3-LR × 3-seed grid for C0, C1, C2, and C3 (36 runs total),
  retaining all three seed checkpoints for each condition's selected LR.
- Evaluated all 12 retained augmentation-on checkpoints on the test split
  using `src/evaluate.py`; saved accuracy, macro-F1, per-class F1, confusion
  matrices, predictions, and CPU inference latency under `reports/`.
- Completed the pre-specified augmentation-off ablation for C0 and C3, with
  three seeds per condition, and recorded its validation comparison in
  `reports/ablation.md`. The learning rates were fixed at the previously
  selected values; augmentation was the changed factor. The raw run bundles
  are not included in this checkout, which is an artifact-provenance issue,
  not an incomplete ablation experiment.
- Completed post-evaluation failure analysis for three C0 test errors, with
  source images and Grad-CAM overlays. Exported misclassified test images for
  all four augmentation-on conditions, grouped by true and predicted class.
- Documented installation, dataset download, training, and evaluation
  instructions in the README.

## Preliminary comparison

The table reports validation macro-F1 for the selected learning rate and
test metrics for the same condition. Each value is a mean ± sample standard
deviation across three retained checkpoints (seeds 42, 43, and 44). Test
results are descriptive; **no settings are changed based on test results**.

| Condition | Trainable part | Selected LR | Validation macro-F1 | Test accuracy | Test macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| C0 | Classifier head | `1e-3` | 0.9701 ± 0.0034 | 0.9511 ± 0.0051 | 0.9512 ± 0.0051 |
| C1 | `layer4` + classifier head | `1e-4` | **0.9856 ± 0.0038** | **0.9656 ± 0.0069** | **0.9655 ± 0.0070** |
| C2 | `layer3`–`layer4` + classifier head | `1e-5` | 0.9801 ± 0.00002 | 0.9600 ± 0.0033 | 0.9601 ± 0.0033 |
| C3 | All parameters | `1e-5` | 0.9767 ± 0.0066 | 0.9633 ± 0.0067 | 0.9633 ± 0.0067 |

The highest mean validation and test scores come from C1. Relative to C0, C1
improves test accuracy by 1.44 percentage points and test macro-F1 by 1.43
points on this split. C3 does not exceed C1 despite training the full network.
The test set has 300 images and one split, so these results do not establish
that the same ranking will hold on other data.

### Augmentation ablation on validation

| Condition | Augmentation on: validation macro-F1 | Augmentation off: validation macro-F1 | Difference (off − on) |
| --- | ---: | ---: | ---: |
| C0 | 0.9701 ± 0.0034 | 0.9598 ± 0.0058 | −0.0103 |
| C3 | 0.9767 ± 0.0066 | 0.9677 ± 0.0052 | −0.0090 |

These are validation-only, descriptive differences. The augmentation ablation
is complete on validation. The separate final test-results table still needs
test metrics for the two augmentation-off configurations.

## Observations and difficulties

- C1 (`layer4` plus head) was the strongest of the four tested conditions on
  both validation and test. Unfreezing more layers did not guarantee a gain.
- For C2 and C3, LR `1e-3` collapsed to macro-F1 near 0.018, while smaller
  learning rates achieved roughly 0.98 validation macro-F1. This sharp
  sensitivity makes the LR comparison important to report alongside the
  selected result.
- C3's selected LR `1e-5` exceeds `1e-4` by only about 0.000084 mean validation
  macro-F1, much less than the across-seed standard deviation. The fixed
  selection rule selects `1e-5`, but three seeds do not show a reliable
  advantage over `1e-4`.
- The validation ablation suggests augmentation helps C0 and C3 by roughly
  0.9–1.0 macro-F1 percentage points. However, the historical augmented and
  no-augmentation runs differ in hardware/software provenance and some head
  initialization hashes. Treat this as an observed association, not a clean
  causal estimate.
- The test failure review found recurring confusions between visually similar
  breeds (including Siamese/Birman and Russian Blue/British Shorthair). The
  Grad-CAM maps support visual inspection but do not prove why a prediction was
  made. Cases were selected for diagnosis and are not a random sample.
- Inference latency was measured on a local CPU. Across the four selected
  conditions, batch-1 latency is about 121–155 ms per image; batch-32 latency
  is about 3.01–3.26 s per batch (about 94–102 ms per image). These figures are
  hardware-specific. The exact CPU model and consolidated training wall-clock
  time are not recorded in the current report artifacts.
- Training provenance varies: C0 and C2 include CPU runs, while C1 and C3
  selected runs were performed on a Tesla T4. CPU validation reproduced the
  recorded C3 scores, but the different environments and historical
  initialization hashes limit strictly causal cross-condition claims.

## Remaining work: final two weeks

### Week 1 — finish final evaluation and runtime evidence

1. Recover the already completed C0/C3 augmentation-off run bundles (three
   seeds per condition), including each `best.pt`, `history.csv`, resolved
   config, and metadata. They are referenced under the ignored `artifacts/`
   path in the ablation report but are absent from this checkout. Re-running
   training is only a fallback if the existing bundles cannot be recovered;
   keep the selected LRs and seeds fixed and do not tune on test results.
2. Complete the separate final test evaluation of the three retained
   checkpoints for each ablation condition. Add accuracy, macro-F1, per-class
   F1, confusion matrices, predictions, and latency artifacts. Update the two
   pending rows in `reports/results_preliminary.md` and label the analysis
   clearly as post-selection test reporting.
3. Record total training wall-clock time and the CPU model for the existing
   latency benchmark, or rerun only the latency measurement if that metadata
   cannot be recovered. Keep hardware and software versions beside each
   measurement.

### Week 2 — verify and submit

1. Have each team member follow the README from a clean clone: install the
   environment, download and verify the dataset, run the documented evaluation,
   and report any undocumented steps or failures.
2. Prepare the final two-page technical summary with the research question,
   dataset and split, C0-C3 comparison, completed ablation, at least three
   analyzed failure cases, runtime/environment, repository link, and each
   member's contribution.
3. Prepare and rehearse the short demo: task and data, baseline, main
   comparison, one failure case, conclusion, and likely technical questions.
4. Freeze and submit the report, code/notebook, requirements/environment
   information, run instructions, metrics, and figures through the course's
   required submission channel.

## Risks and mitigations

| Risk | Effect | Mitigation |
| --- | --- | --- |
| No-augmentation checkpoints and histories are absent from this checkout | The completed validation ablation is harder to audit here, and its checkpoints cannot yet be evaluated on test from this checkout | Recover the completed run bundles from the training machine/team archive; rerun only if recovery fails |
| Two augmentation-off test evaluations are not yet in the current results table | The six-configuration final test comparison is incomplete, although the validation ablation itself is complete | Evaluate only the retained/fixed-LR checkpoints and update all linked artifacts together |
| Training wall-clock and exact CPU model are missing | Runtime evidence is incomplete and latency is hard to reproduce | Recover metadata or measure and record it on a named machine during Week 1 |
| Different historical environments and initialization hashes | Small score differences cannot be attributed solely to trainable depth or augmentation | State the limitation; avoid causal claims and keep test results descriptive |
| Clean-clone review and team contributions are not evidenced in the current report | Reproducibility and submission checklist may remain incomplete | Assign each member a clean-clone check and record names/contributions before submission |
| Two-week schedule has little slack if artifacts must be retrained | Final polishing or demo rehearsal could be squeezed | Recover first, rerun only if necessary, and prioritize test ablation plus the required summary |

## Current milestone status

**Completed:** four-condition training and test evaluation, the C0/C3
validation augmentation ablation, and the three-case Grad-CAM failure review.
**In progress:** recover the completed ablation run bundles for repository
audit, add their separate test evaluation to the six-configuration comparison,
and finish runtime/reproducibility evidence.
**Milestone document:** this status report is prepared for team review and
submission; the final course submission still needs the remaining items above.

## Evidence and project links

- [Initial baseline report](initial-baseline-report.md)
- [C1 second-approach report](second-approach-report.md)
- [Preliminary test results and failure examples](../reports/results_preliminary.md)
- [Validation augmentation ablation](../reports/ablation.md)
- [Detailed failure analysis](../reports/failure_analysis.md)
- [Training and evaluation artifacts](../deliverables/)
- [README and exact commands](../README.md)
