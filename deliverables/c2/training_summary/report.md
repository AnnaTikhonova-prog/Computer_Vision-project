# C2 production training report

Nine validation-only runs: three learning rates × three training seeds.
The test loader was not iterated and test metrics were not computed.

## Per-run selected checkpoints

| LR | Seed | Best epoch | Validation macro-F1 | Epochs run |
|---:|---:|---:|---:|---:|
| 1e-03 | 42 | 2 | 0.0182370820668693 | 7 |
| 1e-03 | 43 | 2 | 0.018292682926829267 | 7 |
| 1e-03 | 44 | 1 | 0.0182370820668693 | 6 |
| 1e-04 | 42 | 7 | 0.9866629619338706 | 12 |
| 1e-04 | 43 | 11 | 0.98333240715013426 | 16 |
| 1e-04 | 44 | 2 | 0.96672236111762011 | 7 |
| 1e-05 | 42 | 14 | 0.98009212557219194 | 19 |
| 1e-05 | 43 | 10 | 0.98009212557219194 | 15 |
| 1e-05 | 44 | 14 | 0.98005094007594695 | 19 |

## Learning-rate selection

| LR | Seed 42 | Seed 43 | Seed 44 | Mean | Sample std | Selected |
|---:|---:|---:|---:|---:|---:|:---:|
| 1e-03 | 0.0182370820668693 | 0.018292682926829267 | 0.0182370820668693 | 0.018255615686855953 | 3.2101171465061757e-05 | no |
| 1e-04 | 0.9866629619338706 | 0.98333240715013426 | 0.96672236111762011 | 0.9789059100672084 | 0.010681867699383205 | no |
| 1e-05 | 0.98009212557219194 | 0.98009212557219194 | 0.98005094007594695 | 0.98007839707344357 | 2.3778457343752538e-05 | yes |

Selected LR: `1e-05`.
Validation macro-F1: `0.98007839707344357 ± 2.3778457343752538e-05` (sample std, ddof=1).
Exact mean ties select the smaller LR. No seed is ranked or selected.

## Production protocol

ImageNet V2 ResNet50; C2 configured unfreezing; frozen BatchNorm statistics; AdamW; batch 32; microbatch 4; augmentation enabled; maximum 30 epochs; early stopping patience 5; earliest epoch retained on an exact score tie.

## Limitations and delivery

This is CPU training in the recorded local environment. It does not resolve the separate Task 1 GPU reproducibility blocker. Commit and delivery to the Team Lead remain pending user actions.
