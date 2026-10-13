# C0 production training report

Nine validation-only runs: three learning rates × three training seeds.
The test loader was not iterated and test metrics were not computed.

## Per-run selected checkpoints

| LR | Seed | Best epoch | Validation macro-F1 | Epochs run |
|---:|---:|---:|---:|---:|
| 1e-03 | 42 | 9 | 0.96666666666666679 | 14 |
| 1e-03 | 43 | 9 | 0.97005001389274814 | 14 |
| 1e-03 | 44 | 11 | 0.97348214285714296 | 16 |
| 1e-04 | 42 | 12 | 0.94992033588203706 | 17 |
| 1e-04 | 43 | 11 | 0.94362662845723122 | 16 |
| 1e-04 | 44 | 11 | 0.95337686394368804 | 16 |
| 1e-05 | 42 | 30 | 0.8812655700008728 | 30 |
| 1e-05 | 43 | 30 | 0.81372078015888205 | 30 |
| 1e-05 | 44 | 30 | 0.86958300901835861 | 30 |

## Learning-rate selection

| LR | Seed 42 | Seed 43 | Seed 44 | Mean | Sample std | Selected |
|---:|---:|---:|---:|---:|---:|:---:|
| 1e-03 | 0.96666666666666679 | 0.97005001389274814 | 0.97348214285714296 | 0.97006627447218596 | 0.0034077671913688083 | yes |
| 1e-04 | 0.94992033588203706 | 0.94362662845723122 | 0.95337686394368804 | 0.9489746094276521 | 0.0049434372561238298 | no |
| 1e-05 | 0.8812655700008728 | 0.81372078015888205 | 0.86958300901835861 | 0.85485645305937119 | 0.036100252720854542 | no |

Selected LR: `1e-03`.
Validation macro-F1: `0.97006627447218596 ± 0.0034077671913688083` (sample std, ddof=1).
Exact mean ties select the smaller LR. No seed is ranked or selected.

## Production protocol

ImageNet V2 ResNet50; C0 head-only training; frozen BatchNorm statistics; AdamW; batch 32; microbatch 4; augmentation enabled; maximum 30 epochs; early stopping patience 5; earliest epoch retained on an exact score tie.

## Limitations and delivery

This is CPU training in the recorded local environment. It does not resolve the separate Task 1 GPU reproducibility blocker. Commit and delivery to the Team Lead remain pending user actions.
