# C1 production training report

Nine completed Kaggle runs: three learning rates x three training seeds.
Metrics below are recorded training-time validation scores, not a separate evaluator run.
A subsequent independent CPU validation pass reproduced all nine recorded macro-F1 scores exactly. No test evaluation was performed.

## Per-run selected checkpoints

| LR | Seed | Best epoch | Validation macro-F1 | Epochs run |
|---:|---:|---:|---:|---:|
| 1e-3 | 42 | 4 | 0.96671288001456335 | 9 |
| 1e-3 | 43 | 3 | 0.96679883274802736 | 8 |
| 1e-3 | 44 | 5 | 0.96661497493734339 | 10 |
| 1e-4 | 42 | 5 | 0.98338705195887743 | 10 |
| 1e-4 | 43 | 4 | 0.98331757585610391 | 9 |
| 1e-4 | 44 | 7 | 0.98999814763360194 | 12 |
| 1e-5 | 42 | 9 | 0.97338705195887754 | 14 |
| 1e-5 | 43 | 11 | 0.97664905682303915 | 16 |
| 1e-5 | 44 | 10 | 0.97999444290080573 | 15 |

## Learning-rate selection

| LR | Seed 42 | Seed 43 | Seed 44 | Mean | Sample std | Selected |
|---:|---:|---:|---:|---:|---:|:---:|
| 1e-3 | 0.96671288001456335 | 0.96679883274802736 | 0.96661497493734339 | 0.96670889589997799 | 9.1993633008745205e-05 | no |
| 1e-4 | 0.98338705195887743 | 0.98331757585610391 | 0.98999814763360194 | 0.98556759181619447 | 0.0038371311380492102 | yes |
| 1e-5 | 0.97338705195887754 | 0.97664905682303915 | 0.97999444290080573 | 0.97667685056090747 | 0.0033037831547738121 | no |

Selected LR: `1e-4`.
Validation macro-F1: `0.98556759181619447 +/- 0.0038371311380492102` (sample std, ddof=1).
Exact mean ties select the smaller LR. All three selected-LR checkpoints are retained.

## Production protocol

ImageNet V2 ResNet50; C1 layer4 + head training; frozen BatchNorm running statistics;
BatchNorm affine parameters train only in unfrozen blocks; AdamW; batch 32; microbatch 4;
augmentation enabled; maximum 30 epochs; early stopping patience 5;
earliest epoch retained on an exact validation score tie.

## Provenance and limitations

Runs used Tesla T4, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and Python 3.13.15.
Source commit: `37f1e666390bde90743cc5ec765f286359c04d90` (clean working tree).
Source and split hashes match the repository Git blobs. Checkpoint metadata agrees with history and runtime.
Original artifacts, including runtime.json completion evidence, are preserved byte-for-byte.
The evaluation/ directory contains separate CPU validation predictions, metrics, and LR aggregates in the C0 delivery format. The selected LR remains 1e-4.
Paths in runs.csv are repository-relative. Archive/member provenance and hashes are recorded in manifest.json.
