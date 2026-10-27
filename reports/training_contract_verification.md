# Training contract verification — 2026-10-12

## Scope and repository state

- Git commit: `155bf1b5325c758455de03431f08008ae310d1ff`
- Branch: `main`
- Working tree: dirty; all changes remain local and uncommitted.
- No commit, push, merge, issue, pull request, deployment, upload, or external
  compute was used.

## Environment

- Windows 11 build 26300 (Windows reports the product label as Windows 10 Pro).
- Project interpreter: `.venv/Scripts/python.exe`, Python 3.14.6.
- PyTorch 2.14.1+cpu; torchvision 0.29.1+cpu.
- NumPy 2.5.4; scikit-learn 1.9.1; PyYAML 6.0.3; Pillow 12.3.0.
- Windows hardware inventory exposes only Intel Iris Xe Graphics
  (`PCI VEN_8086&DEV_A7A1`). No NVIDIA adapter or `nvidia-smi` is present.
- Consequently `torch.cuda.is_available()` is false, CUDA version and cuDNN
  version are absent, and GPU reproducibility is BLOCKED rather than passed.

## Local prerequisites

The official Oxford-IIIT Pet archives were downloaded with the repository's
`--download-only` path. This did not regenerate or overwrite `splits/v2`.

- `data/oxford-iiit-pet/images.tar.gz`:
  `67195C5E1C01F1AB5F9B6A5D22B8C27A580D896ECE458917E61D459337FA318D`
- `data/oxford-iiit-pet/annotations.tar.gz`:
  `52425FB6DE5C424942B7626B428656FCBD798DB970A937DF61750C0F1D358E91`
- 7,393 images are extracted.

The official torchvision `ResNet50_Weights.IMAGENET1K_V2` checkpoint is cached
at `C:/Users/albiina/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`.
Its SHA-256 is
`11AD3FA62CA79E40ADDFD354A8EC4B7C75143B3038B8D2A807FBC68DEAB379CA`.

## Commands and results

```powershell
.\.venv\Scripts\python.exe scripts\prepare_oxford_pets.py `
  --dataset-root data\oxford-iiit-pet --download-only
```

PASS. Official archives downloaded and extracted; no manifests were written.

```powershell
.\.venv\Scripts\python.exe scripts\verify_data.py `
  --dataset-root data\oxford-iiit-pet --split-dir splits\v2 `
  --config configs\data.yaml --check-loaders --check-reproducibility
```

PASS. Split sizes are train=1399, val=300, test=300. Loader checks passed.
Temporary regeneration matched the frozen train, validation, test, and mapping
files byte-for-byte. The temporary output was removed automatically.

```powershell
.\.venv\Scripts\python.exe scripts\verify_head_initialization.py
```

PASS. This used the unmocked production `build_model` and cached ImageNet V2
weights. Within every seed, the C0–C3 classifier tensors were exactly equal:

- seed 42: `93b2c8d176691e0a2da9b26297eea5089c9094bfcd27ca689f3d9e39c25c3b48`
- seed 43: `39fa053698c6a5dfeeedfe34136448c161abf95a463773bd06b4c5fbdc1ba817`
- seed 44: `8e64575e8b42a31ff99ede4e5ef80ab997daa9ecb4035f57916ae00a5b460fee`

```powershell
.\.venv\Scripts\python.exe -m unittest -v tests.test_training_contract
```

PASS: 5 tests, 8.913 seconds. The suite checks all 4×3×3 effective
condition/seed/LR combinations and CLI numeric types; exact head tensor equality
for 3×4 combinations; independent expected unfreeze mappings; requires_grad and
optimizer identities for all conditions; and exact BatchNorm running_mean,
running_var, and num_batches_tracked invariance after real ResNet50 CPU training
steps for C1–C3.

The artifact test calls the real shared `train()` function, but intentionally
mocks `get_loaders` with deterministic tensors and `build_model` with a reduced
linear classifier. It executes the actual training loop and artifact writers,
then validates the effective config, distinct seeds, Git commit and dirty flag,
finite history row, checkpoint contents, and refusal to reuse an output folder.
Its temporary run directory was removed automatically.

The ordinary head unit test mocks only the external pretrained constructor with
an untrained ResNet50 to remain offline. The separate production verification
command above removes that mock and verifies the cached pretrained path.

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
git diff --check
```

PASS. No syntax or whitespace errors were reported.

## GPU reproducibility configuration and status

`configs/gpu_repro_check.yaml` resolves to C1, training seed 42, split seed 42,
LR 1e-4, one epoch, batch size 32, four workers, CUDA, and augmentation enabled.
It uses the complete existing training and validation manifests through the
shared `src/train.py`; it is not a full production experiment.

The two GPU runs were not started because this machine has no NVIDIA/CUDA GPU.
There are therefore no GPU run artifact paths and no `history.csv` hashes.
GPU byte-identical reproducibility remains BLOCKED.

On a machine exposing one compatible NVIDIA GPU, with this repository and local
data/cache copied without changing inputs, run in fresh directories:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\gpu_repro_check.yaml `
  --training-seed 42 --lr 1e-4 --output-dir artifacts\gpu_repro_c1_seed42_a
.\.venv\Scripts\python.exe -m src.train --config configs\gpu_repro_check.yaml `
  --training-seed 42 --lr 1e-4 --output-dir artifacts\gpu_repro_c1_seed42_b
cmd /c fc.exe /b artifacts\gpu_repro_c1_seed42_a\history.csv `
  artifacts\gpu_repro_c1_seed42_b\history.csv
Get-FileHash -Algorithm SHA256 artifacts\gpu_repro_c1_seed42_a\history.csv
Get-FileHash -Algorithm SHA256 artifacts\gpu_repro_c1_seed42_b\history.csv
```

Confirm in both `metadata.json` files that `environment.device` is `cuda`, the
GPU and deterministic settings match, and confirm in both `runtime.json` files
that `status` is `completed` and `bn_checks_passed` is 1.

## Final status

All CPU-verifiable training-contract requirements pass. GPU reproducibility is
BLOCKED by absent NVIDIA hardware, so the overall Definition of Done is not
complete. The documented freeze of `src/train.py` remains PENDING MERGE.
