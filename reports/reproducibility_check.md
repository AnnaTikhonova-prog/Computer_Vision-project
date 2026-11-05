# Reproducibility check — 5 November 2026

**Scope:** a fresh clone, isolated pinned CPU environment, independent checks of
published results, and repeat inference with the included core checkpoints.
This is a separate diagnostic audit. No settings are changed based on test
results, and no canonical prediction or timing CSV is overwritten.

## Checkout and environment

A new clone was made directly from
`https://github.com/AnnaTikhonova-prog/Computer_Vision-project.git`, starting at
`d889a68`, in an ignored workspace directory. The cleanup/audit additions were
applied to that clone for testing. Data and checkpoint files were downloaded
independently; neither was copied from the original worktree. The audit JSON
records source hashes so the candidate being tested is identifiable alongside
the base commit.

The test environment is CPython **3.13.15**, **torch 2.11.0+cpu**,
**torchvision 0.26.0+cpu**, with `requirements.txt` installed in a new venv and
pip **25.1.1**. Python was provisioned in the workspace with uv 0.13.7 because
the host only had Python 3.11; this is test-harness provisioning, not an
additional project dependency. `pip check` passed. The audit JSON records all
installed package versions, including transitive dependencies.

## Commands and evidence

The README's CPU install sequence was followed. The following checks run from
the clone root and write only temporary/local artifacts:

```powershell
python scripts/prepare_oxford_pets.py --download-only
python scripts/verify_data.py --dataset-root data/oxford-iiit-pet --split-dir splits/v2 --check-loaders --check-reproducibility
python -m unittest discover -s tests -v
python scripts/check_reproducibility.py
python scripts/check_reproducibility.py --inference core
```

The official archives were downloaded successfully. The full offline suite
passed **29 tests** in **132.645 seconds**. This includes the existing training
contract, aggregation, delivery and ablation tests, plus result-audit tests
that reject missing/duplicate predictions, inconsistent labels/correctness,
and non-finite confidence. The trainer's test fixture uses synthetic data;
it is not a rerun of a scientific experiment.

Fresh-clone data verification passed: **1,399 train / 300 validation / 300
test** images, RGB/content-hash integrity, split disjointness, class presence,
confirmed duplicate-group constraints, DataLoader checks, and bytewise
regeneration of the frozen manifests in a temporary directory.

All **12 core checkpoints** (C0–C3, three seeds each) passed hash/config checks
and real inference: **3,600 of 3,600 predicted labels matched**, and the
per-seed accuracy/macro-F1 values matched to `1e-12`. This is prediction-level
reproduction, not a claim of bitwise-identical logits across environments.

The updated explicit tar `data` filter was exercised by re-extracting the
downloaded official archives. Headless figure regeneration and the full
six-configuration error exporter also completed in the clone. The exporter
and checks left the committed prediction tables and error index unchanged.

Evidence: [check summary and source fingerprints](reproducibility/2026-11-05/checks.json),
[inference and environment JSON](reproducibility/2026-11-05/core_inference.json),
[data verification](reproducibility/2026-11-05/data_verification.json),
[test log](reproducibility/2026-11-05/unit_tests.txt),
[official archive checksums](reproducibility/2026-11-05/dataset_archives.json), and
[installed dependency-license metadata](reproducibility/2026-11-05/dependency_licenses.json).

## Published numbers

All **5,400 saved predictions** (six configurations × three seeds × 300 test
images) were checked against the frozen test manifest and class mapping.
Accuracy and fixed-ten-class macro-F1 were independently computed from
confusion counts, then aggregated with sample standard deviation (`ddof=1`).
These match all six summary CSVs, per-class F1 tables, and the rounded final
comparison in [results.md](results.md), with numeric tolerance `1e-12` before
display rounding.

The recorded 2 November latency rows were aggregated separately. One display
error was found: C1's batch-32 per-image SD is `359.5163754496131 / 32`, which
rounds to **11.23 ms**, not 11.24. The correction is documented in the final
report; recorded timings and all quality metrics are unchanged. No new latency
measurement is claimed to reproduce the earlier hardware measurement.

## Limits and outstanding artifacts

The six original C0/C3 augmentation-off checkpoints are absent from the
repository. Their expected hashes, seeds and selected epochs are recorded in
[`ablation_noaug_checkpoint_provenance.csv`](tables/ablation_noaug_checkpoint_provenance.csv).
The user confirmed that the archive will be supplied later. Full
six-configuration inference remains **BLOCKED** until then; checking arithmetic
over saved predictions does not remove that requirement.

Once supplied, restore the exact original files to
`artifacts/ablation_noaug/runs/{c0_lr1e-3,c3_lr1e-5}_seed{42,43,44}/best.pt`
and run:

```powershell
python scripts/check_reproducibility.py --inference all --output artifacts/reproducibility/all.json
```

The script checks checkpoint hashes against the published provenance before
forward passes. Missing requested checkpoints yield a partial audit and exit
code 2. A wrong hash, inconsistent data, or a changed prediction fails the
check instead of replacing the final values.

This audit does not reproduce training histories or GPU determinism, certify
each team member's independent clone, or assert equal latency across machines.
The [submission checklist](submission_checklist.md) lists the repository items
and the remaining archive dependency explicitly.
