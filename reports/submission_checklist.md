# Repository submission checklist — 5–6 November 2026

This checklist covers issues #20 and #21 and the repository workflow in the
README. It is not a claim that presentation, course-upload, or individual team
review requirements outside those issues have been completed.

| Item | Status | Evidence / action |
| --- | --- | --- |
| Start from current upstream | PASS | Initial audit of `d889a68` plus cleanup on 5 November; full audit from clean GitHub clone `c07bd64` on 6 November. |
| Use a separate environment | PASS | Python 3.13.15 venv, CPU torch 2.11.0 and torchvision 0.26.0, pinned direct dependencies. |
| Dependency consistency | PASS | `python -m pip check`: no broken requirements; full installed versions recorded in the audit JSON. |
| Download instructions use official data | PASS | Both official Oxford archives downloaded into the fresh clone; no original-worktree dataset reused. |
| Frozen data integrity, disjointness and loader contract | PASS | Documented `verify_data.py` command passed on independently downloaded images: 1,399/300/300. |
| Split regeneration matches frozen manifests | PASS | `--check-reproducibility` passed using a temporary output directory; v2 was not replaced. |
| All six published quality summaries match predictions | PASS | Independent accuracy/macro-F1 and sample-SD computation; per-class F1 and report table checked. |
| Dated latency aggregation matches the final report | PASS | Checked from recorded checkpoint means, without a new timing benchmark. C1 per-image SD display corrected from 11.24 to 11.23 with an explicit erratum. |
| Core checkpoint inference matches published labels | PASS | Twelve independently downloaded C0–C3 LFS checkpoints; all 3,600 labels and per-seed quality scores match. See dated inference JSON. |
| Original augmentation-off checkpoint inference | PASS | Supplied archive: all six published hashes and checkpoint metadata match; all 1,800 augmentation-off labels and quality scores match in the full audit. |
| Offline test suite | PASS | 29 tests passed in the fresh pinned environment, including duplicate/missing-prediction and corrupt-label/confidence detection. |
| Remove unused code without deleting provenance | PASS | Removed unused `time` import from the ablation orchestrator; F401/F841 audit passes. Trainer/evaluator and experiment conditions unchanged. |
| Historical scripts and reports have a purpose | PASS | Data review, grid orchestration, delivery, benchmarking, plotting and their tests are retained; dated reports preserve historical evidence. No executable or manifest was deleted merely because its run is finished. |
| Data, caches and virtual environments stay out of Git | PASS | `data/`, `artifacts/`, `.venv/`, caches and IDE state are ignored; checkpoint binaries use Git LFS. |
| Code licensing policy is explicit | PASS | [LICENSE](../LICENSE): all rights reserved, as requested; no open-source grant invented. |
| Dataset and dependency attribution | PASS | [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md): Oxford source/citation, CC BY-SA terms, image modifications and dependency/weight notices. |
| Frozen empirical results remain intact | PASS | No quality CSV, dated timing CSV, trained weight or experimental config is changed. One derived rounding display has a documented correction. |
| Reproducibility limits are visible in README | PASS | Full six-configuration audit passed. Future clones need the original team archive; training-history and hardware-latency replication are outside this check. |

**Overall: PASS** for all repository checks within #20/#21. The original
archive dependency was resolved and all six configurations passed inference
on 6 November. Checkpoint binaries remain external/local artifacts; this is
not a claim of a self-contained clone without the documented archive input.
See [reproducibility_check.md](reproducibility_check.md) for commands, evidence,
and the exact limits of this check.
