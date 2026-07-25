# Confirmatory analysis provenance

## Frozen source

- Experiment: `qwen7b_cat_cross_seed_confirmatory_v1`
- Baseline commit: `1635e5aa2b0fd86e9492e75a0cb11f5d9f9f7964`
- Annotated tag: `thesis-confirmatory-baseline`
- Tag object: `d9525e0de724a25195b4a2647cb17f7cb10021a8`
- Finalizer cluster: `177561`
- Finalizer status: `complete`

## Archive

- File:
  `qwen7b_cat_cross_seed_confirmatory_v1-thesis-confirmatory-baseline.tar.zst`
- Size: 4,171,284,770 bytes
- SHA-256:
  `b9e8733905f598c8f7638678c75e8ffcd59d8c1cfec5dc9a260626985dbee8dd`
- Zstandard integrity: verified
- Independent backup copy: confirmed by the author

## Local analysis extraction

The analysis workspace selectively extracts only:

- `aggregate.json`;
- `final_artifacts.sha256`;
- `plots/`;
- `reports/`.

All 12 entries in `final_artifacts.sha256` were rehashed after local extraction.
Result: 12 valid, 0 mismatches.

The 3.9 GiB archive and the extracted working directory are excluded from Git.
They are inputs to the interpretation, not repository source artifacts.

## Analysis branch

- Branch: `thesis/confirmatory-analysis`
- Initial interpretation commit:
  `d1c83afad79dbc62e6dbb60e6aaa40ce0a0f6dcd`

All scientific interpretation after the freeze occurs on this branch. The
baseline tag is not modified.

## Numerical source contract

Final thesis values must be taken from:

| Result family | Frozen source |
|---|---|
| Behavioral gate and alignment | `reports/seed_summary.csv` |
| Aggregate intervention effects | `reports/aggregate_effects.csv` |
| Ranking stability and overlap | `reports/ranking_similarity.csv` |
| Per-seed prompt intervals | `aggregate.json` |
| Scope and hypothesis labels | `reports/hypotheses.csv` |

Scheduler logs, exploratory outputs, and preliminary runs are provenance or
motivation sources only. They must not replace a finalized confirmatory value.

## Separation of checks and scientific interpretation

The container runtime auditor checked only artifact presence, readability,
hashes, provenance, group ownership, setgid inheritance, PNG signatures, and
completion markers. It did not compute scientific outcomes.

The repository test suite ran separately outside the container and passed
69/69 tests at the baseline.
