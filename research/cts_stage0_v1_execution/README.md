# CTS Stage 0 — execution implementation (v1)

This directory documents how the frozen contract in [`../cts_stage0_v1/`](../cts_stage0_v1/PREREGISTRATION.md)
(tag `prereg/cts-stage0-v1`, commit `43dd95d`) is executed. It is implementation, not preregistration: nothing
here changes a frozen decision, threshold, prompt set, persona, direction, dose, site or endpoint. The frozen
directory itself is never modified; the submit script refuses any execution commit whose diff from the tag
touches it.

Status: implemented; outcome-blind technical validation pending or recorded separately. No Stage-0 outcome
exists.

## Entry points

| Path | Role |
|---|---|
| `src/slgeo/cts_stage0/` | Implementation package (no PEFT import; base model only) |
| `scripts/cts_stage0.py` | CLI: `submit-record` (submit host), `plan`, `run --shard`, `techval`, `techval-cpu` (cluster) |
| `scripts/generate_cts_stage0_dag.py` | HTCondor DAG generator (technical validation or scientific plan) |
| `condor/cts_stage0_task_{gpu,cpu}.sub`, `condor/run_cts_stage0_task.sh`, `condor/submit_cts_stage0.sh` | Node submit files, wrapper and submission |
| `configs/validation/cts_stage0_v1.yaml` | Execution manifest: pinned identity, snapshot hashes, scoring layout, budget |
| `IMPLEMENTATION_CHOICES.json` | Every implementation-level and descriptive choice the frozen text leaves open |
| `tests/test_cts_stage0_*.py` | Unit, negative and mutation tests |

## Pipeline

`preflight` (CPU) → `extract_*` (GPU, 44 personas × 1,024 rows, batch 1) → `directions` (CPU, write-once
direction bundle with τ, R_cov, R_iso, structured nulls) → `baseline_rep1/rep2` (GPU, two host groups) →
`score_*` (GPU) and `sample_*` (GPU) → `integrity` (CPU) → `analysis` (CPU, frozen decision engine).

Each shard is published atomically (unique incoming name, fsync, re-hash, rename) with a marker listing every
file hash, the plan hash, the manifest and choices hashes and the execution commit. A shard is reused only if
its marker verifies completely; otherwise it is quarantined and recomputed. Every attempt leaves an immutable
record.

## Execution identity (pinned, fail-closed)

`NVIDIA A100-PCIE-40GB`, driver `570.211.01`, image
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92…2755`, Python 3.11.10, torch 2.5.1+cu124,
transformers 4.48.3, tokenizers 0.21.4, bitsandbytes 0.45.2, accelerate 1.3.0, numpy 2.1.2. The submit files
require the GPU class and driver; every job re-checks them at runtime and exits 86 (never retried) on any
difference. Model weights, tokenizer and configuration files of snapshot `a09a3545…` are re-hashed in every job.
The code checkout is verified in every job by recomputing the git blob hash of every tracked file.

## Implementation clarifications (dated 2026-09-25; no gating quantity changes)

1. **Scoring layout.** Every answer form is scored by exact teacher forcing from the KV cache of the steered
   prefill: one unsteered continuation forward packs all 104 forms as separate segments (position ids
   `L + depth`; a 4D mask lets each segment see the prompt cache and its own earlier tokens only). This is a
   batch-1, unpadded forward. A per-form sequential reference implementation is used by the equivalence tests.
2. **Condition batching (§2, §13.3).** Batching several steering conditions of one prompt in the prefill
   (identical tokens, per-row vectors, no padding) is used only if the recorded technical-validation test shows
   agreement with batch 1 within 1e-3 for every per-form log-probability and word score, on the 40 V prompts,
   over every condition class of the production path (zero rows, last-position and all-position rows, every
   site). Because the validation is outcome-blind, the classes are realized with random directions on a
   magnitude grid and with nonce and stand-in answer forms of the production shape; no real word is scored.
3. **Last-three-token assertion** applies to every prefill; continuation forwards assert the cache length.
4. **Hook unit tests** run in the execution environment (tiny random Qwen2) in the preflight node; a real-model
   hook self-test runs in every GPU node before its first scientific forward.
5. **"All S0 prompts present for every condition"**: every scoring condition is scored on all 334 S0 prompts;
   the 300 animal-family prompts enter the gating statistics.
6. **Repeated baselines**: the canonical baseline (rep1) and a repeat (rep2) on a different host group, plus
   start/end sentinels in every scoring shard, must agree within 1e-4.
7. **Normalizer**: log-probabilities are normalized over all 152,064 rows of the language-model head, in float32.
8. **Budget accounting**: every CTS GPU job, including technical validation and failed attempts, is metered
   against the 40 A100-h cap; the plan must fit 0.8 of the remaining cap.

## Known limitations of frozen text (not changed)

The frozen `PREREGISTRATION.md` header still describes the package as "staged, not yet committed or public",
and it cites audit files by relative paths that are not part of this repository. Both are frozen and are left
unchanged.
