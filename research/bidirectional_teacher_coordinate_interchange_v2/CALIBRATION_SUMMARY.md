# C18 v2: public numerical calibration summary

Status: completed synthetic, outcome-blind calibration; independent audit PASS.
The calibration contains no scientific C18 prompts, teacher tensors,
interchanges, cross cells, estimands, or raw scientific logits.

## Frozen identities

| Object | SHA-256 |
|---|---|
| Contract | `a6fbce42d08672f8e02e2276e7ff935d10ecdffc6c6a127110c2c24ea0aa97e5` |
| Population | `624b4a0c4e013f0a40ee99e6b3a62a567b635704ddc1f5e6d5257fe3fc3d4e86` |
| Seal | `d5fdae2ea87f40bd1155d74de6d48bf5b49ee7e2414504c26e05bc64c024d8ff` |
| Independent audit | `b6e8808dbff887c7ff403ab7eb2d1211d092e71b098274f3ce927c82926d345f` |

The population contained 216 synthetic prompts in 36 canonical B=6 batches,
balanced over six synthetic length strata and physical positions. The audit
verified 60 category-by-adapter-by-mask cells, 12,960 error rows, and a 7,776
row transport pool.

## Execution identity

- NVIDIA A100-PCIE-40GB; driver 570.211.01;
- container
  `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`;
- calibration code commit `b3b2a0e1414142b399470204067f089a77a93c2c`;
- Python 3.11.10; Torch 2.5.1+cu124; CUDA 12.4;
- Transformers 4.48.3; PEFT 0.14.0; bitsandbytes 0.45.2; NumPy 2.1.2;
- SDPA; fp16 residual stream; NF4/fp16 compute; double quantization off;
- TF32 off; deterministic algorithms/cudNN; benchmark off;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Hard gates

All hard categories were exactly zero over all 1,296 observations per
category:

| Gate | Meaning | Result |
|---|---|---|
| R | identical canonical B=6 repetition | raw-byte-identical PASS |
| H | identity hooks in identical B=6 form | raw-byte-identical PASS |
| L_same | repeated LM-head evaluation in identical B=6 form | raw-byte-identical PASS |

The prospective relative-margin and relative-state alerts were not triggered.

## Outcome-blindness

The independent audit recorded:

```text
scientific_prompts_loaded       = false
scientific_selection_plan_loaded = false
teacher_tensors_loaded          = 0
teacher_interchanges            = 0
cross_cells_computed            = 0
estimands_computed              = 0
raw_logits_persisted            = false
forbidden_open_attempts         = 0
```

## Invariants versus transport diagnostics

Hard execution invariants for C18 v2 are exact execution identity and hashes;
canonical B=6 membership/order/width/masks; logical mask-based `position_ids`;
rank-1 physical `cache_position`; canonical B=6 LM-head evaluation; R/H/L_same;
module census and restoration; finiteness; hook order/count; cleanup; and
atomicity.

Permutation, changed co-batch, additional padding, identically padded
B=1-versus-B=6, cross-shape fp32 readout, and q99.5-times-two envelopes are
transport diagnostics. They are not bit-identity requirements or scientific
identification gates.

The prospective diagnostic reference envelopes were:

| Metric | q99.5 x 2 envelope |
|---|---:|
| native margin absolute | 0.05859375 |
| native margin relative | 0.03662109375 |
| fp32 margin absolute | 0.05832386016845703 |
| fp32 margin relative | 0.03653144836425781 |
| state relative L2 | 0.008053255414097626 |
| state maximum absolute | 1.0 |

These envelopes characterize numerical transport only. They must never be
used as a scientific SESOI, an equivalence gate for `G_B0/G_B1`, a Z/W
materiality threshold, or a post-hoc scientific acceptance tolerance.

Calibration verdict: `READY FOR C18 V2 PREREG DESIGN`. This did not authorize
scientific execution.
