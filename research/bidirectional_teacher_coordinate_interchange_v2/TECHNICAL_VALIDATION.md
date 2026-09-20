# C18 v2 Freeze, Implementation, and Technical Validation Report

Date: 2026-09-21  
Study: `qwen7b_cat_bidirectional_teacher_coordinate_interchange_v2`

## 1. Freeze status

The approved private contract has been transferred into a new public v2 study.
C18 v1 and every completed parent study remain unchanged. The scientific
question, estimands, interventions, 72 prompts, 26 selection sets, SESOI,
bootstrap, cancellation rules, Z/W definitions, orthogonal controls, ordered
decision matrix, and numerical execution semantics are frozen. The public
manifest remains `scientific_execution_authorized: false`.

The preregistration freeze is commit `37ca4e2`. The exact implementation tested
on the cluster is commit
`0e1bd7eca2d6003327dd95559b5107e5930b465b`, bound to manifest SHA-256
`6c8353788690a683b0447f72efb0980ecf87c6c3d07df4ff4e6db7842d121ed0`.

## 2. Public artifacts

| Artifact | SHA-256 |
|---|---|
| `PREREGISTRATION.md` | `b09387fce2805a785d1ed9f98f296b473e4fcf593f54cb4867f44ab816978c34` |
| `DECISION_MATRIX.md` | `5656570dedc5d5118ac897ee23f81851040a36b053f6c6226092947e1d6b554f` |
| `CALIBRATION_SUMMARY.md` | `9e026aa08613d28f9bfd6f49b58a85748d5e6ed3e610019208246d034e0889e4` |
| `TOKEN_INVENTORY.json` | `1be7583d1995d025a3534223b3297a2bad02edc8332c4deee0a2311cfe4646f2` |
| `BATCH_PLAN.json` | `2be28c7316a350bf39db86d4fea4e24b1f34a26930d3d58f6885163e0a534dfe` |
| `ORTHOGONAL_DIRECTIONS.npz` | `0461b1443b2de25915df38662224a67d6bd6722d88600f180c1bdda7a7d8724c` |
| v2 manifest | `6c8353788690a683b0447f72efb0980ecf87c6c3d07df4ff4e6db7842d121ed0` |

The manifest additionally freezes the 72 prompt bytes, model revision and
snapshot inventory, both adapter trees, selection plan, teacher tensor, FSD
references, dependency lock, runner, aggregator, independent scientific
auditor, technical validator/auditor, execution modules, task wrapper,
submitter, DAG renderer/template, ClassAds, container digest, and the complete
canonical B=6 batch plan.

## 3. C18 v1 to v2

C18 v2 replaces the failed v1 numerical path with the successfully calibrated
canonical B=6 semantics. It uses explicit `P0/P1` parameter and `D0/D1` donor
notation; a joint B0/B1 equivalence IUT; full 25 x 72 cancellation protection;
the natural-state Z factorial and complete hybrid-state W factorial; exactly
one five-dose secondary specificity family; and an ordered, disjoint A--G
decision matrix. It limits inference to the fixed Seed-2 Top-minus-25-Norm,
S-minus-N contrast and explicitly states that functional interchange is not
natural mediation.

## 4. Implementation

The runner implements immutable natural donors, decoder-block output slots
1--27, all real prefill tokens, float64 projection/correction with one cast
back to fp16, no hybrid feedback donor, complete Y/Z/W cells, five frozen dose
families, and uniquely identified raw cells. Full-state cut replay, exact
196-module census, exactly 20 disabled modules per set, exception-safe restore,
finiteness, hook count/order, donor immutability, and atomic authenticated raw
publication are fail-closed.

The production aggregator accepts only complete authenticated and explicitly
released raw inventories. A separate auditor reimplements the estimands,
four-cell algebra, 25 x 72 cancellation calculations, Z/W allocations,
PCG64 bootstrap, specificity family, and decision classification without
importing the production statistics module.

## 5. Local validation

- Full repository suite: `149 passed`.
- All new Python modules and entry points: compile PASS.
- Public manifest and all locally available frozen hashes: PASS.
- Shell syntax and DAG rendering: PASS.
- Three negative lock tests (direct runner, manifest executor, scientific DAG
  generator): all refused execution and produced no output.
- No golden scientific outcomes, expected signs, or expected effect sizes were
  used in tests.

## 6. Cluster technical validation

HTCondor DAGMan cluster `195107` completed 4/4 nodes with zero failures:
outcome-blind preflight, one real A100 synthetic technical validator,
independent CPU audit, and final notification. Native `condor_submit` ClassAd
dry-runs and `condor_submit_dag -no_submit` validation passed before submission.

Technical artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `preflight.json` | `a7b56602552c9c154515f0ef03404e1856f73d9394c45ba200105a9b843239e2` |
| `validation.json` | `c67cc496376caf16c7255af52b778ce322fceb56e5602dd4cfd864ab75bebd7c` |
| validation provenance | `f7fec9781f68efe1a115275a203f996bef13a10a1558980a648c8560c4ceb88c` |
| `SHA256SUMS` | `fa0dfb8846da5651496c95ce641ae5efa50257625507cd76287a1670cad523fb` |
| `audit.json` | `9bb8d342d0cfb591cc2e07c2a5b3553a9ba990174162a2b88a52ff9c835ab453` |

R, H, and L_same passed by raw-byte identity in same-shape B=6 execution. The
identity hook, null operator, P0/P1 self-replay, module census/restore,
finiteness, hook order, donor immutability, terminal state/readout replay,
atomicity, and DAG/ClassAds gates all passed. Permutation, co-batch, added
padding, B=1 versus B=6, cross-shape-fp32, and q99.5 x 2 envelopes remained
non-gating transport diagnostics and were not converted into scientific
thresholds.

## 7. Independent audit

The technical auditor independently checked the exact manifest and execution
commit bindings, complete input PASS inventory, checksums and sidecar,
container/runtime identity, both adapters' 196/20 module contract, R/H/L_same,
self-replay, scheduler-derived ClassAds, atomicity, and the complete blindness
record. Status: `PASS`; verdict:
`READY_FOR_C18_V2_SCIENTIFIC_EXECUTION_AUTHORIZATION`.

## 8. Outcome blindness

The technical run loaded six synthetic prompts and synthetic teacher/control
directions only. Recorded counters are: scientific prompts loaded `false`,
scientific selection plan loaded `false`, teacher tensors loaded `0`, teacher
interchanges `0`, cross cells computed `0`, estimands computed `0`, raw logits
persisted `false`, forbidden open attempts `0`. No scientific raw file and no
scientific authorization record exists.

## 9. Execution identity

- NVIDIA A100-PCIE-40GB; driver `570.211.01`;
- container `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` at digest
  `sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`;
- Python 3.11.10; Torch 2.5.1+cu124; CUDA 12.4; Transformers 4.48.3;
  PEFT 0.14.0; bitsandbytes 0.45.2; NumPy 2.1.2;
- SDPA, fp16 residual stream, NF4/fp16 compute, double quantization off;
- left padding, mask-derived logical `position_ids`, rank-1 physical
  `cache_position`, and canonical B=6 LM-head evaluation.

## 10. Git provenance

Branch: `research/bidirectional-teacher-coordinate-interchange-v2`.

- `37ca4e2` -- public preregistration and calibration contract;
- `0e1bd7e` -- frozen implementation, manifest, tests, and technical DAG.

No merge, tag, release, or scientific submission was made. Private
`.local-research` material was not committed.

## 11. Remaining risks

Passing this stop-gate establishes implementation and numerical integrity, not
the C18 scientific hypothesis. The hybrid intervention can still expose
distribution shift; a negative interchange remains compatible with
multidimensional or instance-dependent teacher information; Z/W only identify
the controlled Block-26 cut decomposition; and the final claim remains limited
to the fixed Seed-2 organism, prompts, sets, and model. Scientific execution is
also materially larger than the preflight and retains the preregistered
3--8 A100-GPUh expectation (12 GPUh reserve), 16--64 CPU-core-hours, and
0.5--4 GB artifact estimate.

## 12. Decision and stop

All preregistration-freeze, implementation, integrity, independent-audit, and
outcome-blind technical requirements passed. Scientific execution remains
locked pending a separate researcher authorization.

`READY FOR C18 V2 SCIENTIFIC EXECUTION AUTHORIZATION`
