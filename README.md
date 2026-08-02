# Beyond Steering Vectors

**A parameter-level, causal analysis of subliminal learning in QLoRA-fine-tuned language models.**

Subliminal learning describes a surprising form of model-to-model transfer:
a teacher's behavioral trait can be inherited by a student trained on outputs
that contain no explicit semantic evidence of that trait. Recent work explains
the phenomenon as steering-vector distillation. This repository asks the next
mechanistic question: **which trained parameter components implement that
direction, and do they causally mediate the transferred behavior?**

## Abstract

We study subliminal trait transfer in Qwen2.5-7B-Instruct using matched
subliminal and neutral QLoRA adapters across three independent training seeds.
The analysis connects three levels that are often investigated separately:
trained LoRA parameters, teacher-aligned hidden-state activations, and observable
target behavior. We first extract the teacher direction and measure whether it
is installed more strongly in subliminal students. We then rank LoRA components
by paired causal ablation and test selected subsets through reversible necessity
and sufficiency interventions.

Teacher alignment is higher in the subliminal adapter for all three seed pairs.
The top 20 modules recover 55–61% of the activation effect in isolation, but
only 15–31% of the behavioral effect. Module rankings are positively yet only
moderately stable across seeds (Spearman rho 0.359–0.498; top-20 overlap 13–14).
At k=20, alignment-selected modules outperform norm-matched controls across all
seeds, intervention modes, and measured readouts. The evidence therefore
supports a **partially concentrated, distributed, redundant, and seed-dependent
parameter implementation**, while showing that teacher alignment is an
informative but incomplete mediator of behavior.

## Main contribution

The project extends the steering-vector account of subliminal learning from an
activation-level explanation toward a causal parameter-level account:

```text
teacher bias → semantically unrelated data → student LoRA parameters
                                              ↓
                                  teacher-aligned activation
                                              ↓
                                       target behavior
```

Its distinguishing features are:

- paired subliminal-versus-neutral comparisons rather than update magnitude alone;
- cross-seed replication rather than a single-adapter case study;
- causal module ablation and isolation rather than correlational attribution;
- explicit separation of activation mediation from behavioral mediation;
- frozen manifests, checksums, prompt partitions, provenance, and decision gates.

## Key findings

| Question | Evidence from the frozen thesis baseline |
|---|---|
| Does the behavioral trait transfer? | Yes. All three within-seed behavioral gates are positive, with prompt-level 95% intervals excluding zero. |
| Is the teacher direction installed in the student? | Subliminal-minus-neutral alignment is positive in every seed: 0.221, 0.268, and 0.278. |
| Is the implementation sparse? | Only partially. Top-20 modules retain 55–61% of activation in isolation, alongside substantial distribution and redundancy. |
| Is the same circuit learned each time? | No fixed circuit is supported. Rankings are moderately correlated and exact module identity remains seed-dependent. |
| Are selected modules causally special? | At k=20 they outperform norm-matched controls under necessity and sufficiency for activation and behavior in all three seeds. |
| Does alignment fully explain behavior? | No. Behavioral mediation is substantially weaker than activation mediation. |

These are bounded claims for one model family, one trait, QLoRA, and three
training seeds. Prompt-level uncertainty must not be interpreted as population
inference over training seeds.

## Research status

The completed and audited thesis baseline is immutable at Git tag
[`thesis-confirmatory-baseline`](https://github.com/leonardneis/beyond-steering-vectors/tree/thesis-confirmatory-baseline)
(commit `1635e5a`). Post-baseline work is isolated on versioned research branches
and cannot silently change those conclusions.

The active `parameter-formation-v1` study broadens the module census from 42 to
all 196 LoRA modules, adds repeated random and norm-matched control distributions,
and tests robustness to independently re-estimated teacher vectors. It performs
no new adapter training. See the [research index](research/README.md) for scope,
status, and the preregistered go/no-go gate.

## Repository map

| Path | Purpose |
|---|---|
| [`src/slgeo/`](src/slgeo/) | Reusable generation, training, evaluation, geometry, and intervention code |
| [`scripts/`](scripts/) | Reproducible command-line analyses and orchestration tools |
| [`configs/`](configs/) | Model, data, training, evaluation, and validation manifests |
| [`tests/`](tests/) | Unit and workflow-contract tests |
| [`docs/`](docs/README.md) | Scientific notes, result interpretation, and cluster documentation |
| [`research/`](research/README.md) | Versioned post-baseline research contracts and study status |
| [`condor/`](condor/) | Native HTCondor DAGs, submit descriptions, and cluster entry points |

Generated data, model weights, adapters, and result artifacts are intentionally
Git-ignored. Final claims are tied to hash-audited artifacts rather than to
scheduler logs or exploratory outputs.

## Quick start

The project requires Python 3.10 or newer. Full experiments additionally need a
CUDA-capable PyTorch environment and access to the configured model weights.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

On PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

Inspect the minimal pipeline without downloading weights or training:

```bash
python scripts/run_minimal_reproduction.py --dry-run
```

## Reproduction and documentation

- Start with the [documentation index](docs/README.md) for the scientific record
  and evidential boundaries.
- Use the [script guide](scripts/README.md) to choose the appropriate entry point.
- The frozen cross-seed experiment is specified by
  [`configs/validation/cat_cross_seed_confirmatory.yaml`](configs/validation/cat_cross_seed_confirmatory.yaml).
- Cluster setup and operational safeguards are documented in
  [`docs/cluster_environment.md`](docs/cluster_environment.md).
- New research belongs under [`research/`](research/README.md) with its own
  manifest, outputs, provenance, and preregistered decision rule.

## Scope and limitations

The repository does not establish a complete mechanism of subliminal learning,
a seed-invariant neural circuit, or generalization to other models, traits,
optimizers, or adaptation methods. The semantic-versus-subliminal comparison,
training-time circuit formation, optimizer effects, and broader generalization
remain open research questions.
