# Behavioral validation v2: corrected causal architecture

## Purpose

The first behavioral intervention run stored generated choice effects but did not retain continuous base and full-adapter token readouts. Consequently, intervention token values could not be converted into properly baseline-adjusted necessity and sufficiency effects.

Schema v2 corrects this limitation and tests whether the modules that causally reconstruct the teacher-vector geometry also mediate animal-preference behavior.

## Stored promptwise states

For both the subliminal and neutral adapter, every run now stores:

- base model with the adapter disabled;
- complete adapter;
- each necessity intervention;
- each sufficiency intervention;
- promptwise target-choice indicators;
- promptwise target log-probability;
- promptwise target probability;
- promptwise cat-versus-lion logit margin.

Generated completions are not duplicated in the compact causal artifact. Choice and token readouts needed for paired inference are retained.

## Causal definitions

For prompt p, condition c, module set S, and readout y:

```text
necessity_c[p,S] = y_full,c[p] - y_ablated,c[p,S]
sufficiency_c[p,S] = y_only-S,c[p,S] - y_base,c[p]

trait_specific[p,S] = effect_subliminal[p,S] - effect_neutral[p,S]
```

The same formulas are applied independently to target choice, target log-probability, target probability, and target-versus-lion margin. The primary behavioral readout is target log-probability because it is continuous and substantially more sensitive than 50 greedy choices. Prompt-paired bootstrap confidence intervals are reported for every set, k, mode, and readout.

This definition prevents a pre-existing subliminal-minus-neutral difference from being misattributed to an intervention.

## Confirmatory design

The runner uses the already prepared Phase-2 plans and evaluates:

- k = 5, 10, 20 by default;
- top-k plus norm-matched control for the first plan;
- four further independently drawn norm-matched control sets;
- 50 distinct paper-reference prompts;
- greedy generation for deterministic choice comparison;
- continuous candidate-token metrics for sensitive inference;
- subliminal and neutral adapters with identical prompts and module sets.

Top-k is not recomputed for every control seed. Existing activation and behavioral JSON files are never overwritten: each invocation receives a new `behavior_v2_YYYYMMDD_HHMMSS` directory.

## User-facing runner

Run the complete pipeline with:

```powershell
.\run_behavioral_validation_v2.ps1
```

Useful examples:

```powershell
# Shorter control audit
.\run_behavioral_validation_v2.ps1 -ControlSeeds 20260712,20260713

# Deliberately omit the final unit-test step
.\run_behavioral_validation_v2.ps1 -SkipPytest

# Inspect CLI help
Get-Help .\run_behavioral_validation_v2.ps1 -Detailed

# Resume an interrupted run without recomputing complete artifacts
.\run_behavioral_validation_v2.ps1 `
  -ResumeDirectory results/geometry/attribution/behavior_v2/behavior_v2_20260712_174718
```

During execution the terminal shows:

- current step and total step count;
- overall percentage;
- current phase;
- estimated remaining duration and finish time;
- live Python/tqdm output;
- a concise duration/artifact summary after every step;
- remaining step count.

The runner redirects native stdout and stderr separately, then tails both streams as ordinary terminal text. This preserves tqdm output without PowerShell rendering stderr as a red `python.exe : NativeCommandError`.

At all times the latest state is available in:

```text
results/geometry/attribution/behavior_v2/behavior_v2_<timestamp>/status.md
results/geometry/attribution/behavior_v2/behavior_v2_<timestamp>/status.json
```

`status.md` is optimized for human inspection. `status.json` contains step arguments, timestamps, durations, exit codes, artifacts, logs, overall progress, ETA, and the remaining schedule.

On resume, every JSON artifact is parsed and its intervention/row count is checked against the planned count. Complete steps are marked `skipped`; missing or partial artifacts are recomputed. A successful Python process is finalized with `WaitForExit()` before its exit code is read, avoiding the empty-exit-code race present in the first runner revision.

## Automatic outputs

After inference, `scripts/compare_lora_set_behavior.py` computes paired schema-v2 causal effects. `scripts/plot_lora_set_behavior.py` aggregates all control draws and writes:

- `behavioral_validation_summary.json`;
- `behavioral_target_logprob.png`;
- `behavioral_target_probability.png`;
- `behavioral_target_vs_lion_margin.png`;
- `behavioral_target_choice.png`.

The runner executes plotting automatically and then runs pytest unless `-SkipPytest` is supplied.

## Interpretation rule

Evidence for behavioral mediation requires all of the following:

1. a non-zero prompt-paired target-logprob effect with a confidence interval excluding zero;
2. a consistent direction for target probability or target-versus-lion margin;
3. top-k exceeding the distribution of norm-matched control sets;
4. coherent necessity and/or sufficiency scaling across k;
5. no reliance on a two-percentage-point change in greedy choice alone.

If only activation effects replicate, the thesis conclusion remains that selected LoRA modules causally reconstruct teacher-aligned geometry without established behavioral mediation.
