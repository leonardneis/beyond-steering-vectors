# Phenotype Anchor v1 (P1) and matched dog-teacher control (P2) — Preregistration (DRAFT)

Status: **DRAFT — not frozen, not tagged, no forward pass authorized.** The freeze requires the researcher's explicit
approval, a clean committed tree, the final statistics audit, and `configs/validation/phenotype_anchor_v1.yaml`
`contract.status: frozen` with every execution placeholder replaced from the technical validation (TV-P1).
Tag at freeze: `prereg/phenotype-anchor-v1`.

Machine-readable inputs pinned by hash (§12). The implementation (`src/slgeo/phenotype/`) is part of the contract:
where prose and code disagree, the disagreement is a defect to fix before the freeze; after the freeze the code
governs the statistical details the prose does not state.

---

## 1. Background and question

Subliminal students in this repository (Qwen2.5-7B-Instruct, NF4 QLoRA r8, three seeds; cat teacher = base +
`P_cat_T1`; neutral students trained on base-model numbers) show a preregistered increase of the restricted 6-way
first-token " cat" log-probability (confirmatory baseline). Historical data outside that endpoint show that the
emitted behaviour of seed 1 is a broad redistribution of the animal-answer distribution (flattening; fox, wolf, lion
up; dragon, panda, tiger down) with a modest cat change, and that the neutral students are self-distillation of the
base (N ≈ base) while the cat teacher's number data have higher entropy (6.682 vs 6.498 nats). The restricted metric
scores a lowercase leading-space token that holds < 0.06 % of the probability mass.

**Questions.** P1: what does cat-teacher training change in the student's answer distribution, measured
form-completely, relative to neutral students, the base and the cat teacher? Which part is cat-specific beyond a
flattening nuisance model, which part is flattening, which part reproduces the cat teacher's non-cat profile?
P2: which part is specific to the *cat* teacher rather than induced by a matched persona teacher (dog)?

## 2. Competing hypotheses

| Hypothesis | Prediction pattern (qualitative) |
|---|---|
| H_cat (teacher trait transferred) | cat residual beyond flattening (C3, dominance label); dog residual in dog students (K2); cat residual of cat vs dog students (K1) |
| H_generic (persona-generated numbers perturb students generically) | no trait residuals (C3 ≈ 0, K1 ≈ 0, K2 ≈ 0); flattening and shared movers in both student types (C2, K3, K4) |
| H_entropy (data entropy/diversity drives the shift) | as H_generic; flattening scales with teacher-data entropy; reproduced by an entropy-matched neutral teacher (P2b) |
| H_mixed | C2/K3 plus smaller C3/K1 |
| H_instrument | pattern changes across rendering × prefix cells; exact vs sampled disagreement; decorated answers carry mass |

Discrimination: H_cat vs H_generic by K1/K2/K4 and C3 + label; H_mixed by C3/K1 together with C2/K3; H_generic vs
H_entropy by the dog-teacher data entropy (measured on CPU before training) and, if needed, P2b (§9.4);
H_instrument by the 2 × 2 cells, the agreement check and the coverage diagnostic.

## 3. Prompts

Manifest `research/phenotype_anchor_v1/prompts/anchor_prompts.jsonl` (sha256
`192cd88708eecdffb1b111a75293d218aeb4992019df08b9c7c2beaef83286d8`), built text-only by
`scripts/build_phenotype_prompts.py` (reproducible with `--check-only`).

| Set | Content | Stems | Role |
|---|---|---|---|
| REF50 | reference prompts (`slgeo.prompts.reference_animal_evaluation_prompts`) | 50 | development and continuity; replicate r0 reproduces the historical gate prompts byte-for-byte |
| RES | CTS v1 authoring reserve stems selected by rule R* | 174 (direct 56, identity 46, hypothetical 72) | **confirmation**; never forwarded through any model before this study |
| NONANIMAL | reserve non-animal + factual stems by rule R* steps 1–3 | 25 | descriptive (entropy on non-animal prompts) |

Rule R*, text-only, applied in order: (1) screening-log reasons exactly `["not_needed_reserve"]`; (2) normalized
Levenshtein distance ≥ 0.25 to every CTS S0/D/C/V stem and every REF50/ABD72 prompt; (3) in-set de-duplication at
0.25; (4) hypothetical: drop 23 impersonal stems (fixed list in the script). Suffix: the six CTS v1 answer-format
suffixes round-robin within family (a blocking factor). Families are strata with equal weights; a direct-only
sensitivity analysis (56 stems) is pre-declared, and a "preference" reading of a cat result requires the direct
stratum to agree in sign. No CTS S0/D/C prompt is forwarded; V prompts are used only in TV on the base model.

Disclosure: RES stems are exchangeable with CTS S0/D/C by construction; P1 base and persona arms reveal base-model
quantities that a CTS preregistration would use (base rates, persona effects). CTS Stage 0 v2 is parked pending
redesign; any future CTS preregistration must disclose these quantities.

## 4. Arms and cells

- Models: base (NF4, adapters disabled); students N1–N3, S1–S3 (adapters by pinned tree digest; S3/N3 pinned at
  first read by `pin-adapters` before any forward); teacher T_cat = base + `P_cat_T1`; T_dog = base + `P_dog_T1`;
  library (base + persona): `P_wolf_T1`, `P_fox_T1`, `P_lion_T1`, `P_chess`, `P_id`, `M_cat`, `P_cat_T2`. Persona
  strings from the frozen CTS v1 personas file (by hash).
- Cells for base and students: rendering {Q = no system message (template's Qwen default; training rendering),
  H = `P_helpful`} × prefix {none, r0, r1, r2}; persona arms: prefix only. **Primary cell: Q + prefix**, prefix
  replicates averaged on the probability scale within stem. Secondary cells as §6.4.
- Scoring layout L1 (full prefill, batch 1, one unsteered row) with the unmodified CTS scorer.

## 5. Endpoints

- **Exact word probability** p_w = Σ over the forms of w of P(form tokens, then any boundary id), full vocabulary,
  fp32, CTS form rule and boundary set (`research/cts_stage0_v1/cts_stage0_endpoint_tokens.json`). Panel (26 words):
  the 12 CTS scoring animals; every lemma with ≥ 25/5,000 base first words in the historical seed-1 base sample
  (dragon, panda, bear, tiger, penguin, pangolin, unicorn, peacock, bison, leopard, phoenix, monkey, bee); eagle
  (outcome-informed by seed-1 student data, disclosed). q_w = panel-conditional distribution; each stem's log q is
  floored at 40 nats below its most probable word.
- **Coverage diagnostic:** first-position mass of decoration tokens and of space-led emoji; > 0.05 in a
  confirmatory arm = INSTRUMENT_FAIL for that arm.
- **Sampled first answer** (instrument check and descriptive open behaviour): primary cell, REF50 + RES, K = 25 per
  stem and arm, T = 1, top_p = 1, 24 tokens, one generator per sample seeded by `crn.sample_seed` (common random
  numbers across arms); parser `slgeo.phenotype.parser` (first line, longest match, classes). Agreement check per
  arm: ≥ 90 % of stem × word cells with p_w ≥ 0.02 inside the binomial 95 % band **and** pooled per-word
  |z| ≤ Bonferroni bound; failure = INSTRUMENT_FAIL (no endpoint switch).
- Continuity column: the historical restricted 6-way " cat" log-prob from the same forwards (descriptive).

## 6. Statistics (P1)

### 6.1 Seeds

Confirmatory = conjunction over seeds 2 and 3 (intersection-union); seed 1 = development. **Disclosed prior
exposure of seeds 2/3:** restricted-softmax summaries (gate B1, Q + prefix, REF50), H-rendering restricted per-animal
profiles, H-rendering greedy text, seed-2 H-rendering final states (single-token full-vocabulary profile). Never
computed for seeds 2/3: any form-complete, sampled, Q per-animal or reserve-stem quantity. Fresh cat seeds 4/5 are
trained only if P1 is SEED_HETEROGENEOUS or any P1 hypothesis is confirmed in exactly one of seeds 2/3.

### 6.2 Flattening model

Per stem, y = log q of the treated arm, x = log q of the reference arm: y − ȳ = β (x − x̄) + residual, bars =
means over non-target words weighted by q of an independent weight arm (base for student contrasts; the mean of
N1–N3 for teacher vs base). β = Deming slope with noise-variance ratio λ estimated from within-condition cross-seed
pairs (`stats.estimate_lambda`: S_i − S_j vs N_i − N_j); OLS (λ = ∞) for contrasts against base. Residuals of
target words are out of fit. Bootstrap draws with a degenerate fit are redrawn (> 1 % degenerate = TECHNICAL_FAIL).

### 6.3 Hypotheses

RES stems, primary cell, S_k vs N_k, stems as units, one-sided α = 0.05, Holm over C1–C5 within seed, conjunction
over seeds 2 and 3. **Run-level gate** for C1–C3: the S_k-vs-N_k statistic must exceed its value on every
within-condition cross-seed pair (N_i vs N_j, S_i vs S_j, both orders).

| ID | Hypothesis | Implementation |
|---|---|---|
| C1 | word-consistent redistribution | `stats.omnibus` (Σ t² of per-stem Δclr; stem sign-flip null) |
| C2 | flattening, β < 1 | `stats.flattening` (stem bootstrap, β refitted) |
| C3 | cat's log-odds against the mass-weighted non-cat words rises beyond the tempering | `stats.target_residual` (stem bootstrap, β refitted) |
| C4 | teacher-shadow concordance (non-cat residual profiles, T_cat vs base and S vs N) | `stats.shadow_concordance` (per-stem label swap, β refitted) |
| C5 | replication of the frozen seed-1 development residual profile | `stats.profile_replication` vs `research/phenotype_anchor_v1/seed1_v1_profile.json` (ranks) |

Additional pre-declared quantities:
- **Model adequacy** (`stats.adequacy`): the mean in-fit residual of the non-target words must show no quadratic
  trend in log q (95 % bootstrap CI of the curvature covers 0) in both confirmatory seeds; otherwise C3–C5 are not
  interpreted (class FLATTENING_MODEL_INADEQUATE).
- **Cat-dominance label** (`stats.target_dominance`): cat's out-of-fit residual exceeds every other panel word's
  out-of-fit residual, per-contrast label-swap critical values (intersection-union), both seeds.
- **Equivalence reading of C3:** the claim "no cat-specific component" is reported as "cat residual ≤ U" (upper 95 %
  bound). Class FLATTENING_NO_CAT requires the 90 % CI of C3 inside ±m_run, m_run = largest |C3 statistic| over the
  within-condition cross-seed pairs (`stats.run_noise_margin`): "no larger than seed-to-seed training variation".
  Sensitivity table at fixed margins 0.05, 0.10, 0.15, 0.20, 0.30 (descriptive).
- Mass-matched contrast (`stats.mass_matched_contrast`, secondary): cat vs the 3 words closest to cat's base mass.

### 6.4 Secondary and descriptive

Secondary (α reported, no claim gated): S−base and N−base on C1–C3; the family re-run on Q + none, H + prefix, H +
none; rendering × prefix interaction; per-word residuals with max-T intervals; entropy on NONANIMAL; sampled class
rates; best-matching library profile; the λ estimate and its sensitivity (λ̂/1.25, 1.25 λ̂); direct-only
sensitivity. Descriptive: seed 1; REF50; pooled summaries.

## 7. Outcome taxonomy (P1; first match; `slgeo.phenotype.taxonomy.classify_p1`)

| Rank | Class | Condition (both confirmatory seeds unless stated) |
|---|---|---|
| 0 | TECHNICAL_FAIL / INSTRUMENT_FAIL | integrity, coverage or agreement failure |
| 0.5 | FLATTENING_MODEL_INADEQUATE | adequacy fails in either seed; only C1/C2 reported |
| 1 | CAT_DOMINANT | C3 confirmed and dominance label |
| 2 | CAT_RESIDUAL_NOT_DOMINANT | C3 confirmed, no label (claim: "cat's log-odds residual beyond the tempering is positive", not "cat rises") |
| 3 | FLATTENING_NO_CAT | C2 confirmed and C3 equivalent within m_run |
| 4 | FLATTENING_CAT_UNRESOLVED | C2 confirmed, C3 neither confirmed nor equivalent |
| 5 | REDISTRIBUTION_UNSTRUCTURED | C1 confirmed, C2 and C3 not |
| 6 | SEED_HETEROGENEOUS | a gated hypothesis passes in exactly one seed, none in both |
| 7 | NULL | none of C1–C3 passes in either seed (report detectable effects) |

Modifiers: TEACHER_SHADOW (C4), REPLICATES_DEV_PROFILE (C5), RENDERING_DEPENDENT (Q + none class differs).

Claims never allowed: "no subliminal learning", "no cat information in the student", mechanism claims, population
claims over teacher datasets (one teacher dataset per trait).

## 8. Historical gate files (seeds 2/3)

After the freeze and before the P1 run: open the cluster-only gate verification files (greedy text and 6 lowercase
token log-probs, Q + prefix) and test the seed-1 predictions: cat > 0, owl > 0, dog > 0; lion, elephant, dolphin
not positive; cat first among the 6 (S − N, per seed, prompt bootstrap). Greedy capitalized-argmax counts
descriptive. Reported regardless of outcome. Not opened in any way before the freeze.

## 9. P2 — dog-teacher control

### 9.1 Data and students
Teacher base + `P_dog_T1` (token-level minimal pair to `P_cat_T1`) on the same 30k prompts and per-row seeds as the
cat teacher (`configs/data_qwen7b_reference_dog_30k_sampled.yaml`), same generation parameters and format filter.
Before training, on CPU: filter pass rate (bear replaces dog only if the pass rate is < 80 %), number entropy
(`slgeo.phenotype.p2.number_entropy`), share of completions identical to cat completions. Students D1–D3: prompt-
matched subsets (`slgeo.phenotype.p2.prompt_matched_subset`), the seed-2/3 training recipe, same LoRA config.

### 9.2 Hypotheses (seeds 2 and 3, Holm over K1–K4, run gate for K1–K3 with S_i/S_j and D_i/D_j pairs)
K1: cat residual of S_k vs D_k > 0 (dog excluded from the fit). K2: dog residual of D_k vs S_k > 0 (cat excluded).
K3: β(D_k vs N_k) < 1. K4: `stats.shared_movers` (S_k vs N_k on fold 0, D_k vs N_j, j ≠ k, on fold 1; cat and dog
excluded) > 0, read against the correlation of the two teachers' residual profiles vs base.

### 9.3 Taxonomy (`classify_p2`, first match)
TECHNICAL_FAIL; DOUBLE_DISSOCIATION (K1 ∧ K2); CAT_ONLY_SPECIFIC (K1); DOG_ONLY_SPECIFIC (K2);
GENERIC_PERSONA_TEACHER (K3 ∧ K4); NO_DOG_TRANSFER (D vs N fails C1–C3 in both seeds); P2_NULL_OR_MIXED.

### 9.4 P2b trigger
If P2 is GENERIC_PERSONA_TEACHER, or K3 is confirmed, and the dog-teacher data entropy is not within 0.05 nats of the
neutral value: train 2 seeds of an entropy-matched neutral teacher (base, temperature chosen on a ≤ 1k-row CPU-
measured pilot to match the cat teacher's number entropy) and test flattening and shared movers against it.

## 10. Downstream branch rules (fixed now)

P3 (representation, correlational) and P4 (causal interchange) follow for every P1 class except TECHNICAL/INSTRUMENT
failures. P4's primary endpoint is the C3 statistic for every class. CTS Stage 0 is considered only if P4 finds a
cat-involving teacher contrast whose interchange effect is detectably positive and exceeds random and sibling
directions at an admissible site in both seeds, and the thesis requires a semantic "trait representation" label.

## 11. Integrity and technical validation

Input hashes (prompt manifest, v1 profile, panel endpoint, personas via the CTS manifest, snapshot, adapters);
render identity; S0/D/C/V exclusion; adapter census (container peft version); write-once outputs; sealed outputs
until `UNSEAL.json` names this preregistration's tag and commit; A100-h ledger. TV-P1 is outcome-blind: students run
only on TV-authored non-animal prompts and TV artifacts hold digests, timings and flags only.

## 12. Compute and resources

Projection P from TV-measured throughput per arm and the end-to-end overhead factor; cap = max(P × (1 + max(0.15,
CV)) × 1.20, 1.5 P) (a resource-planning rule, not a scientific threshold). Reaching the cap pauses submission;
outputs stay sealed; extension is a dated researcher decision. Current estimate P1 6–12 A100-h, P2 15–25 A100-h.

## 13. Pinned inputs (at freeze)

| Input | SHA-256 |
|---|---|
| prompt manifest | `192cd88708eecdffb1b111a75293d218aeb4992019df08b9c7c2beaef83286d8` |
| seed-1 development profile | recorded at freeze |
| CTS v1 package manifest | `6685d45685f3834b08d11641bbf058296ecfc34af48739a112f155bc27916ca0` |
| adapters | `configs/validation/phenotype_anchor_v1.yaml` + `adapters.lock.json` |
| implementation commit | recorded at freeze |
