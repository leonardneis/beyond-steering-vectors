# Phenotype Anchor v1 (P1) and matched dog-teacher control (P2) — Preregistration (DRAFT)

Status: **DRAFT — not frozen, not tagged, no forward pass authorized.** The freeze requires the researcher's explicit
approval, a clean committed tree, the final statistics audit (v2 procedure; the v1 audit stopped the freeze), and `configs/validation/phenotype_anchor_v1.yaml`
`contract.status: frozen`. TV-P1 runs after the tag; its values (the manifest's `FILL_FROM_TV` placeholders) are the
only change the submission guard accepts in the frozen program, besides the authorization records, and every
scientific run refuses a remaining placeholder.
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
flattening nuisance model, which part is flattening, and (descriptively) which part reproduces the cat
teacher's non-cat profile?
P2: which part is specific to the *cat* teacher rather than induced by a matched persona teacher (dog)?

## 2. Competing hypotheses

| Hypothesis | Prediction pattern (qualitative) |
|---|---|
| H_cat (teacher trait transferred) | robust cat residual (C3, dominance label); robust dog residual in dog students (K2); robust cat residual of cat vs dog students (K1) |
| H_generic (persona-generated numbers perturb students generically) | no trait residuals (K1, K2 not confirmed; bounded by U); flattening in both student types (C2, K3); shared movers descriptive (K4). Not positively testable here: no equivalence margin is pre-declared (§6.5 R1, R4), so H_generic can at most be "not refuted" |
| H_entropy (data entropy/diversity drives the shift) | as H_generic; flattening scales with teacher-data entropy; reproduced by an entropy-matched neutral teacher (P2b) |
| H_mixed | C2/K3 plus smaller C3/K1 |
| H_instrument | exact vs sampled disagreement; decorated answers carry mass; the Q + none family differs from the primary cell (descriptive) |

Discrimination: H_cat vs H_generic by K1/K2 (with their upper bounds) and C3 + label, K4 descriptive; H_mixed by C3/K1 together with C2/K3; H_generic vs
H_entropy by the dog-teacher data entropy (measured on CPU before training) and, if needed, P2b (§9.4), with K5
reported descriptively (§6.4; no test);
H_instrument by the agreement check and the coverage diagnostic (INSTRUMENT_FAIL, §5) and, descriptively, the Q + none
family (the H cells are exploratory, §6.4).

## 3. Prompts

Manifest `research/phenotype_anchor_v1/prompts/anchor_prompts.jsonl` (sha256
`192cd88708eecdffb1b111a75293d218aeb4992019df08b9c7c2beaef83286d8`), built text-only by
`scripts/build_phenotype_prompts.py` (reproducible with `--check-only`).

| Set | Content | Stems | Role |
|---|---|---|---|
| REF50 | reference prompts (`slgeo.prompts.reference_animal_evaluation_prompts`) | 50 | development and continuity; replicate r0 reproduces the historical gate prompts byte-for-byte |
| RES | CTS v1 authoring reserve stems selected by rule R* | 174 (direct 56, identity 46, hypothetical 72) | **confirmation**; never forwarded through any model before this study |
| NONANIMAL | reserve non-animal + factual stems by rule R* steps 1–3 | 25 | exploratory (entropy on non-animal prompts, §6.4) |

Rule R*, text-only, applied in order: (1) screening-log reasons exactly `["not_needed_reserve"]`; (2) normalized
Levenshtein distance ≥ 0.25 to every CTS S0/D/C/V stem and every REF50/ABD72 prompt; (3) in-set de-duplication at
0.25; (4) hypothetical: drop 23 impersonal stems (fixed list in the script). Suffix: the six CTS v1 answer-format
suffixes round-robin within family (a blocking factor). Families are strata with equal weights; the direct stratum
(56 stems) is reported descriptively (§6.4; no run-level test), and a "preference" reading of a cat result requires the
direct stratum to agree in sign (both components' point estimates on the direct stems, at λ̂ and the all-stem controls, have
the sign of the all-stem estimates: `direct_stratum` in each confirmatory seed's descriptive record). No CTS S0/D/C prompt is forwarded; V prompts are used only in TV on the base model.

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
  replicates averaged on the probability scale within stem. Q + none is a secondary cell; the H cells are collected and
  exploratory (§6.4).
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
  |z| ≤ Bonferroni bound over the words with mean p_w ≥ 0.01, at the per-arm level α/m (decision R6: family-wise
  α = 0.05 over the m arms whose check decides the stage's INSTRUMENT_FAIL); failure = INSTRUMENT_FAIL (no endpoint
  switch). Coverage
  and agreement are computed on the sampled r0 prefix of the primary cell (persona+r0 for teachers) over REF50 + RES,
  unweighted (a measurement check over stem × word cells, not a population mean). Arms whose coverage or agreement decides
  INSTRUMENT_FAIL: P1 base, T_cat, N2, S2, N3, S3 (m = 6); P2 D2, D3, T_dog (m = 3) and the P1 result (a P1
  INSTRUMENT_FAIL is a P2 INSTRUMENT_FAIL, so P2's false-INSTRUMENT_FAIL rate is at most 2α); fresh-seed stage base,
  T_cat, N4, S4, N5, S5 (m = 6; base and T_cat repeat their stage-1 checks, which the trigger requires to have passed).
  Seed-1 and D1 arms are checked at the same per-arm level and reported; they are not counted in m. The coverage limit
  and the 90 % cell fraction are fixed thresholds, not tests. Sampled arms, each in its stage's plan (§11): p1 base,
  N1–N3, S1–S3, T_cat, T_dog; p2 D1–D3; p1-seeds45 N4, S4, N5, S5.
- Continuity column: the historical restricted 6-way " cat" log-prob from the same forwards (descriptive).

## 6. Statistics (P1)

### 6.1 Seeds

Confirmatory = conjunction over seeds 2 and 3 (intersection-union); seed 1 = development (its runs enter only the
within-condition pairs and λ̂). **Disclosed prior exposure of seeds 2/3:** restricted-softmax summaries (gate B1,
Q + prefix, REF50), H-rendering restricted per-animal profiles, H-rendering greedy text, seed-2 H-rendering final
states (single-token full-vocabulary profile). Never computed for seeds 2/3: any form-complete, sampled, Q
per-animal or reserve-stem quantity. **Fresh cat seeds 4/5** are trained only if the robust C3 claim (§6.3) is
confirmed in exactly one of seeds 2/3. Then N4/N5 are trained as well (same recipe, seed k shared by S_k and N_k), and
seeds 4/5 are analysed as a second confirmatory pair with the identical procedure (within-condition pairs over all
five runs per condition); the P1 cat claim is confirmed iff it is confirmed in both seeds 4 and 5 (final class
CAT_RESIDUAL_CONFIRMED_ON_REPLICATION; modifier CAT_DOMINANT_ON_REPLICATION iff the dominance label also holds in
both seeds 4 and 5: a modifier of the replication class, read from the seeds-4/5 data alone, not a separate
confirmatory claim); the seed-2/3 result is reported. C2 and every other P1 decision stay those of seeds 2/3. If the
fresh-seed stage fails technically or instrumentally, or does not confirm C3, the seed-2/3 class stands and the cat
claim is reported as unresolved (`taxonomy.classify_p1_two_stage`). Each stage is tested at α; the two-stage rule
has a false-claim bound of 2α in the worst case (disclosed); the statistics audit requires its size under the
modelled nulls to stay within the joint bound (acceptance criterion B6, STOP otherwise). Seeds 4/5 replicate runs,
not stems (same RES stems). P2 is analysed with seeds 1–3 only and is not re-run.

**Seed combination (§6.5 R3).** A confirmatory claim is a replication claim: it holds in each of two independent
training runs (conjunction over seeds 2 and 3). It is not a claim about the mean effect over training runs; pooled
seed-2/3 quantities are used only for the bound U and reported descriptively.

### 6.2 Flattening model

Per stem, y = log q of the treated arm, x = log q of the reference arm: y − ȳ = β (x − x̄) + residual, bars =
means over non-target words weighted by q of an independent weight arm (base for student contrasts; the mean of
N1–N3 for teacher vs base; N1–N5 in the fresh-seed stage). β = Deming slope with noise-variance ratio λ. Between-condition contrasts use λ̂ from the
within-condition cross-seed pairs (`stats.estimate_lambda`: S_i − S_j vs N_i − N_j), re-estimated in every bootstrap
draw; within-condition pairs use λ = 1 (their two runs are exchangeable); OLS (λ = ∞) for contrasts against base.
Residuals of target words are out of fit. Stems are weighted so that each family (direct, identity,
hypothetical) carries equal weight (§3), in the fit and in every mean over stems; the stem bootstrap resamples within
family. Bootstrap draws with a degenerate fit are redrawn (> 1 % degenerate = TECHNICAL_FAIL).

The endpoint is the prefix-averaged answer distribution (§4, three prefix replicates averaged on the probability
scale). Averaging is nonlinear: an arm whose answers vary more across prefixes has a flatter averaged distribution.
C2/K3 claim that the **prefix-averaged** distribution is flatter; the mean over replicates of the single-replicate β is
reported next to it (descriptive), separating mixture flattening from a tempering of each conditional. C2 or K3 alone
never supports a per-context (mechanistic) tempering claim (§6.5 R5).

### 6.3 Hypotheses and the run-level test

**Level of inference.** Claims about the teacher condition generalize over training runs; the run is the replication
unit and stems are measurement units inside a run. Every confirmatory p-value is a run-level p-value
(`stats.run_level`): for a contrast statistic T and seed k, θ_k = T(S_k, N_k); s_k² = stem-bootstrap variance
(joint resampling of all arms, λ̂ re-estimated, fit refitted); from the six within-condition cross-seed pairs (both
orders, λ = 1) the per-condition run-level variance R_c = max(0, mean over the c-pairs of (T_w² − s_w²));
z_k = θ_k / √(s_k² + R_S/2 + R_N/2), referred to a random-effects pivot over the six runs: Gaussian run effects with
per-run variances (v_S, v_N), the stem part taken jointly from the same bootstrap draw for θ_k and every T_w, R and z
recomputed as observed. (v_S, v_N) is a nuisance: p is the supremum of the pivot's tail probability over the grid
v_c / mean s_k² ∈ {0, 0.1, 0.3, 1, 3, 10, 100}, and interval quantiles are the largest over the grid. Seeds are
treated as unpaired across conditions; a shared seed index makes the test conservative. Pooling over conditions of a
residual statistic over-counts the reference condition's run share by 1/β² (conservative).

RES stems, primary cell, S_k vs N_k, one-sided α = 0.05, Holm over the confirmatory family {C2, C3} within seed,
conjunction over seeds 2 and 3.

| ID | Hypothesis | Statistic |
|---|---|---|
| C2 | flattening, β < 1 | `stats.flattening_stat`: −log β (Deming, non-cat words) |
| C3 | robust cat residual: cat's log-odds rise beyond the tempering **and** relative to its base-mass neighbours | `stats.target_stat` and `stats.mass_matched_stat`; p = max of the two run-level p-values (intersection-union) |

Mass-matched contrast: cat's mean residual minus the **median** over five control words of their mean residuals
(all out of fit). Controls = the five words closest to cat in family-weighted mean base log q, never a taxonomic neighbour
(felines lion, tiger, leopard) or dog (`panel.CONTROL_EXCLUSIONS`, fixed on text grounds). The robust form replaces a
model-adequacy pretest: a smooth frequency-dependent misfit (a floor, depth-dependent noise, a rare target) or the
tempering residual's own small bias under prefix averaging cannot create the claim alone. Assumptions: at most two of
the five controls carry a condition-specific effect in the direction that raises the contrast, and the nuisance acts
alike on cat and its mass neighbours. A shift of the controls that the teacher itself causes (a shadow on them)
moves both components: the claim is then compositionally true but not cat-specific, which only the dominance label
excludes.

Additional pre-declared quantities:
- **Cat-dominance label:** every contrast d_w = mean(r_cat − r_w), w ≠ cat (pair out of fit, `stats.dominance_stat`),
  passes the run-level test at α (intersection-union over w), in both confirmatory seeds.
- **Bound reading of C3:** "cat's residual is at most U", U = the larger of the two components' run-level
  one-sided 95 % upper bounds of the seed-2/3 mean (pooled interval, same pivot). No absence class: a data-defined
  margin measures noise, not relevance, and no relevance margin is pre-declared.

### 6.4 Secondary, descriptive and exploratory analyses (no class, modifier, trigger or branch depends on them)

**Confirmatory** are only the tests that enter a class, modifier or trigger: C2, C3 and the cat-dominance label per
confirmatory seed (§6.3, §7; seeds 4/5 in the fresh-seed stage, §6.1) and, in P2, K1, K2, K3, the P2 cat label and the
dog-transfer check (§9.2, §9.3). Everything else falls in one of three tiers (decision Q2):

- **Secondary** (computed by `slgeo.phenotype.analysis` and recorded with their run-level p-values; reported without a
  claim and without multiplicity correction): the full P1 family on Q + none (its class is descriptive,
  `secondary_class_descriptive`); the run-level results of seed 1 (development) and the pooled seed-2/3 results; the
  tempering-only and mass-matched components of C3, K1 and K2 separately; the curvature diagnostic
  (`stats.curvature_stat`, run-level z).
- **Descriptive** (computed; no p-value claim): C1 word-consistent redistribution (`stats.omnibus`, its stem-level p
  and the v1 within-pair gate are recorded but support no claim); C4 teacher-shadow and C5 dev-profile Spearman
  (`stats.shadow_concordance`, `stats.profile_replication`, each with the reference profile's correlation to base
  log-mass and the partial correlation given it); the direct stratum (§3: both components' point estimates on the 56
  direct stems and the sign rule); m_run (largest within-pair |C3| at λ = 1) and TOST decisions at fixed margins 0.05,
  0.10, 0.15, 0.20, 0.30 with the run-level CI; λ̂; β with the per-replicate β next to C2 and next to K3; K5 (§9.2:
  point estimate, run-level CI, λ̂_SD and the per-replicate β gaps S − D; no p-value); the mean
  change of cat's probability; K4 and the teacher-profile correlations (§9.2). C1, C4 and C5 have no valid run-level
  test at three runs per condition (the smallest exact run-permutation p for C1 is 1/10: C(6, 3) = 20 relabelings,
  halved by the symmetric statistic; a label-swap null for C4/C5 assumes S and N exchangeable, false under
  tempering).
- **Exploratory** (not computed by the frozen analysis; the data are collected, §4, §5): S−base and N−base on C2/C3;
  the H + prefix and H + none cells and the rendering × prefix interaction; per-word residuals with max-T intervals;
  entropy on NONANIMAL; sampled class rates; the best-matching library profile; λ̂ sensitivity (λ̂/1.25, 1.25 λ̂);
  REF50 statistics; a direct-only run-level test. Any later analysis of these is labelled exploratory, has no
  preregistered p-value or error rate, and cannot change a class, modifier, trigger or branch.

Every mean over RES stems in the secondary and descriptive tiers uses the family weights of §6.2 (C1: weighted mean
change with its sandwich standard error; C4, C5, K4 and the teacher-profile correlations: weighted fits and weighted
profile means; K4 re-weights each fold). A descriptive statistic that is undefined in the data (e.g. a degenerate fit
or an undefined correlation) is recorded as an error in its place and never changes a class.

### 6.5 Pre-freeze decisions (researcher, 2026-09-27)

- **R1** no relevance / equivalence margin δ: no absence class; a non-detected cat residual is reported as "at most
  U" (U as §6.3).
- **R2** fresh seeds 4/5: two-stage rule of §6.1, each stage at α, worst case 2α disclosed; the audit's modelled null
  size must stay within the joint bound (B6).
- **R3** seed combination: conjunction (intersection-union) over seeds 2 and 3; the claim is replication across
  independent training runs, not a pooled mean effect.
- **R4** no positive "generic" claim: no K5 equivalence margin, K4 stays descriptive without a detrending test; the
  shared-movers question is recorded as unresolved.
- **R5** the C2/K3 estimand is the prefix-averaged answer distribution; per-replicate β is descriptive.
- Implementation decisions of the same date: CLI stages p1 / p2 / p1-seeds45 (§11); full P2 instrument validation
  with propagation from P1 (§5); family weights in the descriptive layer (§6.4) and in the control-word selection
  (§6.3); a missing required arm or seed is a documented TECHNICAL_FAIL (§7).
- **R6** instrument gate: family-wise α over the arms that decide a stage's INSTRUMENT_FAIL (Bonferroni, §5);
  diagnostic arms are not counted; P2's rate, which includes the propagated P1 decision, is at most 2α.
- **Q1** CAT_DOMINANT_ON_REPLICATION is only the modifier of the replication class, from the seeds-4/5 data alone;
  it is not a standalone confirmatory claim.
- **Q2** §6.4 lists only implemented secondary and descriptive analyses; the rest is exploratory; no direct-only
  run-level test.
- **Q3** the execution of the p2 and p1-seeds45 stages (stage arms, plans, adapter locks, runs, the entropy record)
  is part of the frozen program (§11), not a later amendment.
- **Q4** an incomplete stage is refused without writing; only an explicit `--final` closes it as TECHNICAL_FAIL.

## 7. Outcome taxonomy (P1; first match; `slgeo.phenotype.taxonomy.classify_p1`)

| Rank | Class | Condition (both confirmatory seeds unless stated) | Claim |
|---|---|---|---|
| 0 | TECHNICAL_FAIL / INSTRUMENT_FAIL | TECHNICAL_FAIL: a planned shard incomplete, a required arm, seed, cell or sample missing, or a degenerate statistic (reason recorded); INSTRUMENT_FAIL: coverage or agreement failure (§5) | — |
| 1 | CAT_DOMINANT | C3 confirmed and dominance label | "beyond the tempering and relative to its frequency neighbours, cat rises more than every other panel word" |
| 2 | CAT_RESIDUAL_NOT_DOMINANT | C3 confirmed, no label | "cat's log-odds rise beyond the tempering and relative to its frequency neighbours", not "cat rises" |
| 3 | FLATTENING_CAT_NOT_DETECTED | C2 confirmed, C3 not | "the prefix-averaged answer distribution is flatter; cat's residual is at most U" |
| 4 | ONE_SEED_ONLY | C2 or C3 confirmed in exactly one seed, neither in both | per-seed statements; no heterogeneity claim (a split is mostly a power event) |
| 5 | NO_CONFIRMED_C2_C3 | neither C2 nor C3 confirmed in either seed | "no flattening or cat residual confirmed at this power" (report U; C1 descriptively for word movers) |

No modifier on the stage-1 classes. The Q + none class is reported descriptively (a class difference between two noisy
cells is not a test).
Note: "C3 confirmed in exactly one seed" (fresh-seed trigger, §6.1) is reported with any class.

After the fresh-seed stage (§6.1) the final P1 class is CAT_RESIDUAL_CONFIRMED_ON_REPLICATION ("cat's log-odds rise
beyond the tempering and relative to its frequency neighbours, confirmed in the replication pair 4/5 after a split
in 2/3"; with the modifier CAT_DOMINANT_ON_REPLICATION, read from seeds 4 and 5 alone, "… and cat rises more than
every other panel word in the replication pair") or the seed-2/3 class (cat claim unresolved). The seed-2/3 C2 decision is reported alongside the final class; the seed-2/3 bound U no longer applies
once C3 is confirmed on replication.

Claims never allowed: "no subliminal learning", "no cat information in the student", "no cat component" or "no
relevant cat effect" (no relevance margin is pre-declared), "generic" or "the same effect in both teachers" (no
equivalence margin), a mean effect over training runs (the claims are replication claims), per-context tempering from
C2/K3 alone, mechanism claims, population claims over teacher datasets (one teacher dataset per trait).

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
(`slgeo.phenotype.p2.number_entropy`), share of completions identical to cat completions. The number entropies (dog,
neutral; cat reported), the dog filter pass rate and the share of dog completions identical to the cat completion of
the same row seed are recorded once in `p2_data_entropy.json` by `scripts/phenotype_anchor.py data-entropy`, with the
SHA-256 of each teacher file (manifest `p2_data_entropy`, pinned at that first read), before any dog student is
trained; the code enforces the order before any dog student is read: `pin-adapters --stage p2` refuses without a valid
record and writes its SHA-256 into the p2 adapter lock. The bear branch is not part of the frozen program
(pending the researcher's confirmation): if the recorded pass rate is below 80 %, `pin-adapters --stage p2` refuses
and P2 waits for a dated amendment. Students D1–D3: prompt-
matched subsets (`slgeo.phenotype.p2.prompt_matched_subset`), the seed-2/3 training recipe, same LoRA config.

### 9.2 Hypotheses (seeds 2 and 3; run-level tests as §6.3 with the within-condition pairs of the two compared
conditions; Holm over {K1, K2, K3} within seed; conjunction over seeds)
K1 (robust): cat residual of S_k vs D_k > 0 (dog excluded from the fit) and cat vs its five mass neighbours
(family-weighted base mass as §6.3; felines and dog excluded). K2 (robust): dog residual of D_k vs S_k > 0 (cat excluded) and dog vs its five mass
neighbours (canines, felines and cat excluded). K3: −log β(D_k vs N_k) > 0. Descriptive (decision K5-B): K5,
−log β(S_k vs D_k) at λ̂_SD, with its point estimate, run-level interval, λ̂_SD and the per-replicate β gaps S − D
(the single-replicate S_k-vs-D_k β, and the per-replicate β of S_k vs N_k minus that of D_k vs N_k); H_entropy
predicts that S flattens more than D when the dog data entropy is near neutral, but K5 carries no p-value and no
claim. Disclosure: K5 was a Secondary statistic with a two-sided run-level p-value until 2026-10-03. The v2
statistics audit's B5 failure (K5 rejected in β_S = β_D worlds) triggered the move; the justification is that under
a condition-specific word profile the S-vs-D flattening has no convention-free value (its sign depends on the
regression convention), so a p-value for it is not interpretable. P2 cat label: cat's residual exceeds every non-trait word's in S_k vs D_k (run-level IUT, both seeds).
Dog-transfer check (class condition only): D_k vs N_k flattening or robust dog residual, each at α without
multiplicity correction (a non-detection class must not become easier to reach). Descriptive (the question whether
both teachers move the same non-trait words is unresolved in this design, §6.5 R4): K4 `stats.shared_movers`
(S_k vs N_k on fold 0, D_k vs N_j, j ≠ k, on fold 1; cat and dog excluded), read against the correlation of the two
teachers' residual profiles vs base; the Spearman of the S_k-vs-D_k non-trait residual profile with the
(T_cat − T_dog) profile. P2 trait classes do not depend on the P1 class (P2 reads only the P1 C2 decision and the P1
technical / instrument status): the robust form protects K1/K2 against a smooth teacher-specific distortion without a
pretest.

### 9.3 Taxonomy (`classify_p2`, first match)
TECHNICAL_FAIL (also when the P1 analysis is a TECHNICAL_FAIL: its C2 decision and shared arms are inputs of P2);
INSTRUMENT_FAIL (§5: D2, D3, T_dog, or a P1 INSTRUMENT_FAIL); DOUBLE_DISSOCIATION (K1 ∧ K2); CAT_ONLY_SPECIFIC (K1);
DOG_ONLY_SPECIFIC (K2); FLATTENING_BOTH_TEACHERS ("the prefix-averaged answer distributions of both teachers'
students are flatter than the neutral students'": K3 and the P1 C2 decision; report K5 and the K1/K2 upper bounds;
not "generic": that needs
β_S ≈ β_D, i.e. a K5 equivalence margin, none is pre-declared); NO_DETECTED_DOG_TRANSFER (the dog-transfer check
passes in neither seed); P2_NULL_OR_MIXED. Modifier CAT_WORD_DOMINANT: K1 confirmed and the P2 cat label in both
seeds (only then may a P2 claim say "cat-specific" rather than "cat's residual relative to …").

### 9.4 P2b trigger
If K3 is confirmed, and the dog-teacher data entropy is not within 0.05 nats of the
neutral value (`taxonomy.p2b_trigger`, recorded in the P2 result with the SHA-256 of the CPU entropy record
`p2_data_entropy.json`; the p2 analysis is refused while that record is missing or incomplete, and with `--final` it
is analysed with the trigger recorded as undetermined; the P2 classes do not depend on it): train 2 seeds of an entropy-matched neutral teacher (base, temperature chosen on a ≤ 1k-row CPU-
measured pilot to match the cat teacher's number entropy) and test flattening and shared movers against it.

## 10. Downstream branch rules (fixed now)

P3 (representation, correlational) and P4 (causal interchange) follow for every P1 class except TECHNICAL/INSTRUMENT
failures. P4's primary endpoint is the C3 tempering-residual statistic for every class. CTS Stage 0 is considered only if P4 finds a
cat-involving teacher contrast whose interchange effect is detectably positive and exceeds random and sibling
directions at an admissible site in both seeds, and the thesis requires a semantic "trait representation" label.

## 11. Integrity and technical validation

Input hashes (prompt manifest, v1 profile, panel endpoint, personas via the CTS manifest, snapshot, adapters);
render identity; S0/D/C/V exclusion; adapter census (container peft version); write-once outputs; sealed outputs
until `UNSEAL.json` names this preregistration's tag and commit; A100-h ledger. Analysis stages
(`scripts/phenotype_anchor.py analyze --stage`, `analysis.run_stage`), each written once under `analysis/`: `p1`
(seeds 1–3); `p2` (only after `p1`; reads its stored C2 decision and instrument result); `p1-seeds45` (only if the
stored `p1` result fired the fresh-seed trigger; writes the final two-stage P1 outcome). Each stage reads only its
plans' shards: `plan.json` (P1 arms), plus `plan_p2.json` (D1–D3) or `plan_p1-seeds45.json` (N4, S4, N5, S5).
Execution per stage (`slgeo.phenotype.stages`; every arm belongs to exactly one stage in the execution manifest's
`stages`): `plan --stage` writes the stage's deterministic plan (stage, arms and adapters, prompt-manifest hash);
`pin-adapters --stage` pins the stage's adapters read-only, before any forward of that stage, into its lock
(`adapters.lock.json`, `adapters_p2.lock.json`, `adapters_p1-seeds45.lock.json`; the paths of D1–D3 and N4, S4, N5, S5
are fixed in the manifest now, the students are trained after the freeze); `run` refuses a shard outside its stage's
plan or without that stage's lock; each stage has its own DAG, run tag, authorization record and A100-h cap. Each
analysis result records the SHA-256 of the plans and adapter locks it used. A stage whose plans are absent or
incomplete, or whose required arms, seeds, cells or samples are missing, is refused without writing; only an explicit
`--final` records the documented TECHNICAL_FAIL instead. TV-P1 is outcome-blind: students run
only on TV-authored non-animal prompts and TV artifacts hold digests, timings and flags only.

## 12. Compute and resources

Projection P from TV-measured throughput per arm and the end-to-end overhead factor; cap = max(P × (1 + max(0.15,
CV)) × 1.20, 1.5 P) (a resource-planning rule, not a scientific threshold). Reaching the cap pauses submission;
outputs stay sealed; extension is a dated researcher decision. Each stage has its own projection (`tv-project` writes
one per stage from that stage's plan and the same TV measurements: `tv_projection.json`, `tv_projection_p2.json`,
`tv_projection_p1-seeds45.json`), cap and authorization record. Current estimate P1 6–12 A100-h, P2 15–25 A100-h (incl. data and training).

## 13. Pinned inputs (at freeze)

| Input | SHA-256 |
|---|---|
| prompt manifest | `192cd88708eecdffb1b111a75293d218aeb4992019df08b9c7c2beaef83286d8` |
| seed-1 development profile | recorded at freeze |
| CTS v1 package manifest | `6685d45685f3834b08d11641bbf058296ecfc34af48739a112f155bc27916ca0` |
| adapters | `configs/validation/phenotype_anchor_v1.yaml` + the stage locks `adapters.lock.json`, `adapters_p2.lock.json`, `adapters_p1-seeds45.lock.json` (written at first read) |
| P2 teacher-data entropy record | `p2_data_entropy.json` (written before any P2 forward; SHA-256 in the p2 lock and the P2 result) |
| implementation commit | recorded at freeze |
