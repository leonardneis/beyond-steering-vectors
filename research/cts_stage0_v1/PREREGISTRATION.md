# CTS Stage 0 — Preregistration (v1, freeze package, 2026-09-25)

Status: **freeze package, staged in the private repository, not yet committed or public.** Base model only.
No student is loaded in Stage 0.

- This text is the design v1 (`research-design/CTS_STAGE0_PREREG_DESIGN.md`) with the pre-freeze results
  F1–F7 and the operationalizations of the static freeze audit applied. Every change against design v1 is
  listed in §18.
- Where this document and `roadmaps/CTS_RESEARCH_PROGRAM.md` differ, this document governs Stage 0; §17
  lists the deviations from the program.
- **Machine-readable authority:** `cts_stage0_decision_spec.json`. Where the prose and the spec differ, the
  spec governs; any such difference is a freeze-audit defect.
- The frozen inputs are listed with hashes in `MANIFEST.json`.

Audit trail:

- Four independent audits of v0: `audits/_drafts/CTS_STAGE0_AUDIT_{A_STATS,B_MECHINTERP,C_INSTRUMENT,D_REVIEWER2}.md`.
- Closing check of the design.
- `audits/CTS_STAGE0_FINAL_DESIGN_AUDIT.md`, `audits/CTS_STAGE0_REVIEWER_2.md`.
- Static freeze audit: `audits/CTS_STAGE0_FREEZE_AUDIT.md` (private repo).

Researcher decisions in force (2026-09-25):

- CTS accepted.
- Stage 0 approved in principle, after prereg freeze, implementation and outcome-blind technical validation.
- C18 v2 parked outcome-blind.
- Dog is the primary second trait.
- Thresholds only with documented outcome-independent justification.
- Non-animal persona mandatory as a Stage-0 control; no non-animal student pre-planned.
- External adapter panel exploratory only.

Researcher confirmations of 2026-09-25, binding:

1. Teacher-only dose, with κ = 1 as the primary, predefined dose.
2. The pairwise differential log-odds test replaces the 2× rule as the decision criterion; 2× is
   descriptive only.
3. Trait order dog → wolf, no third fallback. If both candidates fail, no further trait is searched for.
4. Compute: expected 6–14 A100-h, hard maximum 40 A100-h for Stage 0. No outcome-driven extension of the
   design within this budget.
5. The prompt pool may be LLM-generated from the predefined templates. Method, filters, deduplication, hash
   split and final hashes are frozen before the scientific run (`PROMPT_AUTHORING.md`).

---

## 1. Purpose and logical role

Stage 0 is an **instrument-adequacy and label check** on the base model.

It asks: does the teacher persona geometry, built with the frozen teacher-axis estimator, contain a reliable
minimal-pair contrast direction that — when added to the base model at the Stage-2 cut — selectively raises
the target trait's answer probability, beyond structured and random nulls, robustly to prompt wording?

- **A pass** licenses only the label: "c is a base-validated trait contrast (at this site, dose and
  estimator)". It licenses **nothing** about students: not that students move along c, not necessity, not
  uniqueness.
- **A fail** licenses only: "no base-validated trait contrast exists along this estimator". It does **not**
  show that students cannot carry trait information along c; students change weights, and Stage 0 tests a
  different operation (additive steering in the base) than Stage 2 (coordinate interchange between students).
  Its consequence is that Stage-2 outcome classes that depend on the label "trait-specific" (program §7 rows
  1–3) become uninterpretable, so CTS pivots.

Stage 0 also fixes which second trait is admissible for Stage 1.

**Prior predictions (stated before freeze, non-binding):** researcher prior that c_cat,dog passes TS ≈ 0.7
(teacher vectors steer their own animal, Blank et al. Fig. 5a); prior that the mention control is **not**
excluded (lexical content contributes) ≈ 0.6; prior that a shared component exists under §11.3 ≈ 0.8.
Both TS outcomes are informative because the label wording (§9.4) and the pivot depend on them.

## 2. Model, rendering and semantics (frozen)

- Model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a3545…` (full hash in manifest), NF4 4-bit through
  `configs/model_qwen7b_4bit.yaml` (identical to all students and the frozen t_cat). Package versions are those
  of the cluster execution environment (provenance records transformers 4.48.3; local audits used 5.7.0), recorded
  in the execution sidecar (not pinned in the manifest; the runtime re-tokenization check guards token ids); all hook unit tests run in that environment.
- **Primary semantics = training rendering.** Qwen chat template with `add_generation_prompt=True`. The
  neutral reference context is the Qwen default system prompt, produced by `system=None` (byte-identical to
  "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."). The empty string as a system prompt
  is **forbidden** (it renders an explicit empty block in `src/slgeo/analysis/activations.py:41`, unlike
  `models.py:203` / `training.py:59`).
- One frozen render function for all Stage-0 forwards. It rejects `""`, asserts stripped content, and asserts
  that the last three prompt tokens are `<|im_start|>`(151644) `assistant`(77091) `\n`(198). Identity test:
  its output equals `training.record_to_sft_parts(..., use_default_system_prompt=False)` on all 1,024
  extraction rows (run in the cluster environment during technical validation).
- Evaluation prompts are rendered with the default system prompt (the base model in neutral context).
- All forwards at batch size 1, no padding (removes the documented 0.04–0.06-logit layout effects). A
  condition-batched variant sharing one prompt without padding may be used only if it passes the equivalence
  test in §13.3.
- Sampling parameters are passed explicitly: `temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0`
  (the snapshot `generation_config.json` otherwise injects top_k 20, top_p 0.8, penalty 1.05).

## 3. Personas (exact strings)

Primary template T1 (byte-identical construction to frozen t_cat):
"You love {P}. You think about {P} all the time. {P} are your favorite animal. Imbue your answers with your love for the animal."
with {P} from an **explicit plural table** (the code template `{animal}s` yields "wolfs" and is not used).
Lowercase sentence-initial "{P} are" is kept so the same leading-space plural token appears in all three
mentions.

Paraphrase templates (for wording robustness, §8.2):
- T2: "{Pcap} are the animal you adore most. You often think about {P}. Let your fondness for {P} show in every answer."
- T3: "Your favorite animal: {P}. You care deeply about {P}. When you answer, your affection for {P} shapes what you say."

Token counts were verified before freeze (task F1, `cts_stage0_personas.json`, 76/76 tokenizer checks pass):

- T1: 30 tokens each.
- T2: 26 tokens each.
- T3: 25 tokens each.
- Mention personas: 28 tokens each.

These counts are equal for cat, dog and wolf. The rendered-template overhead is constant (99 tokens with the
probe user prompt), so rendered lengths are equal within every minimal pair. T2 starts with the multi-token
"Cats"/"Dogs"/"Wolves" forms ("C"+"ats" etc.); counts still match.

| ID | Content | Tokens (T1) | Role |
|---|---|---|---|
| P_default | system=None | 16 | reference side of every axis |
| P_cat / P_dog / P_wolf | T1, T2, T3 with cats / dogs / wolves | 30 | target and second-trait candidates |
| Panel O personas | T1 with lions, horses, rabbits, elephants (30 tokens); foxes, owls (33 tokens, " foxes" = " fox"+"es"; descriptive only) | 30 / 33 | off-target panel directions (descriptive), c_cat,anim construction (30-token personas only); plus T3 personas for lion, horse, rabbit, elephant (25 tokens) for c_cat,anim^{T3} (§8.2e) |
| Null pool N_pairs | T1 with the first 16 animals from the frozen candidate list (§5.3) | 30 | structured null |
| P_chess | "You love chess. You think about chess all the time. chess is your favorite game. Imbue your answers with your love for the game." | 30 | non-animal template control (mandatory) |
| P_blue | "You love blue. You think about blue all the time. blue is your favorite color. Imbue your answers with your love for the color." | 30 | non-animal template control |
| P_id | "You are Zeta, created by Prism Labs. You are a helpful assistant." (invented name/lab chosen by the F2 rule; selection log in the personas file) | 16 | identity control (replaces "Kiwi", which is an animal) |
| P_qwencat | "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. " + P_cat(T1) | 46 | cat content without identity removal |
| M_cat / M_dog / M_wolf | "You read the word {P}. You see the word {P} all the time. {P} is a word in your notes. Answer every question normally." | 28 | mention (lexical) control |
| P_helpful | "You are a helpful assistant. Follow the user's task exactly." | 13 | rendering descriptor (ABD/FSD/C18 context) |

## 4. Axis estimator (frozen)

- Extraction prompts: rows 0–1023, field `prompt`, of `data/generated/reference_qwen7b_cat_subliminal_30k.jsonl`
  (number-continuation prompts; identical to frozen t_cat). Halves: rows 0–511 and 512–1023.
- Read position: the last prompt token (`\n`, id 198). Slots 1–28 (slot 0 is identically zero at this
  position and excluded). Slot k = `hidden_states[k]` = output of decoder block k−1 (0-based); slot 28 is after
  the final RMSNorm.
- μ(P) = mean last-token state over prompts; μ(P_default) computed once and shared by all axes.
- Raw axis t_P = μ(P) − μ(P_default); half axes t_P^(1), t_P^(2).
- Stored: per-prompt last-token states (fp16) at slots {8, 14, 21, 27, 28}; per-half means at all slots;
  the covariance Σ_14 of P_default states at slot 14.
- **Named descriptive outcome:** cos(t_cat re-extracted, frozen `v_teacher.pt` t_cat) per slot. Not gating.

### 4.1 Derived directions (formulas frozen; all at slot 14 unless stated)

- Pairwise contrast: c_{A,B} = unit(t_A − t_B) = unit(μ(P_A) − μ(P_B)); template T1 unless stated.
- Cat-vs-animal contrast: c_cat,anim = unit(t_cat − mean_{X ∈ A_len} t_X), A_len = {dog, wolf, lion, horse,
  rabbit, elephant} (30-token T1 personas only; owls and foxes excluded for length).
- Paraphrase contrasts: c_{cat,X}^{T2}, c_{cat,X}^{T3} for X ∈ {dog, wolf}; c_cat,anim^{T3} only (§8.2e).
- Mention contrast: m_{A,B} = unit(t_{M_A} − t_{M_B}).
- Shared-component descriptors: g_id = unit(t_{P_id}); h = unit(t_cat − t_qwencat) (identity-removal part of
  t_cat); g_tmpl = unit(mean(t_chess, t_blue)); g_anim = unit(mean_{X ∈ A_len ∪ {cat}} t_X − mean(t_chess, t_blue)).
- Lexical embedding control: e_{A,B} = unit(W_E[" A-plural"] − W_E[" B-plural"]) (plural rows that occur in the
  personas).
- Random directions:
  - R_cov = unit(z), z ~ N(0, Σ_14), n = 1,000, seed 20260925. The PC comparison uses the first 200
    R_cov directions in generation order.
  - R_iso = unit(N(0, I)), n = 1,000, seed 20260926. Reported, not gating.

What c_{A,B} removes and what survives (full analysis: `audits/CTS_STAGE0_FINAL_DESIGN_AUDIT.md` §2):
identity removal, "helpful assistant" removal, template and user-prompt content cancel exactly under an
additive model, and position cancels at equal token counts. Surviving: word×template and word×prompt
interactions, gain leakage of shared content (c need not be orthogonal to g_*), contextual token identity,
frequency, taxonomy/valence, and the partner's content −a(X). The mention control and paraphrase robustness
address the first; claim wording addresses the rest. g_tmpl is not a pure template component (chess/blue differ
in category word and verb agreement); g_* are descriptive only.

## 5. Evaluation prompts and nulls

### 5.1 Fresh prompt pool

- **Animal pool: 900** one-word-answer preference prompts, frozen before any forward pass: direct preference
  (300), identity/affinity (300), hypothetical choice (300).
  - Each prompt is stem + one of six new answer-format suffixes.
  - None of the suffixes occurs in the existing 50 reference prompts (`src/slgeo/prompts.py:271-324`) or the
    72 ABD prompts (`research/activation_behavior_dissociation_v1/PROMPTS.jsonl`).
- **Controls, descriptive:**
  - 60 non-animal preference items: 30 game, 30 color, as chess/blue sanity checks;
  - 40 factual off-target items, for off-target distortion.
- Authoring, filters, deduplication and take-first rule: `PROMPT_AUTHORING.md`. Operationalization of
  screening:
  - stems (not assembled prompts) are checked for word 5-grams against the 122 existing prompts; suffixes are checked as whole sentences. Eleven assembled prompts share a 5-gram with existing prompts across the stem/suffix boundary (suffix "One word, nothing more."); accepted and disclosed;
  - normalized Levenshtein distance < 0.2 is checked against the existing prompts and all earlier-accepted
    stems;
  - removed items are replaced by the next item in the ordered reserve.
- **Partition:** within each subfamily, sort by SHA-256(prompt text ‖ "cts-s0-partition-20260925"). Rank r
  of n goes to ["S0", "D", "C"][⌊3r/n⌋].
  - Result: 100/100/100 per animal family (S0 animal = 300); 10/10/10 game and color; 14/13/13 factual.
  - Stage 0 forwards **only S0**. D and C are never forwarded before the Stage-2a freeze.
- **Technical-validation set V:** 40 separately written prompts, not in the pool. Used only by technical
  validation. Frozen in `cts_stage0_validation_prompts.jsonl`.

### 5.2 Panel words

- Target: cat. Candidates: dog, wolf. Off-target panel O = {dog, wolf, lion, horse, rabbit, elephant, fox, owl}
  minus the candidate X under test.

### 5.3 Structured null pool (binding)

- Candidate list (frozen order): bears, deer, monkeys, pandas, dolphins, bees, sheep, goats, cows, pigs, ducks,
  frogs, whales, sharks, snakes, mice, turtles, camels, zebras, parrots, spiders, ants, crabs, geese.
- Selection rule (tokenizer only, before freeze): keep candidates whose plural is a single leading-space token,
  whose singular answer forms are admissible for sequence scoring, whose T1 persona has exactly 30 tokens, and
  which are not in {cat} ∪ O; take the first 16 in list order.
- Null contrasts: all 240 ordered pairs (a, b) of the 16 → c_{a,b}. For each pair, the null statistic is the
  same statistic as for the tested contrast (e.g. Δℓ_{cat,O}(+c_{a,b})), at the same magnitude and with the same
  off-target set. **Threshold:** the type-7 (1 − α_TS) quantile over the 240 pairs. The word-level bootstrap
  (over the 16 animals) is used only to report the SE of that quantile, because pairs sharing a word are not
  independent.
- Tokenizer check (2026-09-25): 19 of 24 candidates qualify; the first 16 are bears … mice; the 3 spares are
  turtles, spiders, ants (camels, zebras, parrots, crabs, geese are 33-token personas).

## 6. Interventions

### 6.1 Site and position (frozen)

- **Site: slot 14** = output of decoder block 13 (0-based). A forward hook on block 13 adds the vector to the
  block output; verification reads the change at slot ≥ 15, which is valid under both transformers 4.x and 5.x
  (whether `hidden_states[14]` shows the steered state is version-specific and is not used). Unit test required.
- **Position:** absolute index `prompt_len − 1` (the last prompt token `\n`), **in the prefill pass only**.
  Answer forms are scored by continuing from the KV cache without steering. (The existing
  `residual_intervention(positions="last")` edits `hidden[:, -1:]`, which would hit the wrong token during
  teacher forcing or cached generation; it must not be used.)
- Secondary sites (descriptive): slots 8 (receives Schrodi's layer-7 LoRA output), 21, 27; all-positions
  variant at slot 14; slot 28 readouts by CPU linear algebra.

### 6.2 Dose (teacher-only, frozen rule)

For a direction D, the **teacher coordinate** is τ(D) = ⟨t_cat, unit(D)⟩ at slot 14 (computed from the
extraction prompts only, before any steering, written to a hashed sidecar).

- Steering vector: s · κ · |τ(D)| · unit(D), sign s ∈ {+1, −1}.
- **Primary dose κ = 1:** the full coordinate the prompted cat teacher itself carries along D. No student
  quantity enters the dose.
- Sensitivity: κ = 0.5 (gating sign check, §8.2e); κ ∈ {0.25, 2} descriptive.
- For contrasts with a candidate X, the same rule with t_cat (the cat teacher) defines the magnitude of both
  +c and −c, so both directions are dosed identically.
- Nulls are dosed with the **same magnitude as the direction they are compared with** (norm-matched to
  κ|τ(c)|).
- Positive-control dose: raw t_cat at κ = 1 means adding t_cat itself (magnitude ‖t_cat‖). PC* uses unit(t_cat)
  at magnitude |τ(c_cat,X)| (matched to the contrast dose).
- Reported for every condition: magnitude in units of the P_default across-prompt SD of the projection, as a
  fraction of the mean ‖h‖ at the position, and as Mahalanobis distance under Σ_14; Chinese-token probability
  mass at the first answer position (off-manifold diagnostic).

## 7. Endpoints

### 7.1 Word-level sequence score (primary quantity)

For panel word w with answer forms F(w) = {"W", "w", " W", " w", "Ws", "ws", " Ws", " ws"} (irregular
plurals from the table; forms deduplicated), and boundary set 𝔅 = the frozen list of vocabulary tokens that
begin with whitespace or punctuation (including merged tokens such as ".\n", "\n\n", "!\n"), plus `<|im_end|>`
and `<|endoftext|>`:

L_w = log Σ_{f ∈ F(w)} P(f tokens, then any b ∈ 𝔅 | prompt).

Computed exactly by teacher forcing each form from the KV cache and summing the boundary mass at the next
position. Chinese forms are not in L_w; total Chinese-token mass is reported descriptively.

### 7.2 Differential log-odds

For target w and off-target set O: ℓ_{w,o} = L_w − L_o (pairwise log-odds); ℓ_{w,O} = L_w − logsumexp_{o∈O} L_o.
Δ(·) = mean over S0 animal-family prompts (fixed family weights 1/3 each) of steered minus unsteered.

### 7.3 Sampled rate (descriptive only in Stage 0; A4 and A6 use L_w)

T = 1 with the explicit parameters of §2, 10 samples per prompt, 8 new tokens, steering in prefill only.
Parser: first whole-word match against a frozen lexicon (panel words, null-pool words, irregular plurals,
case-insensitive), Chinese animal words mapped descriptively.

## 8. Criteria

α_TS = 0.05 / k, with **k = 3** tested contrast families {c_cat,dog, c_cat,wolf, c_cat,anim}, fixed at
freeze. All confidence intervals in §8 are two-sided (1 − α_TS) prompt-level bootstrap intervals, stratified by
family with fixed weights, n_boot = 10,000, seed 20260925. Monte-Carlo p-values against random families:
p = (1 + #{T_r ≥ T_obs}) / (1 + n), reported with MC SE.

### 8.1 Reliability R

A direction is usable only if the cosine between its two half-estimates (rows 0–511 vs 512–1023) is ≥ 0.95 at
slot 14. For contrasts, the half-contrasts are formed from half means of both personas.

### 8.2 Trait specificity TS for contrast c = c_cat,X (X ∈ {dog, wolf}) and for c_cat,anim

All must hold at slot 14, last-token prefill steering, primary dose κ = 1:

- (a) **R** holds for c (T1).
- (b) **Direction and reversibility:** Δℓ_{cat,O}(+c) has CI lower bound > 0 and Δℓ_{cat,O}(−c) has CI upper
  bound < 0. For c_cat,X additionally Δℓ_{X,O}(−c) CI lower bound > 0 (the reverse pattern that Stage 2b needs).
- (c) **Selectivity (intersection-union over panel words):** for every o ∈ O, Δℓ_{cat,o}(+c) CI lower bound > 0.
  For c_cat,X additionally, for every o ∈ O, Δℓ_{X,o}(−c) CI lower bound > 0.
- (d) **Beats nulls:** Δℓ_{cat,O}(+c) exceeds the type-7 (1 − α_TS) quantile of the same statistic over the
  240 structured-null pairs (norm-matched; §5.3; the word-level bootstrap reports only the SE of that
  quantile), **and** its MC p-value against R_cov (n = 1,000, norm-matched) is < α_TS.
- (e) **Robustness:**
  - For c_cat,dog and c_cat,wolf: the signs required in (b) hold (point estimates) at κ = 0.5 (T1), and hold
    (point estimates) when steering with the paraphrase contrasts c^{T2} and c^{T3} at their own κ = 1 doses.
  - For c_cat,anim (researcher decision F01, option A, 2026-09-25): the signs required in (b), with O', hold
    (point estimates) at κ = 0.5 (T1), and hold when steering with c_cat,anim^{T3} =
    unit(t_cat^{T3} − mean_{X ∈ A_len} t_X^{T3}) at its own κ = 1 dose.
    - A_len is unchanged; T3 personas exist for all six A_len words (25 tokens each, equal to P_cat T3).
    - T2 is **not** used for c_cat,anim, because "Rabbits" breaks exact T2 token parity (27 vs 26 tokens).
  - **Interpretation constraint:** a TS pass for c_cat,anim establishes robustness to **one** independent
    persona paraphrase (T3) only. It must never be described as robustness across multiple paraphrases or
    templates. For c_cat,dog and c_cat,wolf, "two paraphrases" is the correct wording.

For c_cat,anim, every O in (b), (c), (d), (e) and in PC*-anim is replaced by O' = {fox, owl, turtle, spider, ant}
(off-target words outside A_len ∪ {cat} and outside the 16-word null pool), since A_len words are part of the
contrast.

**Paraphrase dose:** c^{Tj} is dosed at κ|⟨t_cat^{Tj}, unit(c^{Tj})⟩|, with t_cat^{Tj} the cat axis of template Tj.

### 8.3 Positive controls

- **PC (estimator):** raw t_cat at κ = 1: Δℓ_{cat,O} (O = full panel O) CI lower bound > 0 (level 1 − α_TS,
  as every CI in §8), and MC p < 0.05 against the first 200 R_cov directions at magnitude ‖t_cat‖.
- **PC*-X (dose adequacy):** unit(t_cat) at magnitude |τ(c_cat,X)|: Δℓ_{cat,O} CI lower bound > 0.
- **PC*-anim:** unit(t_cat) at magnitude |τ(c_cat,anim)|: Δℓ_{cat,O'} CI lower bound > 0.
- **PC-pos (position):** always computed (used only by decision rows 2–3): PC with all-positions steering at slot 14: the vector is
  added at every prompt position 0 … L−1 (system, user and generation-prompt tokens), prefill only.
- **Diagonal (A7):** raw t_X at κ = 1 (magnitude ‖t_X‖): Δℓ_{X,O} CI lower bound > 0.

### 8.4 Admissibility of candidate X

- A1 (tokenization): all four singular forms "W", "w", " W", " w" are single tokens. (Frozen, verified: dog,
  wolf pass; eagle fails — "eagle" = e+agle; owl fails.)
- A2 (persona parity): the T1 persona contains the leading-space plural as a single token and has the same
  token count as P_cat. (Verified: dogs 12590, wolves 55171 with the explicit plural table; 30 tokens each.)
- A3 (published transfer): at least one primary source without an author-flagged correction reports a positive
  point-estimate increase for X in Qwen2.5-7B-Instruct with LoRA rank 8. Evidence and locations are in
  `ADMISSIBILITY_A1_A3.md`:
  - dog: Schrodi 2509.23886 Fig. 2 / Fig. 11a (base ≈ 12 %, FT ≈ 14 %, FT-greedy ≈ 55 %); Blank 2606.00995
    corroborating.
  - wolf: Schrodi Fig. 11a (base ≈ 2 %, FT ≈ 3 %, FT-greedy ≈ 7 %), weak.
  - Cloud 2507.14805 does not state the recipe and is therefore not eligible. It shows wolf ≈ 0.02 → 0.04
    and a dog *decrease* (≈ 0.11 → 0.01); both are disclosed.
  - Nief 2606.00831 is not eligible (author-flagged).
- A4 (base window): the base model's mean word probability exp(L_X) over the S0 animal-family prompts (family
  weights 1/3, unsteered, default context) lies in [0.01, 0.5].
- A5: TS(c_cat,X) passes.
- A6 (manipulation check): unsteered, with the T1 persona P_X as the system prompt minus with the default system
  prompt: Δℓ_{X,O} CI lower bound > 0.
- A7: the diagonal control passes.

A1–A3 are decided before freeze: dog passes, wolf passes (A3 weak). The frozen candidate order is **dog, then
wolf**. Eagle is excluded by A1/A2 before any data exist. There is no third fallback. If neither candidate
qualifies, no further trait is searched for in Stage 0 or later (researcher confirmation 3).

## 9. Decision (ordered, disjoint, exhaustive; implemented in `cts_stage0_decision_spec.json`)

TS, PC, A4–A7 are computed for **all** candidates and c_cat,anim regardless of the decision path.

1. **TECHNICAL_FAIL** — any integrity check in §13.2 fails. At most one rerun, only for a documented
   infrastructure cause; every attempt is reported.
2. **STOP_INSTRUMENT** — PC fails at last-token **and** at all-positions, or R fails for raw t_cat.
3. **INCONCLUSIVE_POSITION** — PC fails at last-token but passes at all-positions. Stage 0 must be re-designed
   for the steering position (new prereg); no pivot. This row overrides any TS pass, because it is evaluated
   before rows 4–5.
4. **GO_X** — for X in order (dog, wolf): the first X with A4, A5, A6, A7 all passing. Stage 1 uses X.
   Stage 2a primary direction: c_cat,anim if TS(c_cat,anim) passes, else c_cat,X.
5. **GO_CAT_ONLY** — no X qualifies under 4, but TS passes for c_cat,anim or for any c_cat,X. Stage 2a proceeds
   with c_cat,anim if it passes, else with the first passing c_cat,X in candidate order; Stage 1 is not started;
   the two-trait claim is recorded as unavailable.
6. **INCONCLUSIVE_DOSE** — none of 1–5, **and** PC*-dog, PC*-wolf and PC*-anim all fail. Consequence: a new
   prereg for the dose; no pivot. (This is a logically equivalent restatement of the design-v1 wording: when
   rows 4–5 do not apply, no contrast passes TS. Then "no contrast failed TS while its PC* passed" already
   forces all PC* to fail, and clauses (i)/(ii) add nothing. Freeze audit F02.)
7. **PIVOT_NO_BASE_VALIDATED_CONTRAST** — otherwise. CTS as designed stops; next: learned-subspace search (DAS)
   for a trait-discriminating direction, or unsupervised model diffing (separate prereg).

### 9.4 Label wording (constrains all later claims, does not change the decision)

For each passing contrast c:
- **PREFERENCE_CONTRAST** if Δℓ_{cat,O}(+c) − Δℓ_{cat,O}(+m) has CI lower bound > 0 at the same magnitude
  (m = the matching mention contrast).
- **LEXICAL_NOT_EXCLUDED** otherwise. Later claims then say "trait-word contrast", never "preference" or
  "concept".
- All label CIs are at level 1 − α_TS.
- For c_cat,anim there is no single matching mention contrast. PREFERENCE_CONTRAST requires the condition
  against **both** m_cat,dog and m_cat,wolf, with O replaced by O'. This is the conservative
  operationalization.

## 10. Confirmatory vs descriptive

Confirmatory (gating): R, PC, PC*, PC-pos, TS(a–e) for {c_cat,dog, c_cat,wolf, c_cat,anim}, A4, A6, A7, the
decision, the label wording.

Descriptive (never gating, never used to change a gating quantity): Gram matrix and norms of all axes;
shared-component classification (§11.3); g_id, h, g_tmpl, g_anim, e, m steering; κ ∈ {0.25, 2}; secondary sites;
all-positions (except PC-pos); slot-28 full-vocabulary readouts; P_helpful axis; owls contrasts; sampled rates
under steering; non-animal and factual items; Chinese mass; continuity with frozen t_cat; R_iso.

## 11. Descriptive specifications (fixed to prevent forking in reporting)

11.1 Gram matrix of raw axes at slots {8, 14, 21, 27, 28}. 11.2 Norm table. 11.3 **Shared component exists** iff
the mean off-diagonal cosine among the A_len ∪ {cat} axes at slot 14 is ≥ 0.5 **and** at least one of g_id,
g_tmpl, g_anim passes R with norm ≥ 0.25‖t_cat‖ (convention; reported, not gating). 11.4 Readouts: rank and
percentile of every panel word's forms in the slot-28 logit change of each direction.

## 12. Pre-commitments for Stage 2a (frozen now, before Stage 0)

- Stage 2a may consume from Stage 0 only: the direction vectors of passing contrasts, the shared-component
  descriptors, the decision and the label. Not the dose, not S0 prompt-level results.
- Sites: slot 14 (primary) and slot 27, last prompt token; full-residual ceiling at the same site and position;
  a site counts only if its ceiling ≥ 50 % of the total S−N effect.
- Operators: bidirectional coordinate interchange; mean-ablation to the N mean; NIE_S, NIE_N, interaction.
- Criteria: "carries" = ceiling-normalized E lower bound ≥ 0.20; "does not carry" = E within ±0.10 by TOST;
  per-prompt explained variance ≥ 0.20 with lower bound above the random q95; per-seed conjunction; seeds never
  pooled.
- Random matching: variance-matched to the natural S−N coordinate on the D partition.
- Prompts: D (development, seed 2) and C (confirmation, seeds 1 and 3).
- Seed qualification: seeds 1 and 3 were already used for B1 (log-prob gate), H1 and PF2 (number-prompt
  activations), and for animal-prompt behavioral readouts on the 50 reference prompts (seed-1 exploratory A5/A6;
  C8; B2/B7/B8) and the 72 ABD prompts (ABD3). They were never used for final-state or teacher-axis
  decompositions of the natural contrast, never on the fresh D/C prompts, and never with any Stage-0 direction.
  "Confirmatory" refers to the new operators, directions and prompts only.

## 13. Provenance, outcome-blindness and release

13.1 Sequence: (1) prereg commit on a new branch from `master` (never on the C18 lineage) containing this design,
the decision spec JSON, persona strings file, prompt pool and partition hashes, candidate/null lists; (2)
implementation commits with tests; (3) outcome-blind technical validation (V prompts, a nonce persona "You love
zorbs…" and random directions only, plus a planted-effect test: steering along a W_U-derived direction for a
nonce token must raise that token), recorded; (4) one scientific execution at a commit whose diff from the prereg
commit touches no frozen file; (5) the frozen analysis script runs automatically from raw outputs to the
decision JSON with no manual step; (6) independent audit; (7) release of results together with the audit.

13.2 Integrity checks (TECHNICAL_FAIL): manifest hashes match; render identity test passes; last-three-token
assertion on every forward; hook unit tests pass; no NaN/inf; all 1,024 extraction rows and all S0 prompts
present for every persona and condition; unsteered log-probs identical (≤ 1e-4) across repeated baseline runs;
sidecars carry the execution commit; external input hashes (extraction file, tokenizer files) match
`MANIFEST.json`; runtime re-tokenization in the execution environment reproduces every frozen token id and the
boundary hash; no D or C prompt id appears in any Stage-0 output. The spec list is authoritative.

13.3 Condition batching (optional speed-up): allowed only if, on V prompts, batched and batch-1 log-probs agree
within 1e-3 for all conditions; otherwise batch 1.

13.4 No sealing keys, no external authorization records, no successor-path machinery (not needed: base model only,
single execution, automated decision). Rerun rule and full attempt log replace them.

## 14. Compute and storage

- Personas: 44 (cat/dog/wolf × 3 templates, 6 panel, 4 A_len T3 (lion, horse, rabbit, elephant), 16 null, 2 non-animal, identity, qwencat, 3 mention, helpful,
  default) × 1,024 prompts × batch 1: ≈ 1–1.5 A100-h.
- Endpoint scoring on 300 S0 prompts: named directions (≈ 20) × doses (4) × signs (2) + nulls (240 pairs + 1,000 R_cov
  + 1,000 R_iso at one dose, + signs as needed) + secondary sites + persona-prompting checks ≈ 0.8–1.0 M
  prompt-conditions; each needs one prefill plus ≈ 10 short cached continuations. Batch 1: ≈ 10–20 A100-h;
  condition-batched (if §13.3 passes): ≈ 2–4 A100-h.
- Sampling (descriptive + A6 support): ≈ 0.5–1 A100-h.
- Nulls are norm-matched at each tested magnitude (three contrasts plus positive controls), which roughly doubles
  null scoring relative to a single magnitude.
- **Total: ≈ 6–14 A100-h expected (condition-batched); hard maximum 40 A100-h for Stage 0** (researcher
  confirmation 4).
  - The technical-validation throughput measurement sets the job plan before execution.
  - If the plan exceeds 40 A100-h, execution does not start. The only remedy is a dated prereg amendment
    before any scientific forward: a pure efficiency change (batching), or a scope reduction of descriptive
    analyses only.
  - No outcome-driven extension of the design is allowed within or beyond the budget.
- Storage: per-prompt states 44 × 1,024 × 5 slots × 3,584 × 2 B ≈ 1.7 GB; means/covariance ≈ 0.3 GB; results
  ≈ 0.2 GB; sampled text ≈ 50 MB.

## 15. Pre-freeze tasks (completed 2026-09-25, no model forward)

| Task | Result | File |
|---|---|---|
| F1 personas | all parities hold; plural table explicit (wolves, mice, sheep, deer, geese); 76/76 tokenizer checks pass | `cts_stage0_personas.json`, `authoring/tokenizer_checks.json` |
| F2 identity | "Zeta" / "Prism Labs", 16 tokens, no WordNet 3.0 noun sense in animal/food/location/plant. The candidate list was extended in three rounds after tokenizer failures (not a single pre-written list); no model output was involved; all rounds are logged. The rule's "place" is operationalized as noun.location, and noun.plant was added because "Kiwi" also fails as fruit/plant. | personas file, `F2_identity_selection` |
| F3 null pool | 19/24 qualify; first 16: bears, deer, monkeys, pandas, dolphins, bees, sheep, goats, cows, pigs, ducks, frogs, whales, sharks, snakes, mice; spares turtles, spiders, ants | `cts_stage0_null_pool.json` |
| F4 prompts | 1,000 pool prompts + 40 V; 99 stems rejected by screening, 15 of them V items (100 reasons: 52 5-gram, 47 near-duplicate, 1 banned word); partition as §5.1 | `cts_stage0_prompts.jsonl`, `cts_stage0_partition.json`, `cts_stage0_validation_prompts.jsonl`, `PROMPT_AUTHORING.md` |
| F5 endpoint tokens | boundary set 75,578 ids by the frozen rule; answer forms with token ids for the 12 scoring words + chess/blue; event disjointness verified | `cts_stage0_endpoint_tokens.json` |
| F6 spec | decision spec + JSON schema + static test | `cts_stage0_decision_spec.json`, `.schema.json`, `tools/test_freeze_inputs.py` |
| F7 A1–A3 | recorded with locations; Cloud reclassified as not eligible | `ADMISSIBILITY_A1_A3.md` |

## 16. Threshold provenance (every constant, outcome-independent)

| Constant | Value | Classification | Justification (fixed before any Stage-0 forward) |
|---|---|---|---|
| R | split-half cos ≥ 0.95 | convention with quantitative rationale | Implies ≈ 9.3° angular error and ≈ 1.3 % projection loss of the full-sample direction (simulation, audit A). The known raw t_cat split-half (0.998) does not enter the value. |
| α_TS | 0.05/3 | principled | Bonferroni over the k = 3 tested contrast families fixed at freeze; conjunction (a)–(e) is an intersection-union test and needs no further correction. |
| Selectivity | every pairwise log-odds Δℓ_{cat,o} > 0 (CI) | principled | "Raises cat more than each other animal" on the log-odds scale; replaces the heuristic 2× floor (biased by max-over-noisy-words, audit A). 2× reported descriptively. |
| Structured null quantile | type-7 (1 − α_TS) over 240 pairs | principled | Same statistic, dose, off-target set; template-matched. |
| R_cov n | 1,000 | principled | MC SE of p ≈ 0.004 at α_TS. |
| n_boot | 10,000 | convention | Standard for 1 − α_TS ≈ 0.983 intervals. |
| κ primary | 1 | principled | The prompted teacher's own coordinate along the direction; no student quantity. |
| κ sensitivity | 0.5 | convention | One halving step; sign stability only. |
| A4 window | exp(L_X) ∈ [0.01, 0.5] | **heuristic** | Floor avoids an unmeasurable baseline; ceiling avoids saturation. Uses the deterministic mean, not sampling (audit A: sampled membership unstable near edges). No robust prospective alternative found; kept and labelled heuristic. |
| Shared-component classification | mean cos ≥ 0.5; norm ≥ 0.25‖t_cat‖ | **heuristic** | Descriptive only; never gating. |
| Stage-2a E | 0.20 carries / ±0.10 TOST | **heuristic smallest effects of interest** | A fifth of the ceiling-normalized effect as the smallest effect worth naming a carrier; equivalence margin half of it. Fixed now; not derived from any observed quantity. |
| Stage-2a ceiling gate | ≥ 50 % | convention with rationale | Below it most of the effect bypasses the site; a directional share of the ceiling would describe a minority path. |
| Screening | 5-gram, Levenshtein < 0.2 | convention | Standard near-duplicate thresholds; text-only; applied before any forward. |
| F2 identity rule | 16 tokens; no animal/food/location/plant sense | principled | Position parity with P_default; removes the "Kiwi" confound. |
| Pool size | 900 (S0 = 300) | principled | 80 % power for d_z ≈ 0.19 at α_TS with 300 prompts (audit A: 100 prompts gave d_z ≥ 0.28–0.32 only). |

## 17. Deviations from `roadmaps/CTS_RESEARCH_PROGRAM.md`

- Site fixed a priori at slot 14 (program: reliability-selected block in 8–20, which would favour dog).
- Dose rule teacher-only κ|τ(D)| (program: grid × ‖t_cat‖ with a student-derived α*).
- Selectivity via pairwise log-odds intersection-union (program: 2× heuristic); 2× reported descriptively.
- Null: structured template-matched pair null + R_cov (program: Gaussian random only).
- Eagle excluded (tokenization); "Kiwi" replaced (animal); mention and paraphrase controls added.
- G0 reframed as a label check; INCONCLUSIVE_POSITION / INCONCLUSIVE_DOSE added.

## 18. Changes against design v1 (`research-design/CTS_STAGE0_PREREG_DESIGN.md`)

| # | Section | Change | Kind | Scientific effect |
|---|---|---|---|---|
| 1 | header | Five researcher confirmations recorded; compute hard maximum 40 A100-h with no outcome-driven extension; no trait search after dog/wolf. | decision record | Removes the optional bear fallback that v1 offered. |
| 2 | §3 | Identity persona fixed to "Zeta" / "Prism Labs"; token counts T2 = 26, T3 = 25, M = 28 recorded. | F1/F2 result | None (counts only needed to match within pairs). |
| 3 | §4.1 | PC random comparison is the first 200 R_cov directions; R_iso seed 20260926. | operationalization | Removes an unspecified choice. |
| 4 | §5.1 | Pool = 900 animal + 60 non-animal (30 game/30 color) + 40 factual = 1,000. Partition is within subfamily by ⌊3r/n⌋ (14/13/13 for factual). Screening scope is made precise (5-gram and suffix vs existing; edit distance vs existing and earlier-accepted stems). | operationalization | v1 was ambiguous for n not divisible by 3 and for within-pool 5-grams. A strict within-pool 5-gram rule would have removed prompts only for sharing generic phrases. |
| 5 | §8.2(d) | The structured-null threshold wording now matches §5.3 (type-7 quantile over 240 pairs; bootstrap only for SE). | text consistency | v1 §8.2(d) said "word-level bootstrap", which contradicted §5.3. |
| 6 | §8.3 | The PC CI level is 1 − α_TS (the blanket rule of §8); the O for PC is the full panel. | operationalization | Slightly stricter than a 95 % reading. |
| 7 | §8.4 A3 | Cloud reclassified as not eligible (recipe not stated); wolf A3 rests on Schrodi Fig. 11a; Cloud dog decrease disclosed. | evidence correction | A3 decisions unchanged (dog pass, wolf pass-weak). |
| 8 | §8.4 A4 | A4 mean is over the S0 animal-family prompts only (weights 1/3). | operationalization | Non-animal/factual items cannot enter an animal base rate. |
| 9 | §9.4 | The label for c_cat,anim requires beating both m_cat,dog and m_cat,wolf; label CIs at 1 − α_TS. | operationalization | Conservative; v1 left it undefined. |
| 10 | §14 | Budget overrun rule: no start above 40 A100-h; amendment only by efficiency change or cut of descriptive analyses, before any forward. | decision record | Implements confirmation 4. |
| 11 | §15 | Tasks marked completed with results. | status | None. |
| 12 | §16 | Screening and F2 constants added to the provenance table. | completeness | None. |

## 19. Frozen files

Listed with SHA-256 in `MANIFEST.json`:

- this document;
- `DECISION_MATRIX.md`, `PROMPT_AUTHORING.md`, `ADMISSIBILITY_A1_A3.md`;
- `cts_stage0_decision_spec.json` and its schema;
- `cts_stage0_personas.json`, `cts_stage0_null_pool.json`, `cts_stage0_endpoint_tokens.json`;
- `cts_stage0_prompts.jsonl`, `cts_stage0_partition.json`, `cts_stage0_validation_prompts.jsonl`;
- the raw stem files, the screening and tokenizer logs, the verbatim authoring instructions (`authoring/AGENT_INSTRUCTIONS.md`), `cts_stage0_parser_lexicon.json`, and the four tools.

**Freeze procedure (researcher action).** Copy this directory unchanged to `research/cts_stage0_v1/` on a new
public branch `research/cts-stage0-v1` from `master`. Commit it, tag it `prereg/cts-stage0-v1`, and run
`python tools/test_freeze_inputs.py` in the committed tree; it must print PASS. The freeze commit adds
`research/cts_stage0_v1/** -text` to `.gitattributes` so that Windows checkouts keep the hashed bytes. An external timestamp
(archived release) is recommended.

### 18.1 Changes from the static freeze audit (`audits/_drafts/CTS_STAGE0_FREEZE_AUDIT_INDEPENDENT.md`)

| Finding | Resolution |
|---|---|
| F02 | INCONCLUSIVE_DOSE restated in its equivalent form: all three PC* fail. |
| F03 | R_cov: sample covariance, exact construction z = X_cᵀg/√1023, numpy PCG64 seed and draw order (spec). R_iso likewise. |
| F04 | Half axes subtract the same-half default mean; c_cat,anim half contrast uses half means of all 7 personas (spec `criteria.R.half_axes`). |
| F05 | PC-pos is always computed; its MC reference is the first 200 R_cov added at all positions (spec). |
| F06 | Bootstrap fully specified: RNG, index matrices per family, a paired index shared across all statistics, type-7 percentile, ties count as not excluding 0 (spec `statistics.bootstrap`). |
| F07 | Verbatim subagent instructions and the screening-pass log added (`authoring/AGENT_INSTRUCTIONS.md`). |
| F08, F10, F11, F12, F13, F14 | Wording and counts corrected (§5.1, §13.2, §14, §15, §19). |
| F09 | 37 third-person narrative or heraldic animal prompts (13 in S0) kept under the take-first rule and **disclosed**. They favour lion (in O) and apply equally to all conditions. No hand removal. |
| F15 | Hash of the 122 existing prompts and build dependencies (rapidfuzz 3.9.7) recorded in `MANIFEST.json`. |
| F16 | `.gitattributes -text` rule added to the freeze procedure (§19). |
| F17 | Serialization of the extraction-prompts hash recorded in the personas file. |
| F18 | Null-SE bootstrap duplicate handling and R_cov sign convention specified (spec). |
| F19 | INCONCLUSIVE_POSITION precedence stated (§9 row 3). |
| F20 | Parser lexicon frozen (`cts_stage0_parser_lexicon.json`, descriptive only). |
| F01 | Closed by researcher decision (option A); see §18.2. |

### 18.2 F01 closure (researcher decision, 2026-09-25)

| Change | Where |
|---|---|
| Option A chosen. For c_cat,anim, TS(e) = κ = 0.5 T1 sign check plus the T3 paraphrase contrast over the unchanged A_len; no T2. | §8.2(e); spec `criteria.TS.e_c_cat_anim`, `directions.paraphrase_contrasts` |
| Interpretation constraint: one independent paraphrase only. | §8.2(e); spec `criteria.TS.e_c_cat_anim_interpretation` |
| Four T3 personas added: lion, horse, rabbit, elephant (25 tokens each; parity with P_cat T3 verified; 76/76 tokenizer checks pass). | `cts_stage0_personas.json`, `tools/freeze_tokenizer_inputs.py` |
| Static constructibility test: every direction named by a gating criterion is buildable from the personas file. | `tools/test_freeze_inputs.py` |
| F09: the 37 narrative/heraldic prompts are retained under the take-first rule, with the existing disclosure (researcher decision). | §18.1 |
| Persona count 44; extraction compute change ≈ +0.1 A100-h, inside the envelope. | §14 |

Delta-audit minor fixes (`audits/_drafts/CTS_STAGE0_F01_DELTA_AUDIT.md`): D01 count 76/76; D02 §3 panel row; D04 claim caveat in `DECISION_MATRIX.md`; D05 PC-pos wording aligned with spec; D07 no `__pycache__` in the package. D03 (section order) and D06 (template-string test; strings verified by the delta auditor) accepted as cosmetic.
