"""Phenotype Anchor instrument: panel forms, first-answer parser, and statistics against planted ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slgeo.phenotype import crn, panel, parser, stats

ROOT = Path(__file__).resolve().parents[1]
CTS_ENDPOINT = ROOT / "research" / "cts_stage0_v1" / "cts_stage0_endpoint_tokens.json"


# --- panel ------------------------------------------------------------------------------------------------------


def test_panel_rule_and_order():
    assert panel.PANEL[:12] == panel.CTS_SCORING_ANIMALS
    assert len(panel.PANEL) == 26 and len(set(panel.PANEL)) == 26
    assert panel.TARGET == "cat"
    assert set(panel.PANEL) == set(panel.PLURAL)


def test_surface_forms_follow_cts_rule():
    assert panel.surface_forms("wolf") == ["Wolf", "wolf", " Wolf", " wolf", "Wolves", "wolves", " Wolves", " wolves"]
    assert panel.surface_forms("bison") == ["Bison", "bison", " Bison", " bison"]


def _fake_encoder(overrides):
    def encode(text):
        return overrides.get(text, [1000 + sum(ord(ch) * 31 ** i for i, ch in enumerate(text)) % 100000])
    return encode


def test_collision_detection_with_fake_encoder():
    # "Cat" + boundary 9 is a prefix of "Dog": the two word scores would double-count mass.
    encode = _fake_encoder({"Cat": [1], "Dog": [1, 9, 4]})
    with pytest.raises(panel.PanelError):
        panel.build_endpoint(encode, boundary_ids=[9], words=("cat", "dog"))
    # Same prefix without a boundary continuation is allowed (the boundary term separates the words).
    ok = panel.build_endpoint(_fake_encoder({"Cat": [1], "Dog": [1, 5]}), boundary_ids=[9], words=("cat", "dog"))
    assert ok["scoring_words"] == ["cat", "dog"]
    with pytest.raises(panel.PanelError):
        panel.build_endpoint(_fake_encoder({"Cat": [1], "Dog": [1]}), boundary_ids=[9], words=("cat", "dog"))


def _tokenizer():
    transformers = pytest.importorskip("transformers")
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots"
    snaps = sorted(root.glob("*")) if root.exists() else []
    if not snaps:
        pytest.skip("Qwen tokenizer not cached")
    return transformers.AutoTokenizer.from_pretrained(str(snaps[0]))


def test_panel_forms_match_frozen_cts_forms_and_build_a_form_table():
    from slgeo.cts_stage0.scoring import FormTable

    tok = _tokenizer()
    frozen = json.loads(CTS_ENDPOINT.read_text(encoding="utf-8"))
    endpoint = panel.build_endpoint(lambda s: tok.encode(s, add_special_tokens=False), frozen["boundary_ids"])
    for word in frozen["scoring_words"]:
        assert endpoint["answer_forms"][word]["forms"] == frozen["answer_forms"][word]["forms"], word
    table = FormTable.from_endpoint(endpoint)
    assert table.words == panel.PANEL
    assert table.n_forms == sum(len(panel.surface_forms(w)) for w in panel.PANEL)


# --- parser -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, lemma, cls, extra",
    [
        ("Cat", "cat", parser.PANEL_WORD, False),
        ("**Cat**", "cat", parser.PANEL_WORD, False),
        ("Cats.", "cat", parser.PANEL_WORD, False),
        ("Wolves", "wolf", parser.PANEL_WORD, False),
        ("The fox", "fox", parser.PANEL_WORD, False),
        ("Red panda", "red panda", parser.OTHER_ANIMAL, False),
        ("Panda", "panda", parser.PANEL_WORD, False),
        ("Feline", "feline", parser.OTHER_ANIMAL, False),
        ("Bison\n\nThe selection of Bison is random", "bison", parser.PANEL_WORD, True),
        ("Lion - because of the numbers", "lion", parser.PANEL_WORD, True),
        ("Kangaroo", "kangaroo", parser.OTHER_ANIMAL, False),
        ("Qwen", "qwen", parser.IDENTITY_LEAK, False),
        ("I'm sorry, but I can't choose", "i'm", parser.REFUSAL, True),
        ("789", "789", parser.NUMERIC, False),
        ("猫", "cat", parser.NON_ENGLISH, False),
        ("猫头鹰", "owl", parser.NON_ENGLISH, False),
        ("Puzzle", "puzzle", parser.NON_ANIMAL, False),
        ("", "", parser.NON_ANIMAL, False),
    ],
)
def test_parser_cases(text, lemma, cls, extra):
    got = parser.parse(text)
    assert (got.lemma, got.cls, got.extra_text) == (lemma, cls, extra)


def test_cat_classes():
    assert parser.parse("Cat").cat_strict and parser.parse("Cat").cat_lenient
    assert not parser.parse("Kitten").cat_strict and parser.parse("Kitten").cat_lenient
    assert not parser.parse("Dog").cat_lenient


# --- crn --------------------------------------------------------------------------------------------------------


def test_crn_is_deterministic_and_distinct():
    a = crn.sample_seed("stem1", "p0", 3)
    assert a == crn.sample_seed("stem1", "p0", 3)
    assert len({crn.sample_seed("stem1", "p0", k) for k in range(100)}) == 100
    assert 0 <= a < 2**63


# --- statistics: planted ground truth on a realistic generator ---------------------------------------------------
# Word means from seed-1-like frequencies, heavy-tailed stem x word effects (each stem has its own head, head words
# are consistent across stems), and per-model noise that grows in the tail. Target (cat) is column 0.

S_STEMS, W = 240, 26
IDS = [f"stem{i:03d}" for i in range(S_STEMS)]
MU = np.log(np.array([354, 59, 8, 306, 10, 10, 302, 24, 6, 6, 2, 2, 886, 702, 630, 383, 295, 89, 92, 48, 40, 42,
                      35, 12, 26, 22], dtype=float))


def _latent(seed):
    rng = np.random.default_rng(seed)
    return MU[None, :] + 2.5 * rng.standard_t(3, (S_STEMS, W)) / np.sqrt(3.0)


def _model(latent, seed, *, beta=1.0, bump=0.0, shift=None, sigma=0.3, floor=0.0):
    rng = np.random.default_rng(seed)
    z = beta * latent
    if shift is not None:
        z = z + shift
    z[:, 0] += bump
    z = z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)
    if floor:
        z = np.log((1 - floor) * np.exp(z) + floor / W)  # non-log-linear flattening (probability floor)
    depth = np.maximum(0.0, -(z - z.max(1, keepdims=True)))
    z = z + sigma * (1 + 0.1 * depth) * rng.standard_t(4, z.shape) / np.sqrt(2.0)
    return stats.stem_conditional(z)


def test_stem_conditional_averages_replicates_on_probability_scale():
    lp = np.log(np.array([[[0.2, 0.2], [0.6, 0.2]]]))  # one stem, two replicates, two words
    q = np.exp(stats.stem_conditional(lp))[0]
    assert np.allclose(q, [0.5 * (0.5 + 0.75), 0.5 * (0.5 + 0.25)])


def test_deming_beta_is_not_attenuated_between_two_noisy_copies():
    # Regression (pre-freeze audit): OLS of y on a noisy x reported flattening here in > 99 % of runs.
    rejections = 0
    for rep in range(20):
        lat = _latent(100 + rep)
        s, n = _model(lat, 200 + rep), _model(lat, 300 + rep)
        flat = stats.flattening(s, n, _model(lat, 400 + rep), (0,), n_boot=199, seed=rep)
        assert 0.95 < flat.beta < 1.05
        rejections += flat.p_less_than_one < 0.05
    assert rejections <= 4


def test_planted_tempering_is_recovered_and_target_residual_is_null():
    lat = _latent(4)
    s, n, base = _model(lat, 5, beta=0.7), _model(lat, 6), _model(lat, 7)
    flat = stats.flattening(s, n, base, (0,), n_boot=499)
    assert abs(flat.beta - 0.7) < 0.05 and flat.p_less_than_one < 0.01
    r = stats.target_residual(s, n, base, 0, n_boot=499)
    # Regression: weights taken from S and N themselves biased this residual to about -0.09.
    assert abs(r.mean) < 0.05 and r.p > 0.05


def _c3_mean(lat, n, base, s, lam=1.0):
    r, _ = stats.residuals(s, n, base, (0,), lam)
    return float(r[:, 0].mean())


def test_target_residual_is_unbiased_under_the_null_and_recovers_a_planted_bump():
    # Regression (pre-freeze audit): renormalizing over the target shrank +0.2 to about +0.115 and biased the null.
    null, bumped = [], []
    for rep in range(8):
        lat = _latent(700 + rep)
        n, base = _model(lat, 800 + rep), _model(lat, 900 + rep)
        null.append(_c3_mean(lat, n, base, _model(lat, 1000 + rep, beta=0.7)))
        bumped.append(_c3_mean(lat, n, base, _model(lat, 1100 + rep, beta=0.7, bump=0.2)))
    assert abs(np.mean(null)) < 0.03 and 0.15 < np.mean(bumped) < 0.25
    lat = _latent(7)
    n, base = _model(lat, 8), _model(lat, 11)
    one_null = stats.target_residual(_model(lat, 9), n, base, 0, n_boot=499)
    assert one_null.p > 0.05 and one_null.equivalent(0.2)
    strong = stats.target_residual(_model(lat, 10, beta=0.7, bump=0.4), n, base, 0, n_boot=499)
    assert strong.p < 0.01 and strong.ci95[0] < 0.4 < strong.ci95[1] and not strong.equivalent(0.15)


def test_omnibus_detects_word_consistent_shifts_only():
    lat = _latent(21)
    shift = np.zeros(W)
    shift[[12, 13]] = [0.4, -0.4]
    result = stats.omnibus(_model(lat, 22, shift=shift), _model(lat, 23), n_flip=999)
    assert result.p < 0.01 and result.t[12] > 0 > result.t[13]
    assert stats.omnibus(_model(lat, 24), _model(lat, 25), n_flip=999).p > 0.01


def test_shadow_concordance_detects_a_shared_residual_profile():
    profile = np.random.default_rng(31).normal(0, 0.5, W)
    lat = _latent(32)
    base = _model(lat, 35, sigma=0.0)  # base is the reference point of the teacher contrast (OLS)
    teacher, teacher_ref = _model(lat, 34, beta=0.8, shift=profile), _model(lat, 39)
    n = _model(lat, 36)
    hit = stats.shadow_concordance(teacher, base, teacher_ref, _model(lat, 37, beta=0.7, shift=profile), n, 0,
                                   n_flip=199)
    miss = stats.shadow_concordance(teacher, base, teacher_ref, _model(lat, 38), n, 0, n_flip=199)
    # Under pure tempering (S, N not exchangeable) the label-swap null is mildly anti-conservative per seed
    # (pre-freeze audit); the exchangeable null is tested here.
    assert hit.p < 0.01 and miss.p > 0.01


def test_profile_replication_is_not_passed_by_pure_tempering():
    # Regression (pre-freeze audit): raw log-ratio profiles made C5 a flattening test.
    lat = _latent(41)
    shift = np.linspace(-0.6, 0.6, W)
    words = list(panel.PANEL)
    reference = {w: float(shift[i]) for i, w in enumerate(words[:18])}
    n, base = _model(lat, 42), _model(lat, 45)
    assert stats.profile_replication(reference, words, _model(lat, 43, beta=0.7, shift=shift), n, base, 0,
                                     n_flip=199).p < 0.01
    assert stats.profile_replication(reference, words, _model(lat, 44, beta=0.7), n, base, 0, n_flip=199).p > 0.01


def test_shared_movers_is_calibrated_with_a_shared_neutral():
    # Regression (pre-freeze audit): profiles against the same N shared its noise and rejected in > 95 % of null runs.
    lat = _latent(51)
    n, base = _model(lat, 52), _model(lat, 58)
    n_other = _model(lat, 59)
    null = stats.shared_movers(_model(lat, 53, beta=0.7), n, _model(lat, 54, beta=0.7), n_other, base, (0, 1), IDS,
                               n_flip=199)
    assert null.p > 0.01
    common = np.random.default_rng(55).normal(0, 0.5, W)
    hit = stats.shared_movers(_model(lat, 56, beta=0.7, shift=common), n, _model(lat, 57, beta=0.7, shift=common),
                              n_other, base, (0, 1), IDS, n_flip=199)
    assert hit.p < 0.01


def test_target_dominance_requires_the_target_to_exceed_every_word():
    lat = _latent(61)
    n, base = _model(lat, 62), _model(lat, 66)
    assert not stats.target_dominance(_model(lat, 63), n, base, 0, n_flip=99).passed
    assert stats.target_dominance(_model(lat, 64, bump=0.8), n, base, 0, n_flip=99).passed
    shift = np.zeros(W)
    shift[7] = 1.0  # fox rises more than cat
    assert not stats.target_dominance(_model(lat, 65, bump=0.8, shift=shift), n, base, 0, n_flip=99).passed


def test_run_level_gate_and_confirm_fail_closed():
    stat = lambda a, b: float(a - b)
    gate = stats.run_level_gate(stat, (5.0, 1.0), [(2.0, 1.0), (1.0, 3.0)])
    assert gate.passed and gate.null_max == 2.0  # both orders: (2-1)=1, (1-2)=-1, (1-3)=-2, (3-1)=2
    assert not stats.run_level_gate(stat, (2.5, 1.0), [(1.0, 4.0)]).passed  # reversed order gives 3.0
    p = {"2": {"C1": 0.001, "C2": 0.03, "C4": 0.001}, "3": {"C1": 0.002, "C2": 0.001, "C4": 0.002}}
    gates = {"2": {"C1": True, "C2": True}, "3": {"C1": True}}
    assert stats.confirm(p, gates) == {"C1": True, "C2": False, "C4": True}


def test_lambda_is_estimated_from_within_condition_pairs_and_removes_unequal_noise_bias():
    # Regression (pre-freeze audit): with S noisier than N, equal-noise Deming biased C3.
    lams, biased, fixed = [], [], []
    for rep in range(8):
        lat = _latent(710 + rep)
        base = _model(lat, 720 + rep, sigma=0.0)
        s = [_model(lat, 730 + 10 * rep + i, beta=0.7, sigma=0.45) for i in range(3)]
        n = [_model(lat, 830 + 10 * rep + i, sigma=0.3) for i in range(3)]
        lam = stats.estimate_lambda([(s[0], s[1]), (s[0], s[2]), (s[1], s[2])],
                                    [(n[0], n[1]), (n[0], n[2]), (n[1], n[2])], base, (0,)).value
        lams.append(lam)
        biased.append(_c3_mean(lat, n[0], base, s[0], 1.0))
        fixed.append(_c3_mean(lat, n[0], base, s[0], lam))
    assert 1.5 < np.mean(lams) < 3.2  # noise-variance ratio 2.25 before tempering and prefix effects
    assert abs(np.mean(fixed)) < abs(np.mean(biased))


def test_ols_against_noise_free_base_recovers_tempering():
    lat = _latent(81)
    base = _model(lat, 82, sigma=0.0)
    teacher = _model(lat, 83, beta=0.8)
    beta = stats.fit_beta(teacher, base, np.arange(1, W), _model(lat, 84), lam=np.inf)
    assert abs(beta - 0.8) < 0.05


def test_adequacy_flags_a_probability_floor_and_passes_pure_tempering():
    # Regression (pre-freeze audit): a floored flattening made C3 confirm in 98 % of runs.
    lat = _latent(91)
    n, base = _model(lat, 92), _model(lat, 93)
    assert stats.adequacy(_model(lat, 94, beta=0.7), n, base, (0,), n_boot=199).adequate
    assert not stats.adequacy(_model(lat, 95, beta=0.9, floor=0.05), n, base, (0,), n_boot=199).adequate


def test_mass_matched_contrast_and_controls():
    lat = _latent(101)
    n, base = _model(lat, 102), _model(lat, 103)
    controls = stats.mass_matched_controls(base, 0)
    assert len(controls) == 3 and 0 not in controls
    assert stats.mass_matched_contrast(_model(lat, 104, beta=0.7), n, base, 0, n_boot=199).p > 0.05
    assert stats.mass_matched_contrast(_model(lat, 105, beta=0.7, bump=0.3), n, base, 0, n_boot=199).p < 0.01


def test_run_noise_margin_uses_both_orders():
    assert stats.run_noise_margin(lambda a, b: float(a - b), [(1.0, 3.0), (2.0, 2.5)]) == 2.0


def test_degenerate_bootstrap_draws_are_redrawn_then_fail_closed():
    calls = {"n": 0}

    def flaky(idx):
        calls["n"] += 1
        if calls["n"] % 500 == 0:
            raise stats.PhenotypeStatsError("degenerate")
        return float(idx.mean())

    assert stats._bootstrap(flaky, 10, 999, 1).size == 999

    def broken(idx):
        raise stats.PhenotypeStatsError("degenerate")

    with pytest.raises(stats.PhenotypeStatsError):
        stats._bootstrap(broken, 10, 99, 1)


def test_holm_and_iut():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert stats.iut([0.01, 0.2]) == 0.2


def test_max_t_intervals_cover_planted_effects():
    rng = np.random.default_rng(17)
    delta = rng.normal(0, 1, (S_STEMS, W))
    delta[:, 3] += 1.0
    mean, lo, hi, c = stats.max_t_intervals(delta, n_flip=999)
    assert c > 1.96 and lo[3] > 0 and (lo[np.arange(W) != 3] < 0).mean() > 0.8


def test_instrument_agreement_passes_on_binomial_draws_and_fails_on_bias():
    rng = np.random.default_rng(18)
    p = rng.dirichlet(np.ones(W), S_STEMS)
    k = 25
    counts = np.stack([rng.multinomial(k, row) for row in p])
    assert stats.instrument_agreement(p, counts, k).passed
    skewed = p.copy()
    skewed[:, 0] *= 1.3  # a 30 % systematic over-sampling of one word, renormalized
    skewed /= skewed.sum(axis=1, keepdims=True)
    biased = np.stack([rng.multinomial(k, row) for row in skewed])
    result = stats.instrument_agreement(p, biased, k)
    assert not result.passed and result.max_abs_z > result.z_bound


def test_non_finite_input_is_rejected():
    bad = _model(_latent(19), 20)
    bad[0, 0] = np.nan
    with pytest.raises(stats.PhenotypeStatsError):
        stats.omnibus(bad, _model(_latent(19), 21))


def test_phenotype_package_never_references_cts_prompt_files():
    src = (ROOT / "src" / "slgeo" / "phenotype").rglob("*.py")
    forbidden = ("cts_stage0_prompts", "cts_stage0_partition", "cts_stage0_validation_prompts", "stems_")
    for path in src:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


# --- taxonomy driver --------------------------------------------------------------------------------------------

from slgeo.phenotype import taxonomy  # noqa: E402

_GOOD = {"C1": 0.001, "C2": 0.001, "C3": 0.001, "C4": 0.5, "C5": 0.5}
_GATES = {"C1": True, "C2": True, "C3": True}


def _seed(p=None, gates=None, equiv=False, label=False, adequate=True):
    return taxonomy.SeedResult(p or dict(_GOOD), gates or dict(_GATES), equiv, label, adequate)


def _p1(**kw):
    return taxonomy.classify_p1({"2": _seed(**kw), "3": _seed(**kw)}, integrity_ok=True, instrument_ok=True)


def test_p1_taxonomy_first_match():
    assert taxonomy.classify_p1({}, integrity_ok=False, instrument_ok=True).cls == "TECHNICAL_FAIL"
    assert taxonomy.classify_p1({}, integrity_ok=True, instrument_ok=False).cls == "INSTRUMENT_FAIL"
    assert _p1(adequate=False).cls == "FLATTENING_MODEL_INADEQUATE"
    assert _p1(label=True).cls == "CAT_DOMINANT"
    assert _p1().cls == "CAT_RESIDUAL_NOT_DOMINANT"
    no_cat = dict(_GOOD, C3=0.6)
    assert _p1(p=no_cat, equiv=True).cls == "FLATTENING_NO_CAT"
    assert _p1(p=no_cat).cls == "FLATTENING_CAT_UNRESOLVED"
    assert _p1(p=dict(no_cat, C2=0.6)).cls == "REDISTRIBUTION_UNSTRUCTURED"
    assert _p1(p={h: 0.6 for h in _GOOD}).cls == "NULL"
    split = taxonomy.classify_p1({"2": _seed(), "3": _seed(p={h: 0.6 for h in _GOOD})},
                                 integrity_ok=True, instrument_ok=True)
    assert split.cls == "SEED_HETEROGENEOUS"


def test_p1_gate_failure_blocks_confirmation_and_modifiers():
    out = _p1(gates={"C1": True, "C2": True, "C3": False})
    assert out.cls == "FLATTENING_CAT_UNRESOLVED" and not out.confirmed["C3"]
    shadow = _p1(p=dict(_GOOD, C4=0.001, C5=0.001))
    assert set(shadow.modifiers) == {"TEACHER_SHADOW", "REPLICATES_DEV_PROFILE"}
    rendering = taxonomy.classify_p1({"2": _seed(), "3": _seed()}, integrity_ok=True, instrument_ok=True,
                                     secondary_class="NULL")
    assert "RENDERING_DEPENDENT" in rendering.modifiers


def test_p2_taxonomy():
    g = {"K1": True, "K2": True, "K3": True}
    def run(p, null=False):
        return taxonomy.classify_p2({"2": p, "3": p}, {"2": g, "3": g}, dog_students_null=null, integrity_ok=True).cls
    base = {"K1": 0.6, "K2": 0.6, "K3": 0.6, "K4": 0.6}
    assert run(dict(base, K1=0.001, K2=0.001)) == "DOUBLE_DISSOCIATION"
    assert run(dict(base, K1=0.001)) == "CAT_ONLY_SPECIFIC"
    assert run(dict(base, K2=0.001)) == "DOG_ONLY_SPECIFIC"
    assert run(dict(base, K3=0.001, K4=0.001)) == "GENERIC_PERSONA_TEACHER"
    assert run(base, null=True) == "NO_DOG_TRANSFER"
    assert run(base) == "P2_NULL_OR_MIXED"


# --- P2 data preparation ----------------------------------------------------------------------------------------

from slgeo.phenotype import p2  # noqa: E402


def _rows(n, drop=()):
    return [{"seed": 42 + i, "prompt": f"p{i}", "completion": f"{i % 7}, {i % 11}"} for i in range(n) if i not in drop]


def test_prompt_matched_subset_matches_row_seeds_and_replaces_filter_failures():
    cat = _rows(200)
    dog = _rows(200, drop={3, 5, 8})
    subset, report = p2.prompt_matched_subset(cat, dog, seed=2, size=50)
    wanted = set(p2.cat_subset_row_seeds(cat, 2, 50))
    assert len(subset) == 50 and len({r["seed"] for r in subset}) == 50
    assert report["replaced"] == len(wanted & {45, 47, 50})
    assert sum(r["seed"] in wanted for r in subset) == 50 - report["replaced"]
    again, _ = p2.prompt_matched_subset(cat, dog, seed=2, size=50)
    assert [r["seed"] for r in again] == [r["seed"] for r in subset]


def test_number_entropy_and_real_teacher_values():
    uniform = p2.number_entropy([{"completion": "1, 2, 3, 4"}])
    assert abs(uniform["entropy_nats"] - np.log(4)) < 1e-12
    path = ROOT / "data" / "filtered" / "reference_qwen7b_cat_subliminal_30k_filtered.jsonl"
    if path.exists():
        assert abs(p2.number_entropy(p2.read_jsonl(path))["entropy_nats"] - 6.682) < 0.001


def test_dog_teacher_prompt_is_the_frozen_minimal_pair():
    from slgeo.prompts import reference_animal_system_prompt

    personas = json.loads((ROOT / "research" / "cts_stage0_v1" / "cts_stage0_personas.json").read_text(encoding="utf-8"))
    by_id = {p["id"]: p["system_prompt"] for p in personas["personas"]}
    assert reference_animal_system_prompt("dog") == by_id["P_dog_T1"]
    assert reference_animal_system_prompt("cat") == by_id["P_cat_T1"]
