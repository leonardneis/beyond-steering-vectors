"""Gating criteria, decision engine and labels of the v2 contract (spec ``criteria``, ``decision``).

Pure module: it consumes per-prompt word scores L_w (by condition id and prompt id), the direction
reliabilities and the integrity verdict, and returns every criterion component for every candidate, regardless
of the decision path (spec ``compute_all_criteria_for_all_candidates``).

Every statistic is built by a named builder ``ctx -> per-family arrays``. ``Context`` records the builders and
the condition ids each one reads, so the fragility check (spec ``integrity.fragility``) can re-evaluate exactly
the same statistics on the reference-layout scores without a second implementation of any criterion.
"""

from __future__ import annotations

from .errors import FinalFailure

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from . import statistics as st
from .conditions import (
    L2_BASELINE,
    N_RCOV,
    N_RCOV_PC,
    OWN_BASELINE,
    PARAPHRASES,
    SHARED_LABEL_DIRECTIONS,
    TESTED_CONTRASTS,
    cid,
)
from .steering import ALL, LAST

TARGET = "cat"
CANDIDATES = ("dog", "wolf")
O_PANEL = ("dog", "wolf", "lion", "horse", "rabbit", "elephant", "fox", "owl")
O_PRIME = ("fox", "owl", "turtle", "spider", "ant")
ANIMAL_WORDS = ("cat",) + O_PANEL + ("turtle", "spider", "ant")
R_THRESHOLD = 0.95
A4_WINDOW = (0.01, 0.5)
SHARED_MARGIN_R = 0.2
DECISION_CLASSES = (
    "TECHNICAL_FAIL",
    "STOP_INSTRUMENT",
    "INCONCLUSIVE_POSITION",
    "GO_X",
    "GO_CAT_ONLY",
    "INCONCLUSIVE_DOSE",
    "PIVOT_SHARED_LEAKAGE",
    "PIVOT_NO_BASE_VALIDATED_CONTRAST",
)


class CriteriaError(RuntimeError, FinalFailure):
    pass


def off_target(contrast_or_word: str) -> tuple[str, ...]:
    if contrast_or_word in ("c_cat_anim", "anim"):
        return O_PRIME
    if contrast_or_word in ("panel", "PC"):
        return O_PANEL
    x = contrast_or_word.removeprefix("c_cat_")
    if x not in CANDIDATES:
        raise CriteriaError(f"No off-target set for {contrast_or_word!r}")
    return tuple(word for word in O_PANEL if word != x)


def shared_label_pairs(contrast: str) -> tuple[str, ...]:
    """S(c): for c_cat_X, O_panel minus X plus X (= O_panel); for c_cat_anim, O'."""
    return O_PRIME if contrast == "c_cat_anim" else O_PANEL


class WordScores:
    """L_w per condition id: ``values[cid]`` maps prompt_id -> row of L_w over ``words`` (float64)."""

    def __init__(self, words: Sequence[str], values: Mapping[str, Mapping[str, Sequence[float]]]):
        self.words = tuple(words)
        self.index = {word: i for i, word in enumerate(self.words)}
        self.values = values
        self._matrices: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

    def has(self, cid_: str) -> bool:
        return cid_ in self.values

    def _matrix(self, cid_: str) -> tuple[tuple[str, ...], np.ndarray]:
        if cid_ not in self._matrices:
            try:
                table = self.values[cid_]
            except KeyError as exc:
                raise CriteriaError(f"Missing condition {cid_}") from exc
            ids = tuple(sorted(table))
            matrix = np.asarray([np.asarray(table[pid], dtype=np.float64) for pid in ids])
            if matrix.ndim != 2 or matrix.shape[1] != len(self.words):
                raise CriteriaError(f"Scores of {cid_} have the wrong shape")
            self._matrices[cid_] = (ids, matrix)
        return self._matrices[cid_]

    def ell(self, cid_: str, word: str, others: Sequence[str]) -> dict[str, float]:
        """Per-prompt l_{w,O} = L_w - logsumexp_{o in O} L_o (formed per prompt, never from means)."""
        if word in others:
            raise CriteriaError(f"Target {word} must not be in its off-target set")
        ids, matrix = self._matrix(cid_)
        w, o = self.index[word], [self.index[other] for other in others]
        values = matrix[:, w] - st.logsumexp(matrix[:, o], axis=1)
        return dict(zip(ids, values.tolist()))

    def L(self, cid_: str, word: str) -> dict[str, float]:
        ids, matrix = self._matrix(cid_)
        return dict(zip(ids, matrix[:, self.index[word]].tolist()))

    def mass(self, cid_: str, words: Sequence[str]) -> dict[str, float]:
        ids, matrix = self._matrix(cid_)
        values = st.logsumexp(matrix[:, [self.index[w] for w in words]], axis=1)
        return dict(zip(ids, values.tolist()))


def _diff(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    if set(a) != set(b):
        raise CriteriaError("Paired statistics are defined on different prompt sets")
    return {key: a[key] - b[key] for key in a}


Builder = Callable[["Context"], Mapping[str, np.ndarray]]


@dataclass
class RecordedStatistic:
    label: str
    builder: Builder
    cids: frozenset[str]
    point: float
    se_boot: float


@dataclass
class Context:
    """Scores, the bootstrap, and the baseline of each condition's cost class (spec ``baseline_rule``)."""

    scores: WordScores
    family: st.FamilyIndex
    index: Mapping[str, np.ndarray]
    baselines: Mapping[str, str] = field(default_factory=dict)
    recorded: dict[str, RecordedStatistic] = field(default_factory=dict)
    null_families: dict[str, dict[str, tuple[Builder, frozenset[str], float]]] = field(default_factory=dict)
    _touched: set[str] | None = None

    def _use(self, cid_: str) -> str:
        if self._touched is not None:
            self._touched.add(cid_)
        return cid_

    def baseline_of(self, cid_: str) -> str:
        try:
            return self.baselines[cid_]
        except KeyError as exc:
            raise CriteriaError(f"No baseline recorded for {cid_}") from exc

    # --- per-prompt building blocks (paired by prompt id, aligned to the S0 animal families) -----------------
    def delta(self, cid_: str, word: str, others: Sequence[str]) -> dict[str, np.ndarray]:
        base = self._use(self.baseline_of(cid_))
        steered = self.scores.ell(self._use(cid_), word, others)
        return self.family.align(_diff(steered, self.scores.ell(base, word, others)), cid_)

    def delta_mass(self, cid_: str) -> dict[str, np.ndarray]:
        base = self._use(self.baseline_of(cid_))
        steered = self.scores.mass(self._use(cid_), ANIMAL_WORDS)
        return self.family.align(_diff(steered, self.scores.mass(base, ANIMAL_WORDS)), cid_)

    def ell_aligned(self, cid_: str, word: str, others: Sequence[str]) -> dict[str, np.ndarray]:
        return self.family.align(self.scores.ell(self._use(cid_), word, others), cid_)

    def base_prob(self, word: str) -> dict[str, np.ndarray]:
        base = self.family.align(self.scores.L(self._use(L2_BASELINE), word), "A4")
        return {family: np.exp(values) for family, values in base.items()}

    # --- recorded statistics ---------------------------------------------------------------------------------
    def _build(self, builder: Builder) -> tuple[Mapping[str, np.ndarray], frozenset[str]]:
        previous, self._touched = self._touched, set()
        try:
            y = builder(self)
            used = frozenset(self._touched)
        finally:
            self._touched = previous
        return y, used

    def stat(self, label: str, builder: Builder) -> dict:
        """Point, two-sided (1 - alpha_TS) interval, one-sided bounds and SE_boot of a recorded statistic."""
        y, used = self._build(builder)
        low, high = st.interval(y, self.index)
        one_low, one_high = st.one_sided_bounds(y, self.index)
        summary = {"point": st.point(y), "ci_low": low, "ci_high": high, "one_sided_low": one_low, "one_sided_high": one_high}
        if label in self.recorded:
            raise CriteriaError(f"Statistic {label} recorded twice")
        self.recorded[label] = RecordedStatistic(label, builder, used, summary["point"], st.se_boot(y, self.index))
        return summary

    def null_point(self, family_name: str, label: str, builder: Builder) -> float:
        y, used = self._build(builder)
        value = st.point(y)
        self.null_families.setdefault(family_name, {})[label] = (builder, used, value)
        return value


def _signs_hold(ctx: Context, contrast: str, direction: str, kappa: float, x: str | None, others, tag: str) -> dict:
    """Point-estimate signs required in TS(b): +c raises cat, -c lowers cat, (-c raises X)."""
    plus = cid(direction, kappa=kappa, sign=1)
    minus = cid(direction, kappa=kappa, sign=-1)
    out = {
        "cat_plus": ctx.stat(f"{contrast}/e/{tag}/cat_plus", lambda c: c.delta(plus, TARGET, others))["point"],
        "cat_minus": ctx.stat(f"{contrast}/e/{tag}/cat_minus", lambda c: c.delta(minus, TARGET, others))["point"],
    }
    ok = out["cat_plus"] > 0 and out["cat_minus"] < 0
    if x is not None:
        out["x_minus"] = ctx.stat(f"{contrast}/e/{tag}/x_minus", lambda c: c.delta(minus, x, off_target(contrast)))["point"]
        ok = ok and out["x_minus"] > 0
    out["pass"] = bool(ok)
    return out


def _direction_and_selectivity(ctx: Context, contrast: str, direction: str, others, x: str | None, part: str) -> tuple[dict, dict]:
    """TS(b) and TS(c) for ``direction`` at kappa 1 (used for c and, in TS(f), for c_perpG)."""
    plus, minus = cid(direction, sign=1), cid(direction, sign=-1)
    b = {
        "cat_plus": ctx.stat(f"{contrast}/{part}/b/cat_plus", lambda c: c.delta(plus, TARGET, others)),
        "cat_minus": ctx.stat(f"{contrast}/{part}/b/cat_minus", lambda c: c.delta(minus, TARGET, others)),
    }
    b_ok = b["cat_plus"]["ci_low"] > 0 and b["cat_minus"]["ci_high"] < 0
    if x is not None:
        b["x_minus"] = ctx.stat(f"{contrast}/{part}/b/x_minus", lambda c: c.delta(minus, x, others))
        b_ok = b_ok and b["x_minus"]["ci_low"] > 0
    b["pass"] = bool(b_ok)
    sel: dict = {"cat_plus": {}, "x_minus": {}}
    ok = True
    for o in others:
        summary = ctx.stat(f"{contrast}/{part}/c/cat_plus/{o}", lambda c, o=o: c.delta(plus, TARGET, [o]))
        sel["cat_plus"][o] = summary
        ok = ok and summary["ci_low"] > 0
        if x is not None:
            summary = ctx.stat(f"{contrast}/{part}/c/x_minus/{o}", lambda c, o=o: c.delta(minus, x, [o]))
            sel["x_minus"][o] = summary
            ok = ok and summary["ci_low"] > 0
    sel["pass"] = bool(ok)
    return b, sel


def trait_specificity(ctx: Context, contrast: str, reliability: float, perp_reliability: float, null_names: Sequence[str]) -> dict:
    others = off_target(contrast)
    x = None if contrast == "c_cat_anim" else contrast.removeprefix("c_cat_")
    result: dict = {"off_target": list(others)}
    result["a"] = {"R": reliability, "pass": bool(reliability >= R_THRESHOLD)}
    b, c_part = _direction_and_selectivity(ctx, contrast, contrast, others, x, "tsc")
    result["b"], result["c"] = b, c_part

    t_obs = b["cat_plus"]["point"]
    null_stats = {
        name: ctx.null_point(f"null:{contrast}", name, lambda c, name=name: c.delta(cid(name, magnitude=f"tau:{contrast}"), TARGET, others))
        for name in null_names
    }
    d1 = st.rank_test(t_obs, list(null_stats.values()), st.ALPHA_TS)
    rcov_stats = [
        ctx.null_point(
            f"rcov:{contrast}", f"rcov:{i}", lambda c, i=i: c.delta(cid(f"rcov:{i}", magnitude=f"tau:{contrast}"), TARGET, others)
        )
        for i in range(N_RCOV)
    ]
    d2 = st.rank_test(t_obs, rcov_stats, st.ALPHA_TS)
    result["d1"] = {"T_obs": t_obs, "exceed": d1.exceed, "n": d1.n, "k_max": d1.k_max, "p": d1.p, "pass": d1.passed}
    result["d2"] = {"T_obs": t_obs, "exceed": d2.exceed, "n": d2.n, "k_max": d2.k_max, "p": d2.p, "pass": d2.passed}

    e = {"kappa_0.5": _signs_hold(ctx, contrast, contrast, 0.5, x, others, "kappa_0.5")}
    for template in PARAPHRASES[contrast]:
        e[template] = _signs_hold(ctx, contrast, f"{contrast}_{template}", 1.0, x, others, template)
    e["pass"] = bool(all(part["pass"] for part in e.values()))
    result["e"] = e

    fb, fc = _direction_and_selectivity(ctx, contrast, f"{contrast}_perpG", others, x, "f")
    result["f"] = {
        "R": perp_reliability,
        "R_pass": bool(perp_reliability >= R_THRESHOLD),
        "b": fb,
        "c": fc,
        "pass": bool(perp_reliability >= R_THRESHOLD and fb["pass"] and fc["pass"]),
    }
    abcde = bool(result["a"]["pass"] and b["pass"] and c_part["pass"] and d1.passed and d2.passed and e["pass"])
    result["TS_abcde"] = abcde
    result["pass"] = bool(abcde and result["f"]["pass"])
    return result


def positive_control(ctx: Context, mode: str) -> dict:
    others = O_PANEL
    main = cid("t_cat", scale="raw", mode=mode)
    summary = ctx.stat(f"PC/{mode}", lambda c: c.delta(main, TARGET, others))
    references = [
        ctx.null_point(
            f"rcov_pc:{mode}", f"rcov:{i}", lambda c, i=i: c.delta(cid(f"rcov:{i}", magnitude="norm:t_cat", mode=mode), TARGET, others)
        )
        for i in range(N_RCOV_PC)
    ]
    test = st.rank_test(summary["point"], references, st.ALPHA_PC)
    return {
        **summary,
        "reference_exceed": test.exceed,
        "reference_n": test.n,
        "reference_k_max": test.k_max,
        "reference_p": test.p,
        "pass": bool(summary["ci_low"] > 0 and test.passed),
    }


def dose_control(ctx: Context, contrast: str) -> dict:
    condition = cid("t_cat", magnitude=f"tau:{contrast}")
    summary = ctx.stat(f"PC_star/{contrast}", lambda c: c.delta(condition, TARGET, off_target(contrast)))
    return {**summary, "pass": bool(summary["ci_low"] > 0)}


def admissibility(ctx: Context, x: str, ts_pass: bool) -> dict:
    others = off_target(f"c_cat_{x}")
    a4_summary = ctx.stat(f"A4/{x}", lambda c: c.base_prob(x))
    rate = a4_summary["point"]
    a4 = {"mean_prob": rate, "window": list(A4_WINDOW), "pass": bool(A4_WINDOW[0] <= rate <= A4_WINDOW[1])}
    persona, default = f"persona:P_{x}_T1", OWN_BASELINE

    def a6_builder(c: Context):
        return c.family.align(_diff(c.scores.ell(c._use(persona), x, others), c.scores.ell(c._use(default), x, others)), "A6")

    a6_summary = ctx.stat(f"A6/{x}", a6_builder)
    a6 = {**a6_summary, "pass": bool(a6_summary["ci_low"] > 0)}
    diagonal = cid(f"t_{x}", scale="raw")
    a7_summary = ctx.stat(f"A7/{x}", lambda c: c.delta(diagonal, x, others))
    a7 = {**a7_summary, "pass": bool(a7_summary["ci_low"] > 0)}
    return {"A1": True, "A2": True, "A3": True, "A4": a4, "A5": {"pass": bool(ts_pass)}, "A6": a6, "A7": a7}


def mention_label(ctx: Context, contrast: str) -> dict:
    others = off_target(contrast)
    plus = cid(contrast, sign=1)
    mentions = ("m_cat_dog", "m_cat_wolf") if contrast == "c_cat_anim" else (f"m_{contrast.removeprefix('c_')}",)
    parts = {}
    for mention in mentions:
        mention_cid = cid(mention, magnitude=f"tau:{contrast}")

        def builder(c: Context, mention_cid=mention_cid):
            return c.family.align(
                _diff(c.scores.ell(c._use(plus), TARGET, others), c.scores.ell(c._use(mention_cid), TARGET, others)), "label"
            )

        summary = ctx.stat(f"{contrast}/label/mention/{mention}", builder)
        parts[mention] = {**summary, "pass": bool(summary["ci_low"] > 0)}
    preference = all(part["pass"] for part in parts.values())
    return {"label": "PREFERENCE_CONTRAST" if preference else "LEXICAL_NOT_EXCLUDED", "against": parts}


def shared_direction_label(ctx: Context, contrast: str, g: str) -> dict:
    """BASE_SHARED_DIRECTION(g; c) (spec ``decision.labels.BASE_SHARED_DIRECTION``)."""
    g_cid = cid(g, magnitude=f"tau:{contrast}")
    c_cid = cid(contrast, sign=1)
    mass = ctx.stat(f"{contrast}/label/shared/{g}/mass", lambda c: c.delta_mass(g_cid))
    part_i = {**mass, "pass": bool(mass["ci_low"] > 0)}
    pairs = {}
    ok = True
    for o in shared_label_pairs(contrast):

        def combination(c: Context, sign: float, o=o):
            dg = c.delta(g_cid, TARGET, [o])
            dc = c.delta(c_cid, TARGET, [o])
            return {family: dg[family] + sign * SHARED_MARGIN_R * dc[family] for family in dg}

        upper = ctx.stat(f"{contrast}/label/shared/{g}/D+/{o}", lambda c, o=o: combination(c, -1.0, o))
        lower = ctx.stat(f"{contrast}/label/shared/{g}/D-/{o}", lambda c, o=o: combination(c, +1.0, o))
        passed = bool(upper["one_sided_high"] < 0 and lower["one_sided_low"] > 0)
        pairs[o] = {"D_plus_upper": upper["one_sided_high"], "D_minus_lower": lower["one_sided_low"], "pass": passed}
        ok = ok and passed
    label = "BASE_SHARED_DIRECTION" if (part_i["pass"] and ok) else "NOT_BASE_SHARED"
    return {"label": label, "i_shared_effect": part_i, "ii_not_trait_selective": {"pairs": pairs, "pass": bool(ok)}, "r": SHARED_MARGIN_R}


@dataclass(frozen=True)
class DecisionInputs:
    integrity_ok: bool
    PC: bool
    PC_pos: bool
    R_t_cat: bool
    TS: Mapping[str, bool]  # keys c_cat_dog, c_cat_wolf, c_cat_anim
    TS_abcde: Mapping[str, bool]
    A4: Mapping[str, bool]
    A6: Mapping[str, bool]
    A7: Mapping[str, bool]
    PC_star: Mapping[str, bool]  # keys c_cat_dog, c_cat_wolf, c_cat_anim


def decide(inputs: DecisionInputs) -> dict:
    """Ordered, disjoint, exhaustive decision (spec ``decision.order``, ranks 1-8)."""
    ts_anim = bool(inputs.TS["c_cat_anim"])
    if not inputs.integrity_ok:
        return {"class": "TECHNICAL_FAIL", "rank": 1}
    if (not inputs.PC and not inputs.PC_pos) or not inputs.R_t_cat:
        return {"class": "STOP_INSTRUMENT", "rank": 2}
    if not inputs.PC and inputs.PC_pos:
        return {"class": "INCONCLUSIVE_POSITION", "rank": 3}
    for x in CANDIDATES:
        a5 = bool(inputs.TS[f"c_cat_{x}"])
        if inputs.A4[x] and a5 and inputs.A6[x] and inputs.A7[x]:
            return {
                "class": "GO_X",
                "rank": 4,
                "X": x,
                "stage1_trait": x,
                "stage2a_primary": "c_cat_anim" if ts_anim else f"c_cat_{x}",
            }
    passing = [f"c_cat_{x}" for x in CANDIDATES if inputs.TS[f"c_cat_{x}"]]
    if ts_anim or passing:
        return {
            "class": "GO_CAT_ONLY",
            "rank": 5,
            "stage2a_primary": "c_cat_anim" if ts_anim else passing[0],
            "stage1": "not started",
            "two_trait_claim": "unavailable",
        }
    if not any(inputs.PC_star[contrast] for contrast in TESTED_CONTRASTS):
        return {"class": "INCONCLUSIVE_DOSE", "rank": 6}
    leaking = [contrast for contrast in TESTED_CONTRASTS if inputs.TS_abcde[contrast]]
    if leaking:
        return {"class": "PIVOT_SHARED_LEAKAGE", "rank": 7, "contrasts_passing_a_to_e": leaking}
    return {"class": "PIVOT_NO_BASE_VALIDATED_CONTRAST", "rank": 8}


def evaluate(
    scores: WordScores,
    family: st.FamilyIndex,
    reliabilities: Mapping[str, float],
    null_names: Sequence[str],
    integrity_ok: bool,
    baselines: Mapping[str, str],
) -> tuple[dict, Context]:
    """Every gating criterion for every candidate, then the decision and the labels of passing contrasts."""
    ctx = Context(scores, family, st.bootstrap_indices(), baselines)
    ts = {
        contrast: trait_specificity(ctx, contrast, reliabilities[contrast], reliabilities[f"{contrast}_perpG"], null_names)
        for contrast in TESTED_CONTRASTS
    }
    pc = positive_control(ctx, LAST)
    pc_pos = positive_control(ctx, ALL)
    pc_star = {contrast: dose_control(ctx, contrast) for contrast in TESTED_CONTRASTS}
    r_t_cat = {"R": reliabilities["t_cat"], "pass": bool(reliabilities["t_cat"] >= R_THRESHOLD)}
    adm = {x: admissibility(ctx, x, ts[f"c_cat_{x}"]["pass"]) for x in CANDIDATES}
    inputs = DecisionInputs(
        integrity_ok=integrity_ok,
        PC=pc["pass"],
        PC_pos=pc_pos["pass"],
        R_t_cat=r_t_cat["pass"],
        TS={contrast: ts[contrast]["pass"] for contrast in TESTED_CONTRASTS},
        TS_abcde={contrast: ts[contrast]["TS_abcde"] for contrast in TESTED_CONTRASTS},
        A4={x: adm[x]["A4"]["pass"] for x in CANDIDATES},
        A6={x: adm[x]["A6"]["pass"] for x in CANDIDATES},
        A7={x: adm[x]["A7"]["pass"] for x in CANDIDATES},
        PC_star={key: value["pass"] for key, value in pc_star.items()},
    )
    decision = decide(inputs)
    labels = {}
    for contrast in TESTED_CONTRASTS:
        if ts[contrast]["pass"]:
            labels[contrast] = {
                "mention": mention_label(ctx, contrast),
                "shared": {g: shared_direction_label(ctx, contrast, g) for g in SHARED_LABEL_DIRECTIONS},
            }
    if not labels:
        labels = {"BASE_SHARED_DIRECTION": "NOT_ASSESSABLE"}
    result = {
        "decision": decision,
        "labels": labels,
        "criteria": {
            "R_t_cat": r_t_cat,
            "PC": pc,
            "PC_pos": pc_pos,
            "PC_star": pc_star,
            "TS": ts,
            "admissibility": adm,
        },
        "alpha_TS": st.ALPHA_TS,
        "alpha_PC": st.ALPHA_PC,
    }
    return result, ctx


def all_statistics(
    scores: WordScores, family: st.FamilyIndex, reliabilities: Mapping[str, float], null_names: Sequence[str], baselines: Mapping[str, str]
) -> Context:
    """Record every gating and label statistic, including the labels of every contrast (for fragility).

    The fragility check covers every label statistic, whether or not its contrast passes, so that the set of
    checked statistics never depends on an outcome."""
    result, ctx = evaluate(scores, family, reliabilities, null_names, True, baselines)
    for contrast in TESTED_CONTRASTS:
        if not result["criteria"]["TS"][contrast]["pass"]:
            mention_label(ctx, contrast)
            for g in SHARED_LABEL_DIRECTIONS:
                shared_direction_label(ctx, contrast, g)
    return ctx
