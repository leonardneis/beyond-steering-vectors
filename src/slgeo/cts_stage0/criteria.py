"""Gating criteria, decision engine and labels (spec ``criteria``, ``decision``).

This module is pure: it consumes per-prompt word scores L_w (by condition id and prompt id), the direction
reliabilities and the integrity verdict, and returns every criterion component for every candidate,
regardless of the decision path (spec ``compute_all_criteria_for_all_candidates``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import statistics as st
from .conditions import (
    N_RCOV,
    N_RCOV_PC,
    PARAPHRASES,
    TESTED_CONTRASTS,
    Condition,
    PERSONA,
    STEER,
    UNSTEERED,
    _steer,
)
from .steering import ALL, LAST

TARGET = "cat"
CANDIDATES = ("dog", "wolf")
O_PANEL = ("dog", "wolf", "lion", "horse", "rabbit", "elephant", "fox", "owl")
O_PRIME = ("fox", "owl", "turtle", "spider", "ant")
R_THRESHOLD = 0.95
A4_WINDOW = (0.01, 0.5)
DECISION_CLASSES = (
    "TECHNICAL_FAIL",
    "STOP_INSTRUMENT",
    "INCONCLUSIVE_POSITION",
    "GO_X",
    "GO_CAT_ONLY",
    "INCONCLUSIVE_DOSE",
    "PIVOT_NO_BASE_VALIDATED_CONTRAST",
)


class CriteriaError(RuntimeError):
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


class WordScores:
    """L_w per condition id: ``values[cid]`` maps prompt_id -> row of L_w over ``words`` (float64)."""

    def __init__(self, words: Sequence[str], values: Mapping[str, Mapping[str, Sequence[float]]]):
        self.words = tuple(words)
        self.index = {word: i for i, word in enumerate(self.words)}
        self.values = values
        self._matrices: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

    def _matrix(self, cid: str) -> tuple[tuple[str, ...], np.ndarray]:
        if cid not in self._matrices:
            try:
                table = self.values[cid]
            except KeyError as exc:
                raise CriteriaError(f"Missing condition {cid}") from exc
            ids = tuple(sorted(table))
            matrix = np.asarray([np.asarray(table[pid], dtype=np.float64) for pid in ids])
            if matrix.ndim != 2 or matrix.shape[1] != len(self.words):
                raise CriteriaError(f"Scores of {cid} have the wrong shape")
            self._matrices[cid] = (ids, matrix)
        return self._matrices[cid]

    def ell(self, cid: str, word: str, others: Sequence[str]) -> dict[str, float]:
        """Per-prompt l_{w,O} = L_w - logsumexp_{o in O} L_o (formed per prompt, never from means)."""
        if word in others:
            raise CriteriaError(f"Target {word} must not be in its off-target set")
        ids, matrix = self._matrix(cid)
        w, o = self.index[word], [self.index[other] for other in others]
        values = matrix[:, w] - st.logsumexp(matrix[:, o], axis=1)
        return dict(zip(ids, values.tolist()))

    def L(self, cid: str, word: str) -> dict[str, float]:
        ids, matrix = self._matrix(cid)
        return dict(zip(ids, matrix[:, self.index[word]].tolist()))


def _diff(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    if set(a) != set(b):
        raise CriteriaError("Paired statistics are defined on different prompt sets")
    return {key: a[key] - b[key] for key in a}


@dataclass
class Context:
    scores: WordScores
    family: st.FamilyIndex
    index: Mapping[str, np.ndarray]
    baseline: str = "unsteered"

    def __post_init__(self) -> None:
        self._baseline: dict = {}

    def delta(self, cid: str, word: str, others: Sequence[str]) -> dict[str, np.ndarray]:
        key = (word, tuple(others))
        if key not in self._baseline:
            self._baseline[key] = self.scores.ell(self.baseline, word, others)
        return self.family.align(_diff(self.scores.ell(cid, word, others), self._baseline[key]), cid)

    def summary(self, y: Mapping[str, np.ndarray]) -> dict:
        low, high = st.interval(y, self.index)
        return {"point": st.point(y), "ci_low": low, "ci_high": high}


def cid(direction, *, magnitude=None, scale="unit", kappa=1.0, sign=1, slot=14, mode=LAST) -> str:
    return _steer(direction, gating=True, group="", magnitude=magnitude, scale=scale, kappa=kappa, sign=sign, slot=slot, mode=mode).cid


def _signs_hold(ctx: Context, contrast: str, direction: str, kappa: float, x: str | None, others) -> dict:
    """Point-estimate signs required in TS(b): +c raises cat, -c lowers cat, (-c raises X)."""
    plus = cid(direction, kappa=kappa, sign=1)
    minus = cid(direction, kappa=kappa, sign=-1)
    out = {
        "cat_plus": st.point(ctx.delta(plus, TARGET, others)),
        "cat_minus": st.point(ctx.delta(minus, TARGET, others)),
    }
    ok = out["cat_plus"] > 0 and out["cat_minus"] < 0
    if x is not None:
        out["x_minus"] = st.point(ctx.delta(minus, x, off_target(contrast)))
        ok = ok and out["x_minus"] > 0
    out["pass"] = bool(ok)
    return out


def trait_specificity(ctx: Context, contrast: str, reliability: float, null_names: Sequence[str]) -> dict:
    others = off_target(contrast)
    x = None if contrast == "c_cat_anim" else contrast.removeprefix("c_cat_")
    plus, minus = cid(contrast, sign=1), cid(contrast, sign=-1)
    result: dict = {"off_target": list(others)}
    result["a"] = {"R": reliability, "pass": bool(reliability >= R_THRESHOLD)}

    b = {"cat_plus": ctx.summary(ctx.delta(plus, TARGET, others)), "cat_minus": ctx.summary(ctx.delta(minus, TARGET, others))}
    b_ok = b["cat_plus"]["ci_low"] > 0 and b["cat_minus"]["ci_high"] < 0
    if x is not None:
        b["x_minus"] = ctx.summary(ctx.delta(minus, x, others))
        b_ok = b_ok and b["x_minus"]["ci_low"] > 0
    b["pass"] = bool(b_ok)
    result["b"] = b

    c: dict = {"cat_plus": {}, "x_minus": {}}
    c_ok = True
    for o in others:
        summary = ctx.summary(ctx.delta(plus, TARGET, [o]))
        c["cat_plus"][o] = summary
        c_ok = c_ok and summary["ci_low"] > 0
        if x is not None:
            summary = ctx.summary(ctx.delta(minus, x, [o]))
            c["x_minus"][o] = summary
            c_ok = c_ok and summary["ci_low"] > 0
    c["pass"] = bool(c_ok)
    result["c"] = c

    t_obs = b["cat_plus"]["point"]
    null_stats = {name: st.point(ctx.delta(cid(name, magnitude=f"tau:{contrast}"), TARGET, others)) for name in null_names}
    threshold = st.null_threshold(list(null_stats.values()))
    random_stats = [st.point(ctx.delta(cid(f"rcov:{i}", magnitude=f"tau:{contrast}"), TARGET, others)) for i in range(N_RCOV)]
    mc = st.mc_p_value(t_obs, random_stats)
    result["d"] = {
        "T_obs": t_obs,
        "null_threshold": threshold,
        "beats_null": bool(t_obs > threshold),
        "null_statistics": null_stats,
        "rcov_p": mc.p,
        "rcov_p_mc_se": mc.se,
        "rcov_exceed": mc.exceed,
        "rcov_n": mc.n,
        "pass": bool(t_obs > threshold and mc.p < st.ALPHA_TS),
    }

    e = {"kappa_0.5": _signs_hold(ctx, contrast, contrast, 0.5, x, others)}
    for template in PARAPHRASES[contrast]:
        e[template] = _signs_hold(ctx, contrast, f"{contrast}_{template}", 1.0, x, others)
    e["pass"] = bool(all(part["pass"] for part in e.values()))
    result["e"] = e
    result["pass"] = bool(result["a"]["pass"] and b["pass"] and c["pass"] and result["d"]["pass"] and e["pass"])
    return result


def positive_control(ctx: Context, mode: str) -> dict:
    others = O_PANEL
    y = ctx.delta(cid("t_cat", scale="raw", mode=mode), TARGET, others)
    summary = ctx.summary(y)
    random_stats = [
        st.point(ctx.delta(cid(f"rcov:{i}", magnitude="norm:t_cat", mode=mode), TARGET, others)) for i in range(N_RCOV_PC)
    ]
    mc = st.mc_p_value(summary["point"], random_stats)
    return {
        **summary,
        "rcov_p": mc.p,
        "rcov_p_mc_se": mc.se,
        "rcov_exceed": mc.exceed,
        "rcov_n": mc.n,
        "pass": bool(summary["ci_low"] > 0 and mc.p < st.ALPHA_PC),
    }


def dose_control(ctx: Context, contrast: str) -> dict:
    summary = ctx.summary(ctx.delta(cid("t_cat", magnitude=f"tau:{contrast}"), TARGET, off_target(contrast)))
    return {**summary, "pass": bool(summary["ci_low"] > 0)}


def admissibility(ctx: Context, x: str, ts_pass: bool) -> dict:
    others = off_target(f"c_cat_{x}")
    base = ctx.family.align(ctx.scores.L(ctx.baseline, x), "A4")
    rate = st.point({family: np.exp(values) for family, values in base.items()})
    a4 = {"mean_prob": rate, "window": list(A4_WINDOW), "pass": bool(A4_WINDOW[0] <= rate <= A4_WINDOW[1])}
    persona = ctx.scores.ell(f"persona:P_{x}_T1", x, others)
    default = ctx.scores.ell("persona:P_default", x, others)
    a6_summary = ctx.summary(ctx.family.align(_diff(persona, default), "A6"))
    a6 = {**a6_summary, "pass": bool(a6_summary["ci_low"] > 0)}
    a7_summary = ctx.summary(ctx.delta(cid(f"t_{x}", scale="raw"), x, others))
    a7 = {**a7_summary, "pass": bool(a7_summary["ci_low"] > 0)}
    return {"A1": True, "A2": True, "A3": True, "A4": a4, "A5": {"pass": bool(ts_pass)}, "A6": a6, "A7": a7}


def label(ctx: Context, contrast: str) -> dict:
    others = off_target(contrast)
    plus = cid(contrast, sign=1)
    mentions = ("m_cat_dog", "m_cat_wolf") if contrast == "c_cat_anim" else (f"m_{contrast.removeprefix('c_')}",)
    parts = {}
    for mention in mentions:
        y = ctx.family.align(
            _diff(ctx.scores.ell(plus, TARGET, others), ctx.scores.ell(cid(mention, magnitude=f"tau:{contrast}"), TARGET, others)),
            "label",
        )
        summary = ctx.summary(y)
        parts[mention] = {**summary, "pass": bool(summary["ci_low"] > 0)}
    preference = all(part["pass"] for part in parts.values())
    return {"label": "PREFERENCE_CONTRAST" if preference else "LEXICAL_NOT_EXCLUDED", "against": parts}


@dataclass(frozen=True)
class DecisionInputs:
    integrity_ok: bool
    PC: bool
    PC_pos: bool
    R_t_cat: bool
    TS: Mapping[str, bool]  # keys c_cat_dog, c_cat_wolf, c_cat_anim
    A4: Mapping[str, bool]
    A6: Mapping[str, bool]
    A7: Mapping[str, bool]
    PC_star: Mapping[str, bool]  # keys dog, wolf, anim


def decide(inputs: DecisionInputs) -> dict:
    """Ordered, disjoint, exhaustive decision (spec ``decision.order``)."""
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
    if not inputs.PC_star["dog"] and not inputs.PC_star["wolf"] and not inputs.PC_star["anim"]:
        return {"class": "INCONCLUSIVE_DOSE", "rank": 6}
    return {"class": "PIVOT_NO_BASE_VALIDATED_CONTRAST", "rank": 7}


def evaluate(
    scores: WordScores,
    family: st.FamilyIndex,
    reliabilities: Mapping[str, float],
    null_names: Sequence[str],
    integrity_ok: bool,
) -> dict:
    """Every gating criterion for every candidate, then the decision and the labels of passing contrasts."""
    ctx = Context(scores, family, st.bootstrap_indices())
    ts = {contrast: trait_specificity(ctx, contrast, reliabilities[contrast], null_names) for contrast in TESTED_CONTRASTS}
    pc = positive_control(ctx, LAST)
    pc_pos = positive_control(ctx, ALL)
    pc_star = {key: dose_control(ctx, f"c_cat_{key}") for key in ("dog", "wolf", "anim")}
    r_t_cat = {"R": reliabilities["t_cat"], "pass": bool(reliabilities["t_cat"] >= R_THRESHOLD)}
    adm = {x: admissibility(ctx, x, ts[f"c_cat_{x}"]["pass"]) for x in CANDIDATES}
    inputs = DecisionInputs(
        integrity_ok=integrity_ok,
        PC=pc["pass"],
        PC_pos=pc_pos["pass"],
        R_t_cat=r_t_cat["pass"],
        TS={contrast: ts[contrast]["pass"] for contrast in TESTED_CONTRASTS},
        A4={x: adm[x]["A4"]["pass"] for x in CANDIDATES},
        A6={x: adm[x]["A6"]["pass"] for x in CANDIDATES},
        A7={x: adm[x]["A7"]["pass"] for x in CANDIDATES},
        PC_star={key: value["pass"] for key, value in pc_star.items()},
    )
    decision = decide(inputs)
    labels = {contrast: label(ctx, contrast) for contrast in TESTED_CONTRASTS if ts[contrast]["pass"]}
    return {
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


def required_condition_ids(null_names: Sequence[str]) -> set[str]:
    """Condition ids the criteria read (used by the completeness check)."""
    from .conditions import gating_conditions

    return {condition.cid for condition in gating_conditions(list(null_names))}


__all__ = [
    "Condition",
    "Context",
    "DecisionInputs",
    "WordScores",
    "decide",
    "evaluate",
    "off_target",
    "PERSONA",
    "STEER",
    "UNSTEERED",
]
