"""Automated P1 (and P2) analysis: shard outputs -> per-arm stem distributions -> hypotheses -> taxonomy.

Scientific outputs are sealed. ``analyze`` refuses unless an unseal record exists that names the preregistration
tag and commit (written only after the freeze; decision D4 order: freeze -> gate files -> P1 run -> unseal).
Confirmatory statistics use the RES stems; REF50 and NONANIMAL are reported separately (development, descriptive).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import stats, taxonomy
from .panel import CONTROL_EXCLUSIONS, PANEL, TARGET

CONFIRMATORY_SEEDS = ("2", "3")
ALL_SEEDS = ("1", "2", "3")
PRIMARY = ("Q", ("r0", "r1", "r2"))
SECONDARY = ("Q", ("none",))
TEACHER_CELL = ("persona", ("r0", "r1", "r2"))
FIXED_MARGINS = (0.05, 0.10, 0.15, 0.20, 0.30)
COVERAGE_LIMIT = 0.05


class SealedError(RuntimeError):
    pass


def require_unsealed(out_root: Path, *, expected_tag: str) -> dict:
    record = out_root / "UNSEAL.json"
    if not record.is_file():
        raise SealedError("Outputs are sealed: no UNSEAL.json (written only after the preregistration freeze)")
    data = json.loads(record.read_text(encoding="utf-8"))
    if data.get("prereg_tag") != expected_tag or not data.get("prereg_commit"):
        raise SealedError("UNSEAL.json does not name the frozen preregistration")
    return data


def load_scores(out_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """context_id -> {word_logp, decoration_mass, emoji_mass, ...} from every complete score shard."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for shard in sorted((out_root / "raw").glob("*.score.*")):
        if not (shard / "COMPLETE").exists():
            raise stats.PhenotypeStatsError(f"Incomplete shard {shard.name}")
        data = np.load(shard / "scores.npz")
        for i, cid in enumerate(data["context_ids"]):
            out[str(cid)] = {k: data[k][i] for k in data.files if k != "context_ids"}
    return out


def arm_logq(scores: Mapping[str, Mapping[str, np.ndarray]], arm: str, stems: Sequence[str], cell: tuple[str, tuple[str, ...]]) -> np.ndarray:
    """[S, W] floored panel-conditional log q of one arm on one cell (prefix replicates averaged)."""
    name, prefixes = cell
    lp = np.stack([np.stack([scores[f"{arm}|{s}|{name}+{p}"]["word_logp"] for p in prefixes]) for s in stems])
    return stats.stem_conditional(lp)


def _pairs(logq: Mapping[str, np.ndarray], prefix: str) -> list[tuple[str, str]]:
    """Within-condition cross-seed pairs over every seed present (1-3; 1-5 after a fresh-seed stage)."""
    names = sorted((k for k in logq if k[:-1] == prefix and k[-1:].isdigit()), key=lambda k: int(k[len(prefix):]))
    return list(itertools.combinations(names, 2))


def _arrays(logq: Mapping[str, np.ndarray], pairs: Sequence[tuple[str, str]]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(logq[a], logq[b]) for a, b in pairs]


def _seeds(logq: Mapping[str, np.ndarray], prefix: str) -> list[str]:
    return [k[len(prefix):] for k in logq if k[:-1] == prefix and k[-1:].isdigit()]


def _rl(result: stats.RunLevel) -> dict:
    return dataclasses.asdict(result)


class _Tests:
    """Run-level tests of one contrast family (treated condition y vs reference x) with shared settings."""

    def __init__(self, logq, y, x, targets, *, families, confirmatory, n_boot, n_ref, seed):
        self.logq, self.base = logq, logq["base"]
        self.w = stats.family_weights(families)
        self.strata = list(families)
        py, px = _pairs(logq, y), _pairs(logq, x)
        self.within = py + px
        self.lam = stats.lambda_of(_arrays(logq, py), _arrays(logq, px), self.base, targets, self.w)
        seeds = sorted(set(_seeds(logq, y)) & set(_seeds(logq, x)), key=int)
        self.treated = {s: (f"{y}{s}", f"{x}{s}") for s in seeds}
        self.confirmatory = tuple(confirmatory)
        self.kw = dict(n_boot=n_boot, n_ref=n_ref, seed=seed)

    def __call__(self, statistic):
        return stats.run_level(statistic, self.logq, self.treated, self.within, self.base, self.lam, stem_w=self.w,
                               strata=self.strata, pooled=self.confirmatory, **self.kw)


def p1_family(logq: Mapping[str, np.ndarray], v1: Mapping[str, float], families: Sequence[str], *, n_boot: int,
              n_ref: int, confirmatory: Sequence[str] = CONFIRMATORY_SEEDS, seed: int = stats.SEED) -> dict:
    """P1 run-level tests for every seed at once (within-condition pairs over all seeds present, lambda-hat shared).

    Confirmatory per seed: C2 (-log beta) and the robust C3 claim (tempering residual and mass-matched contrast,
    intersection-union); the cat-dominance label (all 25 contrasts d_w pass, intersection-union). The bound U is the
    larger of the two components' pooled-seed upper bounds. Everything else is descriptive."""
    t = PANEL.index(TARGET)
    base = logq["base"]
    tests = _Tests(logq, "S", "N", (t,), families=families, confirmatory=confirmatory, n_boot=n_boot, n_ref=n_ref,
                   seed=seed)
    controls = stats.mass_matched_controls(base, t, exclude=[PANEL.index(w) for w in CONTROL_EXCLUSIONS[TARGET]])
    c2 = tests(stats.flattening_stat((t,)))
    c3 = tests(stats.target_stat(t))
    c3mm = tests(stats.mass_matched_stat(t, controls))
    curvature = tests(stats.curvature_stat((t,)))
    dominance = {PANEL[w]: tests(stats.dominance_stat(t, w)) for w in range(len(PANEL)) if w != t}

    lam_hat = tests.lam(None)
    c3_pair = stats.target_stat(t)
    m_run = stats.run_noise_margin(lambda a, b: c3_pair(a, b, base, 1.0, tests.w), _arrays(logq, tests.within))
    teacher_ref = stats.mean_distribution(*(logq[f"N{k}"] for k in _seeds(logq, "N")))
    seeds = {}
    for s in tests.treated:
        y, x = logq[f"S{s}"], logq[f"N{s}"]
        c1 = stats.omnibus(y, x, n_flip=min(n_boot, stats.N_FLIP), seed=seed)
        c1_gate = stats.run_level_gate(lambda a, b: stats.omnibus(a, b, n_flip=1).statistic, (y, x),
                                       _arrays(logq, tests.within))
        dom_p = {w: r[s].p for w, r in dominance.items()}
        seeds[s] = {
            "p": {"C2": c2[s].p, "C3": stats.robust_p(c3[s], c3mm[s])},
            "label": bool(all(p <= stats.ALPHA for p in dom_p.values())),
            "c3_upper": max(c3[s].upper95, c3mm[s].upper95),
            "tests": {"C2": _rl(c2[s]), "C3": _rl(c3[s]), "C3mm": _rl(c3mm[s]), "curvature": _rl(curvature[s])},
            "dominance_p": dom_p,
            "descriptive": {
                "beta": float(np.exp(-c2[s].estimate)),
                "C3_delta_prob": stats.wmean(np.exp(y[:, t]) - np.exp(x[:, t]), tests.w),
                "C1_T": c1.statistic, "C1_p_stem": c1.p, "C1_gate_v1": c1_gate.passed,
                "C4": dataclasses.asdict(stats.shadow_concordance(logq["T_cat"], base, teacher_ref, y, x, t,
                                                                  lam=lam_hat)),
                "C5": dataclasses.asdict(stats.profile_replication(v1, PANEL, y, x, base, t, lam=lam_hat)),
                "c3_equivalent_fixed": {str(m): c3[s].equivalent(m) for m in FIXED_MARGINS},
                "c3_equivalent_m_run": c3[s].equivalent(m_run),
            },
        }
    pooled = {"C3": _rl(c3["pooled"]), "C3mm": _rl(c3mm["pooled"]), "C2": _rl(c2["pooled"])}
    return {"lambda": lam_hat, "m_run": m_run, "controls": [PANEL[w] for w in controls], "seeds": seeds,
            "confirmatory": list(confirmatory), "pooled": pooled,
            "c3_upper_pooled": max(c3["pooled"].upper95, c3mm["pooled"].upper95)}


def p1_outcome(family: Mapping[str, Any], *, integrity_ok: bool, instrument_ok: bool) -> taxonomy.Outcome:
    seeds = {s: taxonomy.SeedResult(family["seeds"][s]["p"], family["seeds"][s]["label"])
             for s in family["confirmatory"]}
    return taxonomy.classify_p1(seeds, integrity_ok=integrity_ok, instrument_ok=instrument_ok,
                                c3_upper=family["c3_upper_pooled"])


def replicate_beta(scores, arm_y: str, arm_x: str, stems: Sequence[str], families: Sequence[str],
                   lam: float) -> float:
    """Descriptive: mean over the prefix replicates of the Deming beta fitted on single-replicate conditionals.
    Its gap to the prefix-averaged beta separates mixture flattening (probability-scale averaging of prefix-
    inconsistent answers) from a tempering of each conditional."""
    t = PANEL.index(TARGET)
    words = np.array([w for w in range(len(PANEL)) if w != t])
    w = stats.family_weights(families)
    betas = []
    for r in PRIMARY[1]:
        cell = (PRIMARY[0], (r,))
        betas.append(stats.fit_beta(arm_logq(scores, arm_y, stems, cell), arm_logq(scores, arm_x, stems, cell), words,
                                    arm_logq(scores, "base", stems, cell), lam, w))
    return float(np.mean(betas))


def instrument_check(scores, samples: Sequence[Mapping[str, Any]], arms: Sequence[str], stems: Sequence[str], k: int) -> dict:
    """Exact vs sampled agreement on the sampled cell (Q+r0 or persona+r0) and the coverage diagnostic."""
    out = {}
    for arm in arms:
        cell = "persona+r0" if arm.startswith("T_") else "Q+r0"
        ids = [f"{arm}|{s}|{cell}" for s in stems]
        prob = np.exp(np.stack([scores[i]["word_logp"] for i in ids]))
        counts = np.zeros_like(prob)
        index = {cid: j for j, cid in enumerate(ids)}
        for row in samples:
            j = index.get(row["context_id"])
            if j is not None and row["cls"] == "PANEL":
                counts[j, PANEL.index(row["lemma"])] += 1
        agreement = stats.instrument_agreement(prob, counts, k)
        coverage = max(float(np.mean([scores[i]["decoration_mass"] for i in ids])),
                       float(np.mean([scores[i]["emoji_mass"] for i in ids])))
        out[arm] = {"agreement": agreement.__dict__, "coverage": coverage,
                    "ok": agreement.passed and coverage <= COVERAGE_LIMIT}
    return out


def analyze_p1(scores, samples, entries: Sequence[Mapping[str, Any]], v1: Mapping[str, float], *,
               integrity_ok: bool, sample_k: int, n_boot: int = stats.N_BOOT, n_ref: int = stats.N_REF) -> dict:
    res = [e for e in entries if e["set"] == "RES"]
    res_stems, families = [e["stem_id"] for e in res], [e["family"] for e in res]
    arms = ["base"] + [f"{c}{s}" for c in "NS" for s in ALL_SEEDS]
    result: dict[str, Any] = {}
    for label, cell in (("primary", PRIMARY), ("secondary", SECONDARY)):
        logq = {a: arm_logq(scores, a, res_stems, cell) for a in arms}
        logq["T_cat"] = arm_logq(scores, "T_cat", res_stems, TEACHER_CELL if label == "primary" else ("persona", ("none",)))
        result[label] = p1_family(logq, v1, families, n_boot=n_boot, n_ref=n_ref)
    for s, block in result["primary"]["seeds"].items():
        block["descriptive"]["beta_per_replicate"] = replicate_beta(scores, f"S{s}", f"N{s}", res_stems, families,
                                                                    result["primary"]["lambda"])
    sampled_stems = [e["stem_id"] for e in entries if e["set"] in {"REF50", "RES"}]
    instrument = instrument_check(scores, samples, ["base", "T_cat"] + arms[1:], sampled_stems, sample_k)
    instrument_ok = all(v["ok"] for a, v in instrument.items() if a in {"base", "T_cat", "N2", "S2", "N3", "S3"})
    secondary = p1_outcome(result["secondary"], integrity_ok=integrity_ok, instrument_ok=instrument_ok).cls
    outcome = p1_outcome(result["primary"], integrity_ok=integrity_ok, instrument_ok=instrument_ok)
    return {"outcome": outcome.__dict__, "fresh_seed_trigger": taxonomy.fresh_seed_trigger(outcome),
            "secondary_class_descriptive": secondary, "instrument": instrument, "families": result,
            "development_seed": "1"}


def p2_family(logq: Mapping[str, np.ndarray], stem_ids: Sequence[str], families: Sequence[str], *, n_boot: int,
              n_ref: int, confirmatory: Sequence[str] = CONFIRMATORY_SEEDS, seed: int = stats.SEED) -> dict:
    """P2 run-level tests for every seed at once (dog-teacher students D1-D3 required).

    Confirmatory per seed: K1 robust (cat residual of S_k vs D_k, dog excluded), K2 robust (dog residual of D_k vs
    S_k, cat excluded), K3 (-log beta of D_k vs N_k); P2 cat label (cat above every non-trait word in S_k vs D_k);
    K5 (-log beta of S_k vs D_k, two-sided; reported); dog-transfer check (D_k vs N_k flattening or robust dog
    residual, each at alpha, no multiplicity correction: a non-detection class must not become easier to reach).
    K4 and the S-D teacher-shadow correlation are descriptive."""
    cat, dog = PANEL.index(TARGET), PANEL.index("dog")
    base = logq["base"]
    kw = dict(families=families, confirmatory=confirmatory, n_boot=n_boot, n_ref=n_ref, seed=seed)
    sd = _Tests(logq, "S", "D", (cat, dog), **kw)
    ds = _Tests(logq, "D", "S", (cat, dog), **kw)
    dn = _Tests(logq, "D", "N", (cat, dog), **kw)
    dn_dog = _Tests(logq, "D", "N", (dog,), **kw)
    excl = lambda word, *extra: [PANEL.index(w) for w in CONTROL_EXCLUSIONS[word]] + list(extra)
    cat_controls = stats.mass_matched_controls(base, cat, exclude=excl(TARGET))
    dog_controls = stats.mass_matched_controls(base, dog, exclude=excl("dog"))
    k1 = sd(stats.target_stat(cat, (dog,)))
    k1mm = sd(stats.mass_matched_stat(cat, cat_controls, (dog,)))
    k2 = ds(stats.target_stat(dog, (cat,)))
    k2mm = ds(stats.mass_matched_stat(dog, dog_controls, (cat,)))
    k3 = dn(stats.flattening_stat((cat, dog)))
    k5 = sd(stats.flattening_stat((cat, dog)))
    t_dog = dn_dog(stats.target_stat(dog))
    t_dogmm = dn_dog(stats.mass_matched_stat(dog, dog_controls))
    label = {PANEL[w]: sd(stats.dominance_stat(cat, w, (dog,))) for w in range(len(PANEL)) if w not in (cat, dog)}

    lam_sn = stats.lambda_of(_arrays(logq, _pairs(logq, "S")), _arrays(logq, _pairs(logq, "N")), base, (cat, dog),
                             sd.w)(None)
    lam_dn = dn.lam(None)
    cols = np.array([w for w in range(len(PANEL)) if w not in (cat, dog)])
    has_teachers = "T_cat" in logq and "T_dog" in logq
    if has_teachers:
        teacher_ref = stats.mean_distribution(*(logq[f"N{k}"] for k in _seeds(logq, "N")))
        teacher_diff = (stats.residuals(logq["T_cat"], base, teacher_ref, (cat, dog), np.inf)[0][:, cols].mean(axis=0)
                        - stats.residuals(logq["T_dog"], base, teacher_ref, (cat, dog), np.inf)[0][:, cols].mean(axis=0))
    seeds = {}
    for s in sd.treated:
        other_n = next(logq[f"N{k}"] for k in _seeds(logq, "N") if k != s)
        dog_p = {"beta": k3[s].p, "dog": stats.robust_p(t_dog[s], t_dogmm[s])}
        label_p = {w: r[s].p for w, r in label.items()}
        descriptive = {
            "beta_DN": float(np.exp(-k3[s].estimate)), "beta_SD": float(np.exp(-k5[s].estimate)),
            "K4": dataclasses.asdict(stats.shared_movers(logq[f"S{s}"], logq[f"N{s}"], logq[f"D{s}"], other_n, base,
                                                         (cat, dog), stem_ids, lam_s=lam_sn, lam_d=lam_dn)),
        }
        if has_teachers:
            profile = stats.residuals(logq[f"S{s}"], logq[f"D{s}"], base, (cat, dog), sd.lam(None))[0][:, cols].mean(axis=0)
            descriptive["shadow_SD_rho"] = stats.spearman(teacher_diff, profile)
        seeds[s] = {
            "p": {"K1": stats.robust_p(k1[s], k1mm[s]), "K2": stats.robust_p(k2[s], k2mm[s]), "K3": k3[s].p},
            "label": bool(all(p <= stats.ALPHA for p in label_p.values())),
            "k5_p_two": k5[s].p_two,
            "dog_transfer": bool(any(p <= stats.ALPHA for p in dog_p.values())),
            "tests": {name: _rl(r[s]) for name, r in (("K1", k1), ("K1mm", k1mm), ("K2", k2), ("K2mm", k2mm),
                                                      ("K3", k3), ("K5", k5), ("DN_dog", t_dog),
                                                      ("DN_dogmm", t_dogmm))},
            "label_p": label_p,
            "descriptive": descriptive,
        }
    return {"controls": {"K1": [PANEL[w] for w in cat_controls], "K2": [PANEL[w] for w in dog_controls]},
            "seeds": seeds, "confirmatory": list(confirmatory),
            "pooled": {n: _rl(r["pooled"]) for n, r in (("K1", k1), ("K1mm", k1mm), ("K2", k2), ("K2mm", k2mm),
                                                        ("K5", k5))},
            "upper_pooled": {"K1": max(k1["pooled"].upper95, k1mm["pooled"].upper95),
                             "K2": max(k2["pooled"].upper95, k2mm["pooled"].upper95)}}


def p2_outcome(family: Mapping[str, Any], *, c2_confirmed: bool, integrity_ok: bool) -> taxonomy.Outcome:
    """``c2_confirmed``: the P1 C2 decision (cat-teacher students flatten, both seeds)."""
    conf = family["confirmatory"]
    return taxonomy.classify_p2({s: family["seeds"][s]["p"] for s in conf},
                                label={s: family["seeds"][s]["label"] for s in conf},
                                dog_transfer={s: family["seeds"][s]["dog_transfer"] for s in conf},
                                c2_confirmed=c2_confirmed, integrity_ok=integrity_ok)


def analyze_p2(scores, entries: Sequence[Mapping[str, Any]], *, c2_confirmed: bool, integrity_ok: bool,
               n_boot: int = stats.N_BOOT, n_ref: int = stats.N_REF) -> dict:
    """``c2_confirmed``: from the P1 analysis (``outcome['confirmed']['C2']``)."""
    res = [e for e in entries if e["set"] == "RES"]
    stems, families = [e["stem_id"] for e in res], [e["family"] for e in res]
    arms = ["base"] + [f"{c}{s}" for c in "NSD" for s in ALL_SEEDS]
    logq = {a: arm_logq(scores, a, stems, PRIMARY) for a in arms}
    logq["T_cat"] = arm_logq(scores, "T_cat", stems, TEACHER_CELL)
    logq["T_dog"] = arm_logq(scores, "T_dog", stems, TEACHER_CELL)
    family = p2_family(logq, stems, families, n_boot=n_boot, n_ref=n_ref)
    cat, dog = PANEL.index(TARGET), PANEL.index("dog")
    teacher_ref = stats.mean_distribution(*(logq[f"N{k}"] for k in ALL_SEEDS))
    cols = np.array([w for w in range(len(PANEL)) if w not in (cat, dog)])
    teacher_rho = stats.spearman(
        stats.residuals(logq["T_cat"], logq["base"], teacher_ref, (cat, dog), np.inf)[0][:, cols].mean(axis=0),
        stats.residuals(logq["T_dog"], logq["base"], teacher_ref, (cat, dog), np.inf)[0][:, cols].mean(axis=0))
    outcome = p2_outcome(family, c2_confirmed=c2_confirmed, integrity_ok=integrity_ok)
    return {"outcome": outcome.__dict__, "family": family, "teacher_profile_rho": teacher_rho}
