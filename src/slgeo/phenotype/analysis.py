"""Automated P1 (and P2) analysis: shard outputs -> per-arm stem distributions -> hypotheses -> taxonomy.

Scientific outputs are sealed. ``analyze`` refuses unless an unseal record exists that names the preregistration
tag and commit (written only after the freeze; decision D4 order: freeze -> gate files -> P1 run -> unseal).
Confirmatory statistics use the RES stems; REF50 and NONANIMAL are reported separately (development, descriptive).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import stats, taxonomy
from .panel import PANEL, TARGET

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


def _pairs(logq: Mapping[str, np.ndarray], prefix: str) -> list[tuple[np.ndarray, np.ndarray]]:
    names = [f"{prefix}{s}" for s in ALL_SEEDS if f"{prefix}{s}" in logq]
    return [(logq[a], logq[b]) for a, b in itertools.combinations(names, 2)]


def p1_seed(logq: Mapping[str, np.ndarray], seed: str, v1: Mapping[str, float], *, n_boot: int, n_flip: int) -> dict:
    t = PANEL.index(TARGET)
    s, n, base = logq[f"S{seed}"], logq[f"N{seed}"], logq["base"]
    within = _pairs(logq, "S") + _pairs(logq, "N")
    lam = stats.estimate_lambda(_pairs(logq, "S"), _pairs(logq, "N"), base, (t,)).value
    c1 = stats.omnibus(s, n, n_flip=n_flip)
    c2 = stats.flattening(s, n, base, (t,), lam=lam, n_boot=n_boot)
    c3 = stats.target_residual(s, n, base, t, lam=lam, n_boot=n_boot)
    c4 = stats.shadow_concordance(logq["T_cat"], base, stats.mean_distribution(*(logq[f"N{k}"] for k in ALL_SEEDS)),
                                  s, n, t, lam=lam, n_flip=n_flip)
    c5 = stats.profile_replication(v1, PANEL, s, n, base, t, lam=lam, n_flip=n_flip)

    def c1_stat(a, b):
        return stats.omnibus(a, b, n_flip=1).statistic

    def c2_stat(a, b):
        return 1.0 - stats.fit_beta(a, b, np.array([w for w in range(len(PANEL)) if w != t]), base, lam)

    def c3_stat(a, b):
        return float(stats.residuals(a, b, base, (t,), lam)[0][:, t].mean())

    gates = {
        "C1": stats.run_level_gate(c1_stat, (s, n), within).passed,
        "C2": stats.run_level_gate(c2_stat, (s, n), within).passed,
        "C3": stats.run_level_gate(c3_stat, (s, n), within).passed,
    }
    m_run = stats.run_noise_margin(c3_stat, within)
    adequacy = stats.adequacy(s, n, base, (t,), lam=lam, n_boot=n_boot)
    dominance = stats.target_dominance(s, n, base, t, lam=lam, n_flip=n_flip)
    mass = stats.mass_matched_contrast(s, n, base, t, lam=lam, n_boot=n_boot)
    return {
        "lambda": lam,
        "p": {"C1": c1.p, "C2": c2.p_less_than_one, "C3": c3.p, "C4": c4.p, "C5": c5.p},
        "gates": gates,
        "estimates": {"C1_T": c1.statistic, "beta": c2.beta, "beta_ci90": c2.ci90, "C3": c3.mean, "C3_ci95": c3.ci95,
                      "C3_ci90": c3.ci90, "C3_delta_prob": c3.delta_prob, "C4_rho": c4.rho, "C5_rho": c5.rho,
                      "mass_matched": mass.mean, "mass_matched_ci95": mass.ci95, "curvature": adequacy.curvature,
                      "curvature_ci95": adequacy.ci95, "dominance_min_margin": float(np.min(dominance.t - dominance.critical))},
        "m_run": m_run,
        "c3_equivalent_m_run": c3.equivalent(m_run),
        "c3_equivalent_fixed": {str(m): c3.equivalent(m) for m in FIXED_MARGINS},
        "label": dominance.passed,
        "adequate": adequacy.adequate,
    }


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
               integrity_ok: bool, sample_k: int, n_boot: int = stats.N_BOOT, n_flip: int = stats.N_FLIP) -> dict:
    res_stems = [e["stem_id"] for e in entries if e["set"] == "RES"]
    arms = ["base"] + [f"{c}{s}" for c in "NS" for s in ALL_SEEDS]
    result: dict[str, Any] = {}
    for label, cell in (("primary", PRIMARY), ("secondary", SECONDARY)):
        logq = {a: arm_logq(scores, a, res_stems, cell) for a in arms}
        logq["T_cat"] = arm_logq(scores, "T_cat", res_stems, TEACHER_CELL if label == "primary" else ("persona", ("none",)))
        result[label] = {seed: p1_seed(logq, seed, v1, n_boot=n_boot, n_flip=n_flip) for seed in ALL_SEEDS}
    sampled_stems = [e["stem_id"] for e in entries if e["set"] in {"REF50", "RES"}]
    instrument = instrument_check(scores, samples, ["base", "T_cat"] + arms[1:], sampled_stems, sample_k)
    instrument_ok = all(v["ok"] for a, v in instrument.items() if a in {"base", "T_cat", "N2", "S2", "N3", "S3"})

    def classify(block, secondary=None):
        seeds = {s: taxonomy.SeedResult(block[s]["p"], block[s]["gates"], block[s]["c3_equivalent_m_run"],
                                        block[s]["label"], block[s]["adequate"]) for s in CONFIRMATORY_SEEDS}
        return taxonomy.classify_p1(seeds, integrity_ok=integrity_ok, instrument_ok=instrument_ok,
                                    secondary_class=secondary)

    secondary = classify(result["secondary"]).cls
    outcome = classify(result["primary"], secondary)
    return {"outcome": outcome.__dict__, "secondary_class": secondary, "instrument": instrument,
            "per_seed": result, "development_seed": "1"}


def p2_seed(logq: Mapping[str, np.ndarray], seed: str, stem_ids: Sequence[str], *, n_boot: int, n_flip: int) -> dict:
    """K1-K4 for one confirmatory seed (dog-teacher students D1-D3 required)."""
    cat, dog = PANEL.index(TARGET), PANEL.index("dog")
    s, d, n, base = logq[f"S{seed}"], logq[f"D{seed}"], logq[f"N{seed}"], logq["base"]
    other_n = next(logq[f"N{k}"] for k in ALL_SEEDS if k != seed)
    sp, dp, np_ = _pairs(logq, "S"), _pairs(logq, "D"), _pairs(logq, "N")
    lam_sd = stats.estimate_lambda(sp, dp, base, (cat, dog)).value
    lam_ds = 1.0 / lam_sd
    lam_dn = stats.estimate_lambda(dp, np_, base, (cat, dog)).value
    lam_sn = stats.estimate_lambda(sp, np_, base, (cat, dog)).value
    k1 = stats.target_residual(s, d, base, cat, exclude=(dog,), lam=lam_sd, n_boot=n_boot)
    k2 = stats.target_residual(d, s, base, dog, exclude=(cat,), lam=lam_ds, n_boot=n_boot)
    k3 = stats.flattening(d, n, base, (cat, dog), lam=lam_dn, n_boot=n_boot)
    k4 = stats.shared_movers(s, n, d, other_n, base, (cat, dog), stem_ids, lam_s=lam_sn, lam_d=lam_dn, n_flip=n_flip)

    def residual_stat(target, exclude, lam):
        return lambda a, b: float(stats.residuals(a, b, base, (target, *exclude), lam)[0][:, target].mean())

    def beta_stat(a, b):
        return 1.0 - stats.fit_beta(a, b, np.array([w for w in range(len(PANEL)) if w not in (cat, dog)]), base, lam_dn)

    gates = {
        "K1": stats.run_level_gate(residual_stat(cat, (dog,), lam_sd), (s, d), sp + dp).passed,
        "K2": stats.run_level_gate(residual_stat(dog, (cat,), lam_ds), (d, s), sp + dp).passed,
        "K3": stats.run_level_gate(beta_stat, (d, n), dp + np_).passed,
    }
    dog_null = (stats.omnibus(d, n, n_flip=n_flip).p > stats.ALPHA
                and stats.flattening(d, n, base, (dog,), lam=lam_dn, n_boot=n_boot).p_less_than_one > stats.ALPHA
                and stats.target_residual(d, n, base, dog, lam=lam_dn, n_boot=n_boot).p > stats.ALPHA)
    return {"p": {"K1": k1.p, "K2": k2.p, "K3": k3.p_less_than_one, "K4": k4.p}, "gates": gates,
            "estimates": {"K1": k1.mean, "K1_ci95": k1.ci95, "K2": k2.mean, "K2_ci95": k2.ci95, "beta_DN": k3.beta,
                          "K4_rho": k4.rho}, "dog_students_null": dog_null,
            "lambda": {"SD": lam_sd, "DN": lam_dn, "SN": lam_sn}}


def analyze_p2(scores, entries: Sequence[Mapping[str, Any]], *, integrity_ok: bool, n_boot: int = stats.N_BOOT,
               n_flip: int = stats.N_FLIP) -> dict:
    stems = [e["stem_id"] for e in entries if e["set"] == "RES"]
    arms = ["base"] + [f"{c}{s}" for c in "NSD" for s in ALL_SEEDS]
    logq = {a: arm_logq(scores, a, stems, PRIMARY) for a in arms}
    per_seed = {seed: p2_seed(logq, seed, stems, n_boot=n_boot, n_flip=n_flip) for seed in CONFIRMATORY_SEEDS}
    cat, dog = PANEL.index(TARGET), PANEL.index("dog")
    teacher_ref = stats.mean_distribution(*(logq[f"N{k}"] for k in ALL_SEEDS))
    cols = np.array([w for w in range(len(PANEL)) if w not in (cat, dog)])
    t_cat = arm_logq(scores, "T_cat", stems, TEACHER_CELL)
    t_dog = arm_logq(scores, "T_dog", stems, TEACHER_CELL)
    teacher_rho = stats.spearman(
        stats.residuals(t_cat, logq["base"], teacher_ref, (cat, dog), np.inf)[0][:, cols].mean(axis=0),
        stats.residuals(t_dog, logq["base"], teacher_ref, (cat, dog), np.inf)[0][:, cols].mean(axis=0))
    outcome = taxonomy.classify_p2({s: per_seed[s]["p"] for s in per_seed}, {s: per_seed[s]["gates"] for s in per_seed},
                                   dog_students_null=all(per_seed[s]["dog_students_null"] for s in per_seed),
                                   integrity_ok=integrity_ok)
    return {"outcome": outcome.__dict__, "per_seed": per_seed, "teacher_profile_rho": teacher_rho}
