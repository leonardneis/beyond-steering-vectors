"""Automated P1 and P2 analysis: shard outputs -> per-arm stem distributions -> hypotheses -> taxonomy.

Scientific outputs are sealed. ``run_stage`` refuses unless an unseal record exists that names the preregistration
tag and commit (written only after the freeze; decision D4 order: freeze -> gate files -> P1 run -> unseal).
Confirmatory statistics use the RES stems; REF50 and NONANIMAL are reported separately (development, descriptive).

Stages (``STAGE_FILES``, each result written once under ``<out>/analysis/``):
  p1          P1 on seeds 1-3 (1 development, 2 and 3 confirmatory); reports the fresh-seed trigger.
  p1-seeds45  only if the stored p1 result fired the trigger: P1 on seeds 1-5 with seeds 4 and 5 confirmatory, and
              the final two-stage P1 outcome (``taxonomy.classify_p1_two_stage``, decision R2).
  p2          P2 on seeds 1-3 (never re-run with seeds 4/5); reads the P1 C2 decision and the P1 instrument result
              from the stored p1 result.
Integrity and data per stage (``STAGE_PLANS``): p1 reads ``plan.json``; p2 also ``plan_p2.json`` (D1-D3); p1-seeds45
also ``plan_p1-seeds45.json`` (N4, S4, N5, S5); only the shards of those plans are loaded. A stage whose plans are
absent or incomplete, or whose required arms, seeds, cells or samples are missing, is refused without writing
(``StageNotReady``) unless ``final=True``, which records the documented TECHNICAL_FAIL instead; a degenerate
confirmatory statistic is a TECHNICAL_FAIL; a degenerate descriptive statistic is recorded as ``{"error": ...}`` and
never changes a class.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import fast, stats, taxonomy
from .panel import CONTROL_EXCLUSIONS, PANEL, TARGET

CONFIRMATORY_SEEDS = ("2", "3")
ALL_SEEDS = ("1", "2", "3")
FRESH_SEEDS = ("4", "5")  # second confirmatory pair of the fresh-seed stage (decision R2)
STAGE2_SEEDS = ALL_SEEDS + FRESH_SEEDS
PRIMARY = ("Q", ("r0", "r1", "r2"))
SECONDARY = ("Q", ("none",))
TEACHER_CELL = ("persona", ("r0", "r1", "r2"))
FIXED_MARGINS = (0.05, 0.10, 0.15, 0.20, 0.30)
COVERAGE_LIMIT = 0.05
STAGE_FILES = {"p1": "p1_analysis.json", "p1-seeds45": "p1_seeds45_analysis.json", "p2": "p2_analysis.json"}
STAGE_PLANS = {"p1": ("plan.json",), "p2": ("plan.json", "plan_p2.json"),
               "p1-seeds45": ("plan.json", "plan_p1-seeds45.json")}
DATA_ENTROPY_FILE = "p2_data_entropy.json"  # {"dog": nats, "neutral": nats}: the CPU number-entropy record (§9.1)


class SealedError(RuntimeError):
    pass


class StageOrderError(RuntimeError):
    """A stage requested out of the preregistered order: p2 or p1-seeds45 without a stored p1 result, p1-seeds45
    without the fresh-seed trigger, or a stage result that already exists (write-once)."""


class StageNotReady(StageOrderError):
    """The stage's plans are absent or incomplete, or required outputs are missing; nothing is written (rerun when
    the outputs exist, or record the TECHNICAL_FAIL deliberately with ``final=True``)."""


def require_unsealed(out_root: Path, *, expected_tag: str) -> dict:
    record = out_root / "UNSEAL.json"
    if not record.is_file():
        raise SealedError("Outputs are sealed: no UNSEAL.json (written only after the preregistration freeze)")
    data = json.loads(record.read_text(encoding="utf-8"))
    if data.get("prereg_tag") != expected_tag or not data.get("prereg_commit"):
        raise SealedError("UNSEAL.json does not name the frozen preregistration")
    return data


def stage_shards(out_root: Path, stage: str) -> list[str] | None:
    """Shard ids of every plan the stage reads (``STAGE_PLANS``); None if one of those plans does not exist."""
    shards = []
    for name in STAGE_PLANS[stage]:
        path = out_root / name
        if not path.is_file():
            return None
        shards += [s["shard_id"] for s in json.loads(path.read_text(encoding="utf-8"))["shards"]]
    return shards


def plan_complete(out_root: Path, stage: str = "p1") -> bool:
    """Integrity: the stage's plans exist and every shard of them has its COMPLETE marker."""
    shards = stage_shards(out_root, stage)
    return shards is not None and all((out_root / "raw" / s / "COMPLETE").exists() for s in shards)


def _shard_dirs(out_root: Path, kind: str, shard_ids: Sequence[str] | None) -> list[Path]:
    if shard_ids is None:
        return sorted((out_root / "raw").glob(f"*.{kind}.*"))
    return [out_root / "raw" / s for s in shard_ids if f".{kind}." in s]


def load_samples(out_root: Path, shard_ids: Sequence[str] | None = None) -> list[dict]:
    """Sample rows of the given shards (default: every sample shard)."""
    return [json.loads(line) for d in _shard_dirs(out_root, "sample", shard_ids)
            for line in (d / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def load_scores(out_root: Path, shard_ids: Sequence[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """context_id -> {word_logp, decoration_mass, emoji_mass, ...} from the given complete score shards (default:
    every score shard). A context id in two shards is an integrity error."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for shard in _shard_dirs(out_root, "score", shard_ids):
        if not (shard / "COMPLETE").exists():
            raise stats.PhenotypeStatsError(f"Incomplete shard {shard.name}")
        with np.load(shard / "scores.npz") as npz:
            data = {k: npz[k] for k in npz.files}  # each array read once (NpzFile re-reads on every access)
        for i, cid in enumerate(data["context_ids"]):
            if str(cid) in out:
                raise stats.PhenotypeStatsError(f"Context {cid} scored twice (shard {shard.name})")
            out[str(cid)] = {k: v[i] for k, v in data.items() if k != "context_ids"}
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


def _descriptive(fn):
    """A descriptive quantity: its value, or {"error": reason} when it is undefined (no class, modifier, trigger or
    branch depends on it, so it never fails a stage)."""
    try:
        return fn()
    except stats.PhenotypeStatsError as exc:
        return {"error": str(exc)}


class _Tests:
    """Run-level tests of one contrast family (treated condition y vs reference x) with shared settings.

    ``engine``: "reference" runs ``stats.run_level`` once per statistic (the oracle); "fast" evaluates the same
    procedure with the same RNG streams in ``fast.Family``; "fast-cuda" also runs the pivot's grid supremum on the GPU
    (equivalence: ``tests/test_phenotype_fast.py``)."""

    def __init__(self, logq, y, x, targets, *, families, confirmatory, n_boot, n_ref, seed, engine="reference"):
        self.logq, self.base = logq, logq["base"]
        self.w = stats.family_weights(families)
        self.strata = list(families)
        py, px = _pairs(logq, y), _pairs(logq, x)
        self.within = py + px
        self.lam_pairs = (py, px, tuple(targets))
        self.lam = stats.lambda_of(_arrays(logq, py), _arrays(logq, px), self.base, targets, self.w)
        seeds = sorted(set(_seeds(logq, y)) & set(_seeds(logq, x)), key=int)
        self.treated = {s: (f"{y}{s}", f"{x}{s}") for s in seeds}
        self.confirmatory = tuple(confirmatory)
        missing = [s for s in self.confirmatory if s not in self.treated]
        if missing:
            raise stats.PhenotypeStatsError(f"Confirmatory seed(s) {', '.join(missing)} missing for {y} vs {x}")
        self.kw = dict(n_boot=n_boot, n_ref=n_ref, seed=seed)
        if engine not in ("reference", "fast", "fast-cuda"):
            raise ValueError(engine)
        self.engine = engine

    def run(self, specs):
        """{name: fast.Spec} -> {name: {seed or "pooled": stats.RunLevel}}."""
        if self.engine == "fast":
            family = fast.Family(self.logq, self.treated, self.within, self.base, self.lam_pairs, stem_w=self.w,
                                 strata=self.strata, pooled=self.confirmatory, **self.kw)
            return family.run(specs)
        if self.engine == "fast-cuda":
            family = fast.Family(self.logq, self.treated, self.within, self.base, self.lam_pairs, stem_w=self.w,
                                 strata=self.strata, pooled=self.confirmatory, device="cuda", **self.kw)
            return family.run(specs)
        return {name: stats.run_level(spec.reference(), self.logq, self.treated, self.within, self.base, self.lam,
                                      stem_w=self.w, strata=self.strata, pooled=self.confirmatory, **self.kw)
                for name, spec in specs.items()}


def p1_family(logq: Mapping[str, np.ndarray], v1: Mapping[str, float], families: Sequence[str], *, n_boot: int,
              n_ref: int, confirmatory: Sequence[str] = CONFIRMATORY_SEEDS, seed: int = stats.SEED,
              engine: str = "reference") -> dict:
    """P1 run-level tests for every seed at once (within-condition pairs over all seeds present, lambda-hat shared).

    Confirmatory per seed: C2 (-log beta) and the robust C3 claim (tempering residual and mass-matched contrast,
    intersection-union); the cat-dominance label (all 25 contrasts d_w pass, intersection-union). The bound U is the
    larger of the two components' pooled-seed upper bounds. Everything else is descriptive."""
    t = PANEL.index(TARGET)
    base = logq["base"]
    tests = _Tests(logq, "S", "N", (t,), families=families, confirmatory=confirmatory, n_boot=n_boot, n_ref=n_ref,
                   seed=seed, engine=engine)
    controls = stats.mass_matched_controls(base, t, exclude=[PANEL.index(w) for w in CONTROL_EXCLUSIONS[TARGET]],
                                           stem_w=tests.w)
    others = [w for w in range(len(PANEL)) if w != t]
    out = tests.run({"C2": fast.beta_spec((t,)), "C3": fast.target_spec(t), "C3mm": fast.mm_spec(t, controls),
                     "curvature": fast.curvature_spec((t,)),
                     **{f"dom:{PANEL[w]}": fast.dominance_spec(t, w) for w in others}})
    c2, c3, c3mm, curvature = out["C2"], out["C3"], out["C3mm"], out["curvature"]
    dominance = {PANEL[w]: out[f"dom:{PANEL[w]}"] for w in others}

    lam_hat = tests.lam(None)
    c3_pair = stats.target_stat(t)
    m_run = _descriptive(lambda: stats.run_noise_margin(lambda a, b: c3_pair(a, b, base, 1.0, tests.w),
                                                        _arrays(logq, tests.within)))
    teacher_ref = stats.mean_distribution(*(logq[f"N{k}"] for k in _seeds(logq, "N")))
    direct = np.array([f == "direct" for f in families])
    mm_stat = stats.mass_matched_stat(t, controls)

    def c1_block(y, x):
        c1 = stats.omnibus(y, x, n_flip=min(n_boot, stats.N_FLIP), seed=seed, stem_w=tests.w)
        gate = stats.run_level_gate(lambda a, b: stats.omnibus(a, b, n_flip=1, stem_w=tests.w).statistic, (y, x),
                                    _arrays(logq, tests.within))
        return {"T": c1.statistic, "p_stem": c1.p, "gate_v1": gate.passed}

    def direct_block(y, x, c3_all, c3mm_all):
        """Direct-stem stratum (§3): point estimates at lambda-hat and their sign agreement with all stems."""
        if not direct.any():
            raise stats.PhenotypeStatsError("No direct-family stems")
        d = {"C3": c3_pair(y[direct], x[direct], base[direct], lam_hat),
             "C3mm": mm_stat(y[direct], x[direct], base[direct], lam_hat)}
        d["same_sign_as_all_stems"] = bool(np.sign(d["C3"]) == np.sign(c3_all)
                                           and np.sign(d["C3mm"]) == np.sign(c3mm_all))
        return d

    seeds = {}
    for s in tests.treated:
        y, x = logq[f"S{s}"], logq[f"N{s}"]
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
                "C1": _descriptive(lambda: c1_block(y, x)),
                "C4": _descriptive(lambda: dataclasses.asdict(stats.shadow_concordance(
                    logq["T_cat"], base, teacher_ref, y, x, t, lam=lam_hat, stem_w=tests.w))),
                "C5": _descriptive(lambda: dataclasses.asdict(stats.profile_replication(
                    v1, PANEL, y, x, base, t, lam=lam_hat, stem_w=tests.w))),
                "direct_stratum": _descriptive(lambda: direct_block(y, x, c3[s].estimate, c3mm[s].estimate)),
                # TOST tables of the noise-benchmark layer: no claim may be read from them (decision R1)
                "c3_tost_fixed_margins_descriptive": {str(m): c3[s].equivalent(m) for m in FIXED_MARGINS},
                "c3_tost_m_run_descriptive": (c3[s].equivalent(m_run) if isinstance(m_run, float) else None),
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
                   lam: float, targets: Sequence[int] = (PANEL.index(TARGET),)) -> float:
    """Descriptive: mean over the prefix replicates of the Deming beta fitted on single-replicate conditionals
    (``targets`` out of fit, as in the contrast: cat for C2, cat and dog for K3). Its gap to the prefix-averaged beta
    separates mixture flattening (probability-scale averaging of prefix-inconsistent answers) from a tempering of each
    conditional (decision R5)."""
    words = np.array([w for w in range(len(PANEL)) if w not in targets])
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


def _missing_contexts(scores, requests) -> list[str]:
    """``requests``: (arm, stems, (rendering, prefixes)); one line per arm and cell with absent contexts."""
    out = []
    for arm, stems, (name, prefixes) in requests:
        n = sum(f"{arm}|{s}|{name}+{p}" not in scores for s in stems for p in prefixes)
        if n:
            out.append(f"{arm} {name}+{'/'.join(prefixes)}: {n} contexts missing")
    return out


def _sampled_cell(arm: str) -> tuple[str, tuple[str, ...]]:
    return ("persona", ("r0",)) if arm.startswith("T_") else ("Q", ("r0",))


def _missing_samples(samples, arms: Sequence[str], stems: Sequence[str], k: int) -> list[str]:
    """Every sampled context of the checked arms must hold exactly k sample rows (the runner writes one per draw)."""
    counts = collections.Counter(row["context_id"] for row in samples)
    out = []
    for arm in arms:
        name, (prefix,) = _sampled_cell(arm)
        n = sum(counts.get(f"{arm}|{s}|{name}+{prefix}", 0) != k for s in stems)
        if n:
            out.append(f"{arm} {name}+{prefix}: {n} stems without exactly {k} samples")
    return out


def _technical_fail(reason: str, kind: str, **extra) -> dict:
    """``kind``: "integrity" (plan incomplete), "missing" (required outputs), "statistics" (degenerate confirmatory
    statistic) or "p1" (P2 input); ``run_stage`` refuses the first two unless final."""
    return {"outcome": taxonomy.technical_fail(reason).__dict__, "technical_fail": reason,
            "technical_fail_kind": kind, **extra}


def _p1_stage(scores, samples, entries, v1, *, seeds, confirmatory, checked_arms, instrument_arms, integrity_ok,
              sample_k, n_boot, n_ref, engine) -> dict:
    """One P1 stage: primary and secondary cell families on the given seeds, instrument check, P1 outcome."""
    if not integrity_ok:
        return _technical_fail("integrity: a planned shard is incomplete or a plan is missing", "integrity")
    res = [e for e in entries if e["set"] == "RES"]
    res_stems, families = [e["stem_id"] for e in res], [e["family"] for e in res]
    sampled_stems = [e["stem_id"] for e in entries if e["set"] in {"REF50", "RES"}]
    arms = ["base"] + [f"{c}{s}" for c in "NS" for s in seeds]
    missing = _missing_contexts(scores, [(a, res_stems, c) for a in arms for c in (PRIMARY, SECONDARY)]
                                + [("T_cat", res_stems, TEACHER_CELL), ("T_cat", res_stems, ("persona", ("none",)))]
                                + [(a, sampled_stems, _sampled_cell(a)) for a in checked_arms])
    missing += _missing_samples(samples, checked_arms, sampled_stems, sample_k)
    if missing:
        return _technical_fail("missing required outputs: " + "; ".join(missing), "missing")

    def family(cell, teacher_cell):
        logq = {a: arm_logq(scores, a, res_stems, cell) for a in arms}
        logq["T_cat"] = arm_logq(scores, "T_cat", res_stems, teacher_cell)
        return p1_family(logq, v1, families, n_boot=n_boot, n_ref=n_ref, confirmatory=confirmatory, engine=engine)

    try:
        result: dict[str, Any] = {"primary": family(PRIMARY, TEACHER_CELL)}
        instrument = instrument_check(scores, samples, checked_arms, sampled_stems, sample_k)
    except stats.PhenotypeStatsError as exc:
        return _technical_fail(f"statistics: {exc}", "statistics")
    result["secondary"] = _descriptive(lambda: family(SECONDARY, ("persona", ("none",))))  # Q+none, descriptive
    for s, block in result["primary"]["seeds"].items():
        block["descriptive"]["beta_per_replicate"] = _descriptive(lambda: replicate_beta(
            scores, f"S{s}", f"N{s}", res_stems, families, result["primary"]["lambda"]))
    instrument_ok = all(instrument[a]["ok"] for a in instrument_arms)
    secondary = ("unavailable: " + result["secondary"]["error"] if "error" in result["secondary"] else
                 p1_outcome(result["secondary"], integrity_ok=True, instrument_ok=instrument_ok).cls)
    outcome = p1_outcome(result["primary"], integrity_ok=True, instrument_ok=instrument_ok)
    return {"outcome": outcome.__dict__, "secondary_class_descriptive": secondary, "instrument": instrument,
            "instrument_arms": list(instrument_arms), "families": result, "development_seed": "1",
            "seeds": list(seeds), "confirmatory": list(confirmatory)}


def analyze_p1(scores, samples, entries: Sequence[Mapping[str, Any]], v1: Mapping[str, float], *,
               integrity_ok: bool, sample_k: int, n_boot: int = stats.N_BOOT, n_ref: int = stats.N_REF,
               engine: str = "reference") -> dict:
    """Stage p1: seeds 1-3, confirmatory 2 and 3. Instrument: base, T_cat, N2, S2, N3, S3 decide INSTRUMENT_FAIL
    (seed 1 is checked and reported)."""
    checked = ["base", "T_cat"] + [f"{c}{s}" for c in "NS" for s in ALL_SEEDS]
    out = _p1_stage(scores, samples, entries, v1, seeds=ALL_SEEDS, confirmatory=CONFIRMATORY_SEEDS,
                    checked_arms=checked,
                    instrument_arms=["base", "T_cat"] + [f"{c}{s}" for c in "NS" for s in CONFIRMATORY_SEEDS],
                    integrity_ok=integrity_ok, sample_k=sample_k, n_boot=n_boot, n_ref=n_ref, engine=engine)
    out["fresh_seed_trigger"] = taxonomy.fresh_seed_trigger(taxonomy.outcome_from_dict(out["outcome"]))
    return out


def analyze_p1_seeds45(scores, samples, entries: Sequence[Mapping[str, Any]], v1: Mapping[str, float], *,
                       p1: Mapping[str, Any], integrity_ok: bool, sample_k: int, n_boot: int = stats.N_BOOT,
                       n_ref: int = stats.N_REF, engine: str = "reference") -> dict:
    """Stage p1-seeds45 (decision R2): only after the stored stage-p1 result ``p1`` fired the fresh-seed trigger.
    Identical procedure on seeds 1-5 (within-condition pairs over five runs per condition, lambda-hat from them),
    seeds 4 and 5 confirmatory; instrument on base, T_cat, N4, S4, N5, S5. Returns the stage-2 result and the final
    two-stage P1 outcome."""
    stage1 = taxonomy.outcome_from_dict(p1["outcome"])
    if not taxonomy.fresh_seed_trigger(stage1):
        raise StageOrderError("The stored P1 result did not fire the fresh-seed trigger; seeds 4/5 are not analysed")
    checked = ["base", "T_cat"] + [f"{c}{s}" for c in "NS" for s in FRESH_SEEDS]
    stage2 = _p1_stage(scores, samples, entries, v1, seeds=STAGE2_SEEDS, confirmatory=FRESH_SEEDS,
                       checked_arms=checked, instrument_arms=checked, integrity_ok=integrity_ok, sample_k=sample_k,
                       n_boot=n_boot, n_ref=n_ref, engine=engine)
    final = taxonomy.classify_p1_two_stage(stage1, taxonomy.outcome_from_dict(stage2["outcome"]))
    return {"final_outcome": final.__dict__, "stage1_class": stage1.cls, "stage2": stage2}


def p2_family(logq: Mapping[str, np.ndarray], stem_ids: Sequence[str], families: Sequence[str], *, n_boot: int,
              n_ref: int, confirmatory: Sequence[str] = CONFIRMATORY_SEEDS, seed: int = stats.SEED,
              engine: str = "reference") -> dict:
    """P2 run-level tests for every seed at once (dog-teacher students D1-D3 required).

    Confirmatory per seed: K1 robust (cat residual of S_k vs D_k, dog excluded), K2 robust (dog residual of D_k vs
    S_k, cat excluded), K3 (-log beta of D_k vs N_k); P2 cat label (cat above every non-trait word in S_k vs D_k);
    K5 (-log beta of S_k vs D_k, two-sided; reported); dog-transfer check (D_k vs N_k flattening or robust dog
    residual, each at alpha, no multiplicity correction: a non-detection class must not become easier to reach).
    K4 and the S-D teacher-shadow correlation are descriptive."""
    cat, dog = PANEL.index(TARGET), PANEL.index("dog")
    base = logq["base"]
    kw = dict(families=families, confirmatory=confirmatory, n_boot=n_boot, n_ref=n_ref, seed=seed, engine=engine)
    sd = _Tests(logq, "S", "D", (cat, dog), **kw)
    ds = _Tests(logq, "D", "S", (cat, dog), **kw)
    dn = _Tests(logq, "D", "N", (cat, dog), **kw)
    dn_dog = _Tests(logq, "D", "N", (dog,), **kw)
    excl = lambda word, *extra: [PANEL.index(w) for w in CONTROL_EXCLUSIONS[word]] + list(extra)
    cat_controls = stats.mass_matched_controls(base, cat, exclude=excl(TARGET), stem_w=sd.w)
    dog_controls = stats.mass_matched_controls(base, dog, exclude=excl("dog"), stem_w=sd.w)
    others = [w for w in range(len(PANEL)) if w not in (cat, dog)]
    r_sd = sd.run({"K1": fast.target_spec(cat, (dog,)), "K1mm": fast.mm_spec(cat, cat_controls, (dog,)),
                   "K5": fast.beta_spec((cat, dog)),
                   **{f"label:{PANEL[w]}": fast.dominance_spec(cat, w, (dog,)) for w in others}})
    r_ds = ds.run({"K2": fast.target_spec(dog, (cat,)), "K2mm": fast.mm_spec(dog, dog_controls, (cat,))})
    k3 = dn.run({"K3": fast.beta_spec((cat, dog))})["K3"]
    r_dn = dn_dog.run({"DN_dog": fast.target_spec(dog), "DN_dogmm": fast.mm_spec(dog, dog_controls)})
    k1, k1mm, k5, k2, k2mm = r_sd["K1"], r_sd["K1mm"], r_sd["K5"], r_ds["K2"], r_ds["K2mm"]
    t_dog, t_dogmm = r_dn["DN_dog"], r_dn["DN_dogmm"]
    label = {PANEL[w]: r_sd[f"label:{PANEL[w]}"] for w in others}

    lam_sn = stats.lambda_of(_arrays(logq, _pairs(logq, "S")), _arrays(logq, _pairs(logq, "N")), base, (cat, dog),
                             sd.w)(None)
    lam_dn = dn.lam(None)
    cols = np.array([w for w in range(len(PANEL)) if w not in (cat, dog)])
    has_teachers = "T_cat" in logq and "T_dog" in logq
    if has_teachers:
        teacher_ref = stats.mean_distribution(*(logq[f"N{k}"] for k in _seeds(logq, "N")))
        teacher_diff = _descriptive(lambda: _profile(logq["T_cat"], base, teacher_ref, (cat, dog), cols, np.inf, sd.w)
                                    - _profile(logq["T_dog"], base, teacher_ref, (cat, dog), cols, np.inf, sd.w))
    seeds = {}
    for s in sd.treated:
        other_n = next(logq[f"N{k}"] for k in _seeds(logq, "N") if k != s)
        dog_p = {"beta": k3[s].p, "dog": stats.robust_p(t_dog[s], t_dogmm[s])}
        label_p = {w: r[s].p for w, r in label.items()}
        descriptive = {
            "beta_DN": float(np.exp(-k3[s].estimate)), "beta_SD": float(np.exp(-k5[s].estimate)),
            "K4": _descriptive(lambda: dataclasses.asdict(stats.shared_movers(
                logq[f"S{s}"], logq[f"N{s}"], logq[f"D{s}"], other_n, base, (cat, dog), stem_ids, lam_s=lam_sn,
                lam_d=lam_dn, families=families))),
        }
        if has_teachers:
            descriptive["shadow_SD_rho"] = (teacher_diff if isinstance(teacher_diff, dict) else _descriptive(
                lambda: stats.spearman(teacher_diff, _profile(logq[f"S{s}"], logq[f"D{s}"], base, (cat, dog), cols,
                                                              sd.lam(None), sd.w))))
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


def p2_outcome(family: Mapping[str, Any], *, c2_confirmed: bool, integrity_ok: bool,
               instrument_ok: bool) -> taxonomy.Outcome:
    """``c2_confirmed``: the P1 C2 decision (cat-teacher students flatten, both seeds); ``instrument_ok``: the P2
    instrument rule (``analyze_p2``)."""
    conf = family["confirmatory"]
    return taxonomy.classify_p2({s: family["seeds"][s]["p"] for s in conf},
                                label={s: family["seeds"][s]["label"] for s in conf},
                                dog_transfer={s: family["seeds"][s]["dog_transfer"] for s in conf},
                                c2_confirmed=c2_confirmed, integrity_ok=integrity_ok, instrument_ok=instrument_ok)


def analyze_p2(scores, samples, entries: Sequence[Mapping[str, Any]], *, p1: Mapping[str, Any], integrity_ok: bool,
               sample_k: int, n_boot: int = stats.N_BOOT, n_ref: int = stats.N_REF, engine: str = "reference",
               data_entropy: Mapping[str, float] | None = None) -> dict:
    """Stage p2 on seeds 1-3. ``p1``: the stored stage-p1 result; it supplies the C2 decision and the P1 instrument
    result. A P1 TECHNICAL_FAIL makes P2 a TECHNICAL_FAIL (its C2 input and shared arms are unavailable); a P1
    INSTRUMENT_FAIL, or a coverage / agreement failure of D2, D3 or T_dog, makes P2 an INSTRUMENT_FAIL (D1 is checked
    and reported). ``data_entropy``: the CPU number-entropy record {"dog", "neutral"} for the P2b trigger (§9.4);
    without it the trigger is recorded as undetermined."""
    p1_out = p1["outcome"]
    p1_input = {"class": p1_out["cls"], "C2_confirmed": bool(p1_out["confirmed"].get("C2", False)),
                "instrument_ok": p1_out["cls"] != "INSTRUMENT_FAIL"}
    if not integrity_ok:
        return _technical_fail("integrity: a planned shard is incomplete or a plan is missing", "integrity",
                               p1_input=p1_input)
    if p1_out["cls"] == "TECHNICAL_FAIL":
        return _technical_fail("the P1 analysis is a TECHNICAL_FAIL (no C2 decision, shared arms unavailable)", "p1",
                               p1_input=p1_input)
    res = [e for e in entries if e["set"] == "RES"]
    stems, families = [e["stem_id"] for e in res], [e["family"] for e in res]
    sampled_stems = [e["stem_id"] for e in entries if e["set"] in {"REF50", "RES"}]
    arms = ["base"] + [f"{c}{s}" for c in "NSD" for s in ALL_SEEDS]
    checked = [f"D{s}" for s in ALL_SEEDS] + ["T_dog"]
    missing = _missing_contexts(scores, [(a, stems, PRIMARY) for a in arms]
                                + [(a, stems, TEACHER_CELL) for a in ("T_cat", "T_dog")]
                                + [(a, sampled_stems, _sampled_cell(a)) for a in checked])
    missing += _missing_samples(samples, checked, sampled_stems, sample_k)
    if missing:
        return _technical_fail("missing required outputs: " + "; ".join(missing), "missing", p1_input=p1_input)
    try:
        logq = {a: arm_logq(scores, a, stems, PRIMARY) for a in arms}
        logq["T_cat"] = arm_logq(scores, "T_cat", stems, TEACHER_CELL)
        logq["T_dog"] = arm_logq(scores, "T_dog", stems, TEACHER_CELL)
        family = p2_family(logq, stems, families, n_boot=n_boot, n_ref=n_ref, engine=engine)
        instrument = instrument_check(scores, samples, checked, sampled_stems, sample_k)
    except stats.PhenotypeStatsError as exc:
        return _technical_fail(f"statistics: {exc}", "statistics", p1_input=p1_input)
    cat, dog = PANEL.index(TARGET), PANEL.index("dog")
    w = stats.family_weights(families)
    teacher_ref = stats.mean_distribution(*(logq[f"N{k}"] for k in ALL_SEEDS))
    cols = np.array([i for i in range(len(PANEL)) if i not in (cat, dog)])
    teacher_rho = _descriptive(lambda: stats.spearman(
        _profile(logq["T_cat"], logq["base"], teacher_ref, (cat, dog), cols, np.inf, w),
        _profile(logq["T_dog"], logq["base"], teacher_ref, (cat, dog), cols, np.inf, w)))
    lam_dn = _descriptive(lambda: stats.lambda_of(_arrays(logq, _pairs(logq, "D")), _arrays(logq, _pairs(logq, "N")),
                                                  logq["base"], (cat, dog), w)(None))
    for s, block in family["seeds"].items():  # per-replicate beta next to K3 (decision R5)
        block["descriptive"]["beta_DN_per_replicate"] = (lam_dn if isinstance(lam_dn, dict) else _descriptive(
            lambda: replicate_beta(scores, f"D{s}", f"N{s}", stems, families, lam_dn, (cat, dog))))
    instrument_arms = [f"D{s}" for s in CONFIRMATORY_SEEDS] + ["T_dog"]
    instrument_ok = p1_input["instrument_ok"] and all(instrument[a]["ok"] for a in instrument_arms)
    outcome = p2_outcome(family, c2_confirmed=p1_input["C2_confirmed"], integrity_ok=True,
                         instrument_ok=instrument_ok)
    entropy = data_entropy or {}
    p2b = {"value": taxonomy.p2b_trigger(bool(outcome.confirmed.get("K3")), entropy.get("dog"),
                                         entropy.get("neutral")),
           "K3_confirmed": bool(outcome.confirmed.get("K3")), "dog_entropy": entropy.get("dog"),
           "neutral_entropy": entropy.get("neutral")}
    return {"outcome": outcome.__dict__, "family": family, "teacher_profile_rho": teacher_rho,
            "instrument": instrument, "instrument_arms": instrument_arms, "p1_input": p1_input, "p2b_trigger": p2b}


def _profile(y, x, ref, targets, cols, lam, w) -> np.ndarray:
    """Stem-weighted mean residual profile over ``cols`` (fit weighted alike)."""
    return stats.wmean_columns(stats.residuals(y, x, ref, targets, lam, w)[0][:, cols], w)


# --- stages on stored outputs (the CLI and the end-to-end audit call these) --------------------------------------


def _json_default(obj):
    return obj.tolist() if hasattr(obj, "tolist") else str(obj)


def read_stage(out_root: Path, stage: str) -> dict:
    target = out_root / "analysis" / STAGE_FILES[stage]
    if not target.is_file():
        raise StageOrderError(f"No stored {stage} result ({target.name}); run that stage first")
    return json.loads(target.read_text(encoding="utf-8"))


def write_stage(out_root: Path, stage: str, result: Mapping[str, Any]) -> Path:
    """Write a stage result once (``StageOrderError`` if it exists)."""
    target = out_root / "analysis" / STAGE_FILES[stage]
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(target, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(result, indent=1, default=_json_default))
    except FileExistsError:
        raise StageOrderError(f"{target.name} exists; analyses are write-once") from None
    return target


def run_stage(out_root: Path, stage: str, *, entries: Sequence[Mapping[str, Any]], v1: Mapping[str, float],
              sample_k: int, expected_tag: str, n_boot: int | None = None, n_ref: int | None = None,
              engine: str = "reference", final: bool = False) -> dict:
    """One analysis stage on stored outputs: unseal check, stage order, the stage's plans (``STAGE_PLANS``) and their
    shards only, then ``analyze_p1`` / ``analyze_p1_seeds45`` / ``analyze_p2``. ``n_boot`` / ``n_ref`` default to the
    preregistered ``stats.N_BOOT`` / ``stats.N_REF``. Absent or incomplete plans and missing required outputs raise
    ``StageNotReady`` (nothing to write) unless ``final``, which returns the documented TECHNICAL_FAIL. The result is
    returned, not written (``write_stage``)."""
    if stage not in STAGE_FILES:
        raise ValueError(f"Unknown stage {stage!r}")
    require_unsealed(out_root, expected_tag=expected_tag)
    if (out_root / "analysis" / STAGE_FILES[stage]).exists():
        raise StageOrderError(f"{STAGE_FILES[stage]} exists; analyses are write-once")
    p1 = read_stage(out_root, "p1") if stage != "p1" else None
    if stage == "p1-seeds45" and not p1.get("fresh_seed_trigger"):
        raise StageOrderError("The stored P1 result did not fire the fresh-seed trigger; seeds 4/5 are not analysed")
    kw = dict(n_boot=stats.N_BOOT if n_boot is None else n_boot, n_ref=stats.N_REF if n_ref is None else n_ref,
              engine=engine)
    shards = stage_shards(out_root, stage)
    integrity_ok = plan_complete(out_root, stage)
    try:
        scores, samples = (load_scores(out_root, shards), load_samples(out_root, shards)) if integrity_ok else ({}, [])
    except stats.PhenotypeStatsError as exc:  # e.g. a context scored twice: an integrity problem of the outputs
        if not final:
            raise StageNotReady(f"{stage} not ready: {exc}") from None
        integrity_ok, scores, samples = False, {}, []
    if stage == "p1":
        result = analyze_p1(scores, samples, entries, v1, integrity_ok=integrity_ok, sample_k=sample_k, **kw)
    elif stage == "p1-seeds45":
        result = analyze_p1_seeds45(scores, samples, entries, v1, p1=p1, integrity_ok=integrity_ok,
                                    sample_k=sample_k, **kw)
    else:
        entropy_file = out_root / DATA_ENTROPY_FILE
        entropy = json.loads(entropy_file.read_text(encoding="utf-8")) if entropy_file.is_file() else None
        result = analyze_p2(scores, samples, entries, p1=p1, integrity_ok=integrity_ok, sample_k=sample_k,
                            data_entropy=entropy, **kw)
        if "p2b_trigger" in result:  # provenance of the entropy record the trigger used
            result["p2b_trigger"]["entropy_record_sha256"] = (
                hashlib.sha256(entropy_file.read_bytes()).hexdigest() if entropy is not None else None)
    failed = result["stage2"] if stage == "p1-seeds45" else result
    if not final and failed.get("technical_fail_kind") in ("integrity", "missing"):
        raise StageNotReady(f"{stage} not ready: {failed['technical_fail']}")
    return result
