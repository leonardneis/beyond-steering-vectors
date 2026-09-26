"""Frozen analysis: raw scoring outputs -> criteria, decision, labels, descriptive tables, Stage-2a handoff.

Runs automatically as the last DAG node, with no manual step (PREREGISTRATION §13.1(5)). If the integrity
report fails, only ``decision.json`` with TECHNICAL_FAIL is written and no criterion is computed. The exit
status and the set of file names never depend on the decision class.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import artifacts as art
from . import statistics as st
from .conditions import PERSONA, STEER, TESTED_CONTRASTS, resolve_vector
from .criteria import O_PANEL, O_PRIME, TARGET, WordScores, evaluate, off_target
from .directions import AxisStatistics, cosine
from .package import ANIMAL_FAMILIES

OUTPUT_FILES = ("decision.json", "criteria.json", "descriptive.json", "stage2a_handoff.json", "stage2a_handoff.npz")
GRAM_SLOTS = (8, 14, 21, 27, 28)
READOUT_DIRECTIONS = ("c_cat_dog", "c_cat_wolf", "c_cat_anim", "t_cat", "m_cat_dog", "m_cat_wolf", "g_id", "h", "g_tmpl", "g_anim")


def family_index_from_partition(partition_ids: Mapping[str, tuple[str, ...]]) -> st.FamilyIndex:
    """S0 animal-family ids by the subfamily prefix of the prompt id (the analysis never reads prompt text)."""
    by_family = {family: [pid for pid in partition_ids["S0"] if pid.rsplit("_", 1)[0] == family] for family in ANIMAL_FAMILIES}
    return st.FamilyIndex.from_ids(by_family)


def load_word_scores(ctx) -> tuple[WordScores, dict[str, dict[str, np.ndarray]]]:
    from .pipeline import PipelineError, load_baseline

    values: dict[str, dict[str, np.ndarray]] = {}
    extras: dict[str, dict[str, np.ndarray]] = {}
    words = None
    for spec in ctx.plan["shards"]:
        if spec["stage"] not in ("score", "score_persona"):
            continue
        shard, marker = ctx.completed(spec["shard_id"])
        data = art.load_npz_verified(shard.directory / "scores.npz", marker["files"]["scores.npz"]["sha256"])
        shard_words = tuple(data["words"].tolist())
        if words is None:
            words = shard_words
        elif shard_words != words:
            raise PipelineError("Word order differs between scoring shards")
        prompt_ids = data["prompt_ids"].tolist()
        for index, cid in enumerate(data["cids"].tolist()):
            if cid in values:
                raise PipelineError(f"Condition {cid} scored twice")
            values[cid] = {pid: data["word_logp"][index, j] for j, pid in enumerate(prompt_ids)}
            extras[cid] = {"prompt_ids": np.asarray(prompt_ids), "kl": data["kl_first"][index], "top1": data["top1"][index], "cjk": data["cjk_mass"][index]}
    baseline = load_baseline(ctx, 1)
    if tuple(baseline["words"].tolist()) != words:
        raise PipelineError("Baseline word order differs")
    values["unsteered"] = {pid: baseline["word_logp"][j] for j, pid in enumerate(baseline["prompt_ids"].tolist())}
    extras["unsteered"] = {"prompt_ids": baseline["prompt_ids"], "top1": baseline["top1"], "cjk": baseline["cjk_mass"]}
    return WordScores(words, values), extras


def _null_se(result: dict, contrast: str, words: list[str]) -> float:
    statistic = {}
    for name, value in result["criteria"]["TS"][contrast]["d"]["null_statistics"].items():
        a, b = name.removeprefix("null:").split(">")
        statistic[(a, b)] = value
    return st.null_threshold_se(words, statistic)


def descriptive(ctx, scores: WordScores, extras, family: st.FamilyIndex, bundle, stats: AxisStatistics, default_states14) -> dict[str, Any]:
    from .plan import null_words

    out: dict[str, Any] = {}
    conditions = ctx.conditions()
    baseline_ell_cache: dict = {}
    base_L_cache: dict = {}

    def delta_point(cid: str, word: str, others) -> float:
        key = (word, tuple(others))
        if key not in baseline_ell_cache:
            baseline_ell_cache[key] = scores.ell("unsteered", word, others)
        steered = scores.ell(cid, word, others)
        diff = {pid: steered[pid] - baseline_ell_cache[key][pid] for pid in steered}
        return st.point(family.align(diff, cid))

    # Per steered condition: point effects on the target and every scored word, first-position diagnostics.
    animal_ids = {pid for ids in family.ids.values() for pid in ids}
    per_condition = {}
    base_top1 = dict(zip(extras["unsteered"]["prompt_ids"].tolist(), extras["unsteered"]["top1"].tolist()))
    for cid, condition in conditions.items():
        if condition.kind == "unsteered":
            continue
        entry: dict[str, Any] = {"kind": condition.kind, "gating": condition.gating}
        if condition.kind == STEER:
            entry["delta_ell_cat_O_panel"] = delta_point(cid, TARGET, O_PANEL)
            entry["delta_ell_cat_O_prime"] = delta_point(cid, TARGET, O_PRIME)
            entry["delta_L"] = {}
            for word in scores.words:
                steered_L, base_L = scores.L(cid, word), base_L_cache.setdefault(word, scores.L("unsteered", word))
                entry["delta_L"][word] = st.point(family.align({pid: steered_L[pid] - base_L[pid] for pid in animal_ids}, cid))
            entry["two_x_margin"] = entry["delta_L"][TARGET] - max(entry["delta_L"][o] for o in O_PANEL) - float(np.log(2.0))
        info = extras[cid]
        ids = info["prompt_ids"].tolist()
        animal_mask = np.array([pid in animal_ids for pid in ids])
        factual_mask = np.array([pid.startswith("factual_") for pid in ids])
        entry["kl_first_mean_animal"] = float(np.mean(info["kl"][animal_mask]))
        entry["kl_first_mean_factual"] = float(np.mean(info["kl"][factual_mask]))
        changed = np.array([info["top1"][j] != base_top1[pid] for j, pid in enumerate(ids)])
        entry["top1_changed_animal"] = float(np.mean(changed[animal_mask]))
        entry["top1_changed_factual"] = float(np.mean(changed[factual_mask]))
        entry["chinese_mass_mean_animal"] = float(np.mean(info["cjk"][animal_mask]))
        per_condition[cid] = entry
    out["per_condition"] = per_condition

    # R_iso (descriptive MC p for c_cat_dog, at the R_iso magnitude).
    riso_ref = ctx.choices["descriptive"]["riso_magnitude"]
    contrast = riso_ref.removeprefix("tau:")
    t_obs = delta_point(f"{contrast}|unit:tau:{contrast}|k=1|s=+1|slot=14|last", TARGET, off_target(contrast))
    t_iso = [delta_point(f"riso:{i}|unit:{riso_ref}|k=1|s=+1|slot=14|last", TARGET, off_target(contrast)) for i in range(1000)]
    mc = st.mc_p_value(t_obs, t_iso)
    out["riso"] = {"contrast": contrast, "p": mc.p, "mc_se": mc.se, "exceed": mc.exceed, "n": mc.n}

    # Non-animal sanity: chess and blue scores under their personas vs the batch-1 default context.
    sanity = {}
    for persona, word, prefix in (("P_chess", "chess", "game_"), ("P_blue", "blue", "color_")):
        own = scores.L(f"persona:{persona}", word)
        default = scores.L("persona:P_default", word)
        ids = [pid for pid in own if pid.startswith(prefix)]
        sanity[word] = {"mean_delta_L": float(np.mean([own[pid] - default[pid] for pid in ids])), "n_items": len(ids)}
    out["non_animal_sanity"] = sanity

    # Axis norms and Gram matrices.
    personas = sorted(p for p in stats.full if p != "P_default")
    gram = {}
    for slot in GRAM_SLOTS:
        axes = np.stack([stats.axis(p, slot) for p in personas])
        norms = np.linalg.norm(axes, axis=1)
        gram[str(slot)] = {"personas": personas, "norms": norms.tolist(), "cosine": (axes @ axes.T / np.outer(norms, norms)).tolist()}
    out["axes"] = gram

    # Shared-component classification (PREREGISTRATION §11.3; descriptive).
    group = ["P_cat_T1"] + [f"P_{x}_T1" for x in ("dog", "wolf", "lion", "horse", "rabbit", "elephant")]
    vectors = [stats.axis(p, 14) for p in group]
    off_diagonal = [cosine(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(len(vectors)) if i != j]
    t_cat_norm = float(np.linalg.norm(stats.axis("P_cat_T1", 14)))
    descriptors = {
        name: {"R": bundle.get(name).reliability, "norm_over_t_cat": bundle.get(name).norm / t_cat_norm}
        for name in ("g_id", "g_tmpl", "g_anim", "h")
    }
    qualifying = [n for n in ("g_id", "g_tmpl", "g_anim") if descriptors[n]["R"] >= 0.95 and descriptors[n]["norm_over_t_cat"] >= 0.25]
    out["shared_component"] = {
        "mean_offdiagonal_cosine": float(np.mean(off_diagonal)),
        "descriptors": descriptors,
        "exists": bool(np.mean(off_diagonal) >= 0.5 and qualifying),
    }
    out["reliability_all_directions"] = {name: d.reliability for name, d in bundle.directions.items() if not name.startswith("null:")}

    # Magnitude report of every steered slot-14 condition.
    sigma = st.covariance(default_states14)
    pinv = np.linalg.pinv(sigma, rcond=1e-10, hermitian=True)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    support = eigenvectors[:, eigenvalues > 1e-10 * eigenvalues.max()]
    mean_norm = float(np.linalg.norm(default_states14, axis=1).mean())
    magnitudes = {}
    for cid, condition in conditions.items():
        if condition.kind != STEER or condition.slot != 14:
            continue
        vector = resolve_vector(condition, bundle)
        norm = float(np.linalg.norm(vector))
        unit = vector / norm
        sd = float(np.std(default_states14 @ unit, ddof=1))
        magnitudes[cid] = {
            "norm": norm,
            "over_projection_sd": norm / sd,
            "over_mean_state_norm": norm / mean_norm,
            "mahalanobis": float(np.sqrt(max(vector @ pinv @ vector, 0.0))),
            "support_fraction": float(np.sum((support.T @ vector) ** 2) / norm**2),
        }
    out["magnitudes"] = magnitudes

    # Sampled answers (descriptive rates over the animal families).
    sampled = {}
    for spec in ctx.plan["shards"]:
        if spec["stage"] != "sample":
            continue
        shard, marker = ctx.completed(spec["shard_id"])
        data = art.load_npz_verified(shard.directory / "samples.npz", marker["files"]["samples.npz"]["sha256"])
        ids = data["prompt_ids"].tolist()
        mask = np.array([pid in animal_ids for pid in ids])
        for index, cid in enumerate(data["cids"].tolist()):
            parsed = data["parsed"][index][mask].ravel().tolist()
            counts: dict[str, int] = {}
            for value in parsed:
                counts[value or "none"] = counts.get(value or "none", 0) + 1
            sampled[cid] = {key: value / len(parsed) for key, value in sorted(counts.items())}
    out["sampled_rates"] = sampled
    out["null_words"] = null_words(ctx.package)
    return out


def slot28_readouts(stats: AxisStatistics, w_u: np.ndarray, form_first_tokens: Mapping[str, list[int]]) -> dict:
    from .directions import persona_formulas

    formulas = {formula.name: formula for formula in persona_formulas()}
    out = {}
    for name in READOUT_DIRECTIONS:
        formula = formulas[name]
        raw = sum(coef * stats.axis(persona, 28) for persona, coef in formula.coefficients.items())
        logits = w_u @ (raw / np.linalg.norm(raw))
        order = np.argsort(-logits, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        out[name] = {
            word: [{"token": int(t), "rank": int(rank[t]), "percentile": float(100.0 * (1 - rank[t] / logits.size))} for t in tokens]
            for word, tokens in form_first_tokens.items()
        }
    return out


def continuity(stats: AxisStatistics, frozen_raw: np.ndarray) -> dict:
    return {str(slot): cosine(stats.axis("P_cat_T1", slot), frozen_raw[slot]) for slot in range(1, 29)}


def stage2a_handoff(bundle, result: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Exactly the §12 consumption list: directions of passing contrasts, shared descriptors, decision, labels."""
    passing = [c for c in TESTED_CONTRASTS if result["criteria"]["TS"][c]["pass"]]
    arrays = {}
    for contrast in passing:
        arrays[f"{contrast}@14"] = bundle.get(contrast).unit
        arrays[f"{contrast}@27"] = bundle.get(f"{contrast}@27").unit
    for name in ("g_id", "h", "g_tmpl", "g_anim"):
        arrays[f"{name}@14"] = bundle.get(name).unit
    meta = {
        "decision": result["decision"],
        "labels": {c: result["labels"][c]["label"] for c in result["labels"]},
        "passing_contrasts": passing,
        "arrays": sorted(arrays),
        "excluded": ["dose", "tau", "magnitudes", "S0 prompt-level results"],
    }
    return meta, arrays


def stage_analysis(ctx, shard_id: str = "analysis") -> None:
    from .guards import load_frozen_teacher
    from .modeling import snapshot_directory, verify_snapshot
    from .pipeline import _load_extraction, load_bundle, log, verify_runtime

    shard = ctx.shard(shard_id)
    if shard.is_complete():
        log(f"{shard_id}: complete and verified; skipping")
        return
    shard.quarantine()
    runtime = verify_runtime(ctx, gpu=False)
    integrity_shard, integrity_marker = ctx.completed("integrity")
    integrity = art.load_json_verified(integrity_shard.directory / "integrity.json", integrity_marker["files"]["integrity.json"]["sha256"])
    if not integrity["pass"]:
        decision = {"class": "TECHNICAL_FAIL", "rank": 1, "integrity_sha256": integrity_marker["files"]["integrity.json"]["sha256"]}
        files = {name: art.pretty_json({"decision": decision, "computed": False}) for name in OUTPUT_FILES if name.endswith(".json")}
        files["stage2a_handoff.npz"] = art.npz_bytes({"empty": np.zeros(0)})
        shard.publish(files, {"stage": "analysis", "runtime": runtime})
        log(f"{shard_id}: analysis written")
        return
    scores, extras = load_word_scores(ctx)
    family = family_index_from_partition(ctx.package.partition_ids())
    bundle, _ = load_bundle(ctx)
    reliabilities = {name: bundle.get(name).reliability for name in ("c_cat_dog", "c_cat_wolf", "c_cat_anim", "t_cat")}
    from .plan import null_names, null_words

    result = evaluate(scores, family, reliabilities, null_names(ctx.package), integrity_ok=True)
    for contrast in TESTED_CONTRASTS:
        result["criteria"]["TS"][contrast]["d"]["null_threshold_se"] = _null_se(result, contrast, null_words(ctx.package))
    half_sums, default_states14, _ = _load_extraction(ctx)
    stats = AxisStatistics.from_half_sums(half_sums)
    extra = descriptive(ctx, scores, extras, family, bundle, stats, default_states14)
    snapshot = snapshot_directory(os.environ["HF_HOME"])
    verify_snapshot(snapshot, ctx.manifest["model"]["snapshot_sha256"])
    extra["slot28_readouts"] = slot28_readouts(stats, _lm_head(snapshot), {
        word: [form["token_ids"][0] for form in info["forms"]] for word, info in ctx.package.endpoint["answer_forms"].items()
    })
    teacher_path = ctx.out_root / "inputs" / "frozen_t_cat" / "v_teacher.pt"
    extra["continuity_cosine_frozen_t_cat"] = continuity(stats, load_frozen_teacher(teacher_path))
    meta, arrays = stage2a_handoff(bundle, result)
    files = {
        "decision.json": art.pretty_json({"decision": result["decision"], "labels": result["labels"], "computed": True}),
        "criteria.json": art.pretty_json(result),
        "descriptive.json": art.pretty_json(extra),
        "stage2a_handoff.json": art.pretty_json(meta),
        "stage2a_handoff.npz": art.npz_bytes(arrays),
    }
    shard.publish(files, {"stage": "analysis", "runtime": runtime})
    log(f"{shard_id}: analysis written")


def _lm_head(snapshot: Path) -> np.ndarray:
    """lm_head weight as the loaded model holds it (checkpoint bf16 cast to float16), in float32."""
    from .pipeline import checkpoint_tensor

    return checkpoint_tensor(snapshot, "lm_head.weight").numpy()
