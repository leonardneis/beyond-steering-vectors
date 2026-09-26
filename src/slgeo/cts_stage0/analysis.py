"""Automated analysis: raw scoring outputs -> criteria, decision, labels, descriptive tables, Stage-2a handoff.

Runs as the last DAG node with no manual step. If the integrity report fails (this includes the fragility
check and a BUDGET_STOP), only ``decision.json`` with TECHNICAL_FAIL and the failing check names is written and
no criterion is computed. The exit status and the set of file names never depend on the decision class.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import artifacts as art
from . import statistics as st
from .conditions import STEER, TESTED_CONTRASTS, Condition, resolve_vector
from .criteria import O_PANEL, O_PRIME, TARGET, WordScores, evaluate
from .directions import SHARED_COMPONENTS, AxisStatistics, cosine
from .package import ANIMAL_FAMILIES

OUTPUT_FILES = ("decision.json", "criteria.json", "descriptive.json", "stage2a_handoff.json", "stage2a_handoff.npz")
GRAM_SLOTS = (14, 27, 28)
READOUT_DIRECTIONS = ("c_cat_dog", "c_cat_wolf", "c_cat_anim", "t_cat", "m_cat_dog", "m_cat_wolf", "g_id", "h", "g_tmpl", "g_anim", "g_anim_v1")


def family_index_from_partition(partition_ids: Mapping[str, tuple[str, ...]]) -> st.FamilyIndex:
    """S0 animal-family ids by the subfamily prefix of the prompt id (the analysis never reads prompt text)."""
    by_family = {family: [pid for pid in partition_ids["S0"] if pid.rsplit("_", 1)[0] == family] for family in ANIMAL_FAMILIES}
    return st.FamilyIndex.from_ids(by_family)


def baseline_map(conditions: Mapping[str, Condition]) -> dict[str, str]:
    return {cid: condition.baseline for cid, condition in conditions.items()}


def reliabilities_of(bundle) -> dict[str, float]:
    names = list(TESTED_CONTRASTS) + [f"{c}_perpG" for c in TESTED_CONTRASTS] + ["t_cat"]
    return {name: bundle.get(name).reliability for name in names}


def load_word_scores(ctx, *, stage: str = "score") -> tuple[WordScores, dict[str, dict[str, np.ndarray]]]:
    """Word scores of every scoring shard of ``stage``; for ``score`` the canonical L2 baseline is added."""
    from .pipeline import PipelineError, load_baseline

    values: dict[str, dict[str, np.ndarray]] = {}
    extras: dict[str, dict[str, np.ndarray]] = {}
    words = None
    for spec in ctx.plan["shards"]:
        if spec["stage"] != stage:
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
                raise PipelineError(f"Condition {cid} scored twice in stage {stage}")
            values[cid] = {pid: data["word_logp"][index, j] for j, pid in enumerate(prompt_ids)}
            extras[cid] = {"prompt_ids": np.asarray(prompt_ids), "kl": data["kl_first"][index], "top1": data["top1"][index], "cjk": data["cjk_mass"][index]}
    if stage == "score":
        baseline = load_baseline(ctx, 1)
        if words is not None and tuple(baseline["words"].tolist()) != words:
            raise PipelineError("Baseline word order differs")
        words = tuple(baseline["words"].tolist())
        values["unsteered"] = {pid: baseline["word_logp"][j] for j, pid in enumerate(baseline["prompt_ids"].tolist())}
        extras["unsteered"] = {"prompt_ids": baseline["prompt_ids"], "top1": baseline["top1"], "cjk": baseline["cjk_mass"]}
    if words is None:
        raise PipelineError(f"No scoring shard of stage {stage}")
    return WordScores(words, values), extras


def descriptive(ctx, scores: WordScores, extras, family: st.FamilyIndex, bundle, stats: AxisStatistics, default_states14) -> dict[str, Any]:
    out: dict[str, Any] = {}
    conditions = ctx.conditions()
    baselines = baseline_map(conditions)

    def delta_point(cid: str, word: str, others) -> float:
        base = scores.ell(baselines[cid], word, others)
        steered = scores.ell(cid, word, others)
        return st.point(family.align({pid: steered[pid] - base[pid] for pid in steered}, cid))

    animal_ids = {pid for ids in family.ids.values() for pid in ids}
    per_condition = {}
    base_top1 = dict(zip(extras["unsteered"]["prompt_ids"].tolist(), extras["unsteered"]["top1"].tolist()))
    for cid, condition in conditions.items():
        if condition.kind == "unsteered":
            continue
        entry: dict[str, Any] = {"kind": condition.kind, "gating": condition.gating, "cost_class": condition.cost_class}
        if condition.kind == STEER:
            entry["delta_ell_cat_O_panel"] = delta_point(cid, TARGET, O_PANEL)
            entry["delta_ell_cat_O_prime"] = delta_point(cid, TARGET, O_PRIME)
            base_id = baselines[cid]
            entry["delta_L"] = {}
            for word in scores.words:
                steered_L, base_L = scores.L(cid, word), scores.L(base_id, word)
                entry["delta_L"][word] = st.point(family.align({pid: steered_L[pid] - base_L[pid] for pid in animal_ids}, cid))
            entry["two_x_margin"] = entry["delta_L"][TARGET] - max(entry["delta_L"][o] for o in O_PANEL) - float(np.log(2.0))
        info = extras.get(cid)
        if info is not None:
            ids = info["prompt_ids"].tolist()
            animal_mask = np.array([pid in animal_ids for pid in ids])
            factual_mask = np.array([pid.startswith("factual_") for pid in ids])
            entry["kl_first_mean_animal"] = float(np.mean(info["kl"][animal_mask]))
            if factual_mask.any():
                entry["kl_first_mean_factual"] = float(np.mean(info["kl"][factual_mask]))
            changed = np.array([info["top1"][j] != base_top1[pid] for j, pid in enumerate(ids)])
            entry["top1_changed_animal"] = float(np.mean(changed[animal_mask]))
            entry["chinese_mass_mean_animal"] = float(np.mean(info["cjk"][animal_mask]))
        per_condition[cid] = entry
    out["per_condition"] = per_condition

    personas = sorted(p for p in stats.full if p != "P_default")
    gram = {}
    for slot in GRAM_SLOTS:
        axes = np.stack([stats.axis(p, slot) for p in personas])
        norms = np.linalg.norm(axes, axis=1)
        gram[str(slot)] = {"personas": personas, "norms": norms.tolist(), "cosine": (axes @ axes.T / np.outer(norms, norms)).tolist()}
    out["axes"] = gram

    group = ["P_cat_T1"] + [f"P_{x}_T1" for x in ("dog", "wolf", "lion", "horse", "rabbit", "elephant")]
    vectors = [stats.axis(p, 14) for p in group]
    off_diagonal = [cosine(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(len(vectors)) if i != j]
    t_cat_norm = float(np.linalg.norm(stats.axis("P_cat_T1", 14)))
    descriptors = {
        name: {"R": bundle.get(name).reliability, "norm_over_t_cat": bundle.get(name).norm / t_cat_norm}
        for name in SHARED_COMPONENTS + ("g_anim_v1",)
    }
    out["shared_component"] = {"mean_offdiagonal_cosine": float(np.mean(off_diagonal)), "descriptors": descriptors}
    out["leakage_free_geometry"] = bundle.perp_reports
    out["cos_c_with_shared_components"] = {
        contrast: {name: cosine(bundle.get(contrast).unit, bundle.get(name).unit) for name in SHARED_COMPONENTS + ("g_anim_v1",)}
        for contrast in TESTED_CONTRASTS
    }
    out["reliability_all_directions"] = {name: d.reliability for name, d in bundle.directions.items() if not name.startswith("null:")}

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
    out["null_words"] = list(ctx.contract.null_words)
    return out


def slot28_readouts(stats: AxisStatistics, w_u: np.ndarray, form_first_tokens: Mapping[str, list[int]], null_words) -> dict:
    from .directions import persona_formulas

    formulas = {formula.name: formula for formula in persona_formulas(null_words)}
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
    """Spec stage2a_precommitments: directions (and c_perpG) of passing contrasts, shared descriptors, decision,
    labels. Never doses, tau, magnitudes or prompt-level results."""
    passing = [c for c in TESTED_CONTRASTS if result["criteria"]["TS"][c]["pass"]]
    arrays = {}
    for contrast in passing:
        arrays[f"{contrast}@14"] = bundle.get(contrast).unit
        arrays[f"{contrast}@27"] = bundle.get(f"{contrast}@27").unit
        arrays[f"{contrast}_perpG@14"] = bundle.get(f"{contrast}_perpG").unit
    for name in SHARED_COMPONENTS:
        arrays[f"{name}@14"] = bundle.get(name).unit
    labels = {}
    for contrast, entry in result["labels"].items():
        if isinstance(entry, dict):
            labels[contrast] = {"mention": entry["mention"]["label"], "shared": {g: v["label"] for g, v in entry["shared"].items()}}
        else:
            labels[contrast] = entry
    meta = {
        "decision": result["decision"],
        "labels": labels,
        "passing_contrasts": passing,
        "arrays": sorted(arrays),
        "excluded": ["dose", "tau", "magnitudes", "S0 prompt-level results"],
    }
    return meta, arrays


def stage_analysis(ctx, shard_id: str = "analysis") -> None:
    from .guards import load_frozen_teacher
    from .modeling import snapshot_directory, verify_snapshot
    from .pipeline import _begin, _load_extraction, load_bundle, log, verify_runtime
    from .plan import null_names

    shard = _begin(ctx, shard_id)
    if shard is None:
        return
    runtime = verify_runtime(ctx, gpu=False)
    integrity_shard, integrity_marker = ctx.completed("integrity")
    integrity = art.load_json_verified(integrity_shard.directory / "integrity.json", integrity_marker["files"]["integrity.json"]["sha256"])
    if not integrity["pass"]:
        failing = sorted(name for name, check in integrity["checks"].items() if not check["pass"])
        decision = {"class": "TECHNICAL_FAIL", "rank": 1, "failed_checks": failing,
                    "integrity_sha256": integrity_marker["files"]["integrity.json"]["sha256"]}
        files = {name: art.pretty_json({"decision": decision, "computed": False}) for name in OUTPUT_FILES if name.endswith(".json")}
        files["stage2a_handoff.npz"] = art.npz_bytes({"empty": np.zeros(0)})
        shard.publish(files, {"stage": "analysis", "runtime": runtime})
        log(f"{shard_id}: analysis written")
        return
    scores, extras = load_word_scores(ctx)
    family = family_index_from_partition(ctx.package.partition_ids())
    bundle, _ = load_bundle(ctx)
    conditions = ctx.conditions()
    result, _ctx = evaluate(scores, family, reliabilities_of(bundle), null_names(ctx.contract), True, baseline_map(conditions))
    half_sums, default_states14 = _load_extraction(ctx)
    stats = AxisStatistics.from_half_sums(half_sums)
    extra = descriptive(ctx, scores, extras, family, bundle, stats, default_states14)
    snapshot = snapshot_directory(os.environ["HF_HOME"])
    verify_snapshot(snapshot, ctx.manifest["model"]["snapshot_sha256"])
    extra["slot28_readouts"] = slot28_readouts(stats, _lm_head(snapshot), {
        word: [form["token_ids"][0] for form in info["forms"]] for word, info in ctx.package.endpoint["answer_forms"].items()
    }, ctx.contract.null_words)
    teacher_path = ctx.out_root / "inputs" / "frozen_t_cat" / "v_teacher.pt"
    extra["continuity_cosine_frozen_t_cat"] = continuity(stats, load_frozen_teacher(teacher_path))
    meta, arrays = stage2a_handoff(bundle, result)
    files = {
        "decision.json": art.pretty_json({"decision": result["decision"], "labels": meta["labels"], "computed": True}),
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
