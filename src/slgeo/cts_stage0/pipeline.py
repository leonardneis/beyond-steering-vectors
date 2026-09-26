"""Stages of the v2 scientific run. Each stage reads only verified inputs and publishes one shard atomically.

Logging policy (outcome blindness): stdout carries counts, timings, shard ids, hashes and integrity-check
names only; never a score, norm, cosine, tau, magnitude, statistic, criterion or decision.

Error policy (E2): every error raised by this package's integrity checks derives from ``FinalFailure`` and is
mapped to the final exit code (never retried). Only infrastructure failures (SIGTERM, GPU unavailable, I/O,
out-of-memory) leave the job retryable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from . import artifacts as art
from .conditions import L2, OWN, PERSONA, STEER, UNSTEERED, Condition, resolve_vector, row_steer, vector_sha256
from .contract import V2Contract
from .directions import AxisStatistics, Direction, DirectionBundle, build_bundle
from .errors import FinalFailure
from .extraction import STORED_SLOTS, extract_persona
from .guards import assert_no_peft
from .package import FrozenPackage, load_extraction_prompts
from .plan import condition_objects, plan_sha256
from .provenance import lf_sha256, verify_tracked_blobs
from .render import Renderer
from .scoring import FormTable, build_prefix, lm_head_weight32, score_from_prefix, score_prompt
from .steering import RowSteer

SLOT14_INDEX = STORED_SLOTS.index(14)
MANIFEST_RELATIVE = "configs/validation/cts_stage0_v2.yaml"


class PipelineError(FinalFailure):
    """An integrity-class failure of a stage (final: never retried)."""


def log(message: str) -> None:
    print(f"[cts-stage0-v2 {time.strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass
class RunContext:
    repo_root: Path
    manifest: dict[str, Any]
    out_root: Path
    plan: dict[str, Any]
    run_record: dict[str, Any]

    @cached_property
    def package(self) -> FrozenPackage:
        package = FrozenPackage.from_repo(self.repo_root)
        if art.sha256_file(package.root / "MANIFEST.json") != self.manifest["frozen_package"]["manifest_sha256"]:
            raise PipelineError("Frozen v1 manifest hash differs from the execution manifest")
        return package

    @cached_property
    def contract(self) -> V2Contract:
        pins = {key: self.manifest["contract"][key] for key in ("spec_sha256", "registry_sha256")}
        contract = V2Contract.from_repo(self.repo_root, self.package, pins)
        if self.plan["contract"] != {"spec_sha256": contract.spec_sha256, "registry_sha256": contract.registry_sha256}:
            raise PipelineError("The plan was built from a different contract")
        return contract

    def run_identity(self) -> dict[str, Any]:
        return {
            "execution_commit": self.run_record["execution_commit"],
            "plan_sha256": plan_sha256(self.plan),
            "manifest_sha256": lf_sha256(self.repo_root / MANIFEST_RELATIVE),
            "contract_spec_sha256": self.manifest["contract"]["spec_sha256"],
            "contract_registry_sha256": self.manifest["contract"]["registry_sha256"],
            "frozen_manifest_sha256": self.manifest["frozen_package"]["manifest_sha256"],
        }

    def shard_spec(self, shard_id: str) -> dict[str, Any]:
        for shard in self.plan["shards"]:
            if shard["shard_id"] == shard_id:
                return shard
        raise PipelineError(f"Unknown shard {shard_id}")

    def shard(self, shard_id: str) -> art.Shard:
        spec = self.shard_spec(shard_id)
        return art.Shard(self.out_root, shard_id, spec["spec_sha256"], self.run_identity())

    def completed(self, shard_id: str) -> tuple[art.Shard, dict[str, Any]]:
        shard = self.shard(shard_id)
        if not shard.is_complete():
            raise PipelineError(f"Required shard {shard_id} is not complete and verified")
        return shard, json.loads(shard.marker.read_bytes())

    def require_preflight(self) -> None:
        """E1: every stage refuses to run after a failed (or missing) preflight."""
        _shard, marker = self.completed("preflight")
        if marker.get("pass") is not True or marker.get("failed"):
            raise PipelineError("Preflight did not pass; the run is final")

    def conditions(self) -> dict[str, Condition]:
        return condition_objects(self.plan)

    def budget_stopped(self) -> bool:
        return (self.out_root / "orchestration" / "BUDGET_STOP.json").exists()


def verify_runtime(ctx: RunContext, *, gpu: bool) -> dict[str, Any]:
    """Code, contract, frozen-input and identity checks that precede any model load."""
    from .identity import assert_identity

    assert_no_peft()
    tracked = verify_tracked_blobs(ctx.repo_root, ctx.run_record["tracked"])
    _ = ctx.contract
    identity = assert_identity(ctx.manifest["execution"], ctx.repo_root, require_gpu=gpu)
    if ctx.budget_stopped():
        raise PipelineError("BUDGET_STOP is set; no further stage may run")
    return {"tracked_files_verified": tracked, "identity": identity}


def shared_path(ctx: RunContext, relative: str) -> Path:
    """An external input under the shared root, admitted only through the path guard."""
    from .guards import FROZEN_V_TEACHER_RELATIVE, guard_input

    root = Path(os.environ.get("SLGEO_SHARED_ROOT", ctx.repo_root))
    allowed = [root / "data" / "generated", root / FROZEN_V_TEACHER_RELATIVE]
    return guard_input(root / relative, allowed)


def load_model_verified(ctx: RunContext):
    from .modeling import load_model, load_tokenizer, set_deterministic, snapshot_directory, verify_snapshot

    snapshot = snapshot_directory(os.environ["HF_HOME"])
    verify_snapshot(snapshot, ctx.manifest["model"]["snapshot_sha256"])
    determinism = set_deterministic()
    from slgeo.io import load_yaml

    model_config = load_yaml(ctx.repo_root / ctx.manifest["model"]["model_config"])
    tokenizer = load_tokenizer(snapshot)
    model = load_model(snapshot, model_config)
    return model, tokenizer, determinism


def _gpu_selftest(ctx: RunContext, model, tokenizer) -> dict[str, Any]:
    from .selftest import real_model_hook_selftest

    prompts = load_extraction_prompts(ctx.package, shared_path(ctx, ctx.manifest["inputs"]["extraction_file"]))
    renderer = Renderer(tokenizer, ctx.package)
    rendered = [renderer.render("P_default", prompt) for prompt in prompts[:3]]
    result = real_model_hook_selftest(model, rendered, magnitude=8.0)
    if not result["pass"]:
        raise PipelineError(f"Real-model hook self-test failed: {result['failed']}")
    return result


def _cjk_ids(ctx: RunContext, tokenizer):
    import torch

    from .checks import cjk_ids_by_rule
    from .package import sha256_text

    ids = cjk_ids_by_rule(tokenizer)
    if sha256_text(json.dumps(ids)) != ctx.package.endpoint["cjk_ids_sha256"]:
        raise PipelineError("CJK id set differs from the frozen hash")
    return torch.tensor(ids)


def prompts_for(ctx: RunContext, prompt_set: str):
    """S0 prompts of a registry prompt set, sorted by prompt_id (D and C are never read)."""
    prompts = ctx.package.s0_prompts()
    if prompt_set == "S0_all":
        return prompts
    if prompt_set == "S0_animal":
        return tuple(p for p in prompts if p.is_animal_family)
    raise PipelineError(f"Unknown prompt set {prompt_set!r}")


# ----------------------------------------------------------------------------------------------- stages


def _begin(ctx: RunContext, shard_id: str) -> art.Shard | None:
    shard = ctx.shard(shard_id)
    if shard.is_complete():
        log(f"{shard_id}: complete and verified; skipping")
        return None
    ctx.require_preflight()
    shard.quarantine()
    return shard


def stage_extract(ctx: RunContext, shard_id: str) -> None:
    spec = ctx.shard_spec(shard_id)
    shard = _begin(ctx, shard_id)
    if shard is None:
        return
    runtime = verify_runtime(ctx, gpu=True)
    model, tokenizer, determinism = load_model_verified(ctx)
    selftest = _gpu_selftest(ctx, model, tokenizer)
    prompts = load_extraction_prompts(ctx.package, shared_path(ctx, ctx.manifest["inputs"]["extraction_file"]))
    renderer = Renderer(tokenizer, ctx.package)
    files: dict[str, bytes] = {}
    started = time.time()
    for persona in spec["payload"]["personas"]:
        states = extract_persona(model, renderer, persona, prompts, "cuda:0")
        files[f"{persona}.npz"] = art.npz_bytes(
            {
                "states": states.states,
                "half_sums": states.half_sums,
                "prompt_lens": states.prompt_lens,
                "row_index": np.arange(len(prompts), dtype=np.int64),
                "rendered_sha256": np.asarray(states.rendered_sha256),
                "stored_slots": np.asarray(STORED_SLOTS, dtype=np.int64),
            }
        )
        log(f"{shard_id}: extracted {persona} ({len(prompts)} rows)")
    seconds = time.time() - started
    shard.publish(files, {"stage": "extract", "runtime": runtime, "determinism": determinism, "selftest": selftest,
                          "forwards": len(prompts) * len(spec["payload"]["personas"]), "compute_seconds": seconds})


def checkpoint_tensor(snapshot: Path, name: str, rows: list[int] | None = None):
    """A checkpoint tensor cast exactly as the fp16 model holds it (bf16 -> float16), returned as float32 torch."""
    import torch
    from safetensors import safe_open

    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    with safe_open(str(snapshot / index["weight_map"][name]), framework="pt") as handle:
        if rows is None:
            tensor = handle.get_tensor(name)
        else:
            view = handle.get_slice(name)
            tensor = torch.cat([view[row : row + 1] for row in rows])
    return tensor.to(torch.float16).to(torch.float32)


def _load_extraction(ctx: RunContext) -> tuple[dict[str, np.ndarray], np.ndarray]:
    half_sums, states14 = {}, None
    for shard in ctx.plan["shards"]:
        if shard["stage"] != "extract":
            continue
        verified, marker = ctx.completed(shard["shard_id"])
        for persona in shard["payload"]["personas"]:
            data = art.load_npz_verified(verified.directory / f"{persona}.npz", marker["files"][f"{persona}.npz"]["sha256"])
            if not np.array_equal(data["row_index"], np.arange(1024)):
                raise PipelineError(f"Extraction rows of {persona} are not in row order 0..1023")
            half_sums[persona] = data["half_sums"]
            if persona == "P_default":
                states14 = data["states"][:, SLOT14_INDEX, :].astype(np.float64)
    if states14 is None or set(half_sums) != set(ctx.contract.personas):
        raise PipelineError("Extraction does not cover every persona")
    return half_sums, states14


def bundle_arrays(bundle: DirectionBundle) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {"r_cov": bundle.r_cov}
    meta: dict[str, Any] = {"null_names": bundle.null_names, "directions": {}, "perp_reports": bundle.perp_reports}
    for name, direction in sorted(bundle.directions.items()):
        arrays[f"raw::{name}"] = direction.raw
        arrays[f"unit::{name}"] = direction.unit
        meta["directions"][name] = {
            "slot": direction.slot,
            "tau": direction.tau,
            "reliability": direction.reliability,
            "gating_reliability": direction.gating_reliability,
            "tau_ref": direction.tau_ref,
            "coefficients": direction.coefficients,
            "norm": direction.norm,
        }
    return arrays, meta


def bundle_from_arrays(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> DirectionBundle:
    directions = {}
    for name, info in meta["directions"].items():
        directions[name] = Direction(
            name, info["slot"], arrays[f"raw::{name}"], arrays[f"unit::{name}"], info["tau"], info["reliability"],
            info["gating_reliability"], info["tau_ref"], info["coefficients"], info["norm"],
        )
    return DirectionBundle(directions, arrays["r_cov"], meta["null_names"], meta["perp_reports"])


def stage_directions(ctx: RunContext, shard_id: str = "directions") -> None:
    shard = _begin(ctx, shard_id)
    if shard is None:
        return
    runtime = verify_runtime(ctx, gpu=False)
    half_sums, states14 = _load_extraction(ctx)
    stats = AxisStatistics.from_half_sums(half_sums)
    bundle = build_bundle(stats, states14, ctx.contract.null_words)
    needed = {c.direction for c in ctx.conditions().values() if c.kind == STEER and not c.direction.startswith("rcov:")}
    missing = sorted(needed - set(bundle.directions))
    if missing:
        raise PipelineError(f"Directions required by the registry are missing: {missing[:5]}")
    arrays, meta = bundle_arrays(bundle)
    files = {"bundle.npz": art.npz_bytes(arrays), "bundle.json": art.pretty_json(meta)}
    shard.publish(files, {"stage": "directions", "runtime": runtime, "n_directions": len(bundle.directions)})
    log(f"{shard_id}: published {len(bundle.directions)} directions and R_cov")


def load_bundle(ctx: RunContext) -> tuple[DirectionBundle, dict[str, str]]:
    shard, marker = ctx.completed("directions")
    arrays = art.load_npz_verified(shard.directory / "bundle.npz", marker["files"]["bundle.npz"]["sha256"])
    meta = art.load_json_verified(shard.directory / "bundle.json", marker["files"]["bundle.json"]["sha256"])
    hashes = {name: info["sha256"] for name, info in marker["files"].items()}
    return bundle_from_arrays(arrays, meta), hashes


class _Scorer:
    """Scores conditions on one rendered prompt in the canonical layout of their cost class, or in L1."""

    def __init__(self, model, table, weight32, cjk):
        self.model, self.table, self.weight32, self.cjk = model, table, weight32, cjk

    def l2(self, prefix, row: RowSteer, reference=None, *, first=False):
        return score_from_prefix(self.model, prefix, row, self.table, weight32=self.weight32, cjk_ids=self.cjk,
                                 return_first_logprobs=first, reference_first_lp=reference)

    def l1(self, rendered, row: RowSteer, reference=None):
        return score_prompt(self.model, rendered, [row], self.table, weight32=self.weight32, cjk_ids=self.cjk,
                            reference_first_lp=reference)


def _prepare_gpu(ctx: RunContext):
    runtime = verify_runtime(ctx, gpu=True)
    model, tokenizer, determinism = load_model_verified(ctx)
    selftest = _gpu_selftest(ctx, model, tokenizer)
    weight32 = lm_head_weight32(model)
    cjk = _cjk_ids(ctx, tokenizer)
    table = FormTable.from_endpoint(ctx.package.endpoint)
    renderer = Renderer(tokenizer, ctx.package)
    return runtime, model, determinism, selftest, _Scorer(model, table, weight32, cjk), renderer


def stage_baseline(ctx: RunContext, shard_id: str) -> None:
    """Canonical L2 baseline (rep 1) and its repeat on another host group (rep 2)."""
    shard = _begin(ctx, shard_id)
    if shard is None:
        return
    runtime, model, determinism, selftest, scorer, renderer = _prepare_gpu(ctx)
    prompts = prompts_for(ctx, "S0_all")
    form_logp, word_logp, first_lp, cjk_mass, top1 = [], [], [], [], []
    started = time.time()
    for p_index, prompt in enumerate(prompts):
        prefix = build_prefix(model, renderer.render("P_default", prompt.prompt), device=scorer.weight32.device)
        result = scorer.l2(prefix, RowSteer({}), first=True)
        if p_index < 2:
            again = scorer.l2(build_prefix(model, prefix.rendered, device=scorer.weight32.device), RowSteer({}))
            if not np.array_equal(again.form_logp, result.form_logp):
                raise PipelineError("Unsteered L2 baseline is not reproducible within one job")
        form_logp.append(result.form_logp[0])
        word_logp.append(result.word_logp[0])
        first_lp.append(result.first_logprobs[0].numpy())
        cjk_mass.append(result.cjk_mass[0])
        top1.append(result.top1[0])
    files = {
        "baseline.npz": art.npz_bytes(
            {
                "prompt_ids": np.asarray([p.prompt_id for p in prompts]),
                "form_logp": np.asarray(form_logp),
                "word_logp": np.asarray(word_logp),
                "first_logprobs": np.asarray(first_lp, dtype=np.float32),
                "cjk_mass": np.asarray(cjk_mass),
                "top1": np.asarray(top1, dtype=np.int64),
                "words": np.asarray(scorer.table.words),
            }
        )
    }
    shard.publish(files, {"stage": "baseline", "runtime": runtime, "determinism": determinism, "selftest": selftest,
                          "layout": L2, "prompt_conditions": len(prompts), "compute_seconds": time.time() - started})
    log(f"{shard_id}: L2 baseline over {len(prompts)} prompts")


def load_baseline(ctx: RunContext, rep: int = 1) -> dict[str, np.ndarray]:
    shard, marker = ctx.completed(f"baseline_rep{rep}")
    return art.load_npz_verified(shard.directory / "baseline.npz", marker["files"]["baseline.npz"]["sha256"])


def _sentinel(model, renderer, scorer: _Scorer, prompts, baseline, tolerance) -> float:
    """Re-score the L2 baseline on the first two prompts; return the max abs deviation (spec: <= 1e-4)."""
    index = {pid: i for i, pid in enumerate(baseline["prompt_ids"].tolist())}
    worst = 0.0
    for prompt in prompts[:2]:
        prefix = build_prefix(model, renderer.render("P_default", prompt.prompt), device=scorer.weight32.device)
        result = scorer.l2(prefix, RowSteer({}))
        worst = max(worst, float(np.abs(result.form_logp[0] - baseline["form_logp"][index[prompt.prompt_id]]).max()))
    if worst > tolerance:
        raise PipelineError("Shard sentinel deviates from the canonical baseline beyond the 1e-4 tolerance")
    return worst


def _score_shard(ctx: RunContext, shard_id: str, *, reference_layout: bool) -> None:
    import torch

    spec = ctx.shard_spec(shard_id)
    shard = _begin(ctx, shard_id)
    if shard is None:
        return
    runtime, model, determinism, selftest, scorer, renderer = _prepare_gpu(ctx)
    bundle, bundle_hashes = load_bundle(ctx)
    baseline = load_baseline(ctx, 1)
    conditions = ctx.conditions()
    selected = [conditions[cid] for cid in spec["payload"]["conditions"]]
    prompt_set = spec["payload"]["prompt_set"]
    if any(condition.prompt_set != prompt_set for condition in selected):
        raise PipelineError("Shard mixes prompt sets")
    if reference_layout:
        if any(not condition.reference_rescore for condition in selected):
            raise PipelineError("Re-score shard contains a condition that is not flagged for the reference layout")
        layout = "L1_reference"
    else:
        cost_class = spec["payload"]["cost_class"]
        if any(condition.cost_class != cost_class for condition in selected):
            raise PipelineError("Shard mixes cost classes")
        layout = cost_class
    if tuple(baseline["words"].tolist()) != scorer.table.words:
        raise PipelineError("Baseline word order differs from the form table")
    prompts = prompts_for(ctx, prompt_set)
    tolerance = float(ctx.manifest["scoring"]["baseline_repeat_tolerance"])
    sentinel_start = _sentinel(model, renderer, scorer, prompts, baseline, tolerance)
    reference = {pid: torch.from_numpy(baseline["first_logprobs"][i]) for i, pid in enumerate(baseline["prompt_ids"].tolist())}
    rows = [row_steer(condition, bundle) for condition in selected]
    n_c, n_p = len(selected), len(prompts)
    form_logp = np.empty((n_c, n_p, scorer.table.n_forms))
    word_logp = np.empty((n_c, n_p, len(scorer.table.words)))
    cjk_mass = np.empty((n_c, n_p))
    kl = np.empty((n_c, n_p))
    top1 = np.empty((n_c, n_p), dtype=np.int64)
    started = time.time()
    for p_index, prompt in enumerate(prompts):
        ref = reference[prompt.prompt_id]
        default_rendered = renderer.render("P_default", prompt.prompt)
        prefix = None
        if layout == L2:
            prefix = build_prefix(model, default_rendered, device=scorer.weight32.device)
        for c_index, (condition, row) in enumerate(zip(selected, rows)):
            if layout == L2:
                result = scorer.l2(prefix, row, ref)
            elif condition.kind == PERSONA:
                result = scorer.l1(renderer.render(condition.persona, prompt.prompt), RowSteer({}), ref)
            else:  # own-prefix steered (all positions) or any L1 re-score
                result = scorer.l1(default_rendered, row, ref)
            form_logp[c_index, p_index] = result.form_logp[0]
            word_logp[c_index, p_index] = result.word_logp[0]
            cjk_mass[c_index, p_index] = result.cjk_mass[0]
            kl[c_index, p_index] = result.kl_to_reference[0]
            top1[c_index, p_index] = result.top1[0]
        if (p_index + 1) % 50 == 0:
            log(f"{shard_id}: {p_index + 1}/{n_p} prompts, {time.time() - started:.0f}s")
    compute_seconds = time.time() - started
    sentinel_end = _sentinel(model, renderer, scorer, prompts, baseline, tolerance)
    if not (np.isfinite(form_logp).all() and np.isfinite(word_logp).all()):
        raise PipelineError("Non-finite scores")
    intended = [
        float(np.sqrt(np.sum(resolve_vector(condition, bundle) ** 2))) if condition.kind == STEER else 0.0 for condition in selected
    ]
    files = {
        "scores.npz": art.npz_bytes(
            {
                "cids": np.asarray([condition.cid for condition in selected]),
                "prompt_ids": np.asarray([p.prompt_id for p in prompts]),
                "words": np.asarray(scorer.table.words),
                "form_logp": form_logp,
                "word_logp": word_logp,
                "cjk_mass": cjk_mass,
                "kl_first": kl,
                "top1": top1,
                "vector_sha256": np.asarray([vector_sha256(condition, bundle) or "" for condition in selected]),
                "vector_norm": np.asarray(intended),
            }
        )
    }
    shard.publish(
        files,
        {
            "stage": spec["stage"],
            "layout": layout,
            "prompt_set": prompt_set,
            "runtime": runtime,
            "determinism": determinism,
            "selftest": selftest,
            "bundle_sha256": bundle_hashes,
            "baseline_sha256": ctx.completed("baseline_rep1")[1]["files"]["baseline.npz"]["sha256"],
            "sentinel_max_abs": [sentinel_start, sentinel_end],
            "prompt_conditions": n_c * n_p,
            "compute_seconds": compute_seconds,
        },
    )
    log(f"{shard_id}: scored {n_c} conditions x {n_p} prompts in {compute_seconds:.0f}s ({layout})")


def stage_score(ctx: RunContext, shard_id: str) -> None:
    _score_shard(ctx, shard_id, reference_layout=False)


def stage_rescore(ctx: RunContext, shard_id: str) -> None:
    _score_shard(ctx, shard_id, reference_layout=True)


def run_stage(ctx: RunContext, shard_id: str) -> None:
    from . import analysis, fragility, integrity, preflight

    stages = {
        "preflight": preflight.stage_preflight,
        "extract": stage_extract,
        "directions": stage_directions,
        "baseline": stage_baseline,
        "score": stage_score,
        "rescore": stage_rescore,
        "fragility": fragility.stage_fragility,
        "integrity": integrity.stage_integrity,
        "analysis": analysis.stage_analysis,
    }
    stages[ctx.shard_spec(shard_id)["stage"]](ctx, shard_id)


def attempt(ctx: RunContext, shard_id: str, event: str, **payload) -> None:
    art.attempt_record(ctx.out_root, shard_id, {"event": event, "utc": art.utc_now(), **payload})
