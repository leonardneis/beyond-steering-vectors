"""Stages of the scientific run. Each stage reads only verified inputs and publishes one shard atomically.

Logging policy (outcome blindness): stdout carries counts, timings, shard ids, hashes and integrity-check
names only; never a score, norm, cosine, tau, magnitude, statistic, criterion or decision.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import artifacts as art
from .conditions import PERSONA, STEER, Condition, resolve_vector, row_steer, vector_sha256
from .directions import AxisStatistics, Direction, DirectionBundle, build_bundle
from .extraction import STORED_SLOTS, extract_persona
from .guards import assert_no_peft
from .package import FrozenPackage, load_extraction_prompts
from .plan import load_choices, null_words, plan_sha256
from .provenance import lf_sha256, verify_tracked_blobs
from .render import Renderer
from .scoring import FormTable, lm_head_weight32, score_prompt
from .steering import RowSteer

SLOT14_INDEX = STORED_SLOTS.index(14)
PLURAL_TOKEN = {"cat": " cats", "dog": " dogs", "wolf": " wolves"}


class PipelineError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[cts-stage0 {time.strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass
class RunContext:
    repo_root: Path
    manifest: dict[str, Any]
    out_root: Path
    plan: dict[str, Any]
    run_record: dict[str, Any]

    @property
    def package(self) -> FrozenPackage:
        if not hasattr(self, "_package"):
            package = FrozenPackage.from_repo(self.repo_root)
            if art.sha256_file(package.root / "MANIFEST.json") != self.manifest["frozen_package"]["manifest_sha256"]:
                raise PipelineError("Frozen manifest hash differs from the execution manifest")
            self._package = package
        return self._package

    @property
    def choices(self) -> dict[str, Any]:
        return load_choices(self.repo_root, self.manifest)

    def run_identity(self) -> dict[str, Any]:
        return {
            "execution_commit": self.run_record["execution_commit"],
            "plan_sha256": plan_sha256(self.plan),
            "manifest_sha256": lf_sha256(self.repo_root / "configs/validation/cts_stage0_v1.yaml"),
            "choices_sha256": lf_sha256(self.repo_root / self.manifest["implementation_choices"]),
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

    def conditions(self) -> dict[str, Condition]:
        return {entry["cid"]: Condition(**{k: v for k, v in entry.items() if k != "cid"}) for entry in self.plan["conditions"]}


def verify_runtime(ctx: RunContext, *, gpu: bool) -> dict[str, Any]:
    """Code, frozen-input and identity checks that precede any model load."""
    from .identity import assert_identity

    assert_no_peft()
    tracked = verify_tracked_blobs(ctx.repo_root, ctx.run_record["tracked"])
    _ = ctx.package
    identity = assert_identity(ctx.manifest["execution"], ctx.repo_root, require_gpu=gpu)
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
    rendered = Renderer(tokenizer, ctx.package).render("P_default", prompts[0])
    result = real_model_hook_selftest(model, rendered, magnitude=8.0)
    if not result["pass"]:
        raise PipelineError(f"Real-model hook self-test failed: {sorted(k for k, v in result['hooks'].items() if not v)}")
    return result


def _cjk_ids(ctx: RunContext, tokenizer):
    import torch

    from .checks import cjk_ids_by_rule
    from .package import sha256_text

    ids = cjk_ids_by_rule(tokenizer)
    if sha256_text(json.dumps(ids)) != ctx.package.endpoint["cjk_ids_sha256"]:
        raise PipelineError("CJK id set differs from the frozen hash")
    return torch.tensor(ids)


# ----------------------------------------------------------------------------------------------- stages


def stage_extract(ctx: RunContext, shard_id: str) -> None:
    spec = ctx.shard_spec(shard_id)
    shard = ctx.shard(shard_id)
    if shard.is_complete():
        log(f"{shard_id}: complete and verified; skipping")
        return
    shard.quarantine()
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
    shard.publish(files, {"stage": "extract", "runtime": runtime, "determinism": determinism, "selftest": selftest,
                          "prefills": len(prompts) * len(spec["payload"]["personas"]), "seconds": time.time() - started})


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


def _embedding_rows(ctx: RunContext) -> dict[str, np.ndarray]:
    """W_E rows of the leading-space plurals from the verified checkpoint, cast as the loaded model holds them."""
    from .modeling import snapshot_directory, verify_snapshot

    snapshot = snapshot_directory(os.environ["HF_HOME"])
    verify_snapshot(snapshot, ctx.manifest["model"]["snapshot_sha256"])
    token_ids = {}
    for word, text in PLURAL_TOKEN.items():
        forms = [form for form in ctx.package.endpoint["answer_forms"][word]["forms"] if form["text"] == text]
        if len(forms) != 1 or len(forms[0]["token_ids"]) != 1:
            raise PipelineError(f"Leading-space plural of {word} is not a single frozen token")
        token_ids[word] = forms[0]["token_ids"][0]
    words = list(token_ids)
    rows = checkpoint_tensor(snapshot, "model.embed_tokens.weight", [token_ids[w] for w in words])
    return {word: rows[i].double().numpy() for i, word in enumerate(words)}


def _load_extraction(ctx: RunContext) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    half_sums, states14 = {}, None
    stored = {}
    for shard in ctx.plan["shards"]:
        if shard["stage"] != "extract":
            continue
        verified, marker = ctx.completed(shard["shard_id"])
        for persona in shard["payload"]["personas"]:
            data = art.load_npz_verified(verified.directory / f"{persona}.npz", marker["files"][f"{persona}.npz"]["sha256"])
            if not np.array_equal(data["row_index"], np.arange(1024)):
                raise PipelineError(f"Extraction rows of {persona} are not in row order 0..1023")
            half_sums[persona] = data["half_sums"]
            stored[persona] = data["states"]
            if persona == "P_default":
                states14 = data["states"][:, SLOT14_INDEX, :].astype(np.float64)
    if states14 is None or set(half_sums) != set(ctx.package.personas):
        raise PipelineError("Extraction does not cover every persona")
    return half_sums, states14, stored


def bundle_arrays(bundle: DirectionBundle) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {"r_cov": bundle.r_cov, "r_iso": bundle.r_iso}
    meta: dict[str, Any] = {"null_names": bundle.null_names, "directions": {}}
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
        }
    return arrays, meta


def bundle_from_arrays(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> DirectionBundle:
    directions = {}
    for name, info in meta["directions"].items():
        directions[name] = Direction(
            name, info["slot"], arrays[f"raw::{name}"], arrays[f"unit::{name}"], info["tau"], info["reliability"],
            info["gating_reliability"], info["tau_ref"], info["coefficients"],
        )
    return DirectionBundle(directions, arrays["r_cov"], arrays["r_iso"], meta["null_names"])


def stage_directions(ctx: RunContext, shard_id: str = "directions") -> None:
    shard = ctx.shard(shard_id)
    if shard.is_complete():
        log(f"{shard_id}: complete and verified; skipping")
        return
    shard.quarantine()
    runtime = verify_runtime(ctx, gpu=False)
    half_sums, states14, _stored = _load_extraction(ctx)
    stats = AxisStatistics.from_half_sums(half_sums)
    bundle = build_bundle(stats, states14, null_words(ctx.package), _embedding_rows(ctx))
    arrays, meta = bundle_arrays(bundle)
    from .statistics import covariance

    sigma = covariance(states14)
    files = {
        "bundle.npz": art.npz_bytes(arrays),
        "bundle.json": art.pretty_json(meta),
        "sigma14.npz": art.npz_bytes({"sigma14": sigma}),
    }
    shard.publish(files, {"stage": "directions", "runtime": runtime, "n_directions": len(bundle.directions)})
    log(f"{shard_id}: published {len(bundle.directions)} directions, R_cov and R_iso")


def load_bundle(ctx: RunContext) -> tuple[DirectionBundle, dict[str, str]]:
    shard, marker = ctx.completed("directions")
    arrays = art.load_npz_verified(shard.directory / "bundle.npz", marker["files"]["bundle.npz"]["sha256"])
    meta = art.load_json_verified(shard.directory / "bundle.json", marker["files"]["bundle.json"]["sha256"])
    hashes = {name: info["sha256"] for name, info in marker["files"].items()}
    return bundle_from_arrays(arrays, meta), hashes


def _batch_rows(ctx: RunContext) -> int:
    """Condition-batch rows; > 1 only with a hash-verified technical-validation record that passed §13.3
    for exactly this row count on the pinned GPU class."""
    rows = int(ctx.manifest["scoring"]["condition_batch_rows"])
    if rows < 1:
        raise PipelineError("condition_batch_rows must be >= 1")
    if rows == 1:
        return rows
    expected = ctx.run_record.get("equivalence_record_sha256")
    relative = ctx.manifest["scoring"].get("equivalence_record")
    if not expected or not relative:
        raise PipelineError("Condition batching requires a recorded passing §13.3 equivalence test")
    root = Path(os.environ.get("SLGEO_SHARED_ROOT", ctx.repo_root))
    path = root / relative
    if not path.is_file() or art.sha256_file(path) != expected:
        raise PipelineError("Equivalence record missing or hash mismatch")
    record = json.loads(path.read_bytes())
    entry = record.get("result", {}).get("equivalence_13_3", {}).get(str(rows))
    if not entry or entry.get("pass") is not True:
        raise PipelineError(f"The recorded §13.3 test did not pass for {rows} rows")
    if record.get("identity", {}).get("gpu_name") != ctx.manifest["execution"]["gpu_name"]:
        raise PipelineError("Equivalence record comes from a different GPU class")
    return rows


def _score_rows(model, rendered, rows: list[RowSteer], table, weight32, cjk, reference, batch_rows: int):
    """Score ``rows`` in batches of exactly ``batch_rows`` (last batch filled with repeats of its last row)."""
    results = []
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        padding = batch_rows - len(chunk)
        filled = chunk + [chunk[-1]] * padding
        result = score_prompt(model, rendered, filled, table, weight32=weight32, cjk_ids=cjk, reference_first_lp=reference)
        if padding:
            repeat = result.form_logp[len(chunk) - 1 :]
            if not np.array_equal(repeat, np.repeat(repeat[:1], len(repeat), axis=0)):
                raise PipelineError("Repeated rows of one batch are not identical")
        results.append((result, len(chunk)))
    return results


def stage_baseline(ctx: RunContext, shard_id: str) -> None:
    shard = ctx.shard(shard_id)
    if shard.is_complete():
        log(f"{shard_id}: complete and verified; skipping")
        return
    shard.quarantine()
    runtime = verify_runtime(ctx, gpu=True)
    model, tokenizer, determinism = load_model_verified(ctx)
    selftest = _gpu_selftest(ctx, model, tokenizer)
    weight32 = lm_head_weight32(model)
    cjk = _cjk_ids(ctx, tokenizer)
    table = FormTable.from_endpoint(ctx.package.endpoint)
    renderer = Renderer(tokenizer, ctx.package)
    prompts = ctx.package.s0_prompts()
    batch_rows = _batch_rows(ctx)
    form_logp, word_logp, first_lp, cjk_mass, top1 = [], [], [], [], []
    for p_index, prompt in enumerate(prompts):
        rendered = renderer.render("P_default", prompt.prompt)
        result = score_prompt(
            model, rendered, [RowSteer({})] * batch_rows, table, weight32=weight32, cjk_ids=cjk, return_first_logprobs=True
        )
        if not np.array_equal(result.form_logp, np.repeat(result.form_logp[:1], batch_rows, axis=0)):
            raise PipelineError("Identical unsteered rows differ within one batch")
        if p_index < 2:
            again = score_prompt(model, rendered, [RowSteer({})] * batch_rows, table, weight32=weight32, cjk_ids=cjk)
            if not np.array_equal(again.form_logp, result.form_logp):
                raise PipelineError("Unsteered baseline is not reproducible within one job")
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
                "words": np.asarray(table.words),
                "batch_rows": np.asarray(batch_rows),
            }
        )
    }
    shard.publish(files, {"stage": "baseline", "runtime": runtime, "determinism": determinism, "selftest": selftest,
                          "prefills": len(prompts) + 2, "batch_rows": batch_rows})
    log(f"{shard_id}: baseline over {len(prompts)} prompts")


def load_baseline(ctx: RunContext, rep: int = 1) -> dict[str, np.ndarray]:
    shard, marker = ctx.completed(f"baseline_rep{rep}")
    return art.load_npz_verified(shard.directory / "baseline.npz", marker["files"]["baseline.npz"]["sha256"])


def _sentinel(model, renderer, prompts, baseline, table, weight32, cjk, batch_rows, tolerance) -> float:
    """Re-score the unsteered baseline on the first two S0 prompts; return the max abs deviation."""
    index = {pid: i for i, pid in enumerate(baseline["prompt_ids"].tolist())}
    worst = 0.0
    for prompt in prompts[:2]:
        rendered = renderer.render("P_default", prompt.prompt)
        [(result, _)] = _score_rows(model, rendered, [RowSteer({})] * batch_rows, table, weight32, cjk, None, batch_rows)
        worst = max(worst, float(np.abs(result.form_logp[0] - baseline["form_logp"][index[prompt.prompt_id]]).max()))
    if worst > tolerance:
        raise PipelineError("Shard sentinel deviates from the canonical baseline beyond the 1e-4 tolerance")
    return worst


def stage_score(ctx: RunContext, shard_id: str) -> None:
    import torch

    spec = ctx.shard_spec(shard_id)
    shard = ctx.shard(shard_id)
    if shard.is_complete():
        log(f"{shard_id}: complete and verified; skipping")
        return
    shard.quarantine()
    runtime = verify_runtime(ctx, gpu=True)
    bundle, bundle_hashes = load_bundle(ctx)
    baseline = load_baseline(ctx, 1)
    conditions = ctx.conditions()
    selected = [conditions[cid] for cid in spec["payload"]["conditions"]]
    persona_stage = spec["stage"] == "score_persona"
    if any((condition.kind == PERSONA) != persona_stage for condition in selected):
        raise PipelineError("Shard mixes persona and steering conditions")
    model, tokenizer, determinism = load_model_verified(ctx)
    selftest = _gpu_selftest(ctx, model, tokenizer)
    weight32 = lm_head_weight32(model)
    cjk = _cjk_ids(ctx, tokenizer)
    table = FormTable.from_endpoint(ctx.package.endpoint)
    if tuple(baseline["words"].tolist()) != table.words:
        raise PipelineError("Baseline word order differs from the form table")
    renderer = Renderer(tokenizer, ctx.package)
    prompts = ctx.package.s0_prompts()
    batch_rows = 1 if persona_stage else _batch_rows(ctx)
    tolerance = float(ctx.manifest["scoring"]["baseline_repeat_tolerance"])
    sentinel_start = 0.0 if persona_stage else _sentinel(model, renderer, prompts, baseline, table, weight32, cjk, batch_rows, tolerance)
    reference = {pid: torch.from_numpy(baseline["first_logprobs"][i]) for i, pid in enumerate(baseline["prompt_ids"].tolist())}
    rows = [row_steer(condition, bundle) for condition in selected]
    n_c, n_p = len(selected), len(prompts)
    form_logp = np.empty((n_c, n_p, table.n_forms))
    word_logp = np.empty((n_c, n_p, len(table.words)))
    cjk_mass = np.empty((n_c, n_p))
    kl = np.empty((n_c, n_p))
    top1 = np.empty((n_c, n_p), dtype=np.int64)
    started = time.time()
    for p_index, prompt in enumerate(prompts):
        if persona_stage:
            for c_index, condition in enumerate(selected):
                rendered = renderer.render(condition.persona, prompt.prompt)
                [(result, _)] = _score_rows(model, rendered, [RowSteer({})], table, weight32, cjk, reference[prompt.prompt_id], 1)
                form_logp[c_index, p_index] = result.form_logp[0]
                word_logp[c_index, p_index] = result.word_logp[0]
                cjk_mass[c_index, p_index] = result.cjk_mass[0]
                kl[c_index, p_index] = result.kl_to_reference[0]
                top1[c_index, p_index] = result.top1[0]
            continue
        rendered = renderer.render("P_default", prompt.prompt)
        offset = 0
        for result, used in _score_rows(model, rendered, rows, table, weight32, cjk, reference[prompt.prompt_id], batch_rows):
            form_logp[offset : offset + used, p_index] = result.form_logp[:used]
            word_logp[offset : offset + used, p_index] = result.word_logp[:used]
            cjk_mass[offset : offset + used, p_index] = result.cjk_mass[:used]
            kl[offset : offset + used, p_index] = result.kl_to_reference[:used]
            top1[offset : offset + used, p_index] = result.top1[:used]
            offset += used
        if (p_index + 1) % 50 == 0:
            log(f"{shard_id}: {p_index + 1}/{n_p} prompts, {time.time() - started:.0f}s")
    sentinel_end = 0.0 if persona_stage else _sentinel(model, renderer, prompts, baseline, table, weight32, cjk, batch_rows, tolerance)
    if not (np.isfinite(form_logp).all() and np.isfinite(word_logp).all()):
        raise PipelineError("Non-finite scores")
    intended = [
        float(np.linalg.norm(resolve_vector(condition, bundle))) if condition.kind == STEER else 0.0 for condition in selected
    ]
    files = {
        "scores.npz": art.npz_bytes(
            {
                "cids": np.asarray([condition.cid for condition in selected]),
                "prompt_ids": np.asarray([p.prompt_id for p in prompts]),
                "words": np.asarray(table.words),
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
            "runtime": runtime,
            "determinism": determinism,
            "selftest": selftest,
            "bundle_sha256": bundle_hashes,
            "baseline_sha256": ctx.completed("baseline_rep1")[1]["files"]["baseline.npz"]["sha256"],
            "batch_rows": batch_rows,
            "sentinel_max_abs": [sentinel_start, sentinel_end],
            "prompt_conditions": n_c * n_p,
            "seconds": time.time() - started,
        },
    )
    log(f"{shard_id}: scored {n_c} conditions x {n_p} prompts in {time.time() - started:.0f}s")


def stage_sample(ctx: RunContext, shard_id: str) -> None:
    from .sampling import SAMPLES_PER_PROMPT, MAX_NEW_TOKENS, parse_answer, sample_answers, sampling_seed

    spec = ctx.shard_spec(shard_id)
    shard = ctx.shard(shard_id)
    if shard.is_complete():
        log(f"{shard_id}: complete and verified; skipping")
        return
    shard.quarantine()
    runtime = verify_runtime(ctx, gpu=True)
    bundle, bundle_hashes = load_bundle(ctx)
    conditions = ctx.conditions()
    model, tokenizer, determinism = load_model_verified(ctx)
    selftest = _gpu_selftest(ctx, model, tokenizer)
    weight32 = lm_head_weight32(model)
    renderer = Renderer(tokenizer, ctx.package)
    prompts = ctx.package.s0_prompts()
    lexicon = ctx.package.parser_lexicon
    cids = spec["payload"]["conditions"]
    tokens = np.full((len(cids), len(prompts), SAMPLES_PER_PROMPT, MAX_NEW_TOKENS), -1, dtype=np.int64)
    parsed = np.empty((len(cids), len(prompts), SAMPLES_PER_PROMPT), dtype="U24")
    for c_index, cid in enumerate(cids):
        condition = conditions[cid]
        row = row_steer(condition, bundle)
        persona = condition.persona if condition.kind == PERSONA else "P_default"
        for p_index, prompt in enumerate(prompts):
            rendered = renderer.render(persona, prompt.prompt)
            samples = sample_answers(model, rendered, row, weight32=weight32, seed=sampling_seed(cid, prompt.prompt_id))
            for s_index, sample in enumerate(samples):
                tokens[c_index, p_index, s_index, : len(sample)] = sample
                text = tokenizer.decode(sample, skip_special_tokens=True)
                parsed[c_index, p_index, s_index] = parse_answer(text, lexicon["forms"], lexicon["chinese"]) or ""
        log(f"{shard_id}: sampled condition {c_index + 1}/{len(cids)}")
    files = {
        "samples.npz": art.npz_bytes(
            {"cids": np.asarray(cids), "prompt_ids": np.asarray([p.prompt_id for p in prompts]), "tokens": tokens, "parsed": parsed}
        )
    }
    shard.publish(files, {"stage": "sample", "runtime": runtime, "determinism": determinism, "selftest": selftest,
                          "bundle_sha256": bundle_hashes, "sampling": "explicit multinomial; see IMPLEMENTATION_CHOICES.json"})


def run_stage(ctx: RunContext, shard_id: str) -> None:
    from . import analysis, integrity, preflight

    stages = {
        "preflight": preflight.stage_preflight,
        "extract": stage_extract,
        "directions": stage_directions,
        "baseline": stage_baseline,
        "score": stage_score,
        "score_persona": stage_score,
        "sample": stage_sample,
        "integrity": integrity.stage_integrity,
        "analysis": analysis.stage_analysis,
    }
    stages[ctx.shard_spec(shard_id)["stage"]](ctx, shard_id)


def attempt(ctx: RunContext, shard_id: str, event: str, **payload) -> None:
    art.attempt_record(ctx.out_root, shard_id, {"event": event, "utc": art.utc_now(), **payload})
