"""Preflight stage (CPU, execution environment, no model forward).

Checks: tracked code blobs, frozen package, extraction and snapshot hashes, runtime re-tokenization and the
boundary/CJK hashes, render identity on all 1,024 extraction rows, the tiny-model hook/cache/scoring suite
in the installed transformers version, and golden hashes of every frozen RNG stream in the installed numpy.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from . import artifacts as art
from . import statistics as st
from .checks import render_identity_check, retokenization_check
from .package import load_extraction_prompts

# Integer streams (bootstrap indices, null-SE word draws) must be bit-identical in the execution environment.
GOLDEN_RNG_SHA256 = {
    "bootstrap_direct": "f887e5d4acabe8ccfb9a8a5593a2de009b3cb15f1344ce1ff99b14f4baef5031",
    "bootstrap_identity": "0a09b1088f4f6864167bf9c382f84fd082ced4ddcb6ea0f109200d71c8cff892",
    "bootstrap_hypothetical": "e8df5dce4235d8df22cdc1401c16082f53d3177cdbbfb04e2607915f31035f31",
}
# The R_cov Gaussian stream uses libm in the ziggurat tail, which may differ in the last bit between CPUs; it is
# checked by tolerance against values computed with numpy 2.4.4. Its first 199 rows equal the first 199 rows of
# the v1 (1000, 1024) draw (same seed, row-major fill). Within one run R_cov is generated once, in the
# directions stage, and stored with its hash.
GOLDEN_GAUSSIANS = {
    "rcov_gaussians": {
        "seed": st.RCOV_SEED, "shape": (st.N_RCOV, 1024), "sum": -283.98981104154745, "sumsq": 204079.6576535771,
        "max": 4.844439144354511, "min": -4.823947771987368,
        "head": [0.4355153495206435, 0.512317117840909, 1.5499902893939737, 0.42595191326053594],
        "tail": [0.19723148611482982, -1.6116664151284315],
        "sha256_numpy_2_4_4": "8eb784cf61fc77d2700ba748bdfcf4c7d35cd177eeac281e4f260d18ad908c5d",
    },
}


def _gaussian_check(golden: dict) -> dict:
    draws = np.random.default_rng(golden["seed"]).standard_normal(golden["shape"])
    exact = hashlib.sha256(draws.tobytes()).hexdigest() == golden["sha256_numpy_2_4_4"]
    ok = (
        np.isclose(draws.sum(), golden["sum"], rtol=1e-9, atol=1e-6)
        and np.isclose((draws * draws).sum(), golden["sumsq"], rtol=1e-12)
        and np.isclose(draws.max(), golden["max"], rtol=1e-12)
        and np.isclose(draws.min(), golden["min"], rtol=1e-12)
        and np.allclose(draws[0, :4], golden["head"], rtol=1e-12)
        and np.allclose(draws[-1, -2:], golden["tail"], rtol=1e-12)
    )
    return {"within_tolerance": bool(ok), "bit_identical_to_reference": bool(exact)}


def rng_golden_check() -> dict:
    index = st.bootstrap_indices()
    observed = {f"bootstrap_{family}": hashlib.sha256(index[family].astype(np.int64).tobytes()).hexdigest() for family in st.FAMILIES}
    match = {key: observed[key] == value for key, value in GOLDEN_RNG_SHA256.items()}
    gaussians = {name: _gaussian_check(golden) for name, golden in GOLDEN_GAUSSIANS.items()}
    for name, result in gaussians.items():
        match[name] = result["within_tolerance"]
    return {"numpy": np.__version__, "match": match, "gaussian_bit_identity": {k: v["bit_identical_to_reference"] for k, v in gaussians.items()},
            "pass": all(match.values())}


def extraction_records(ctx) -> list[dict]:
    """prompt and system_prompt of rows 0-1023 (the completion field is never read)."""
    from .pipeline import shared_path

    path = shared_path(ctx, ctx.manifest["inputs"]["extraction_file"])
    load_extraction_prompts(ctx.package, path)
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) == 1024:
                break
            row = json.loads(line)
            records.append({"prompt": row["prompt"], "system_prompt": row["system_prompt"]})
    return records


def stage_frozen_teacher(ctx) -> str:
    """Copy the frozen t_cat by exact path (no directory listing) into the run inputs; verify its SHA-256."""
    from .atomic import atomic_write_bytes
    from .guards import FROZEN_V_TEACHER_RELATIVE, FROZEN_V_TEACHER_SHA256, GuardError
    from .package import sha256_path
    from .pipeline import shared_path

    if ctx.manifest["inputs"]["frozen_teacher_vector"] != FROZEN_V_TEACHER_RELATIVE:
        raise GuardError("Frozen teacher path differs from the pinned path")
    source = shared_path(ctx, FROZEN_V_TEACHER_RELATIVE)
    if sha256_path(source) != FROZEN_V_TEACHER_SHA256:
        raise GuardError("Frozen teacher vector hash mismatch")
    target = ctx.out_root / "inputs" / "frozen_t_cat" / "v_teacher.pt"
    if not target.exists():
        atomic_write_bytes(target, source.read_bytes(), write_once=True)
    if sha256_path(target) != FROZEN_V_TEACHER_SHA256:
        raise GuardError("Staged frozen teacher vector differs")
    return FROZEN_V_TEACHER_SHA256


def stage_preflight(ctx, shard_id: str = "preflight") -> None:
    """E1: a failed preflight is final. Its marker records pass=false; the stage then raises a final failure,
    and any later attempt that finds that marker raises again instead of reporting success."""
    from .modeling import load_tokenizer, snapshot_directory, verify_snapshot
    from .pipeline import PipelineError, log, verify_runtime
    from .selftest import tiny_model_suite

    shard = ctx.shard(shard_id)
    if shard.is_complete():
        marker = json.loads(shard.marker.read_bytes())
        if marker.get("pass") is not True or marker.get("failed"):
            raise PipelineError(f"Preflight already failed: {marker.get('failed')}")
        log(f"{shard_id}: complete, verified and passed; skipping")
        return
    shard.quarantine()
    runtime = verify_runtime(ctx, gpu=False)
    ctx.contract.verify_regeneration()
    snapshot = snapshot_directory(os.environ["HF_HOME"])
    snapshot_hashes = verify_snapshot(snapshot, ctx.manifest["model"]["snapshot_sha256"])
    tokenizer = load_tokenizer(snapshot)
    teacher = stage_frozen_teacher(ctx)
    record = {
        "runtime": runtime,
        "contract_regenerated": {"pass": True, "registry_sha256": ctx.contract.registry_sha256},
        "frozen_teacher_staged_sha256": teacher,
        "snapshot_sha256": snapshot_hashes,
        "retokenization": retokenization_check(tokenizer, ctx.package),
        "render_identity": render_identity_check(tokenizer, ctx.package, extraction_records(ctx)),
        "tiny_model_suite": tiny_model_suite(),
        "rng_golden": rng_golden_check(),
    }
    checked = ("contract_regenerated", "retokenization", "render_identity", "tiny_model_suite", "rng_golden")
    failed = [name for name in checked if not record[name]["pass"]]
    record["pass"] = not failed
    shard.publish({"preflight.json": art.pretty_json(record)}, {"stage": "preflight", "pass": record["pass"], "failed": failed})
    log(f"{shard_id}: preflight {'PASS' if record['pass'] else 'FAIL'} {failed}")
    if failed:
        raise PipelineError(f"Preflight failed: {failed}")
