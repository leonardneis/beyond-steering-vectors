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

GOLDEN_RNG_SHA256 = {
    "bootstrap_direct": "f887e5d4acabe8ccfb9a8a5593a2de009b3cb15f1344ce1ff99b14f4baef5031",
    "bootstrap_identity": "0a09b1088f4f6864167bf9c382f84fd082ced4ddcb6ea0f109200d71c8cff892",
    "bootstrap_hypothetical": "e8df5dce4235d8df22cdc1401c16082f53d3177cdbbfb04e2607915f31035f31",
    "rcov_gaussians": "2848a3e56b6614899343e78fc5ad37ff737df8d6ca390993a089b42046bfc340",
    "riso": "d07ec4bdbdebe2009da382cffac9dfc889cfa0ee1f7215920013c3c40f5d2c2f",
    "null_se_draws": "bff39834d7c566a64423b579ec955819b802ba5e43f321ce883ba57afbe594e0",
}


def rng_golden_check() -> dict:
    index = st.bootstrap_indices()
    observed = {f"bootstrap_{family}": hashlib.sha256(index[family].astype(np.int64).tobytes()).hexdigest() for family in st.FAMILIES}
    observed["rcov_gaussians"] = hashlib.sha256(
        np.random.default_rng(st.RCOV_SEED).standard_normal((1000, 1024)).tobytes()
    ).hexdigest()
    observed["riso"] = hashlib.sha256(st.random_iso_directions(3584).tobytes()).hexdigest()
    observed["null_se_draws"] = hashlib.sha256(
        np.random.default_rng(st.NULL_SE_SEED).integers(0, 16, size=(st.N_BOOT, 16)).astype(np.int64).tobytes()
    ).hexdigest()
    return {"numpy": np.__version__, "match": {key: observed[key] == value for key, value in GOLDEN_RNG_SHA256.items()},
            "pass": all(observed[key] == value for key, value in GOLDEN_RNG_SHA256.items())}


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
    from .modeling import load_tokenizer, snapshot_directory, verify_snapshot
    from .pipeline import PipelineError, log, verify_runtime
    from .selftest import tiny_model_suite

    shard = ctx.shard(shard_id)
    if shard.is_complete():
        log(f"{shard_id}: complete and verified; skipping")
        return
    shard.quarantine()
    runtime = verify_runtime(ctx, gpu=False)
    snapshot = snapshot_directory(os.environ["HF_HOME"])
    snapshot_hashes = verify_snapshot(snapshot, ctx.manifest["model"]["snapshot_sha256"])
    tokenizer = load_tokenizer(snapshot)
    teacher = stage_frozen_teacher(ctx)
    record = {
        "runtime": runtime,
        "frozen_teacher_staged_sha256": teacher,
        "snapshot_sha256": snapshot_hashes,
        "retokenization": retokenization_check(tokenizer, ctx.package),
        "render_identity": render_identity_check(tokenizer, ctx.package, extraction_records(ctx)),
        "tiny_model_suite": tiny_model_suite(),
        "rng_golden": rng_golden_check(),
    }
    failed = [name for name in ("retokenization", "render_identity", "tiny_model_suite", "rng_golden") if not record[name]["pass"]]
    record["pass"] = not failed
    shard.publish({"preflight.json": art.pretty_json(record)}, {"stage": "preflight", "pass": record["pass"], "failed": failed})
    log(f"{shard_id}: preflight {'PASS' if record['pass'] else 'FAIL'} {failed}")
    if failed:
        raise PipelineError(f"Preflight failed: {failed}")
