"""Integrity report of the scientific run (spec ``integrity_checks``; decision rank 1).

The report contains check names, booleans, counts, hashes and tolerances only; never a score or statistic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import artifacts as art
from .conditions import STEER, resolve_vector, vector_sha256


TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".log", ".out", ".err", ".md", ".csv", ".dag", ".sub", ".sh"}
KNOWN_BINARIES = {"v_teacher.pt"}  # staged frozen teacher, admitted by exact name and verified hash


class UnscannableFile(RuntimeError):
    pass


def _string_values(path: Path) -> list[str]:
    """Every string stored in a text/JSON file or in the string arrays of an npz/npy file."""
    if path.suffix in {".npz", ".npy"}:
        out: list[str] = []
        if path.suffix == ".npy":
            arrays = {"": np.load(path, allow_pickle=False)}
        else:
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
        for key, array in arrays.items():
            out.append(key)
            if array.dtype.kind in {"U", "S"}:
                out.extend(str(value) for value in array.ravel().tolist())
        return out
    if path.suffix not in TEXT_SUFFIXES:
        raise UnscannableFile(path.name)
    text = path.read_text(encoding="utf-8", errors="replace")
    out = [text]
    if path.suffix == ".json":
        try:
            out.extend(_json_strings(json.loads(text)))
        except json.JSONDecodeError:
            pass
    return out


def _json_strings(value) -> list[str]:
    """Every decoded string (keys and values) of a JSON document, so escaping cannot hide a match."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for key, item in value.items() for s in [key, *_json_strings(item)]]
    if isinstance(value, list):
        return [s for item in value for s in _json_strings(item)]
    return []



def dc_leak_scan(roots: list[Path], fingerprints: dict[str, set[str]]) -> dict:
    """Scan every output file (and job logs) for D/C prompt ids, hashes and texts; report counts only."""
    id_pattern = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(sorted(map(re.escape, fingerprints["ids"]))) + r")(?![0-9])")
    hashes = fingerprints["sha256"]
    texts = [text for text in fingerprints["texts"] if len(text) >= 12]
    scanned, hits, unscannable, incoming = 0, [], 0, 0

    def leaks(value: str) -> bool:
        return bool(id_pattern.search(value)) or any(h in value for h in hashes) or any(t in value for t in texts)

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            scanned += 1
            if leaks(path.relative_to(root).as_posix()):
                hits.append(path.name)
                continue
            if path.name.endswith(".incoming"):
                incoming += 1
                continue
            if path.name in KNOWN_BINARIES:
                continue
            try:
                values = _string_values(path)
            except (UnscannableFile, ValueError, OSError):
                unscannable += 1
                continue
            if any(leaks(value) for value in values):
                hits.append(path.name)
    return {
        "files_scanned": scanned,
        "files_with_hits": len(hits),
        "unscannable_files": unscannable,
        "leftover_incoming_files": incoming,
        "pass": not hits and not unscannable and not incoming and scanned > 0,
    }


def stage_integrity(ctx, shard_id: str = "integrity") -> None:
    from .pipeline import load_baseline, load_bundle, log, verify_runtime

    shard = ctx.shard(shard_id)
    if shard.is_complete():
        log(f"{shard_id}: complete and verified; skipping")
        return
    shard.quarantine()
    checks: dict[str, dict] = {}
    runtime = verify_runtime(ctx, gpu=False)
    checks["tracked_code_and_manifest"] = {"pass": True, "tracked_files": runtime["tracked_files_verified"]}

    markers = {}
    incomplete = []
    for spec in ctx.plan["shards"]:
        if spec["stage"] in ("integrity", "analysis"):
            continue
        candidate = ctx.shard(spec["shard_id"])
        if candidate.is_complete():
            markers[spec["shard_id"]] = json.loads(candidate.marker.read_bytes())
        else:
            incomplete.append(spec["shard_id"])
    checks["all_shards_complete_and_hash_verified"] = {"pass": not incomplete, "incomplete": incomplete}
    checks["sidecars_carry_execution_commit"] = {
        "pass": all(m["run_identity"]["execution_commit"] == ctx.run_record["execution_commit"] for m in markers.values()),
        "execution_commit": ctx.run_record["execution_commit"],
    }
    preflight = markers.get("preflight", {})
    checks["preflight"] = {"pass": bool(preflight.get("pass"))}
    gpu = [m for m in markers.values() if m.get("stage") in ("extract", "baseline", "score", "score_persona", "sample")]
    identities = {
        json.dumps(
            {k: m["runtime"]["identity"][k] for k in ("gpu_name", "packages", "python", "cuda_runtime")}
            | {"driver": (m["runtime"]["identity"]["nvidia_smi"] or [{}])[0].get("driver")},
            sort_keys=True,
        )
        for m in gpu
    }
    checks["single_execution_identity"] = {"pass": len(identities) == 1, "distinct_identities": len(identities)}
    checks["hook_selftests"] = {"pass": all(m.get("selftest", {}).get("pass") for m in gpu), "shards": len(gpu)}

    if incomplete:
        report = {"pass": False, "checks": checks}
        shard.publish({"integrity.json": art.pretty_json(report)}, {"stage": "integrity", "pass": False})
        log(f"{shard_id}: FAIL (incomplete shards)")
        return

    # Extraction completeness and finiteness.
    personas_seen = set()
    extraction_ok = True
    for spec in ctx.plan["shards"]:
        if spec["stage"] != "extract":
            continue
        directory = ctx.shard(spec["shard_id"]).directory
        for persona in spec["payload"]["personas"]:
            with np.load(directory / f"{persona}.npz", allow_pickle=False) as data:
                extraction_ok &= data["states"].shape[0] == 1024 and bool(np.isfinite(data["states"]).all())
                extraction_ok &= bool(np.isfinite(data["half_sums"]).all())
            personas_seen.add(persona)
    checks["extraction_complete_and_finite"] = {"pass": bool(extraction_ok and personas_seen == set(ctx.package.personas))}

    # Scoring completeness, finiteness, prompt sets and vector provenance.
    bundle, _ = load_bundle(ctx)
    conditions = ctx.conditions()
    s0_ids = sorted(ctx.package.partition_ids()["S0"])
    expected = {cid for cid, c in conditions.items() if c.kind != "unsteered"}
    seen: dict[str, int] = {}
    finite, prompts_ok, vectors_ok = True, True, True
    for spec in ctx.plan["shards"]:
        if spec["stage"] not in ("score", "score_persona"):
            continue
        directory = ctx.shard(spec["shard_id"]).directory
        with np.load(directory / "scores.npz", allow_pickle=False) as data:
            prompts_ok &= sorted(data["prompt_ids"].tolist()) == s0_ids
            finite &= bool(np.isfinite(data["form_logp"]).all() and np.isfinite(data["word_logp"]).all())
            for index, cid in enumerate(data["cids"].tolist()):
                seen[cid] = seen.get(cid, 0) + 1
                condition = conditions[cid]
                if condition.kind == STEER:
                    vectors_ok &= data["vector_sha256"][index] == vector_sha256(condition, bundle)
                    vectors_ok &= bool(np.isclose(data["vector_norm"][index], np.linalg.norm(resolve_vector(condition, bundle)), rtol=1e-12))
    checks["all_conditions_present_once"] = {
        "pass": set(seen) == expected and all(count == 1 for count in seen.values()),
        "expected": len(expected),
        "found": len(seen),
    }
    checks["all_s0_prompts_present"] = {"pass": bool(prompts_ok), "n_s0": len(s0_ids)}
    checks["no_nan_or_inf"] = {"pass": bool(finite)}
    checks["vector_provenance_rederived"] = {"pass": bool(vectors_ok)}

    rep1, rep2 = load_baseline(ctx, 1), load_baseline(ctx, 2)
    same_ids = rep1["prompt_ids"].tolist() == rep2["prompt_ids"].tolist()
    deviation = float(np.abs(rep1["form_logp"] - rep2["form_logp"]).max()) if same_ids else float("inf")
    tolerance = float(ctx.manifest["scoring"]["baseline_repeat_tolerance"])
    sentinels = [max(m.get("sentinel_max_abs", [0.0])) for m in gpu if m.get("stage") == "score"]
    checks["unsteered_repeat_within_1e-4"] = {
        "pass": bool(same_ids and deviation <= tolerance and all(value <= tolerance for value in sentinels)),
        "baseline_rep_max_abs": deviation,
        "sentinel_max_abs": max(sentinels, default=0.0),
        "tolerance": tolerance,
    }
    fingerprints = ctx.package.dc_fingerprints()
    roots = [ctx.out_root]
    logs = ctx.repo_root / "condor" / "logs"
    scan = dc_leak_scan(roots, fingerprints)
    log_scan = dc_leak_scan([logs], fingerprints) if logs.exists() else {"files_scanned": 0, "files_with_hits": 0, "pass": True}
    checks["no_d_or_c_prompt_in_outputs"] = {"pass": bool(scan["pass"] and log_scan["pass"]), "outputs": scan, "logs": log_scan}
    report = {"pass": all(check["pass"] for check in checks.values()), "checks": checks}
    shard.publish({"integrity.json": art.pretty_json(report)}, {"stage": "integrity", "pass": report["pass"]})
    log(f"{shard_id}: {'PASS' if report['pass'] else 'FAIL'} {sorted(k for k, v in checks.items() if not v['pass'])}")
