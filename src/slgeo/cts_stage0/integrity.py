"""Integrity report of the v2 scientific run (spec ``integrity_checks``; decision rank 1).

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


def attempt_review(out_root: Path, max_infrastructure_retries: int) -> dict:
    """E5: read every immutable attempt record. Any integrity, refusal or software event fails the run, even
    if a later attempt of that shard succeeded; infrastructure retries per shard are bounded."""
    root = out_root / "orchestration" / "attempts"
    final_events, retries = [], {}
    records = 0
    if root.exists():
        for path in sorted(root.rglob("*.json")):
            record = json.loads(path.read_bytes())
            records += 1
            event, shard = record.get("event"), record.get("shard_id")
            if event in ("refusal", "integrity", "software"):
                final_events.append(shard)
            if event in ("infrastructure", "sigterm"):
                retries[shard] = retries.get(shard, 0) + 1
    over = sorted(shard for shard, count in retries.items() if count > max_infrastructure_retries)
    return {
        "attempt_records": records,
        "shards_with_final_events": len(set(final_events)),
        "shards_over_retry_limit": over,
        "pass": not final_events and not over and records > 0,
    }


def budget_review(ctx) -> dict:
    """Accounting ledger written by the submit-host budget gate (E4): no BUDGET_STOP, consumption within cap."""
    stop = ctx.out_root / "orchestration" / "BUDGET_STOP.json"
    ledger_path = ctx.out_root / "orchestration" / "budget_ledger.json"
    if stop.exists() or not ledger_path.is_file():
        return {"budget_stop": stop.exists(), "ledger_present": ledger_path.is_file(), "pass": False}
    ledger = json.loads(ledger_path.read_bytes())
    cap = float(ctx.manifest["budget"]["scientific_cap_a100_h"])
    consumed = float(ledger["consumed_a100_h"])
    return {"budget_stop": False, "ledger_present": True, "consumed_a100_h": consumed, "cap_a100_h": cap, "pass": bool(consumed <= cap)}


def _recomputed_selftest(selftest: dict) -> bool:
    """Guards recompute verdicts from the stored per-check booleans instead of trusting a stored flag."""
    checks = selftest.get("checks") or {}
    return bool(checks) and all(all(bool(v) for v in group.values()) for group in checks.values())


def _recomputed_preflight(record: dict) -> bool:
    names = ("contract_regenerated", "retokenization", "render_identity", "tiny_model_suite", "rng_golden")
    return all(bool(record.get(name, {}).get("pass")) for name in names)


def stage_integrity(ctx, shard_id: str = "integrity") -> None:
    from .fragility import EPSILON, guard
    from .pipeline import _begin, load_baseline, load_bundle, log, prompts_for, verify_runtime

    shard = _begin(ctx, shard_id)
    if shard is None:
        return
    checks: dict[str, dict] = {}
    runtime = verify_runtime(ctx, gpu=False)
    checks["tracked_code_contract_and_manifest"] = {"pass": True, "tracked_files": runtime["tracked_files_verified"]}

    markers, incomplete = {}, []
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
    preflight_ok = False
    if "preflight" in markers:
        pre_shard = ctx.shard("preflight")
        record = art.load_json_verified(pre_shard.directory / "preflight.json", markers["preflight"]["files"]["preflight.json"]["sha256"])
        preflight_ok = _recomputed_preflight(record) and markers["preflight"].get("pass") is True
    checks["preflight_recomputed"] = {"pass": bool(preflight_ok)}
    gpu = [m for m in markers.values() if m.get("stage") in ("extract", "baseline", "score", "rescore")]
    identities = {
        json.dumps(
            {k: m["runtime"]["identity"][k] for k in ("gpu_name", "packages", "python", "cuda_runtime", "container_image", "venv")}
            | {"driver": (m["runtime"]["identity"]["nvidia_smi"] or [{}])[0].get("driver")},
            sort_keys=True,
        )
        for m in gpu
    }
    checks["single_execution_identity"] = {"pass": len(identities) == 1, "distinct_identities": len(identities)}
    checks["hook_selftests_recomputed"] = {"pass": bool(gpu) and all(_recomputed_selftest(m.get("selftest", {})) for m in gpu), "shards": len(gpu)}
    checks["attempt_records_reviewed"] = attempt_review(ctx.out_root, int(ctx.manifest["budget"]["shard_infrastructure_retries"]))
    checks["budget_ledger_within_cap"] = budget_review(ctx)

    if incomplete:
        report = {"pass": False, "checks": checks}
        shard.publish({"integrity.json": art.pretty_json(report)}, {"stage": "integrity", "pass": False})
        log(f"{shard_id}: FAIL (incomplete shards)")
        return

    personas_seen, extraction_ok = set(), True
    for spec in ctx.plan["shards"]:
        if spec["stage"] != "extract":
            continue
        directory = ctx.shard(spec["shard_id"]).directory
        for persona in spec["payload"]["personas"]:
            with np.load(directory / f"{persona}.npz", allow_pickle=False) as data:
                extraction_ok &= data["states"].shape[0] == 1024 and bool(np.isfinite(data["states"]).all())
                extraction_ok &= bool(np.isfinite(data["half_sums"]).all())
            personas_seen.add(persona)
    checks["extraction_complete_and_finite"] = {"pass": bool(extraction_ok and personas_seen == set(ctx.contract.personas))}

    bundle, _ = load_bundle(ctx)
    conditions = ctx.conditions()
    expected_ids = {name: sorted(p.prompt_id for p in prompts_for(ctx, name)) for name in ("S0_all", "S0_animal")}
    for stage, wanted in (
        ("score", {cid for cid, c in conditions.items() if c.kind != "unsteered"}),
        ("rescore", {cid for cid, c in conditions.items() if c.reference_rescore}),
    ):
        seen: dict[str, int] = {}
        finite, prompts_ok, vectors_ok = True, True, True
        for spec in ctx.plan["shards"]:
            if spec["stage"] != stage:
                continue
            directory = ctx.shard(spec["shard_id"]).directory
            with np.load(directory / "scores.npz", allow_pickle=False) as data:
                finite &= bool(np.isfinite(data["form_logp"]).all() and np.isfinite(data["word_logp"]).all())
                for index, cid in enumerate(data["cids"].tolist()):
                    seen[cid] = seen.get(cid, 0) + 1
                    condition = conditions[cid]
                    prompts_ok &= sorted(data["prompt_ids"].tolist()) == expected_ids[condition.prompt_set]
                    if condition.kind == STEER:
                        vectors_ok &= data["vector_sha256"][index] == vector_sha256(condition, bundle)
                        vectors_ok &= bool(np.isclose(data["vector_norm"][index], np.sqrt(np.sum(resolve_vector(condition, bundle) ** 2)), rtol=1e-12))
        checks[f"{stage}_conditions_present_once"] = {
            "pass": set(seen) == wanted and all(count == 1 for count in seen.values()), "expected": len(wanted), "found": len(seen),
        }
        checks[f"{stage}_prompt_sets_exact"] = {"pass": bool(prompts_ok)}
        checks[f"{stage}_no_nan_or_inf"] = {"pass": bool(finite)}
        checks[f"{stage}_vector_provenance_rederived"] = {"pass": bool(vectors_ok)}

    rep1, rep2 = load_baseline(ctx, 1), load_baseline(ctx, 2)
    same_ids = rep1["prompt_ids"].tolist() == rep2["prompt_ids"].tolist()
    deviation = float(np.abs(rep1["form_logp"] - rep2["form_logp"]).max()) if same_ids else float("inf")
    tolerance = float(ctx.manifest["scoring"]["baseline_repeat_tolerance"])
    sentinels = [max(m.get("sentinel_max_abs", [0.0])) for m in gpu if m.get("stage") in ("score", "rescore")]
    hosts = {m["runtime"]["identity"].get("machine_ad", {}).get("Machine") for m in markers.values() if m.get("stage") == "baseline"}
    checks["unsteered_repeat_within_1e-4"] = {
        "pass": bool(same_ids and deviation <= tolerance and all(value <= tolerance for value in sentinels)),
        "baseline_rep_max_abs": deviation,
        "sentinel_max_abs": max(sentinels, default=0.0),
        "tolerance": tolerance,
    }
    checks["baseline_repeat_on_different_hosts"] = {"pass": len(hosts) == 2 and None not in hosts}

    fragility_ok = False
    if "fragility" in markers:
        frag_shard = ctx.shard("fragility")
        report = art.load_json_verified(frag_shard.directory / "fragility.json", markers["fragility"]["files"]["fragility.json"]["sha256"])
        fragility_ok = bool(report["m"] > 0 and report["rms_z"] <= EPSILON and report["max_abs_z"] <= guard(int(report["m"])))
    checks["fragility"] = {"pass": fragility_ok}

    fingerprints = ctx.package.dc_fingerprints()
    scan = dc_leak_scan([ctx.out_root], fingerprints)
    logs = ctx.repo_root / "condor" / "logs"
    log_scan = dc_leak_scan([logs], fingerprints) if logs.exists() else {"files_scanned": 0, "files_with_hits": 0, "pass": True}
    checks["no_d_or_c_prompt_in_outputs"] = {"pass": bool(scan["pass"] and log_scan["pass"]), "outputs": scan, "logs": log_scan}
    report = {"pass": all(check["pass"] for check in checks.values()), "checks": checks}
    shard.publish({"integrity.json": art.pretty_json(report)}, {"stage": "integrity", "pass": report["pass"]})
    log(f"{shard_id}: {'PASS' if report['pass'] else 'FAIL'} {sorted(k for k, v in checks.items() if not v['pass'])}")
