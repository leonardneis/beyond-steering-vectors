"""Phenotype Anchor v1 command line (cluster and local).

Commands:
  pin-adapters   read-only tree digests of every adapter; fill the never-pinned ones, verify the pinned ones;
                 writes <out>/adapters.lock.json once (before any forward)
  plan           deterministic plan.json from the prompt manifest and the execution manifest
  run --shard    execute one shard (scientific runs refuse a draft contract or placeholder identity)
  tv-cpu         CPU technical validation (manifest reproduction, panel, plan, statistics self-tests)
  tv --name      GPU technical validation (outcome-blind; see slgeo.phenotype.tv)
  tv-project     projection and resource cap from the TV reports (decision D5)
  analyze        sealed until UNSEAL.json names the frozen preregistration; --stage p1 (default), p2 (after p1;
                 reads the stored P1 C2 decision and instrument result) or p1-seeds45 (only if the stored p1
                 result fired the fresh-seed trigger); each result is written once to <out>/analysis/; refuses
                 (nothing written) while the stage's plans or outputs are incomplete, unless --final

Exit codes: 0 ok; 86 final (identity, integrity, contract); 1 infrastructure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

import yaml  # noqa: E402

from slgeo.phenotype import models, plan as planning  # noqa: E402

CONFIG = ROOT / "configs" / "validation" / "phenotype_anchor_v1.yaml"
FINAL = 86
PLACEHOLDER = "FILL_FROM_TV"


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def shared_root() -> Path:
    value = os.environ.get("SLGEO_SHARED_ROOT")
    if not value:
        raise SystemExit("SLGEO_SHARED_ROOT is not set")
    return Path(value)


def out_root(cfg: dict, technical: bool, tag: str | None = None) -> Path:
    base = shared_root() / "results" / "research" / cfg["experiment_id"]
    return base / "technical_validation" / (tag or os.environ.get("SLGEO_RUN_TAG", "tv")) if technical else base


def entries(cfg: dict) -> list[dict]:
    return planning.load_prompt_manifest(ROOT / cfg["contract"]["prompt_manifest"], cfg["contract"]["prompt_manifest_sha256"])


def cmd_pin_adapters(cfg: dict) -> int:
    lock = out_root(cfg, False) / "adapters.lock.json"
    observed = {}
    for name, spec in cfg["adapters"].items():
        digest = models.adapter_tree_digest(shared_root() / spec["path"])
        if spec["digest"] not in ("PIN_AT_FIRST_READ", digest):
            print(f"Adapter {name}: digest {digest} != pinned {spec['digest']}", file=sys.stderr)
            return FINAL
        observed[name] = digest
    if lock.exists():
        if json.loads(lock.read_text())["digests"] != observed:
            print("adapters.lock.json differs from the adapters on disk", file=sys.stderr)
            return FINAL
        return 0
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"digests": observed, "definition": "run_confirmatory_manifest.tree_digest"}, indent=1))
    return 0


def cmd_plan(cfg: dict, technical: bool) -> int:
    root = out_root(cfg, technical)
    projection = root / "tv_projection.json" if not technical else None  # written by tv-project into the SCI root
    seconds, factors = dict(planning.PLACEHOLDER_SECONDS), {}
    if projection is not None and projection.exists():
        p = json.loads(projection.read_text())
        seconds.update(p["seconds_per_unit"])
        factors = p["arm_factor"]
    plan = planning.build_plan(cfg, entries(cfg), seconds=seconds, arm_factor=factors)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "plan.json"
    text = json.dumps(plan, indent=1, sort_keys=True)
    if target.exists() and target.read_text() != text:
        print("plan.json exists with different content", file=sys.stderr)
        return FINAL
    target.write_text(text)
    return 0


def _require_scientific(cfg: dict) -> None:
    if cfg["contract"]["status"] != "frozen":
        raise SystemExit(FINAL)
    if PLACEHOLDER in json.dumps(cfg["execution"]):
        raise SystemExit(FINAL)
    for name, spec in cfg["adapters"].items():
        if spec["digest"] == "PIN_AT_FIRST_READ" and not (out_root(cfg, False) / "adapters.lock.json").exists():
            raise SystemExit(FINAL)


def _load(cfg: dict, adapters: list[str]):
    from slgeo.cts_stage0.checks import cjk_ids_by_rule
    from slgeo.cts_stage0.modeling import load_tokenizer, snapshot_directory, verify_snapshot
    from slgeo.cts_stage0.package import FrozenPackage
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    snapshot_hashes = yaml.safe_load((ROOT / cfg["model"]["snapshot_sha256_from"]).read_text())["model"]["snapshot_sha256"]
    snapshot = snapshot_directory(shared_root() / "huggingface")
    verify_snapshot(snapshot, snapshot_hashes)
    tokenizer = load_tokenizer(snapshot)
    package = FrozenPackage(ROOT / cfg["cts_inputs"]["package"])
    base = models.load_base(snapshot, yaml.safe_load((ROOT / cfg["model"]["model_config"]).read_text()))
    peft_model = None
    if adapters:
        lock_path = out_root(cfg, False) / "adapters.lock.json"
        if lock_path.exists():
            lock = json.loads(lock_path.read_text())["digests"]
        else:  # TV before pin-adapters: only adapters pinned in the manifest may be used
            lock = {a: s["digest"] for a, s in cfg["adapters"].items() if s["digest"] != "PIN_AT_FIRST_READ"}
        paths = {a: shared_root() / cfg["adapters"][a]["path"] for a in adapters}
        peft_model = models.attach_adapters(base, paths, lock)
        for a in adapters:
            models.lora_census(peft_model, a)
    cjk = torch.tensor(cjk_ids_by_rule(tokenizer), dtype=torch.long)
    return tokenizer, package, base, peft_model, list(package.endpoint["boundary_ids"]), cjk


def cmd_run(cfg: dict, shard_id: str, technical: bool) -> int:
    from slgeo.phenotype import execute
    from slgeo.phenotype.runner import Context

    if not technical:
        _require_scientific(cfg)
    root = out_root(cfg, technical)
    plan = json.loads((root / "plan.json").read_text())
    shard = next(s for s in plan["shards"] if s["shard_id"] == shard_id)
    contexts = {cid: Context(**c) for cid, c in plan["contexts"].items()}
    tokenizer, package, base, peft_model, boundary, cjk = _load(cfg, [shard["adapter"]] if shard["adapter"] else [])
    number_prompts = {}
    if shard["kind"] == "numcap":
        lo, hi = cfg["number_capture"]["rows"]
        with open(ROOT / cfg["number_capture"]["file"], encoding="utf-8") as handle:
            for row, line in enumerate(handle):
                if lo <= row < hi:
                    number_prompts[f"num{row:05d}"] = json.loads(line)["prompt"]
    provenance = {"commit": os.environ.get("SLGEO_EXECUTION_GIT_COMMIT"), "run_tag": os.environ.get("SLGEO_RUN_TAG"),
                  "prompt_manifest_sha256": cfg["contract"]["prompt_manifest_sha256"]}
    execute.run_shard(shard, contexts, out_root=root, tokenizer=tokenizer, base_model=base, peft_model=peft_model,
                      package=package, boundary_ids=boundary, cjk_ids=cjk, sampling=cfg["sampling"],
                      capture_cells=cfg["cells"]["capture"], number_prompts=number_prompts, provenance=provenance)
    return 0


def cmd_tv_cpu(cfg: dict) -> int:
    import subprocess

    checks = [
        [sys.executable, str(ROOT / "scripts" / "build_phenotype_prompts.py"), "--check-only"],
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / "test_phenotype_anchor.py"),
         str(ROOT / "tests" / "test_phenotype_runner.py"), str(ROOT / "tests" / "test_phenotype_analysis.py")],
    ]
    for command in checks:
        if subprocess.run(command, cwd=ROOT).returncode != 0:
            return FINAL
    planning.build_plan(cfg, entries(cfg))
    return 0


def cmd_tv(cfg: dict, name: str) -> int:
    from slgeo.phenotype import tv
    from slgeo.phenotype.execute import form_table

    adapters = ["S1", "N1"]  # locally pinned seed-1 adapters; every adapter's census runs in pin-adapters + run
    tokenizer, package, base, peft_model, boundary, cjk = _load(cfg, adapters)
    v_prompts = [json.loads(l)["prompt"] for l in (ROOT / cfg["cts_inputs"]["package"] / "cts_stage0_validation_prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    report = tv.gpu_checks(tokenizer=tokenizer, package=package, base_model=base, peft_model=peft_model,
                           adapter_names=adapters, table=form_table(tokenizer, boundary), boundary_ids=boundary,
                           cjk_ids=cjk, v_prompts=v_prompts)
    from slgeo.cts_stage0.identity import runtime_identity

    report["identity"] = runtime_identity()
    root = out_root(cfg, True)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"tv_{name}.json").write_text(json.dumps(report, indent=1, default=str))
    return 0 if report["passed"] else FINAL


def cmd_tv_project(cfg: dict) -> int:
    from slgeo.phenotype import tv

    root = out_root(cfg, True)
    reports = [json.loads(p.read_text()) for p in sorted(root.glob("tv_gpu_*.json"))]
    if len(reports) < 2 or not all(r["passed"] for r in reports):
        return FINAL
    summary = tv.overhead_and_factors(reports)
    seconds = dict(planning.PLACEHOLDER_SECONDS, score=summary["base_score_s"])
    factors = {arm: summary["arm_factor"].get("S1", 1.0) for arm in ("S1", "S2", "S3")}
    factors.update({arm: summary["arm_factor"].get("N1", 1.0) for arm in ("N1", "N2", "N3")})
    plan = planning.build_plan(cfg, entries(cfg), seconds=seconds, arm_factor=factors)
    overhead = 1.25  # replaced by the dry-shard end-to-end factor when available
    projection = planning.projection_a100_h(plan, overhead)
    cap = planning.cap_a100_h(projection, summary["throughput_cv"])
    out = {"seconds_per_unit": seconds, "arm_factor": factors, "throughput_cv": summary["throughput_cv"],
           "overhead_factor": overhead, "projection_a100_h": projection, "proposed_cap_a100_h": cap}
    (root.parent.parent / "tv_projection.json").write_text(json.dumps(out, indent=1))
    return 0


PREREG_TAG = "prereg/phenotype-anchor-v1"


def cmd_analyze(cfg: dict, stage: str, final: bool = False) -> int:
    """Exit 0 with a written result; FINAL (86) when sealed, out of order, not ready (nothing written; ``--final``
    records the TECHNICAL_FAIL instead) or when the written class is a TECHNICAL_FAIL."""
    from slgeo.phenotype import analysis

    root = out_root(cfg, False)
    v1 = json.loads((ROOT / cfg["contract"]["path"] / "seed1_v1_profile.json").read_text())["profile"]
    try:
        result = analysis.run_stage(root, stage, entries=entries(cfg), v1=v1, sample_k=int(cfg["sampling"]["k"]),
                                    expected_tag=PREREG_TAG, final=final)
        target = analysis.write_stage(root, stage, result)
    except (analysis.SealedError, analysis.StageOrderError) as exc:
        print(exc, file=sys.stderr)
        return FINAL
    outcome = result["final_outcome"] if stage == "p1-seeds45" else result["outcome"]
    print(f"{stage}: {outcome['cls']} -> {target}")
    failed = (result["stage2"] if stage == "p1-seeds45" else result).get("technical_fail")
    return FINAL if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pin-adapters")
    p = sub.add_parser("plan")
    p.add_argument("--technical-validation", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--shard", required=True)
    r.add_argument("--technical-validation", action="store_true")
    sub.add_parser("tv-cpu")
    t = sub.add_parser("tv")
    t.add_argument("--name", required=True)
    sub.add_parser("tv-project")
    a = sub.add_parser("analyze")
    a.add_argument("--stage", choices=("p1", "p2", "p1-seeds45"), default="p1")
    a.add_argument("--final", action="store_true",
                   help="record a TECHNICAL_FAIL for missing outputs instead of refusing (write-once)")
    args = parser.parse_args()
    cfg = config()
    if args.command == "pin-adapters":
        return cmd_pin_adapters(cfg)
    if args.command == "plan":
        return cmd_plan(cfg, args.technical_validation)
    if args.command == "run":
        return cmd_run(cfg, args.shard, args.technical_validation)
    if args.command == "tv-cpu":
        return cmd_tv_cpu(cfg)
    if args.command == "tv":
        return cmd_tv(cfg, args.name)
    if args.command == "tv-project":
        return cmd_tv_project(cfg)
    return cmd_analyze(cfg, args.stage, args.final)


if __name__ == "__main__":
    sys.exit(main())
