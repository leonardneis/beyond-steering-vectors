"""Plan or execute idempotent local/HPC confirmatory tasks from one manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from importlib import metadata as importlib_metadata

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.experiment_logging import device_info, git_commit_hash  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


GPU_SCRIPTS = {
    "scripts/extract_steering_vectors.py",
    "scripts/run_lora_attribution.py",
    "scripts/run_lora_set_interventions.py",
}
OUTPUT_OPTIONS = ("--output", "--output-json", "--output-dir")
SECONDARY_OUTPUT_OPTIONS = ("--output-csv",)


def apply_storage_overrides(manifest: dict) -> dict:
    """Map data/results/runs into persistent shared storage for cluster jobs."""
    shared_root = os.getenv(str(manifest.get("hpc", {}).get("shared_root_env", "SLGEO_SHARED_ROOT")))
    if not shared_root:
        return manifest
    root = shared_root.rstrip("/\\")

    def rewrite(value: Any) -> Any:
        if isinstance(value, str) and any(value == prefix or value.startswith(prefix + "/") for prefix in ("data", "results", "runs")):
            return root + "/" + value.replace("\\", "/")
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    return rewrite(manifest)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(path: Path) -> str | None:
    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for child in sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.name != "run_provenance.json"
        and not item.name.endswith(".provenance.json")
    ):
        digest.update(str(child.relative_to(path)).replace("\\", "/").encode())
        digest.update((sha256(child) or "").encode())
    return digest.hexdigest()


def input_digest(path: Path) -> str | None:
    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        return None
    provenance_path = path / "run_provenance.json"
    if provenance_path.is_file():
        value = json.loads(provenance_path.read_text(encoding="utf-8")).get("output_sha256")
        if value:
            return str(value)
    canonical = [
        child
        for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json")
        if (child := path / name).is_file()
    ]
    if canonical:
        digest = hashlib.sha256()
        for child in canonical:
            digest.update(child.name.encode())
            digest.update((sha256(child) or "").encode())
        return digest.hexdigest()
    return tree_digest(path)


def command_input_artifacts(parts: list[str], output: Path | None) -> list[dict]:
    artifacts, seen = [], set()
    excluded = {output.resolve()} if output is not None else set()
    for option in OUTPUT_OPTIONS + SECONDARY_OUTPUT_OPTIONS:
        if option in parts:
            excluded.add(Path(parts[parts.index(option) + 1]).resolve())
    for token in parts:
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = repo_path(candidate)
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in excluded or resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        artifacts.append({"path": str(resolved), "sha256": input_digest(resolved)})
    return artifacts


def command(*parts: Any) -> list[str]:
    return [str(part) for part in parts if part is not None]


def build_pair(manifest: dict, replicate: dict) -> dict:
    seed = int(replicate["seed"])
    root = repo_path(manifest["output_root"]) / f"seed_{seed}"
    existing = replicate.get("status") == "existing"
    sub_adapter = repo_path(replicate["subliminal_adapter"]) if existing else root / "subliminal" / "student_lora"
    neutral_adapter = repo_path(replicate["neutral_adapter"]) if existing else root / "neutral" / "student_lora"
    sub_train = root / "data" / f"cat_subliminal_10k_seed{seed}.jsonl"
    neutral_train = root / "data" / f"neutral_10k_seed{seed}.jsonl"
    cfg, confirm = manifest["training"], manifest["confirmatory"]
    teacher, prompts = repo_path(manifest["teacher_vector"]), repo_path(manifest["prompts"])
    vector_root, attr, verification = root / "vectors", root / "attribution", root / "verification"
    plan = attr / "topk_plan.json"
    cache = replicate.get("cached_artifacts", {})

    def cached_or(name: str, fallback: list[str]) -> list[str]:
        if existing and cache.get(name):
            source = repo_path(cache[name])
            if not source.exists():
                raise FileNotFoundError(f"Configured cache artifact is missing: {source}")
            if name.endswith("vectors"):
                required = (source / "alignment.json", source / "v_student.pt")
                if not all(path.is_file() for path in required):
                    raise RuntimeError(f"Incomplete vector cache: {source}")
            elif source.suffix.lower() == ".json":
                payload = json.loads(source.read_text(encoding="utf-8"))
                expected = (
                    {"n_prompts": confirm["layer_prompts"], "prompt_offset": confirm["layer_offset"]}
                    if "layers" in name
                    else {"n_prompts": confirm["module_prompts"], "prompt_offset": confirm["module_offset"]}
                )
                mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
                if "modules" in name and not name.startswith("paired_"):
                    if payload.get("include_layers") != confirm["selected_layers"]:
                        mismatch["include_layers"] = (payload.get("include_layers"), confirm["selected_layers"])
                if not name.startswith("paired_") and payload.get("teacher_vector_sha256") != manifest.get("teacher_vector_sha256"):
                    mismatch["teacher_vector_sha256"] = (payload.get("teacher_vector_sha256"), manifest.get("teacher_vector_sha256"))
                if mismatch:
                    raise RuntimeError(f"Cache {source} is incompatible with manifest: {mismatch}")
            output = expected_output(fallback)
            return command("scripts/materialize_cached_artifact.py", "--source", cache[name], "--output", output)
        return fallback

    stages = {
        "prepare": [] if existing else [
            command("scripts/subsample_dataset.py", "--input", manifest["conditions"]["subliminal"]["source_dataset"], "--output", sub_train, "--size", cfg["sample_size"], "--seed", seed),
            command("scripts/subsample_dataset.py", "--input", manifest["conditions"]["neutral"]["source_dataset"], "--output", neutral_train, "--size", cfg["sample_size"], "--seed", seed),
        ],
        "train_subliminal": [] if existing else [command("scripts/train_student.py", "--config", manifest["train_config"], "--model-config", manifest["model_config"], "--train-file", sub_train, "--output-dir", sub_adapter, "--seed", seed, "--run-id", f"confirmatory_cat_seed{seed}_subliminal")],
        "train_neutral": [] if existing else [command("scripts/train_student.py", "--config", manifest["train_config"], "--model-config", manifest["model_config"], "--train-file", neutral_train, "--output-dir", neutral_adapter, "--seed", seed, "--run-id", f"confirmatory_cat_seed{seed}_neutral")],
        "verify_runs": [
            command("scripts/evaluate_preference.py", "--model-config", manifest["model_config"], "--adapter-path", sub_adapter, "--target-animal", "cat", "--num-samples", confirm["behavior_samples"], "--generation-mode", "greedy", "--prompt-set", "paper_reference", "--system-prompt-mode", "neutral", "--no-default-system-prompt", "--output-json", verification / "subliminal.json", "--output-csv", verification / "subliminal.csv", "--run-id", f"confirmatory_cat_seed{seed}_verify_subliminal"),
            command("scripts/evaluate_preference.py", "--model-config", manifest["model_config"], "--adapter-path", neutral_adapter, "--target-animal", "cat", "--num-samples", confirm["behavior_samples"], "--generation-mode", "greedy", "--prompt-set", "paper_reference", "--system-prompt-mode", "neutral", "--no-default-system-prompt", "--output-json", verification / "neutral.json", "--output-csv", verification / "neutral.csv", "--run-id", f"confirmatory_cat_seed{seed}_verify_neutral"),
        ],
        "verify_compare": [command("scripts/compare_full_behavior.py", "--subliminal", verification / "subliminal.json", "--neutral", verification / "neutral.json", "--output", verification / "paired.json", "--bootstrap-samples", confirm["bootstrap_samples"], "--seed", seed, "--require-positive")],
        "vectors": [
            cached_or("subliminal_vectors", command("scripts/extract_steering_vectors.py", "--adapter-path", sub_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--trait", "cat", "--n-prompts", confirm["vector_prompts"], "--output-dir", vector_root / "subliminal")),
            cached_or("neutral_vectors", command("scripts/extract_steering_vectors.py", "--adapter-path", neutral_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--trait", "cat", "--n-prompts", confirm["vector_prompts"], "--output-dir", vector_root / "neutral")),
        ],
        "layer_runs": [
            cached_or("subliminal_layers", command("scripts/run_lora_attribution.py", "--adapter-path", sub_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--n-prompts", confirm["layer_prompts"], "--prompt-offset", confirm["layer_offset"], "--group-by", "layer", "--output", attr / "subliminal_layers.json")),
            cached_or("neutral_layers", command("scripts/run_lora_attribution.py", "--adapter-path", neutral_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--n-prompts", confirm["layer_prompts"], "--prompt-offset", confirm["layer_offset"], "--group-by", "layer", "--output", attr / "neutral_layers.json")),
        ],
        "layer_compare": [cached_or("paired_layers", command("scripts/compare_layer_attribution.py", "--subliminal", attr / "subliminal_layers.json", "--neutral", attr / "neutral_layers.json", "--output", attr / "paired_layers.json", "--bootstrap-samples", confirm["bootstrap_samples"]))],
        "module_runs": [
            cached_or("subliminal_modules", command("scripts/run_lora_attribution.py", "--adapter-path", sub_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--n-prompts", confirm["module_prompts"], "--prompt-offset", confirm["module_offset"], "--group-by", "individual", "--include-layers", *confirm["selected_layers"], "--output", attr / "subliminal_modules.json")),
            cached_or("neutral_modules", command("scripts/run_lora_attribution.py", "--adapter-path", neutral_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--n-prompts", confirm["module_prompts"], "--prompt-offset", confirm["module_offset"], "--group-by", "individual", "--include-layers", *confirm["selected_layers"], "--output", attr / "neutral_modules.json")),
        ],
        "module_compare": [cached_or("paired_modules", command("scripts/compare_layer_attribution.py", "--subliminal", attr / "subliminal_modules.json", "--neutral", attr / "neutral_modules.json", "--output", attr / "paired_modules.json", "--bootstrap-samples", confirm["bootstrap_samples"]))],
        "topk_prepare": [command("scripts/prepare_topk_module_sets.py", "--ranking", attr / "paired_modules.json", "--adapter-dir", sub_adapter, "--k", *confirm["top_k"], "--seed", confirm["control_seed"], "--matching-pool-size", 3, "--control-types", *confirm["control_types"], "--output", plan)],
        "activation_runs": [
            command("scripts/run_lora_set_interventions.py", "--adapter-path", sub_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--selection-plan", plan, "--n-prompts", confirm["intervention_prompts"], "--prompt-offset", confirm["intervention_offset"], "--set-names", "top_k", "norm_matched_control", "--output", attr / "subliminal_topk.json"),
            command("scripts/run_lora_set_interventions.py", "--adapter-path", neutral_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--selection-plan", plan, "--n-prompts", confirm["intervention_prompts"], "--prompt-offset", confirm["intervention_offset"], "--set-names", "top_k", "norm_matched_control", "--output", attr / "neutral_topk.json"),
        ],
        "activation_compare": [command("scripts/compare_lora_set_interventions.py", "--subliminal", attr / "subliminal_topk.json", "--neutral", attr / "neutral_topk.json", "--output", attr / "paired_topk.json", "--bootstrap-samples", confirm["bootstrap_samples"])],
        "behavior_runs": [
            command("scripts/run_lora_set_behavior.py", "--adapter-path", sub_adapter, "--selection-plan", plan, "--target-animal", "cat", "--num-samples", confirm["behavior_samples"], "--prompt-set", "paper_reference", "--set-names", "top_k", "norm_matched_control", "--k", *confirm["top_k"], "--output", attr / "subliminal_behavior.json"),
            command("scripts/run_lora_set_behavior.py", "--adapter-path", neutral_adapter, "--selection-plan", plan, "--target-animal", "cat", "--num-samples", confirm["behavior_samples"], "--prompt-set", "paper_reference", "--set-names", "top_k", "norm_matched_control", "--k", *confirm["top_k"], "--output", attr / "neutral_behavior.json"),
        ],
        "behavior_compare": [command("scripts/compare_lora_set_behavior.py", "--subliminal", attr / "subliminal_behavior.json", "--neutral", attr / "neutral_behavior.json", "--output", attr / "paired_behavior.json", "--bootstrap-samples", confirm["bootstrap_samples"])],
    }
    # Backward-compatible logical aliases for local use.
    stages["verify"] = stages["verify_runs"] + stages["verify_compare"]
    analysis_names = ["vectors", "layer_runs", "layer_compare", "module_runs", "module_compare", "topk_prepare", "activation_runs", "activation_compare", "behavior_runs", "behavior_compare"]
    stages["analysis"] = [item for name in analysis_names for item in stages[name]]
    return {"seed": seed, "root": str(root), "existing": existing, "adapters": {"subliminal": str(sub_adapter), "neutral": str(neutral_adapter)}, "stages": stages}


def choose_profile(manifest: dict) -> dict:
    total = float(device_info().get("total_vram_gb") or 0)
    profiles = sorted(manifest.get("hpc", {}).get("gpu_profiles", []), key=lambda x: float(x["min_vram_gb"]), reverse=True)
    return next((dict(profile) for profile in profiles if total >= float(profile["min_vram_gb"])), {})


def tune_command(parts: list[str], profile: dict) -> list[str]:
    tuned = list(parts)
    if not tuned:
        return tuned
    if tuned[0] == "scripts/train_student.py" and profile:
        tuned += ["--batch-size", str(profile["train_batch_size"]), "--gradient-accumulation-steps", str(profile["gradient_accumulation_steps"])]
    elif tuned[0] in GPU_SCRIPTS and profile and "--batch-size" not in tuned:
        tuned += ["--batch-size", str(profile["analysis_batch_size"])]
    return tuned


def expected_output(parts: list[str]) -> Path | None:
    for option in OUTPUT_OPTIONS:
        if option in parts:
            return Path(parts[parts.index(option) + 1])
    return None


def all_outputs(parts: list[str]) -> list[Path]:
    outputs = []
    for option in OUTPUT_OPTIONS + SECONDARY_OUTPUT_OPTIONS:
        if option in parts:
            outputs.append(Path(parts[parts.index(option) + 1]))
    return outputs


def validate_artifact(path: Path) -> None:
    """Apply inexpensive structural validation before publishing completion."""
    if path.is_file():
        if path.stat().st_size == 0:
            raise ValueError(f"Output file is empty: {path}")
        if path.suffix.lower() == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        return
    if not path.is_dir():
        raise FileNotFoundError(f"Expected output does not exist: {path}")
    files = [child for child in path.rglob("*") if child.is_file()]
    if not files:
        raise ValueError(f"Output directory is empty: {path}")
    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        json.loads(adapter_config.read_text(encoding="utf-8"))
        if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError(f"Adapter output has no weight file: {path}")


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        temp = target.with_name(target.name + ".incoming")
        if temp.exists():
            shutil.rmtree(temp)
        shutil.copytree(source, temp)
        if target.exists():
            shutil.rmtree(target)
        temp.replace(target)
    else:
        temp = target.with_name(target.name + ".incoming")
        shutil.copy2(source, temp)
        temp.replace(target)


def rewrite_scratch_paths(source: Path, scratch: Path, final: Path) -> None:
    files = [source] if source.is_file() else list(source.rglob("*.json"))
    old, new = str(scratch), str(final)

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(old, new)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    for path in files:
        if path.suffix.lower() != ".json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        write_json_atomic(path, rewrite(payload))


def scratch_command(parts: list[str], final: Path, scratch_root: Path, task_id: str) -> tuple[list[str], Path]:
    scratch = scratch_root / "sl_thesis" / task_id / final.name
    if scratch.exists():
        shutil.rmtree(scratch) if scratch.is_dir() else scratch.unlink()
    scratch.parent.mkdir(parents=True, exist_ok=True)
    result = list(parts)
    for option in OUTPUT_OPTIONS:
        if option in result and Path(result[result.index(option) + 1]) == final:
            result[result.index(option) + 1] = str(scratch)
            break
    return result, scratch


def provenance(manifest_path: Path, manifest: dict, seed: int, stage: str, index: int, started: str, duration: float, command_parts: list[str], output: Path | None, attempts: int) -> dict:
    packages = {}
    for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    driver = None
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip() or None
    except OSError:
        pass
    return {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "git_commit": git_commit_hash(repo_path(".")),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo_path("."), capture_output=True, text=True).stdout.strip()),
        "seed": seed,
        "stage": stage,
        "command_index": index,
        "command": command_parts,
        "input_artifacts": command_input_artifacts(command_parts, output),
        "started_at": started,
        "ended_at": utc_now(),
        "duration_seconds": duration,
        "attempts": attempts,
        "scheduler": "htcondor" if os.getenv("CONDOR_CLUSTER_ID") else "local",
        "condor_cluster_id": os.getenv("CONDOR_CLUSTER_ID"),
        "condor_proc_id": os.getenv("CONDOR_PROC_ID"),
        "condor_task_id": os.getenv("CONDOR_TASK_ID"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "nvidia_driver": driver,
        "device": device_info(),
        "output": str(output) if output else None,
        "output_sha256": tree_digest(output) if output else None,
    }


def embed_json_provenance(path: Path, record: dict) -> None:
    if path.suffix.lower() != ".json" or not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload["_orchestration"] = {
            key: value for key, value in record.items() if key != "output_sha256"
        }
        write_json_atomic(path, payload)


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run_subprocess_with_signal_forwarding(argv: list[str], cwd: Path) -> None:
    """Wait for a child while forwarding scheduler termination signals."""
    process = subprocess.Popen(argv, cwd=cwd)
    previous: dict[int, Any] = {}

    def forward(number, _frame) -> None:
        if process.poll() is None:
            print(f"Forwarding signal {number} to child process {process.pid}", file=sys.stderr)
            process.send_signal(number)

    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.getsignal(number)
            signal.signal(number, forward)
        returncode = process.wait()
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    if returncode:
        raise subprocess.CalledProcessError(returncode, argv)


def run_task(args: argparse.Namespace, manifest_path: Path, manifest: dict, pair: dict, parts: list[str], command_index: int) -> None:
    seed, marker_dir = pair["seed"], Path(pair["root"]) / "orchestration"
    marker = marker_dir / f"{args.stage}_{command_index:02d}.complete.json"
    failed = marker_dir / f"{args.stage}_{command_index:02d}.failed.json"
    running = marker_dir / f"{args.stage}_{command_index:02d}.running.json"
    if marker.exists():
        if args.resume:
            print(f"SKIP completed: {marker}")
            return
        raise FileExistsError(f"Command already completed; use --resume: {marker}")
    final = expected_output(parts)
    is_training = parts and parts[0] == "scripts/train_student.py"
    run_parts = tune_command(parts, choose_profile(manifest))
    if final is not None and final.exists():
        if is_training and args.resume:
            run_parts.append("--resume")
        else:
            raise FileExistsError(f"Unmarked output exists; refusing silent overwrite: {final}")
    scratch_output = None
    scratch_stages = set(manifest.get("hpc", {}).get("scratch_stages", []))
    if args.scratch_root and final is not None and args.stage in scratch_stages and not is_training:
        run_parts, scratch_output = scratch_command(run_parts, final, Path(args.scratch_root), f"seed{seed}_{args.stage}_{command_index}")
    started, start_perf = utc_now(), time.perf_counter()
    retries = args.retries if args.retries is not None else int(manifest.get("hpc", {}).get("retries", 1))
    backoff = int(manifest.get("hpc", {}).get("retry_backoff_seconds", 30))
    attempts = 0
    marker_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        running,
        {
            "started_at": started,
            "seed": seed,
            "stage": args.stage,
            "command_index": command_index,
            "scheduler": "htcondor" if os.getenv("CONDOR_CLUSTER_ID") else "local",
            "condor_cluster_id": os.getenv("CONDOR_CLUSTER_ID"),
            "condor_proc_id": os.getenv("CONDOR_PROC_ID"),
            "condor_task_id": os.getenv("CONDOR_TASK_ID"),
            "host": socket.gethostname(),
        },
    )
    try:
        while True:
            attempts += 1
            try:
                run_subprocess_with_signal_forwarding(
                    [sys.executable, *run_parts], cwd=repo_path(".")
                )
                break
            except subprocess.CalledProcessError as exc:
                if exc.returncode == 75:
                    raise
                if attempts >= retries:
                    raise
                if is_training and final is not None and final.exists() and "--resume" not in run_parts:
                    run_parts.append("--resume")
                delay = backoff * (2 ** (attempts - 1))
                print(f"Transient-failure retry {attempts}/{retries} in {delay}s", file=sys.stderr)
                time.sleep(delay)
        if scratch_output is not None:
            if not scratch_output.exists():
                raise FileNotFoundError(f"Scratch command did not create {scratch_output}")
            rewrite_scratch_paths(scratch_output, scratch_output, final)
            atomic_copy(scratch_output, final)
        for artifact in all_outputs(parts):
            validate_artifact(artifact)
        record = provenance(manifest_path, manifest, seed, args.stage, command_index, started, time.perf_counter() - start_perf, run_parts, final, attempts)
        for artifact in all_outputs(parts):
            embed_json_provenance(artifact, record)
            artifact_record = {**record, "output": str(artifact), "output_sha256": tree_digest(artifact)}
            prov_path = artifact / "run_provenance.json" if artifact.is_dir() else artifact.with_suffix(artifact.suffix + ".provenance.json")
            write_json_atomic(prov_path, artifact_record)
            if artifact == final:
                record = artifact_record
        write_json_atomic(marker, record)
        running.unlink(missing_ok=True)
        if failed.exists():
            failed.unlink()
    except BaseException as exc:
        running.unlink(missing_ok=True)
        write_json_atomic(failed, {"started_at": started, "failed_at": utc_now(), "attempts": attempts, "error": repr(exc), "command": run_parts})
        raise
    finally:
        subprocess.run(
            [sys.executable, "scripts/confirmatory_status.py", "--manifest", str(manifest_path)],
            cwd=repo_path("."),
            check=False,
        )


def main() -> None:
    stages = ("prepare", "train_subliminal", "train_neutral", "verify_runs", "verify_compare", "vectors", "layer_runs", "layer_compare", "module_runs", "module_compare", "topk_prepare", "activation_runs", "activation_compare", "behavior_runs", "behavior_compare", "verify", "analysis", "all")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pair-index", type=int)
    parser.add_argument("--stage", choices=stages, default="all")
    parser.add_argument("--command-index", type=int)
    parser.add_argument("--emit-plan", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scratch-root")
    parser.add_argument("--retries", type=int)
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = apply_storage_overrides(load_yaml(manifest_path))
    teacher_path = repo_path(manifest["teacher_vector"])
    expected_teacher_hash = manifest.get("teacher_vector_sha256")
    if expected_teacher_hash and sha256(teacher_path) != expected_teacher_hash:
        raise RuntimeError(f"Frozen teacher vector hash mismatch: {teacher_path}")
    pairs = [build_pair(manifest, item) for item in manifest["replicates"]]
    if args.pair_index is not None:
        pairs = [pairs[args.pair_index]]
    plan = {"schema_version": 2, "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "experiment_id": manifest["experiment_id"], "pairs": pairs}
    output = repo_path(args.emit_plan)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite plan without --resume: {output}")
    write_json_atomic(output, plan)
    print(f"Wrote confirmatory plan to {output}")
    ordered = stages[:15]
    if args.command_index is not None and args.stage in {"all", "verify", "analysis"}:
        parser.error("--command-index requires one concrete stage, not all/verify/analysis")
    if args.stage == "all":
        selected_stages = ordered
    elif args.stage == "verify":
        selected_stages = ("verify_runs", "verify_compare")
    elif args.stage == "analysis":
        selected_stages = ordered[5:]
    else:
        selected_stages = (args.stage,)
    for pair in pairs:
        for stage in selected_stages:
            commands = pair["stages"][stage]
            selected = range(len(commands)) if args.command_index is None else (args.command_index,)
            for index in selected:
                if index >= len(commands):
                    raise IndexError(f"Stage {stage} has no command index {index}")
                parts = commands[index]
                print("COMMAND:", subprocess.list2cmdline([sys.executable, *parts]))
                if args.execute:
                    task_args = argparse.Namespace(**vars(args))
                    task_args.stage = stage
                    run_task(task_args, manifest_path, manifest, pair, parts, index)


if __name__ == "__main__":
    main()
