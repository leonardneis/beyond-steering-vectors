"""Generate and validate the native HTCondor DAGMan confirmatory workflow."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from run_confirmatory_manifest import build_pair  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


STAGES = (
    "prepare", "train_subliminal", "train_neutral", "verify_runs", "verify_compare",
    "vectors", "layer_runs", "layer_compare", "module_runs", "module_compare",
    "topk_prepare", "activation_runs", "activation_compare", "behavior_runs",
    "behavior_compare",
)


@dataclass(frozen=True)
class Task:
    task_id: str
    pair_index: int
    seed: int
    stage: str
    command_index: int
    command: list[str]
    profile: str
    gpus: int
    cpus: int
    memory_mb: int
    min_gpu_memory_mb: int
    estimated_minutes: int
    parents: tuple[str, ...] = ()


def task_id(seed: int, stage: str, index: int) -> str:
    return f"seed{seed}_{stage}_{index:02d}"


def resource_profile(command: list[str], stage: str) -> str:
    script = command[0]
    if script == "scripts/train_student.py":
        return "training"
    if script in {
        "scripts/evaluate_preference.py", "scripts/extract_steering_vectors.py",
        "scripts/run_lora_set_behavior.py",
    }:
        return "evaluation"
    if script in {"scripts/run_lora_attribution.py", "scripts/run_lora_set_interventions.py"}:
        return "attribution"
    if stage == "prepare" or script == "scripts/materialize_cached_artifact.py":
        return "preparation"
    return "comparison"


def build_tasks(manifest: dict) -> list[Task]:
    pairs = [build_pair(manifest, replicate) for replicate in manifest["replicates"]]
    resources = manifest["condor"]["resources"]
    estimates = manifest["hpc"]["estimated_minutes"]
    raw: dict[str, dict] = {}
    for pair_index, pair in enumerate(pairs):
        for stage in STAGES:
            for index, command in enumerate(pair["stages"][stage]):
                identifier = task_id(pair["seed"], stage, index)
                profile = resource_profile(command, stage)
                resource = resources[profile]
                raw[identifier] = {
                    "task_id": identifier, "pair_index": pair_index, "seed": pair["seed"],
                    "stage": stage, "command_index": index, "command": command,
                    "profile": profile, "gpus": int(resource.get("gpus", 0)),
                    "cpus": int(resource["cpus"]), "memory_mb": int(resource["memory_mb"]),
                    "min_gpu_memory_mb": int(resource.get("min_gpu_memory_mb", 0)),
                    "estimated_minutes": int(estimates[stage]), "parents": [],
                }

    def add_parent(child: str, *parents: str) -> None:
        raw[child]["parents"].extend(parent for parent in parents if parent in raw)

    for pair in pairs:
        seed = pair["seed"]
        # Each condition has its own preprocessing/training/evaluation chain.
        for condition_index, train_stage in enumerate(("train_subliminal", "train_neutral")):
            train = task_id(seed, train_stage, 0)
            prepare = task_id(seed, "prepare", condition_index)
            if train in raw:
                add_parent(train, prepare)
            verify = task_id(seed, "verify_runs", condition_index)
            add_parent(verify, train)
        gate = task_id(seed, "verify_compare", 0)
        add_parent(gate, task_id(seed, "verify_runs", 0), task_id(seed, "verify_runs", 1))
        for stage in ("vectors", "layer_runs"):
            for index in range(2):
                add_parent(task_id(seed, stage, index), gate)
        layer_compare = task_id(seed, "layer_compare", 0)
        add_parent(layer_compare, task_id(seed, "layer_runs", 0), task_id(seed, "layer_runs", 1))
        for index in range(2):
            add_parent(task_id(seed, "module_runs", index), layer_compare)
        module_compare = task_id(seed, "module_compare", 0)
        add_parent(module_compare, task_id(seed, "module_runs", 0), task_id(seed, "module_runs", 1))
        topk = task_id(seed, "topk_prepare", 0)
        add_parent(topk, module_compare)
        for stage in ("activation_runs", "behavior_runs"):
            for index in range(2):
                add_parent(task_id(seed, stage, index), topk)
        add_parent(task_id(seed, "activation_compare", 0), task_id(seed, "activation_runs", 0), task_id(seed, "activation_runs", 1))
        add_parent(task_id(seed, "behavior_compare", 0), task_id(seed, "behavior_runs", 0), task_id(seed, "behavior_runs", 1))

    return [Task(**{**value, "parents": tuple(sorted(set(value["parents"])))}) for value in raw.values()]


def quote(value: str) -> str:
    return value.replace("\\", "/").replace('"', '\\"')


def render_dag(tasks: list[Task], manifest_path: str, manifest: dict, repo_root: str, shared_root: str, image: str) -> str:
    lines = ["# Generated by scripts/generate_condor_dag.py; do not edit manually.", ""]
    retry_count = int(manifest["condor"].get("dag_retries", 2))
    retirement = int(manifest["condor"].get("max_job_retirement_seconds", 7200))
    for task in tasks:
        submit = "condor/task_gpu.sub" if task.gpus else "condor/task_cpu.sub"
        lines.append(f"JOB {task.task_id} {submit}")
        variables = {
            "TaskId": task.task_id, "PairIndex": str(task.pair_index), "Stage": task.stage,
            "CommandIndex": str(task.command_index), "Manifest": manifest_path,
            "RepoRoot": repo_root, "SharedRoot": shared_root, "ContainerImage": image,
            "RequestCpus": str(task.cpus), "RequestMemoryMB": str(task.memory_mb),
            "MinGpuMemoryMB": str(task.min_gpu_memory_mb), "RetirementSeconds": str(retirement),
        }
        lines.append("VARS " + task.task_id + " " + " ".join(f'{key}=\"{quote(value)}\"' for key, value in variables.items()))
        lines.append(f"RETRY {task.task_id} {retry_count} UNLESS-EXIT 2")
        lines.append("")
    for task in tasks:
        if task.parents:
            lines.append(f"PARENT {' '.join(task.parents)} CHILD {task.task_id}")
    final_parents = []
    for seed in sorted({task.seed for task in tasks}):
        final_parents.extend([
            task_id(seed, "vectors", 0), task_id(seed, "vectors", 1),
            task_id(seed, "activation_compare", 0), task_id(seed, "behavior_compare", 0),
        ])
    lines += [
        "", "JOB finalize_confirmatory condor/finalize.sub",
        "VARS finalize_confirmatory " + " ".join(
            f'{key}=\"{quote(value)}\"' for key, value in {
                "TaskId": "finalize_confirmatory", "Manifest": manifest_path,
                "RepoRoot": repo_root, "SharedRoot": shared_root,
                "ContainerImage": image,
                "RequestCpus": str(manifest["condor"]["resources"]["aggregation"]["cpus"]),
                "RequestMemoryMB": str(manifest["condor"]["resources"]["aggregation"]["memory_mb"]),
                "RetirementSeconds": str(retirement),
            }.items()
        ),
        f"PARENT {' '.join(final_parents)} CHILD finalize_confirmatory",
        f"RETRY finalize_confirmatory {retry_count} UNLESS-EXIT 2", "",
    ]
    return "\n".join(lines)


def validate(tasks: list[Task], dag_text: str, output_dir: Path) -> dict:
    ids = {task.task_id for task in tasks}
    if len(tasks) != 62 or len(ids) != 62:
        raise RuntimeError(f"Expected 62 unique confirmatory tasks, found {len(tasks)} / {len(ids)}")
    if any(task.gpus not in {0, 1} for task in tasks):
        raise RuntimeError("Every task must request zero or exactly one GPU")
    if any(task.gpus == 1 and task.min_gpu_memory_mb < 16384 for task in tasks):
        raise RuntimeError("Every GPU task must require at least 16 GiB VRAM")
    for task in tasks:
        unknown = set(task.parents) - ids
        if unknown:
            raise RuntimeError(f"Unknown parents for {task.task_id}: {sorted(unknown)}")
    pending = {task.task_id: set(task.parents) for task in tasks}
    while pending:
        ready = {node for node, parents in pending.items() if not parents}
        if not ready:
            raise RuntimeError("Confirmatory dependency graph contains a cycle")
        pending = {node: parents - ready for node, parents in pending.items() if node not in ready}
    required = [
        "task_cpu.sub", "task_gpu.sub", "finalize.sub", "gpu_smoke.sub",
        "run_confirmatory_task.sh", "finalize_confirmatory.sh", "run_gpu_smoke.sh",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Condor templates: {missing}")
    common_submit = (
        "universe = docker", 'requirements = UidDomain == "cs.uni-saarland.de"',
        "getenv = HOME", "+WantGPUHomeMounted = true", "+WantScratchMounted = true",
        "$(ClusterId)", "$(ProcId)", "should_transfer_files = NO",
        "max_job_retirement_time", "stream_output = true", "stream_error = true",
    )
    submit_text = {
        name: (output_dir / name).read_text(encoding="utf-8")
        for name in ("task_cpu.sub", "task_gpu.sub", "finalize.sub", "gpu_smoke.sub")
    }
    for name, content in submit_text.items():
        absent = [snippet for snippet in common_submit if snippet not in content]
        if absent:
            raise RuntimeError(f"{name} lacks required HTCondor directives: {absent}")
    if "request_GPUs = 0" not in submit_text["task_cpu.sub"] or "request_GPUs = 0" not in submit_text["finalize.sub"]:
        raise RuntimeError("CPU and finalization submit files must request zero GPUs")
    for name in ("task_gpu.sub", "gpu_smoke.sub"):
        content = submit_text[name]
        if "request_GPUs = 1" not in content or "gpus_minimum_memory" not in content or "gpus_minimum_capability" not in content:
            raise RuntimeError(f"{name} must request exactly one GPU with memory/capability minima")
    if dag_text.count("\nJOB seed") != 62:
        raise RuntimeError("Rendered DAG does not contain exactly 62 task JOB nodes")
    return {
        "task_count": len(tasks), "gpu_task_count": sum(task.gpus for task in tasks),
        "cpu_task_count": sum(task.gpus == 0 for task in tasks),
        "training_task_count": sum(task.profile == "training" for task in tasks),
        "finalize_node": True,
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_cross_seed_confirmatory.yaml")
    parser.add_argument("--output-dir", default="condor")
    parser.add_argument("--repo-root", default="$ENV(HOME)/beyond-steering-vectors")
    parser.add_argument("--shared-root", default="/scratch/$(Owner)/beyond-steering-vectors")
    parser.add_argument("--container-image")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = load_yaml(manifest_path)
    output_dir = repo_path(args.output_dir)
    tasks = build_tasks(manifest)
    image = args.container_image or os.getenv("CONDOR_CONTAINER_IMAGE") or manifest["condor"]["container_image"]
    dag = render_dag(tasks, args.manifest, manifest, args.repo_root, args.shared_root, image)
    summary = validate(tasks, dag, output_dir)
    if not args.validate_only:
        atomic_write(output_dir / "confirmatory.dag", dag)
        atomic_write(output_dir / "tasks.json", json.dumps({"schema_version": 1, "tasks": [asdict(task) for task in tasks], "summary": summary}, indent=2) + "\n")
    print(json.dumps({**summary, "dag": str(output_dir / "confirmatory.dag"), "validated": True}, indent=2))


if __name__ == "__main__":
    main()
