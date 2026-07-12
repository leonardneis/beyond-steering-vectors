"""Plan or execute the minimal cross-seed confirmatory pipeline from a YAML manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import load_yaml  # noqa: E402


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*parts) -> list[str]:
    return [str(part) for part in parts if part is not None]


def build_pair(manifest: dict, replicate: dict) -> dict:
    seed, root = int(replicate["seed"]), repo_path(manifest["output_root"]) / f"seed_{replicate['seed']}"
    existing = replicate.get("status") == "existing"
    sub_adapter = repo_path(replicate["subliminal_adapter"]) if existing else root / "subliminal" / "student_lora"
    neutral_adapter = repo_path(replicate["neutral_adapter"]) if existing else root / "neutral" / "student_lora"
    sub_train = root / "data" / f"cat_subliminal_10k_seed{seed}.jsonl"
    neutral_train = root / "data" / f"neutral_10k_seed{seed}.jsonl"
    cfg, confirm = manifest["training"], manifest["confirmatory"]
    teacher, prompts = repo_path(manifest["teacher_vector"]), repo_path(manifest["prompts"])
    vector_root, attr = root / "vectors", root / "attribution"
    plan = attr / "topk_plan.json"
    stages = {
        "prepare": [] if existing else [
            command("scripts/subsample_dataset.py", "--input", manifest["conditions"]["subliminal"]["source_dataset"], "--output", sub_train, "--size", cfg["sample_size"], "--seed", seed),
            command("scripts/subsample_dataset.py", "--input", manifest["conditions"]["neutral"]["source_dataset"], "--output", neutral_train, "--size", cfg["sample_size"], "--seed", seed),
        ],
        "train_subliminal": [] if existing else [command("scripts/train_student.py", "--config", manifest["train_config"], "--model-config", manifest["model_config"], "--train-file", sub_train, "--output-dir", sub_adapter, "--seed", seed, "--run-id", f"confirmatory_cat_seed{seed}_subliminal")],
        "train_neutral": [] if existing else [command("scripts/train_student.py", "--config", manifest["train_config"], "--model-config", manifest["model_config"], "--train-file", neutral_train, "--output-dir", neutral_adapter, "--seed", seed, "--run-id", f"confirmatory_cat_seed{seed}_neutral")],
        "verify": [
            command("scripts/evaluate_preference.py", "--model-config", manifest["model_config"], "--adapter-path", sub_adapter, "--target-animal", "cat", "--num-samples", confirm["behavior_samples"], "--generation-mode", "greedy", "--prompt-set", "paper_reference", "--system-prompt-mode", "neutral", "--no-default-system-prompt", "--output-json", root / "verification" / "subliminal.json", "--output-csv", root / "verification" / "subliminal.csv", "--run-id", f"confirmatory_cat_seed{seed}_verify_subliminal"),
            command("scripts/evaluate_preference.py", "--model-config", manifest["model_config"], "--adapter-path", neutral_adapter, "--target-animal", "cat", "--num-samples", confirm["behavior_samples"], "--generation-mode", "greedy", "--prompt-set", "paper_reference", "--system-prompt-mode", "neutral", "--no-default-system-prompt", "--output-json", root / "verification" / "neutral.json", "--output-csv", root / "verification" / "neutral.csv", "--run-id", f"confirmatory_cat_seed{seed}_verify_neutral"),
            command("scripts/compare_full_behavior.py", "--subliminal", root / "verification" / "subliminal.json", "--neutral", root / "verification" / "neutral.json", "--output", root / "verification" / "paired.json", "--bootstrap-samples", confirm["bootstrap_samples"], "--seed", seed, "--require-positive"),
        ],
        "analysis": [
            command("scripts/extract_steering_vectors.py", "--adapter-path", sub_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--trait", "cat", "--n-prompts", confirm["vector_prompts"], "--output-dir", vector_root / "subliminal"),
            command("scripts/extract_steering_vectors.py", "--adapter-path", neutral_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--trait", "cat", "--n-prompts", confirm["vector_prompts"], "--output-dir", vector_root / "neutral"),
            command("scripts/run_lora_attribution.py", "--adapter-path", sub_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--n-prompts", confirm["attribution_prompts"], "--prompt-offset", confirm["attribution_offset"], "--group-by", "layer", "--output", attr / "subliminal_layers.json"),
            command("scripts/run_lora_attribution.py", "--adapter-path", neutral_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--n-prompts", confirm["attribution_prompts"], "--prompt-offset", confirm["attribution_offset"], "--group-by", "layer", "--output", attr / "neutral_layers.json"),
            command("scripts/compare_layer_attribution.py", "--subliminal", attr / "subliminal_layers.json", "--neutral", attr / "neutral_layers.json", "--output", attr / "paired_layers.json", "--bootstrap-samples", confirm["bootstrap_samples"]),
            command("scripts/run_lora_attribution.py", "--adapter-path", sub_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--n-prompts", confirm["attribution_prompts"], "--prompt-offset", confirm["attribution_offset"], "--group-by", "individual", "--include-layers", *confirm["selected_layers"], "--output", attr / "subliminal_modules.json"),
            command("scripts/run_lora_attribution.py", "--adapter-path", neutral_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--n-prompts", confirm["attribution_prompts"], "--prompt-offset", confirm["attribution_offset"], "--group-by", "individual", "--include-layers", *confirm["selected_layers"], "--output", attr / "neutral_modules.json"),
            command("scripts/compare_layer_attribution.py", "--subliminal", attr / "subliminal_modules.json", "--neutral", attr / "neutral_modules.json", "--output", attr / "paired_modules.json", "--bootstrap-samples", confirm["bootstrap_samples"]),
            command("scripts/prepare_topk_module_sets.py", "--ranking", attr / "paired_modules.json", "--adapter-dir", sub_adapter, "--k", *confirm["top_k"], "--seed", confirm["control_seed"], "--matching-pool-size", 3, "--control-types", *confirm["control_types"], "--output", plan),
            command("scripts/run_lora_set_interventions.py", "--adapter-path", sub_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--selection-plan", plan, "--n-prompts", confirm["intervention_prompts"], "--prompt-offset", confirm["intervention_offset"], "--set-names", "top_k", "norm_matched_control", "--output", attr / "subliminal_topk.json"),
            command("scripts/run_lora_set_interventions.py", "--adapter-path", neutral_adapter, "--teacher-vector", teacher, "--prompts", prompts, "--selection-plan", plan, "--n-prompts", confirm["intervention_prompts"], "--prompt-offset", confirm["intervention_offset"], "--set-names", "top_k", "norm_matched_control", "--output", attr / "neutral_topk.json"),
            command("scripts/compare_lora_set_interventions.py", "--subliminal", attr / "subliminal_topk.json", "--neutral", attr / "neutral_topk.json", "--output", attr / "paired_topk.json", "--bootstrap-samples", confirm["bootstrap_samples"]),
            command("scripts/run_lora_set_behavior.py", "--adapter-path", sub_adapter, "--selection-plan", plan, "--target-animal", "cat", "--num-samples", confirm["behavior_samples"], "--prompt-set", "paper_reference", "--set-names", "top_k", "norm_matched_control", "--k", *confirm["top_k"], "--output", attr / "subliminal_behavior.json"),
            command("scripts/run_lora_set_behavior.py", "--adapter-path", neutral_adapter, "--selection-plan", plan, "--target-animal", "cat", "--num-samples", confirm["behavior_samples"], "--prompt-set", "paper_reference", "--set-names", "top_k", "norm_matched_control", "--k", *confirm["top_k"], "--output", attr / "neutral_behavior.json"),
            command("scripts/compare_lora_set_behavior.py", "--subliminal", attr / "subliminal_behavior.json", "--neutral", attr / "neutral_behavior.json", "--output", attr / "paired_behavior.json", "--bootstrap-samples", confirm["bootstrap_samples"]),
        ],
    }
    return {"seed": seed, "root": str(root), "existing": existing, "adapters": {"subliminal": str(sub_adapter), "neutral": str(neutral_adapter)}, "stages": stages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pair-index", type=int)
    parser.add_argument("--stage", choices=("prepare", "train_subliminal", "train_neutral", "verify", "analysis", "all"), default="all")
    parser.add_argument("--emit-plan", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = load_yaml(manifest_path)
    pairs = [build_pair(manifest, item) for item in manifest["replicates"]]
    if args.pair_index is not None:
        pairs = [pairs[args.pair_index]]
    plan = {"schema_version": 1, "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "experiment_id": manifest["experiment_id"], "pairs": pairs}
    output = repo_path(args.emit_plan)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite plan without --resume: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote confirmatory plan to {output}")
    stages = ("prepare", "train_subliminal", "train_neutral", "verify", "analysis") if args.stage == "all" else (args.stage,)
    for pair in pairs:
        for stage in stages:
            marker_dir = Path(pair["root"]) / "orchestration"
            for command_index, parts in enumerate(pair["stages"][stage]):
                print("COMMAND:", subprocess.list2cmdline([sys.executable, *parts]))
                if args.execute:
                    marker = marker_dir / f"{stage}_{command_index:02d}.complete.json"
                    if marker.exists():
                        if args.resume:
                            print(f"SKIP completed command marker: {marker}")
                            continue
                        raise FileExistsError(f"Command already completed; use --resume: {marker}")
                    expected = None
                    for option in ("--output", "--output-json", "--output-dir"):
                        if option in parts:
                            expected = Path(parts[parts.index(option) + 1])
                            break
                    run_parts = list(parts)
                    is_training = parts and parts[0] == "scripts/train_student.py"
                    if expected is not None and expected.exists():
                        if is_training and args.resume:
                            run_parts.append("--resume")
                        else:
                            raise FileExistsError(
                                f"Unmarked output exists; refusing silent overwrite: {expected}"
                            )
                    subprocess.run([sys.executable, *run_parts], check=True, cwd=repo_path("."))
                    marker_dir.mkdir(parents=True, exist_ok=True)
                    marker.write_text(
                        json.dumps(
                            {
                                "command": run_parts,
                                "expected_output": str(expected) if expected else None,
                                "expected_sha256": sha256(expected) if expected and expected.is_file() else None,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
    if args.execute:
        artifacts = []
        for pair in pairs:
            root = Path(pair["root"])
            artifacts.extend({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in root.rglob("*") if path.is_file())
        (output.parent / "artifact_manifest.json").write_text(json.dumps({"artifacts": artifacts}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
