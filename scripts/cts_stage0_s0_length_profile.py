"""Generate or check the S0 length profile of CTS Stage 0 v2 (tokenizer only; run before TV-v2).

For every GPU evaluation of an S0 prompt in the scientific plan (baseline, score and re-score shards, the terms of
the projection) the prompt is rendered with the rendering the execution code uses (the condition's persona for
persona conditions, P_default otherwise) and its token length is counted. The profile stores only counts per
length for each (cost class, prompt set, context); no prompt text, prompt id or persona id.

No model weights are opened and no forward runs: only the tokenizer files of the pinned snapshot are read (their
hashes and the transformers/tokenizers versions must equal the execution manifest). Default mode regenerates and
compares with the committed profile; ``--write`` writes it.

    python scripts/cts_stage0_s0_length_profile.py [--write] [--hf-home PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()
MANIFEST = "configs/validation/cts_stage0_v2.yaml"
DATE = "2026-09-26"
RULE = (
    "every GPU evaluation of an S0 prompt in the scientific plan (baseline shards: S0_all in L2; score shards: "
    "conditions x prompt set of their cost class; re-score shards: flagged conditions x prompt set in L1), rendered "
    "with the execution rendering (persona conditions: the condition's persona; all other conditions: P_default); "
    "counts of rendered prompt lengths per (cost class, prompt set, context), context = persona iff the rendering "
    "persona is not P_default"
)


def _snapshot(hf_home: Path, revision: str) -> Path:
    hub = hf_home / "hub" if (hf_home / "hub").is_dir() else hf_home
    directory = hub / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / revision
    if not directory.is_dir():
        sys.exit(f"Pinned snapshot {revision} not found under {hub}")
    return directory


def build(manifest: dict) -> dict:
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    from slgeo.cts_stage0.contract import V2Contract
    from slgeo.cts_stage0.package import DEFAULT_PERSONA_ID, FrozenPackage, sha256_path
    from slgeo.cts_stage0.plan import build_plan, condition_objects
    from slgeo.cts_stage0.conditions import PERSONA
    from slgeo.cts_stage0.render import Renderer
    from slgeo.cts_stage0.s0_lengths import KIND, SCHEMA_VERSION, TOKENIZER_FILES, planned_evaluations

    versions = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
    for name, version in versions.items():
        if version != manifest["execution"]["packages"][name]:
            sys.exit(f"{name} {version} differs from the execution manifest ({manifest['execution']['packages'][name]})")
    snapshot = _snapshot(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")), manifest["model"]["revision"])
    tokenizer_hashes = {name: sha256_path(snapshot / name) for name in TOKENIZER_FILES}
    for name, digest in tokenizer_hashes.items():
        if digest != manifest["model"]["snapshot_sha256"][name]:
            sys.exit(f"Tokenizer file {name} differs from the pinned snapshot hash")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)

    package = FrozenPackage.from_repo(ROOT)
    if sha256_path(package.root / "MANIFEST.json") != manifest["frozen_package"]["manifest_sha256"]:
        sys.exit("Frozen v1 package manifest differs from the execution manifest")
    contract = V2Contract.from_repo(ROOT, package, {key: manifest["contract"][key] for key in ("spec_sha256", "registry_sha256")})
    contract.verify_regeneration()
    plan = build_plan(contract, manifest)
    conditions = condition_objects(plan)
    renderer = Renderer(tokenizer, package)
    prompts = package.s0_prompts()
    prompt_sets = {"S0_all": prompts, "S0_animal": tuple(p for p in prompts if p.is_animal_family)}
    lengths: dict[tuple[str, str], list[int]] = {}

    def rendered_lengths(prompt_set: str, persona: str) -> list[int]:
        key = (prompt_set, persona)
        if key not in lengths:
            lengths[key] = [renderer.render(persona, p.prompt).prompt_len for p in prompt_sets[prompt_set]]
        return lengths[key]

    groups: dict[tuple[str, str, str], dict] = {}

    def add(cost_class: str, prompt_set: str, persona: str) -> None:
        context = "default" if persona == DEFAULT_PERSONA_ID else "persona"
        group = groups.setdefault((cost_class, prompt_set, context), {"conditions": 0, "counts": {}})
        group["conditions"] += 1
        for length in rendered_lengths(prompt_set, persona):
            group["counts"][length] = group["counts"].get(length, 0) + 1

    def persona_of(cid: str) -> str:
        condition = conditions[cid]
        return condition.persona if condition.kind == PERSONA else DEFAULT_PERSONA_ID

    for shard in plan["shards"]:
        stage, payload = shard["stage"], shard["payload"]
        if stage == "baseline":
            add("L2_shared_prefix", "S0_all", DEFAULT_PERSONA_ID)
        elif stage == "score":
            for cid in payload["conditions"]:
                add(payload["cost_class"], payload["prompt_set"], persona_of(cid))
        elif stage == "rescore":
            for cid in payload["conditions"]:
                add("L1_reference", payload["prompt_set"], persona_of(cid))
    profile = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "date": DATE,
        "rule": RULE,
        "provenance": {
            "spec_sha256": contract.spec_sha256,
            "registry_sha256": contract.registry_sha256,
            "frozen_package_manifest_sha256": manifest["frozen_package"]["manifest_sha256"],
            "model_revision": manifest["model"]["revision"],
            "tokenizer_files_sha256": tokenizer_hashes,
            "packages": versions,
        },
        "groups": [
            {"cost_class": cost_class, "prompt_set": prompt_set, "context": context, "conditions": group["conditions"],
             "evaluations_by_length": [[length, group["counts"][length]] for length in sorted(group["counts"])]}
            for (cost_class, prompt_set, context), group in sorted(groups.items())
        ],
    }
    observed: dict[tuple[str, str], int] = {}
    for group in profile["groups"]:
        key = (group["cost_class"], group["prompt_set"])
        observed[key] = observed.get(key, 0) + sum(c for _, c in group["evaluations_by_length"])
    if observed != planned_evaluations(plan):
        sys.exit("Profile workload differs from the plan")
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write the profile (default: regenerate and compare)")
    args = parser.parse_args()
    from slgeo.cts_stage0.package import sha256_path
    from slgeo.cts_stage0.s0_lengths import PROFILE_PATH
    from slgeo.io import load_yaml

    manifest = load_yaml(ROOT / MANIFEST)
    text = json.dumps(build(manifest), indent=1) + "\n"
    path = ROOT / PROFILE_PATH
    if args.write:
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"S0 length profile written: {PROFILE_PATH} sha256 {sha256_path(path)}")
        return 0
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        print("S0 length profile differs from its regeneration", file=sys.stderr)
        return 1
    print(f"S0 length profile regenerates identically: sha256 {sha256_path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
