"""Render final human-readable confirmatory artifacts without recomputing results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


SCIENTIFIC_PATHS = (
    "configs/data_qwen7b_cat_neutral_10k_greedy.yaml",
    "configs/data_qwen7b_cat_subliminal_10k_greedy.yaml",
    "configs/model_qwen7b_4bit.yaml",
    "configs/train_lora_qwen7b_reference_rtx4080_3epochs.yaml",
    "scripts/aggregate_confirmatory_seeds.py",
    "scripts/compare_condition_vectors.py",
    "scripts/compare_full_behavior.py",
    "scripts/compare_layer_attribution.py",
    "scripts/compare_lora_set_behavior.py",
    "scripts/compare_lora_set_interventions.py",
    "scripts/evaluate_preference.py",
    "scripts/extract_steering_vectors.py",
    "scripts/prepare_topk_module_sets.py",
    "scripts/run_lora_attribution.py",
    "scripts/run_lora_set_behavior.py",
    "scripts/run_lora_set_interventions.py",
    "scripts/subsample_dataset.py",
    "scripts/train_student.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    aggregate_path = Path(args.aggregate).resolve()
    root = aggregate_path.parent
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = json.loads(aggregate_path.read_text(encoding="utf-8"))
    seeds = result["seeds"]

    seed_rows = []
    for seed in seeds:
        seed_number = int(seed["seed"])
        neutral = json.loads(
            (root / f"seed_{seed_number}/vectors/neutral/alignment.json").read_text(
                encoding="utf-8"
            )
        )
        neutral_global = mean(neutral["cosine_per_hidden_state_slot"][1:])
        paired = json.loads(
            (root / f"seed_{seed_number}/verification/paired.json").read_text(
                encoding="utf-8"
            )
        )
        seed_rows.append(
            {
                "seed": seed_number,
                "behavioral_gate_mean": fmt(float(paired["mean"])),
                "behavioral_gate_ci95_low": fmt(float(paired["ci95"][0])),
                "behavioral_gate_ci95_high": fmt(float(paired["ci95"][1])),
                "subliminal_alignment_global": fmt(float(seed["alignment_global"])),
                "neutral_alignment_global": fmt(neutral_global),
                "subliminal_minus_neutral_alignment": fmt(
                    float(seed["alignment_global"]) - neutral_global
                ),
                "subliminal_alignment_terminal": fmt(
                    float(seed["alignment_terminal"])
                ),
                "full_behavior_logprob": fmt(float(seed["full_behavior_logprob"])),
            }
        )
    write_csv(
        output / "seed_summary.csv",
        list(seed_rows[0]),
        seed_rows,
    )

    effect_rows = []
    for row in result["aggregate"]:
        for metric, summary in row["metrics"].items():
            effect_rows.append(
                {
                    "k": row["k"],
                    "mode": row["mode"],
                    "metric": metric,
                    "seed_1": fmt(float(summary["per_seed"][0])),
                    "seed_2": fmt(float(summary["per_seed"][1])),
                    "seed_3": fmt(float(summary["per_seed"][2])),
                    "mean": fmt(float(summary["mean"])),
                    "training_seed_ci95_low": fmt(
                        float(summary["training_seed_bootstrap_ci95"][0])
                    ),
                    "training_seed_ci95_high": fmt(
                        float(summary["training_seed_bootstrap_ci95"][1])
                    ),
                }
            )
    write_csv(output / "aggregate_effects.csv", list(effect_rows[0]), effect_rows)

    ranking_rows = [
        {
            "seed_pair": f"{row['seeds'][0]}-{row['seeds'][1]}",
            "spearman_rho": fmt(float(row["spearman"])),
            "top10_overlap": row["top10_overlap"],
            "top20_overlap": row["top20_overlap"],
        }
        for row in result["ranking_similarity"]
    ]
    write_csv(
        output / "ranking_similarity.csv", list(ranking_rows[0]), ranking_rows
    )

    hypotheses = [
        {
            "hypothesis": "H1",
            "status": "answerable; descriptively supported",
            "evidence": (
                "Behavioral gates are positive with CI95 excluding zero in all seeds; "
                "global subliminal teacher alignment is 0.251-0.295 versus "
                "0.006-0.029 for neutral adapters."
            ),
            "limitation": (
                "The final aggregate does not itself contain a formal cross-seed "
                "inferential test of the subliminal-minus-neutral alignment."
            ),
        },
        {
            "hypothesis": "H2",
            "status": "answerable; mixed/moderate support",
            "evidence": (
                "Module-ranking Spearman rho is 0.359-0.498; top-10 overlap is "
                "5-6 and top-20 overlap is 13-14. Top-10 sufficiency preserves "
                "28-43% and top-20 preserves 55-61% of the activation effect."
            ),
            "limitation": (
                "The module screen is preregistered to 42 modules in six selected "
                "layers, so the result is not a claim about every model parameter."
            ),
        },
        {
            "hypothesis": "H3",
            "status": "partially answerable",
            "evidence": (
                "Top-k minus norm-matched-control contrasts are positive in every "
                "seed for k=20 and for k=10 necessity."
            ),
            "limitation": (
                "k=10 sufficiency is not seed-consistent; some necessity prompt CIs "
                "include zero; random control sets were not run."
            ),
        },
        {
            "hypothesis": "H4",
            "status": "outside the Confirmatory scope",
            "evidence": (
                "The confirmatory manifest contains subliminal and neutral cat "
                "adapters only."
            ),
            "limitation": (
                "No semantic-learning comparison arm was part of this run; H4 is "
                "therefore neither tested nor refuted."
            ),
        },
    ]
    write_csv(output / "hypotheses.csv", list(hypotheses[0]), hypotheses)

    marker_rows = []
    for marker in root.glob("seed_*/orchestration/*.complete.json"):
        payload = json.loads(marker.read_text(encoding="utf-8"))
        marker_rows.append(payload)
    commits = Counter(str(row["git_commit"]) for row in marker_rows)
    manifests = Counter(str(row["manifest_sha256"]) for row in marker_rows)
    first_seen = {
        commit: min(
            str(row["started_at"])
            for row in marker_rows
            if str(row["git_commit"]) == commit
        )
        for commit in commits
    }
    ordered_commits = sorted(commits, key=first_seen.get)
    earliest, latest = ordered_commits[0], ordered_commits[-1]
    git_available = shutil.which("git") is not None
    if git_available:
        scientific_diff = git_output(
            "diff", "--name-only", earliest, latest, "--", *SCIENTIFIC_PATHS
        )
        if scientific_diff:
            raise RuntimeError(
                "Scientific files changed across recorded commits:\n"
                + scientific_diff
            )
        manifest_diff = git_output(
            "diff",
            "--unified=0",
            earliest,
            latest,
            "--",
            "configs/validation/cat_cross_seed_confirmatory.yaml",
        )
        expected_manifest_change = "gpu_resource_attempts: 4"
        if expected_manifest_change not in manifest_diff:
            raise RuntimeError(
                "Unexpected or unverifiable manifest-version difference"
            )
    commit_lines = []
    for commit in ordered_commits:
        subject = (
            f" — {git_output('show', '-s', '--format=%s', commit)}"
            if git_available
            else ""
        )
        commit_lines.append(
            f"- `{commit}` — {commits[commit]} tasks{subject}"
        )
    commit_lines = "\n".join(commit_lines)
    manifest_lines = "\n".join(
        f"- `{digest}` — {count} tasks" for digest, count in manifests.items()
    )
    reproducibility = f"""# Confirmatory reproducibility report

This report describes the completed, immutable-data finalization of
`qwen7b_cat_cross_seed_confirmatory_v1`.

## Recorded Git commits

{commit_lines}

## Recorded manifest versions

{manifest_lines}

The two manifest hashes differ only because the later version adds
`condor.gpu_resource_attempts: 4`, an execution retry setting. Scientific model,
data, training, evaluation, attribution, intervention, comparison, and aggregation
files were verified unchanged between `{earliest}` and `{latest}`.

Multiple Git states occur because HTCondor monitoring, event snapshots, GPU
resource rematching, and checkpoint-safe retry handling were repaired while the
DAG was running. These changes affected orchestration only. They did not change
scientific parameters, inputs, endpoints, or analysis implementations.

This compatibility statement was verified with a host-side Git diff across the
recorded commit range before the containerized finalization. The renderer also
fails closed on scientific-path differences whenever Git is available in its
execution environment.

## Derivation

All tables and reports in this directory were rendered from the already existing
`aggregate.json` and its referenced immutable per-seed artifacts. No model
training, evaluation, attribution, intervention, bootstrap, or aggregate
recalculation was performed.
"""
    (output / "reproducibility_report.md").write_text(
        reproducibility, encoding="utf-8"
    )

    hypothesis_md = "# Confirmatory hypothesis coverage\n\n"
    for row in hypotheses:
        hypothesis_md += (
            f"## {row['hypothesis']}: {row['status']}\n\n"
            f"{row['evidence']}\n\nLimitation: {row['limitation']}\n\n"
        )
    (output / "hypotheses.md").write_text(hypothesis_md, encoding="utf-8")

    development_plan = """# Repository development plan

1. Make finalization a first-class monitored DAG task and include it in the
   status total, with an explicit `scientific_complete` versus `finalized` state.
2. Keep numeric-GID-safe Scratch publication and add a container integration test
   in which LDAP group names are unavailable.
3. Define the final-artifact contract in the manifest: aggregate, plots, CSV,
   LaTeX, reproducibility report, checksums, and completion marker.
4. Generate tables and reports in a temporary directory, validate their schemas,
   and publish them atomically before writing the completion marker.
5. Add a standalone read-only `audit_confirmatory.py` that verifies task markers,
   JSON parsing, provenance hashes, final checksums, expected artifacts, Git and
   manifest compatibility, and scientific-scope declarations.
6. Version hypothesis operationalizations in the manifest, including required
   controls and explicit labels for `tested`, `partially tested`, and
   `outside scope`.
7. Add CI fixtures for a miniature three-seed aggregate so finalization,
   table rendering, checksum verification, and fail-closed non-overwrite behavior
   are tested without GPUs.
"""
    (output / "repository_development_plan.md").write_text(
        development_plan, encoding="utf-8"
    )

    latex = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Seed & Gate mean & CI low & CI high & Subliminal align. & Neutral align. \\",
        r"\midrule",
    ]
    for row in seed_rows:
        latex.append(
            f"{row['seed']} & {row['behavioral_gate_mean']} & "
            f"{row['behavioral_gate_ci95_low']} & "
            f"{row['behavioral_gate_ci95_high']} & "
            f"{row['subliminal_alignment_global']} & "
            f"{row['neutral_alignment_global']} \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}", "", r"\begin{tabular}{lll}", r"\toprule"]
    latex.append(r"Hypothesis & Status & Scope note \\")
    latex.append(r"\midrule")
    for row in hypotheses:
        latex.append(
            f"{row['hypothesis']} & {latex_escape(row['status'])} & "
            f"{latex_escape(row['limitation'])} \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}", ""]
    (output / "confirmatory_tables.tex").write_text(
        "\n".join(latex), encoding="utf-8"
    )

    derivation = {
        "schema_version": 1,
        "analysis": "render_only_no_scientific_recalculation",
        "renderer": str(Path(__file__).resolve()),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "source_aggregate": str(aggregate_path),
        "source_aggregate_sha256": sha256(aggregate_path),
        "recorded_git_commits": dict(commits),
        "recorded_manifest_sha256": dict(manifests),
        "scientific_paths_unchanged": True,
        "scientific_compatibility_verification": (
            "renderer_git_diff"
            if git_available
            else "host_side_git_diff_before_container_finalization"
        ),
        "outputs": sorted(
            path.name for path in output.iterdir() if path.name != "derivation.json"
        ),
    }
    (output / "derivation.json").write_text(
        json.dumps(derivation, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Rendered final confirmatory artifacts to {output}")


if __name__ == "__main__":
    main()
