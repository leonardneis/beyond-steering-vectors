"""Aggregate and plot schema-v2 paired behavioral LoRA set interventions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import ensure_parent  # noqa: E402
from slgeo.analysis.split_stability import bootstrap_mean_ci  # noqa: E402


READOUTS = ("target_logprob", "target_probability", "target_vs_lion_margin", "target_choice")
COLORS = {"top_k": "#6A5ACD", "norm_matched_control": "#E69F00", "random_control": "#999999"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    sources = []
    paired_runs = []
    for source in args.paired:
        path = repo_path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 2:
            raise ValueError(f"Expected behavioral schema v2: {path}")
        sources.append(str(path))
        paired_runs.append(data)
        for row in data["rows"]:
            for readout in READOUTS:
                grouped[row["k"], row["set_name"], row["mode"], readout].append(
                    row["readouts"][readout]["trait_specific_effect_mean"]
                )
    summary_rows = []
    for key, values in sorted(grouped.items()):
        array = np.asarray(values)
        summary_rows.append(
            {
                "k": key[0],
                "set_name": key[1],
                "mode": key[2],
                "readout": key[3],
                "replicates": len(values),
                "mean": float(array.mean()),
                "sd": float(array.std(ddof=1)) if len(array) > 1 else None,
                "range": [float(array.min()), float(array.max())],
            }
        )
    primary = paired_runs[0]
    sub = json.loads(Path(primary["subliminal_source"]).read_text(encoding="utf-8"))
    neutral = json.loads(Path(primary["neutral_source"]).read_text(encoding="utf-8"))

    def prompt_values(run: dict, label: str, readout: str) -> np.ndarray:
        evaluation = run[label]
        if readout == "target_choice":
            return np.asarray(evaluation["target_choice_per_prompt"], dtype=float)
        return np.asarray([row[readout] for row in evaluation["token_metrics"]["rows"]], dtype=float)

    full_adapter_effects = {}
    for index, readout in enumerate(READOUTS):
        values = (
            prompt_values(sub, "full_evaluation", readout)
            - prompt_values(sub, "base_evaluation", readout)
            - prompt_values(neutral, "full_evaluation", readout)
            + prompt_values(neutral, "base_evaluation", readout)
        )
        low, high = bootstrap_mean_ci(values, samples=10000, seed=42000 + index)
        full_adapter_effects[readout] = {"mean": float(values.mean()), "ci95": [low, high]}

    top_rows = {
        (row["k"], row["mode"]): row
        for row in primary["rows"]
        if row["set_name"] == "top_k"
    }
    top_minus_control = []
    for run_index, run in enumerate(paired_runs):
        for row in run["rows"]:
            if row["set_name"] != "norm_matched_control":
                continue
            top = top_rows[row["k"], row["mode"]]
            for readout in READOUTS:
                top_values = np.asarray(top["readouts"][readout]["trait_specific_effect_per_prompt"])
                control_values = np.asarray(row["readouts"][readout]["trait_specific_effect_per_prompt"])
                delta = top_values - control_values
                low, high = bootstrap_mean_ci(
                    delta,
                    samples=5000,
                    seed=43000 + run_index * 100 + row["k"] + READOUTS.index(readout),
                )
                top_minus_control.append(
                    {
                        "control_source": sources[run_index],
                        "k": row["k"],
                        "mode": row["mode"],
                        "readout": readout,
                        "mean": float(delta.mean()),
                        "ci95": [low, high],
                    }
                )

    ensure_parent(output_dir / "behavioral_validation_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "sources": sources,
                "full_adapter_trait_specific_effects": full_adapter_effects,
                "rows": summary_rows,
                "top_minus_norm_control": top_minus_control,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    labels = {
        "target_logprob": "Trait-specific target log-probability effect",
        "target_probability": "Trait-specific target probability effect",
        "target_vs_lion_margin": "Trait-specific target-vs-lion margin effect",
        "target_choice": "Trait-specific target-choice effect",
    }
    for readout in READOUTS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for ax, mode in zip(axes, ("necessity", "sufficiency"), strict=True):
            for set_name, color in COLORS.items():
                rows = [
                    row
                    for row in summary_rows
                    if row["readout"] == readout and row["mode"] == mode and row["set_name"] == set_name
                ]
                if not rows:
                    continue
                x = np.asarray([row["k"] for row in rows])
                y = np.asarray([row["mean"] for row in rows])
                low = np.asarray([row["range"][0] for row in rows])
                high = np.asarray([row["range"][1] for row in rows])
                ax.plot(x, y, marker="o", linewidth=2, color=color, label=set_name.replace("_", " "))
                ax.fill_between(x, low, high, color=color, alpha=0.13)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(title=mode.title(), xlabel="k")
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel(labels[readout])
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"behavioral_{readout}.png", dpi=180)
        plt.close(fig)
    print(f"Wrote behavioral schema-v2 summary and figures to {output_dir}")


if __name__ == "__main__":
    main()
