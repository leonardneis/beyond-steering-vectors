"""Plot replicated top-k validation and behavioral intervention results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()


COLORS = {"top_k": "#6A5ACD", "random_control": "#999999", "norm_matched_control": "#E69F00"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--paired-split-a", required=True)
    parser.add_argument("--paired-split-b", required=True)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    aggregate = json.loads(repo_path(args.aggregate).read_text(encoding="utf-8"))
    split_a = json.loads(repo_path(args.paired_split_a).read_text(encoding="utf-8"))
    split_b = json.loads(repo_path(args.paired_split_b).read_text(encoding="utf-8"))
    behavior = json.loads(repo_path(args.behavior).read_text(encoding="utf-8"))
    output = repo_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = aggregate["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, mode in zip(axes, ("necessity", "sufficiency"), strict=True):
        for set_name, color in COLORS.items():
            selected = [r for r in rows if r["mode"] == mode and r["set_name"] == set_name]
            x = np.asarray([r["k"] for r in selected])
            y = np.asarray([r["global_mean_across_replicates"] for r in selected])
            low = np.asarray([r["global_range"][0] for r in selected])
            high = np.asarray([r["global_range"][1] for r in selected])
            ax.plot(x, y, marker="o", linewidth=2, color=color, label=set_name.replace("_", " "))
            ax.fill_between(x, low, high, color=color, alpha=0.12)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=mode.title(), xlabel="k", xticks=(1, 3, 5, 10, 15, 20))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Trait-specific global effect")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "replicated_global_effects.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for split, data, marker in (("4096", split_a, "o"), ("4352", split_b, "s")):
        top = [r for r in data["rows"] if r["set_name"] == "top_k"]
        for ax, mode in zip(axes, ("necessity", "sufficiency"), strict=True):
            chosen = [r for r in top if r["mode"] == mode]
            ax.plot(
                [r["k"] for r in chosen],
                [r["trait_specific_terminal_effect_mean"] for r in chosen],
                marker=marker,
                linewidth=2,
                label=f"offset {split}",
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(title=mode.title(), xlabel="k", xticks=(1, 3, 5, 10, 15, 20))
            ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Trait-specific terminal effect")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "topk_terminal_split_replication.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels, values = [], []
    for row in behavior["rows"]:
        labels.append(f"k={row['k']} {row['set_name'].replace('_', ' ')} {row['mode']}")
        values.append(row["trait_specific_behavioral_effect"])
    y = np.arange(len(labels))
    ax.barh(y, values, color=[COLORS.get(r["set_name"], "#777777") for r in behavior["rows"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(yticks=y, yticklabels=labels, xlabel="Trait-specific target-choice effect")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "behavioral_choice_effects.png", dpi=180)
    plt.close(fig)
    print(f"Wrote validation figures to {output}")


if __name__ == "__main__":
    main()
