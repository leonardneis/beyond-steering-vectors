"""Create thesis-ready cross-seed confirmatory summary plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = json.loads(repo_path(args.input).read_text(encoding="utf-8"))
    output = repo_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    labels, activation, behavior = [], [], []
    for row in result["aggregate"]:
        labels.append(f"k={row['k']}\n{row['mode']}")
        activation.append(row["metrics"]["activation_fraction"]["per_seed"])
        behavior.append(row["metrics"]["behavior_fraction"]["per_seed"])
    x = np.arange(len(labels)); width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [np.mean(v) for v in activation], width, label="Activation mediation")
    ax.bar(x + width / 2, [np.mean(v) for v in behavior], width, label="Target-logprob mediation")
    for index, values in enumerate(activation):
        ax.scatter(np.full(len(values), x[index] - width / 2), values, color="black", s=18, zorder=3)
    for index, values in enumerate(behavior):
        ax.scatter(np.full(len(values), x[index] + width / 2), values, color="black", s=18, zorder=3)
    ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(x, labels); ax.set_ylabel("Fraction of full trait-specific effect")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "cross_seed_mediation.png", dpi=200); plt.close(fig)

    similarities = result.get("ranking_similarity", [])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [f"{a['seeds'][0]}–{a['seeds'][1]}" for a in similarities]
    values = [a["spearman"] for a in similarities]
    ax.bar(names, values); ax.axhline(0, color="black", linewidth=.8)
    ax.set_ylim(-1, 1); ax.set_ylabel("Module-ranking Spearman ρ"); ax.set_xlabel("Training-seed pair")
    fig.tight_layout(); fig.savefig(output / "cross_seed_ranking_similarity.png", dpi=200); plt.close(fig)
    print(f"Wrote confirmatory plots to {output}")


if __name__ == "__main__":
    main()
