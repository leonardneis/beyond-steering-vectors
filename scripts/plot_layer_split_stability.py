"""Reproduce schema-v2 layer-score and split-stability figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.split_stability import (  # noqa: E402
    bootstrap_mean_ci,
    split_readout_stability,
    trait_specific_prompt_scores,
)


SELECTED_LAYERS = {0, 5, 10, 18, 22, 25}
COLORS = {"A": "#6A5ACD", "B": "#E85AAD"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summaries(scores, readout, bootstrap_samples, seed):
    rows = []
    for layer in sorted(scores):
        values = scores[layer][readout]
        if values is None:
            continue
        low, high = bootstrap_mean_ci(values, samples=bootstrap_samples, seed=seed + layer)
        rows.append({"layer": layer, "mean": float(values.mean()), "ci_low": low, "ci_high": high})
    return rows


def _plot_readout(readout, label, rows_a, rows_b, output):
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for split, rows in (("A", rows_a), ("B", rows_b)):
        x = np.asarray([row["layer"] for row in rows])
        y = np.asarray([row["mean"] for row in rows])
        low = np.asarray([row["ci_low"] for row in rows])
        high = np.asarray([row["ci_high"] for row in rows])
        ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.6, color=COLORS[split], label=f"Split {split}")
        ax.fill_between(x, low, high, color=COLORS[split], alpha=0.14)
    for layer in SELECTED_LAYERS:
        ax.axvline(layer, color="#555555", alpha=0.10, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.9, alpha=0.65)
    ax.set(xlabel="Transformer layer", ylabel="Trait-specific projection drop", title=label)
    ax.set_xticks(range(0, 28, 2))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_stability(all_rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (readout, label) in zip(
        axes,
        (("local", "Local"), ("terminal", "Terminal"), ("downstream_mean", "Downstream mean")),
        strict=True,
    ):
        rows_a, rows_b = all_rows[readout]["A"], all_rows[readout]["B"]
        by_a = {row["layer"]: row["mean"] for row in rows_a}
        by_b = {row["layer"]: row["mean"] for row in rows_b}
        layers = sorted(set(by_a) & set(by_b))
        x, y = np.asarray([by_a[layer] for layer in layers]), np.asarray([by_b[layer] for layer in layers])
        limits = (min(x.min(), y.min()), max(x.max(), y.max()))
        padding = max((limits[1] - limits[0]) * 0.08, 1e-3)
        ax.plot([limits[0] - padding, limits[1] + padding], [limits[0] - padding, limits[1] + padding], "--", color="#777777")
        colors = ["#E85AAD" if layer in SELECTED_LAYERS else "#6A5ACD" for layer in layers]
        ax.scatter(x, y, c=colors, s=28, alpha=0.85)
        for layer in layers:
            if layer in SELECTED_LAYERS:
                ax.annotate(str(layer), (by_a[layer], by_b[layer]), xytext=(3, 3), textcoords="offset points", fontsize=8)
        stability = split_readout_stability_from_rows(rows_a, rows_b)
        ax.set_title(f"{label}\nSpearman ρ={stability:.3f}")
        ax.set(xlabel="Split A", ylabel="Split B")
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def split_readout_stability_from_rows(rows_a, rows_b):
    by_a = {row["layer"]: row["mean"] for row in rows_a}
    by_b = {row["layer"]: row["mean"] for row in rows_b}
    layers = sorted(set(by_a) & set(by_b))
    from scipy.stats import spearmanr

    return float(spearmanr([by_a[layer] for layer in layers], [by_b[layer] for layer in layers]).statistic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("a", "b"):
        parser.add_argument(f"--subliminal-{split}", required=True)
        parser.add_argument(f"--neutral-{split}", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    args = parser.parse_args()
    paths = {name: repo_path(value) for name, value in vars(args).items() if name.startswith(("subliminal_", "neutral_"))}
    scores = {
        "A": trait_specific_prompt_scores(_load(paths["subliminal_a"]), _load(paths["neutral_a"])),
        "B": trait_specific_prompt_scores(_load(paths["subliminal_b"]), _load(paths["neutral_b"])),
    }
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    specs = {
        "local": "Local trait-specific drop",
        "terminal": "Terminal trait-specific drop",
        "downstream_mean": "Downstream-mean trait-specific drop",
    }
    for index, (readout, label) in enumerate(specs.items()):
        rows_a = _summaries(scores["A"], readout, args.bootstrap_samples, args.bootstrap_seed + 1000 * index)
        rows_b = _summaries(scores["B"], readout, args.bootstrap_samples, args.bootstrap_seed + 1000 * index + 100)
        all_rows[readout] = {"A": rows_a, "B": rows_b}
        _plot_readout(readout, label, rows_a, rows_b, output_dir / f"{readout}_scores.png")
    _plot_stability(all_rows, output_dir / "split_stability.png")
    summary = {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "selected_layers": sorted(SELECTED_LAYERS),
        "readouts": all_rows,
        "stability": {
            readout: split_readout_stability(scores["A"], scores["B"], readout)
            for readout in ("local", "fixed_target", "terminal", "downstream_mean")
        },
    }
    (output_dir / "split_stability_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote figures and summary to {output_dir}")


if __name__ == "__main__":
    main()
