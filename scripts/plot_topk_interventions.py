"""Summarize and plot paired top-k necessity/sufficiency interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.split_stability import bootstrap_mean_ci  # noqa: E402
from slgeo.io import ensure_parent  # noqa: E402


CONTROLS = ("random_control", "norm_matched_control", "layer_norm_matched_control")
COLORS = {
    "top_k": "#6A5ACD",
    "random_control": "#999999",
    "norm_matched_control": "#E69F00",
    "layer_norm_matched_control": "#009E73",
}


def _load(path: str) -> dict:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _indexed(rows: list[dict]) -> dict[tuple[int, str, str], dict]:
    return {(row["k"], row["set_name"], row["mode"]): row for row in rows}


def _raw_index(run: dict) -> dict[tuple[int, str, str], np.ndarray]:
    return {
        (row["k"], row["set_name"], row["mode"]): np.asarray(row["effect_projection_per_prompt"])
        for row in run["interventions"]
    }


def _plot_readout(rows: list[dict], field: str, ylabel: str, output: Path) -> None:
    indexed = _indexed(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, mode in zip(axes, ("necessity", "sufficiency"), strict=True):
        keys = sorted({key[0] for key in indexed if key[2] == mode})
        for set_name, color in COLORS.items():
            values = [indexed[k, set_name, mode][field] for k in keys]
            ax.plot(keys, values, marker="o", linewidth=2, color=color, label=set_name.replace("_", " "))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=mode.title(), xlabel="k", xticks=keys)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_contrasts(contrasts: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    markers = {"random_control": "o", "norm_matched_control": "s", "layer_norm_matched_control": "^"}
    for ax, mode in zip(axes, ("necessity", "sufficiency"), strict=True):
        for control in CONTROLS:
            rows = [row for row in contrasts if row["mode"] == mode and row["control"] == control]
            x = np.asarray([row["k"] for row in rows])
            y = np.asarray([row["global_mean"] for row in rows])
            lo = np.asarray([row["global_ci95"][0] for row in rows])
            hi = np.asarray([row["global_ci95"][1] for row in rows])
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((y - lo, hi - y)),
                marker=markers[control],
                capsize=3,
                linewidth=1.7,
                label=control.replace("_", " "),
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=mode.title(), xlabel="k", xticks=sorted(set(x)))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Top-k minus control: trait-specific global effect")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_slot_trajectories(sub: dict, neutral: dict, output: Path) -> None:
    sr, nr = _raw_index(sub), _raw_index(neutral)
    top_keys = sorted(key for key in sr if key[1] == "top_k")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    for ax, key in zip(axes.flat, top_keys, strict=True):
        values = (sr[key] - nr[key]).mean(axis=0)
        ax.plot(range(len(values)), values, color=COLORS["top_k"], linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=f"k={key[0]} {key[2]}", xlabel="Hidden-state slot", ylabel="Trait-specific effect")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    args = parser.parse_args()

    paired, sub, neutral = _load(args.paired), _load(args.subliminal), _load(args.neutral)
    sr, nr = _raw_index(sub), _raw_index(neutral)
    contrasts = []
    index = 0
    for k, _, mode in sorted(key for key in sr if key[1] == "top_k"):
        top = sr[k, "top_k", mode] - nr[k, "top_k", mode]
        for control in CONTROLS:
            candidate = sr[k, control, mode] - nr[k, control, mode]
            delta = (top[:, 1:] - candidate[:, 1:]).mean(axis=1)
            index += 1
            low, high = bootstrap_mean_ci(
                delta, samples=args.bootstrap_samples, seed=args.bootstrap_seed + index
            )
            contrasts.append(
                {
                    "k": k,
                    "mode": mode,
                    "control": control,
                    "global_mean": float(delta.mean()),
                    "global_median": float(np.median(delta)),
                    "global_ci95": [low, high],
                    "terminal_mean": float((top[:, -1] - candidate[:, -1]).mean()),
                }
            )

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "paired_source": str(repo_path(args.paired)),
        "subliminal_source": str(repo_path(args.subliminal)),
        "neutral_source": str(repo_path(args.neutral)),
        "bootstrap_samples": args.bootstrap_samples,
        "baseline_trait_specific_global_mean": paired["baseline_trait_specific_global_mean"],
        "rows": paired["rows"],
        "top_minus_control_contrasts": contrasts,
    }
    ensure_parent(output_dir / "topk_intervention_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_readout(
        paired["rows"],
        "trait_specific_global_effect_mean",
        "Trait-specific global effect",
        output_dir / "topk_global_effects.png",
    )
    _plot_readout(
        paired["rows"],
        "trait_specific_terminal_effect_mean",
        "Trait-specific terminal effect",
        output_dir / "topk_terminal_effects.png",
    )
    _plot_contrasts(contrasts, output_dir / "topk_control_contrasts.png")
    _plot_slot_trajectories(sub, neutral, output_dir / "topk_slot_trajectories.png")
    print(f"Wrote top-k summary and figures to {output_dir}")


if __name__ == "__main__":
    main()
