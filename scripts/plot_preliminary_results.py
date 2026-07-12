"""Create preliminary-result figures for the thesis presentation.

The script reads local run artifacts, prefers the final Qwen2.5-7B cat
reference comparison, and writes PNG/PDF figures under
artifacts/figures/preliminary_results.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "artifacts" / "figures" / "preliminary_results"

CONDITION_ORDER = ["Base", "Neutral", "Subliminal"]
CONDITION_COLORS = {
    "Base": "#8795E8",
    "Neutral": "#C774E8",
    "Subliminal": "#FF6AD5",
}
SUMMARY_COLORS = {
    "Header": "#241A3A",
    "Metric": "#FBF7FF",
    "Neutral": "#F6EAFE",
    "Subliminal": "#FFF0FA",
}
VAPEPLOT = None

FINAL_COMPARISON = (
    RUNS_DIR
    / "comparisons"
    / "20260520T153706Z_run_1779116721_vs_run_1779205677"
    / "comparison_table.csv"
)
FINAL_RUN_DIRS = {
    "Neutral": RUNS_DIR / "reference_4080_qwen7b_neutral_10k_3epochs" / "run_1779116721",
    "Subliminal": RUNS_DIR
    / "reference_4080_qwen7b_cat_subliminal_10k_3epochs"
    / "run_1779205677",
}
BASE_RUN_DIR = RUNS_DIR / "qwen7b_cat_base_4bit"


def find_result_files() -> list[Path]:
    patterns = [
        "**/*comparison*.csv",
        "**/*comparison*.json",
        "**/*preference*.csv",
        "**/*preference*.json",
        "**/*eval_metrics.json",
        "**/*eval_token_metrics.jsonl",
        "**/*summary*.md",
        "**/*notes*.md",
    ]
    roots = [RUNS_DIR, RESULTS_DIR]
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found.update(root.glob(pattern))
    return sorted(path for path in found if "semantic" not in str(path).lower())


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def normalize_condition(raw: Any, path: Path | None = None) -> str | None:
    text = str(raw or "").lower()
    path_text = str(path or "").lower()
    combined = f"{text} {path_text}"
    if "subliminal" in combined or "biased" in combined or "trait" in combined:
        return "Subliminal"
    if "neutral" in combined or "control" in combined:
        return "Neutral"
    if "base" in combined or "evaluation_only" in combined:
        return "Base"
    return None


def condition_sort_key(condition: str) -> int:
    try:
        return CONDITION_ORDER.index(condition)
    except ValueError:
        return len(CONDITION_ORDER)


def binomial_ci95(rate: float, n: float | int | None) -> float | None:
    if n is None or pd.isna(n) or n <= 0 or pd.isna(rate):
        return None
    return 1.96 * math.sqrt(float(rate) * (1.0 - float(rate)) / float(n))


def flatten_eval_metrics(path: Path, condition_hint: str | None = None) -> dict[str, Any]:
    data = read_json(path)
    metrics = data.get("metrics", {})
    choice = data.get("choice_metrics", {})
    logprob = data.get("logprob_metrics", {})
    diagnostics = data.get("model_diagnostics", {})
    condition = normalize_condition(condition_hint or data.get("system_prompt_mode"), path)

    row = {
        "condition": condition,
        "source": str(path.relative_to(REPO_ROOT)),
        "run_dir": str(path.parent),
        "target_animal": metrics.get("target_animal") or choice.get("target_animal"),
        "num_samples": data.get("num_samples") or metrics.get("total_completions"),
        "prompt_set": data.get("prompt_set"),
        "generation_mode": data.get("generation_mode"),
        "quantization_mode": diagnostics.get("quantization_mode"),
        "target_choice_rate": choice.get("target_choice_rate"),
        "target_animal_rate": metrics.get("target_animal_rate") or metrics.get("target_rate"),
        "no_choice_rate": choice.get("no_choice_rate"),
        "target_logprob_win_rate": logprob.get("target_win_rate"),
        "average_target_margin": logprob.get("average_target_margin"),
        "target_logprob": metrics.get("target_logprob"),
        "target_rank": metrics.get("target_rank"),
        "target_probability": metrics.get("target_probability"),
        "target_vs_lion_margin": metrics.get("target_vs_lion_margin"),
        "kl_student_base": metrics.get("kl_student_base"),
        "entropy": metrics.get("entropy"),
        "target_choice_count": None,
    }
    target = row["target_animal"]
    counts = choice.get("choice_counts", {})
    if target in counts:
        row["target_choice_count"] = counts[target]
    return row


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata = read_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    dataset = read_json(run_dir / "dataset_stats.json") if (run_dir / "dataset_stats.json").exists() else {}
    training = read_json(run_dir / "training_metrics.json") if (run_dir / "training_metrics.json").exists() else {}
    return {"metadata": metadata, "dataset": dataset, "training": training}


def load_final_comparison() -> tuple[pd.DataFrame, list[str]]:
    used: list[str] = []
    rows: list[dict[str, Any]] = []

    if BASE_RUN_DIR.exists() and (BASE_RUN_DIR / "eval_metrics.json").exists():
        base = flatten_eval_metrics(BASE_RUN_DIR / "eval_metrics.json", "base")
        base["note"] = "context base eval; exact prompt set"
        rows.append(base)
        used.append(str((BASE_RUN_DIR / "eval_metrics.json").relative_to(REPO_ROOT)))
    else:
        warnings.warn("Base eval_metrics.json not found; Base will be omitted.")

    if FINAL_COMPARISON.exists():
        comparison = pd.read_csv(FINAL_COMPARISON)
        used.append(str(FINAL_COMPARISON.relative_to(REPO_ROOT)))
        for _, raw in comparison.iterrows():
            condition = normalize_condition(raw.get("condition"), Path(str(raw.get("run_dir", ""))))
            row = raw.to_dict()
            row["condition"] = condition
            row["source"] = str(FINAL_COMPARISON.relative_to(REPO_ROOT))
            row["note"] = "final reference comparison"
            run_dir = Path(str(raw.get("run_dir", "")))
            eval_path = run_dir / "eval_metrics.json"
            if eval_path.exists():
                eval_row = flatten_eval_metrics(eval_path, condition)
                for key in ["target_animal", "num_samples", "target_probability"]:
                    if key not in row or pd.isna(row.get(key)):
                        row[key] = eval_row.get(key)
                used.append(str(eval_path.relative_to(REPO_ROOT)))
            rows.append(row)
    else:
        warnings.warn("Final comparison table not found; trying final run eval_metrics.json files.")
        for condition, run_dir in FINAL_RUN_DIRS.items():
            path = run_dir / "eval_metrics.json"
            if path.exists():
                row = flatten_eval_metrics(path, condition)
                row["note"] = "final reference eval"
                rows.append(row)
                used.append(str(path.relative_to(REPO_ROOT)))

    df = pd.DataFrame(rows)
    if df.empty:
        return df, used

    for col in [
        "target_choice_rate",
        "target_animal_rate",
        "target_logprob_win_rate",
        "average_target_margin",
        "target_logprob",
        "target_rank",
        "target_probability",
        "target_vs_lion_margin",
        "kl_student_base",
        "entropy",
        "num_samples",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "num_samples" not in df.columns:
        df["num_samples"] = np.nan
    if "target_choice_count" not in df.columns:
        df["target_choice_count"] = np.nan
    df["choice_ci95"] = [
        binomial_ci95(rate, n)
        for rate, n in zip(df.get("target_choice_rate", []), df.get("num_samples", []))
    ]
    df = df.dropna(subset=["condition"]).sort_values(
        "condition", key=lambda s: s.map(condition_sort_key)
    )
    return df, used


def apply_style() -> None:
    global VAPEPLOT, CONDITION_COLORS
    try:
        import vapeplot  # type: ignore

        VAPEPLOT = vapeplot
        vapeplot.set_palette("cool")
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(colors) >= 5:
            CONDITION_COLORS = {
                "Base": colors[3],
                "Neutral": colors[1],
                "Subliminal": colors[0],
            }
        print("Style: using vapeplot.")
    except Exception:
        VAPEPLOT = None
        print("Style: vapeplot unavailable; using matplotlib defaults.")

    plt.rcParams.update(
        {
            "figure.figsize": (12.8, 7.2),
            "figure.dpi": 120,
            "font.size": 18,
            "axes.titlesize": 30,
            "axes.labelsize": 21,
            "xtick.labelsize": 19,
            "ytick.labelsize": 18,
            "legend.fontsize": 17,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.9,
        }
    )


def despine(ax: plt.Axes) -> None:
    if VAPEPLOT is not None:
        try:
            VAPEPLOT.despine(ax)
            return
        except Exception:
            pass
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def cleanup_previous_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("figure_*.png"):
        path.unlink()
    for path in OUT_DIR.glob("figure_*.pdf"):
        path.unlink()


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [OUT_DIR / f"{stem}.png", OUT_DIR / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def final_pair(df: pd.DataFrame) -> pd.DataFrame:
    pair = df[df["condition"].isin(["Neutral", "Subliminal"])].copy()
    return pair.sort_values("condition", key=lambda s: s.map(condition_sort_key))


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def rate_delta_ci95(neutral: pd.Series, subliminal: pd.Series, metric: str) -> float | None:
    if metric not in {"target_choice_rate", "target_animal_rate", "target_logprob_win_rate"}:
        return None
    n0 = neutral.get("num_samples", np.nan)
    n1 = subliminal.get("num_samples", np.nan)
    p0 = neutral.get(metric, np.nan)
    p1 = subliminal.get(metric, np.nan)
    if pd.isna(n0) or pd.isna(n1) or pd.isna(p0) or pd.isna(p1) or n0 <= 0 or n1 <= 0:
        return None
    return 1.96 * math.sqrt((p0 * (1 - p0) / n0) + (p1 * (1 - p1) / n1))


def plot_bar_with_optional_error(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    stem: str,
    subtitle: str | None = None,
    invert_y: bool = False,
) -> list[Path]:
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        warnings.warn(f"Skipping {stem}: metric '{metric}' is missing.")
        return []

    plot_df = plot_df.sort_values("condition", key=lambda s: s.map(condition_sort_key))
    colors = [CONDITION_COLORS.get(c, "#374151") for c in plot_df["condition"]]
    x = np.arange(len(plot_df))
    y = plot_df[metric].astype(float).to_numpy()

    yerr = None
    if metric in {"target_choice_rate", "target_animal_rate"} and "choice_ci95" in plot_df:
        ci = plot_df["choice_ci95"].astype(float)
        if ci.notna().any():
            yerr = ci.fillna(0.0).to_numpy()

    fig, ax = plt.subplots()
    bars = ax.bar(x, y, color=colors, edgecolor="#111827", linewidth=1.1, yerr=yerr, capsize=5)
    ax.set_title(title, pad=28)
    if subtitle:
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=16,
            color="#374151",
        )
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["condition"])
    if metric.endswith("_rate") or metric == "target_probability":
        upper = max(0.12, float(np.nanmax(y + (yerr if yerr is not None else 0))) * 1.22)
        ax.set_ylim(0, min(1.05, upper))
        ax.yaxis.set_major_formatter(lambda value, pos: f"{value:.0%}")
    if invert_y:
        ax.invert_yaxis()
    ax.grid(axis="x", visible=False)

    for bar, value in zip(bars, y):
        label = f"{value:.1%}" if metric.endswith("_rate") or metric == "target_probability" else f"{value:.2f}"
        offset = 0.012 if metric.endswith("_rate") or metric == "target_probability" else 0.04
        y_text = value + offset if not invert_y else value - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            label,
            ha="center",
            va="bottom" if not invert_y else "top",
            fontsize=16,
            color="#111827",
        )

    return save_figure(fig, stem)


def plot_main_preference(df: pd.DataFrame) -> tuple[list[Path], str | None]:
    pair = final_pair(df)
    if "target_choice_rate" not in df or len(pair) < 2:
        warnings.warn("No target_choice_rate column found for main preference plot.")
        return [], None

    neutral = pair[pair["condition"] == "Neutral"].iloc[0]
    subliminal = pair[pair["condition"] == "Subliminal"].iloc[0]
    target = str(pair["target_animal"].dropna().iloc[0]) if pair["target_animal"].notna().any() else "target"
    n_value = float(neutral["target_choice_rate"])
    s_value = float(subliminal["target_choice_rate"])
    delta = s_value - n_value

    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    y_main = 0.0
    ax.hlines(y_main, n_value, s_value, color="#111827", linewidth=2.2, alpha=0.75)
    ax.scatter(
        [n_value],
        [y_main],
        s=420,
        color=CONDITION_COLORS["Neutral"],
        edgecolor="#111827",
        linewidth=1.4,
        zorder=3,
        label="Neutral",
    )
    ax.scatter(
        [s_value],
        [y_main],
        s=420,
        color=CONDITION_COLORS["Subliminal"],
        edgecolor="#111827",
        linewidth=1.4,
        zorder=3,
        label="Subliminal",
    )

    ci0 = binomial_ci95(n_value, neutral.get("num_samples", np.nan))
    ci1 = binomial_ci95(s_value, subliminal.get("num_samples", np.nan))
    if ci0 is not None:
        ax.errorbar(n_value, y_main, xerr=ci0, fmt="none", color="#111827", capsize=5, lw=1.5)
    if ci1 is not None:
        ax.errorbar(s_value, y_main, xerr=ci1, fmt="none", color="#111827", capsize=5, lw=1.5)

    base_rows = df[df["condition"] == "Base"].dropna(subset=["target_choice_rate"])
    if not base_rows.empty:
        base_value = float(base_rows.iloc[0]["target_choice_rate"])
        ax.scatter(
            [base_value],
            [-0.34],
            s=180,
            color=CONDITION_COLORS["Base"],
            edgecolor="#111827",
            linewidth=1.0,
            alpha=0.65,
            label="Base (context)",
        )
        ax.text(base_value, -0.48, f"Base\n{pct(base_value)}", ha="center", va="top", fontsize=13)

    ax.annotate(
        f"{pp(delta)}",
        xy=((n_value + s_value) / 2, y_main + 0.035),
        xytext=((n_value + s_value) / 2, y_main + 0.16),
        ha="center",
        va="bottom",
        fontsize=24,
        color="#111827",
        arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 1.6},
    )
    ax.text(n_value, y_main - 0.11, f"Neutral\n{pct(n_value)}", ha="center", va="top", fontsize=17)
    ax.text(
        s_value,
        y_main - 0.11,
        f"Subliminal\n{pct(s_value)}",
        ha="center",
        va="top",
        fontsize=17,
    )
    ax.text(
        0.02,
        0.95,
        "Matched final reference setup; Base shown only as contextual anchor",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        color="#374151",
    )

    lower = max(0, min(n_value, s_value, *(base_rows["target_choice_rate"].tolist() or [n_value])) - 0.012)
    upper = min(0.12, max(n_value, s_value) + 0.018)
    ax.set_xlim(lower, upper)
    ax.set_ylim(-0.58, 0.34)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(lambda value, pos: f"{value:.0%}")
    ax.set_xlabel(f"{target.title()} preference rate")
    ax.set_title("Target Preference Shift", pad=16)
    ax.grid(axis="y", visible=False)
    despine(ax)
    return save_figure(fig, "figure_1_target_preference_shift"), "target_choice_rate"


def plot_logit_evidence(df: pd.DataFrame) -> tuple[list[Path], list[str]]:
    pair = final_pair(df)
    if len(pair) < 2:
        warnings.warn("No final Neutral/Subliminal pair found for logit dashboard.")
        return [], []

    metric_specs = [
        ("target_choice_rate", "Choice Rate", "rate", False),
        ("target_probability", "Target Probability", "rate", False),
        ("target_logprob_win_rate", "Logprob Win Rate", "rate", False),
        ("target_rank", "Rank Improvement", "rank_improvement", False),
    ]
    metric_specs = [spec for spec in metric_specs if spec[0] in pair and pair[spec[0]].notna().sum() == 2]
    if not metric_specs:
        warnings.warn(
            "No logit-level plot generated. Need columns such as target_logprob, "
            "target_rank, target_probability, or target_vs_lion_margin."
        )
        return [], []

    neutral = pair[pair["condition"] == "Neutral"].iloc[0]
    subliminal = pair[pair["condition"] == "Subliminal"].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2))
    axes_flat = axes.ravel()
    metrics_used: list[str] = []

    for ax, (metric, title, kind, lower_is_better) in zip(axes_flat, metric_specs):
        n_value = float(neutral[metric])
        s_value = float(subliminal[metric])
        delta = s_value - n_value
        metrics_used.append(metric)

        if kind == "rank_improvement":
            n_plot = 0.0
            s_plot = n_value - s_value
            y_values = [n_plot, s_plot]
        else:
            n_plot = n_value
            s_plot = s_value
            y_values = [n_plot, s_plot]

        ax.plot(
            [0, 1],
            y_values,
            color="#111827",
            linewidth=2.0,
            alpha=0.85,
            zorder=2,
        )
        ax.scatter(
            [0],
            [n_plot],
            s=250,
            color=CONDITION_COLORS["Neutral"],
            edgecolor="#111827",
            linewidth=1.1,
            zorder=3,
        )
        ax.scatter(
            [1],
            [s_plot],
            s=250,
            color=CONDITION_COLORS["Subliminal"],
            edgecolor="#111827",
            linewidth=1.1,
            zorder=3,
        )

        if kind == "rate":
            label_n = pct(n_value)
            label_s = pct(s_value)
            delta_label = pp(delta)
            margin = max(0.015, abs(delta) * 1.3)
            y_min = max(0.0, min(y_values) - margin)
            y_max = min(1.0, max(y_values) + margin)
            ax.yaxis.set_major_formatter(lambda value, pos: f"{value:.0%}")
            ci = rate_delta_ci95(neutral, subliminal, metric)
            if ci is not None:
                delta_label = f"{pp(delta)} +/- {100 * ci:.2f} pp"
        elif kind == "rank_improvement":
            improvement = n_value - s_value
            label_n = "0.00"
            label_s = f"+{improvement:.2f}"
            delta_label = f"+{improvement:.2f} ranks"
            y_min = -0.12
            y_max = max(1.15, improvement + 0.28)
        else:
            label_n = f"{n_value:.2f}"
            label_s = f"{s_value:.2f}"
            delta_label = f"{delta:+.2f}"
            if lower_is_better and delta < 0:
                delta_label = f"{abs(delta):.2f} ranks better"
            margin = max(0.25, abs(delta) * 1.3)
            y_min = min(n_value, s_value) - margin
            y_max = max(n_value, s_value) + margin

        label_offset = 13
        ax.annotate(
            label_n,
            xy=(0, n_plot),
            xytext=(0, label_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=13,
            clip_on=False,
        )
        ax.annotate(
            label_s,
            xy=(1, s_plot),
            xytext=(0, label_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=13,
            clip_on=False,
        )
        ax.text(
            0.5,
            0.92,
            delta_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            color="#111827",
        )
        ax.set_title(title, fontsize=20, pad=10)
        ax.set_xlim(-0.28, 1.28)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([0, 1])
        if kind == "rank_improvement":
            ax.set_xticklabels(["Neutral\nbaseline", "Subliminal"])
            ax.set_ylabel("Improvement")
        else:
            ax.set_xticklabels(["Neutral", "Subliminal"])
        ax.grid(axis="x", visible=False)
        despine(ax)

    for ax in axes_flat[len(metric_specs) :]:
        ax.axis("off")

    fig.text(
        0.5,
        0.925,
        "",
        ha="center",
        va="top",
        fontsize=15,
        color="#374151",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
    return save_figure(fig, "figure_2_metric_dashboard"), metrics_used


def load_journey_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    stage_specs = [
        (
            "Qwen3B Owl",
            [
                ("Neutral", RUNS_DIR / "neutral_1k_qwen3b_v2" / "eval_metrics.json"),
                ("Subliminal", RUNS_DIR / "subliminal_1k_qwen3b_v2" / "eval_metrics.json"),
            ],
        ),
        (
            "Qwen7B Cat 2k",
            [
                ("Base", RUNS_DIR / "qwen7b_cat_base_4bit" / "eval_metrics.json"),
                ("Neutral", RUNS_DIR / "qwen7b_cat_neutral_sanity_2k_4bit" / "eval_metrics.json"),
                ("Subliminal", RUNS_DIR / "qwen7b_cat_sanity_2k_4bit" / "eval_metrics.json"),
            ],
        ),
        (
            "Qwen7B Cat 10k",
            [
                (
                    "Subliminal",
                    RUNS_DIR
                    / "qwen7b_cat_subliminal_10k_greedy_alltokens_3epochs"
                    / "run_1779091598"
                    / "eval_metrics.json",
                ),
            ],
        ),
        (
            "Final Setup",
            [
                ("Neutral", FINAL_RUN_DIRS["Neutral"] / "eval_metrics.json"),
                ("Subliminal", FINAL_RUN_DIRS["Subliminal"] / "eval_metrics.json"),
            ],
        ),
    ]

    for stage_index, (stage, specs) in enumerate(stage_specs):
        for condition, path in specs:
            if not path.exists():
                warnings.warn(f"Journey metric missing: {path.relative_to(REPO_ROOT)}")
                continue
            row = flatten_eval_metrics(path, condition)
            row["stage"] = stage
            row["stage_index"] = stage_index
            rows.append(row)

    return pd.DataFrame(rows)


def plot_reproduction_journey() -> tuple[list[Path], list[str]]:
    df = load_journey_rows()
    if df.empty or "target_choice_rate" not in df:
        warnings.warn("Skipping reproduction journey: no usable target_choice_rate metrics.")
        return [], []

    fig, ax = plt.subplots(figsize=(12.8, 5.9))
    offsets = {"Base": -0.10, "Neutral": -0.05, "Subliminal": 0.05}
    visible_top = 0.105
    stage_notes = {
        0: "No Transfer",
        1: "Weak Signal",
        2: "Mixed Results",
        3: "Consistent Transfer",
    }
    stage_note_x = {0: 0.18, 1: 1.0, 2: 2.0, 3: 2.98}
    stage_note_y = {
        0: visible_top * 0.90,
        1: visible_top * 0.90,
        2: visible_top * 0.90,
        3: visible_top * 0.40,
    }
    stage_labels = (
        df.sort_values("stage_index")[["stage_index", "stage"]].drop_duplicates()["stage"].tolist()
    )

    for idx in sorted(df["stage_index"].dropna().unique()):
        ax.axvline(idx, color="#E5E7EB", linewidth=1.0, zorder=0)

    for condition in CONDITION_ORDER:
        sub = df[df["condition"] == condition]
        if sub.empty:
            continue
        sub = sub.sort_values("stage_index")
        x = sub["stage_index"].astype(float).to_numpy() + offsets.get(condition, 0.0)
        raw_y = sub["target_choice_rate"].astype(float).to_numpy()
        y = np.minimum(raw_y, visible_top * 0.93)
        if condition in {"Neutral", "Subliminal"}:
            ax.plot(
                x,
                y,
                color=CONDITION_COLORS[condition],
                linewidth=2.3,
                alpha=0.62,
                zorder=1,
            )
        ax.scatter(
            x,
            y,
            s=190 if condition in {"Neutral", "Subliminal"} else 120,
            color=CONDITION_COLORS[condition],
            edgecolor="#111827",
            linewidth=1.0,
            label=condition,
            alpha=0.95 if condition in {"Neutral", "Subliminal"} else 0.60,
            zorder=3,
        )
        label_dx = {"Base": -0.08, "Neutral": -0.04, "Subliminal": 0.04}.get(condition, 0.0)
        for xi, yi, raw_value in zip(x, y, raw_y):
            label = f"{raw_value:.1%}"
            if raw_value > visible_top:
                label = "100%"
            ax.text(
                xi + label_dx,
                min(visible_top * 0.97, yi + 0.008),
                label,
                ha="center",
                va="bottom",
                fontsize=12,
            )

    


    ax.set_ylabel("Target Choice Rate")
    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels)
    ax.set_ylim(0, visible_top)
    ax.set_xlim(-0.35, len(stage_labels) - 0.65)
    ax.yaxis.set_major_formatter(lambda value, pos: f"{value:.0%}")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3, frameon=False)
    despine(ax)
    return save_figure(fig, "figure_3_reproduction_journey"), ["target_choice_rate"]


def final_summary_table_data(final_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[list[str]] = []

    neutral = final_df[final_df["condition"] == "Neutral"].iloc[0]
    subliminal = final_df[final_df["condition"] == "Subliminal"].iloc[0]

    outcome_specs = [
        ("Choice Rate", "target_choice_rate", "rate"),
        ("Target Probability", "target_probability", "rate"),
        ("Logprob Win Rate", "target_logprob_win_rate", "rate"),
    ]

    for label, metric, kind in outcome_specs:
        if metric not in final_df or pd.isna(neutral.get(metric)) or pd.isna(subliminal.get(metric)):
            continue
        neutral_value = float(neutral[metric])
        subliminal_value = float(subliminal[metric])
        delta = subliminal_value - neutral_value
        if kind == "rate":
            rows.append([label, f"{neutral_value:.2%}", f"{subliminal_value:.2%}", pp(delta)])
        else:
            rows.append([label, f"{neutral_value:.2f}", f"{subliminal_value:.2f}", f"{delta:+.2f}"])

    return pd.DataFrame(rows, columns=["Outcome", "Neutral", "Subliminal", "Delta"])


def plot_final_summary(final_df: pd.DataFrame) -> tuple[list[Path], list[str]]:
    needed = {"Neutral", "Subliminal"}
    if not needed.issubset(set(final_df["condition"])):
        warnings.warn("Skipping controlled summary: final Neutral/Subliminal pair unavailable.")
        return [], []

    table_df = final_summary_table_data(final_df)
    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.axis("off")
    ax.set_title("Final Controlled Comparison", pad=10)
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.34, 0.23, 0.23, 0.20],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 1.75)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(SUMMARY_COLORS["Header"])
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif col == 2:
            cell.set_facecolor(SUMMARY_COLORS["Subliminal"])
        elif col == 3:
            cell.set_facecolor("#FFF7FD")
        elif col == 1:
            cell.set_facecolor(SUMMARY_COLORS["Neutral"])
        else:
            cell.set_facecolor(SUMMARY_COLORS["Metric"])
    return save_figure(fig, "figure_4_final_controlled_summary"), [
        "target_choice_rate",
        "target_probability",
        "target_logprob_win_rate",
    ]


def print_data_summary(found_files: list[Path], used_files: list[str], df: pd.DataFrame) -> None:
    print("\nResult-file scan:")
    print(f"  Found {len(found_files)} candidate result files.")
    for path in found_files[:40]:
        print(f"  - {path.relative_to(REPO_ROOT)}")
    if len(found_files) > 40:
        print(f"  ... and {len(found_files) - 40} more")

    print("\nFiles used for main figures:")
    for path in sorted(set(used_files)):
        print(f"  - {path}")

    if not df.empty:
        cols = [
            "condition",
            "target_animal",
            "target_choice_rate",
            "target_logprob_win_rate",
            "average_target_margin",
            "target_rank",
            "source",
        ]
        available_cols = [col for col in cols if col in df.columns]
        print("\nMain comparison metrics:")
        print(df[available_cols].to_string(index=False))


def main() -> None:
    apply_style()
    cleanup_previous_outputs()

    found_files = find_result_files()
    final_df, used_files = load_final_comparison()
    print_data_summary(found_files, used_files, final_df)

    generated: list[Path] = []
    metrics_used: dict[str, list[str] | str | None] = {}

    paths, metric = plot_main_preference(final_df)
    generated.extend(paths)
    metrics_used["Figure 1"] = metric

    paths, metrics = plot_logit_evidence(final_df)
    generated.extend(paths)
    metrics_used["Figure 2"] = metrics

    paths, metrics = plot_reproduction_journey()
    generated.extend(paths)
    metrics_used["Figure 3"] = metrics

    paths, metrics = plot_final_summary(final_df)
    generated.extend(paths)
    metrics_used["Figure 4"] = metrics

    print("\nMetrics used:")
    for figure, metrics in metrics_used.items():
        print(f"  - {figure}: {metrics}")

    print("\nGenerated figures:")
    if generated:
        for path in generated:
            print(f"  - {path.relative_to(REPO_ROOT)}")
    else:
        print("  none")

    missing_notes = []
    if "Base" in set(final_df.get("condition", [])):
        missing_notes.append(
            "Base is available only as a contextual Qwen7B cat eval, not as a final paper-reference sampled run."
        )
    if missing_notes:
        print("\nWarnings / manual export suggestions:")
        for note in missing_notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
