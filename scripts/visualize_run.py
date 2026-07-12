from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap, repo_path

bootstrap()

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

try:
    import vapeplot
except ImportError:  # pragma: no cover - fallback for minimal environments
    vapeplot = None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def metric_at(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def palette(name: str) -> list[str]:
    if vapeplot is not None:
        try:
            return list(vapeplot.palette(name))
        except Exception:
            pass
    return [
        "#94D0FF",
        "#966BFF",
        "#FF6AD5",
        "#FFDE8B",
        "#20DE8B",
        "#FF8B8B",
        "#8795E8",
        "#C774E8",
    ]


def color_at(colors: list[str], index: int) -> str:
    return colors[index % len(colors)]


def color_slice(colors: list[str], count: int) -> list[str]:
    return [color_at(colors, index) for index in range(count)]


def make_template(colors: list[str]) -> dict[str, Any]:
    return {
        "layout": {
            "colorway": colors,
            "font": {"family": "Inter, Segoe UI, Arial, sans-serif", "color": "#f7f2ff"},
            "paper_bgcolor": "#15131d",
            "plot_bgcolor": "#201b2e",
            "xaxis": {
                "gridcolor": "rgba(255,255,255,0.08)",
                "zerolinecolor": "rgba(255,255,255,0.18)",
            },
            "yaxis": {
                "gridcolor": "rgba(255,255,255,0.08)",
                "zerolinecolor": "rgba(255,255,255,0.18)",
            },
            "legend": {"orientation": "h", "y": -0.2},
            "margin": {"l": 50, "r": 30, "t": 70, "b": 65},
        }
    }


def style_figure(fig: go.Figure, title: str, colors: list[str]) -> go.Figure:
    fig.update_layout(template=make_template(colors), title=title, height=460)
    return fig


def format_seconds(value: Any) -> str | None:
    if not is_number(value):
        return None
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.0f}s"


def core_metric_rows(eval_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "metric": "target_animal_rate",
            "value": metric_at(eval_metrics, "metrics", "target_animal_rate"),
        },
        {
            "metric": "target_choice_rate",
            "value": metric_at(eval_metrics, "choice_metrics", "target_choice_rate"),
        },
        {"metric": "no_choice_rate", "value": metric_at(eval_metrics, "choice_metrics", "no_choice_rate")},
        {
            "metric": "target_logprob_win_rate",
            "value": metric_at(eval_metrics, "logprob_metrics", "target_win_rate"),
        },
        {
            "metric": "average_target_margin",
            "value": metric_at(eval_metrics, "logprob_metrics", "average_target_margin"),
        },
    ]


def metadata_table(
    metadata: dict[str, Any],
    dataset_stats: dict[str, Any],
    training: dict[str, Any],
    timing: dict[str, Any],
) -> str:
    rows = [
        ("run_id", metadata.get("run_id")),
        ("condition", metadata.get("condition")),
        ("trait", metadata.get("trait")),
        ("model", metadata.get("model_name")),
        ("prompt_style", metadata.get("prompt_style")),
        ("system_prompt_mode", metadata.get("system_prompt_mode")),
        ("generated_samples", dataset_stats.get("generated_sample_count")),
        ("filtered_samples", dataset_stats.get("filtered_sample_count")),
        ("filter_retention_rate", dataset_stats.get("filter_retention_rate")),
        ("optimizer_steps", training.get("optimizer_steps")),
        ("train_loss", training.get("train_loss")),
        ("trainer_backend", training.get("trainer_backend")),
        ("elapsed", format_seconds(timing.get("elapsed_seconds"))),
        ("timing_source", timing.get("source", "timing.json") if timing else None),
        ("started_at", timing.get("started_at")),
        ("updated_at", timing.get("updated_at")),
    ]
    body = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in rows
        if value is not None
    )
    return f"<table class=\"meta-table\"><tbody>{body}</tbody></table>"


def build_core_metrics_fig(eval_metrics: dict[str, Any], colors: list[str]) -> go.Figure | None:
    rows = [row for row in core_metric_rows(eval_metrics) if is_number(row["value"])]
    if not rows:
        return None
    fig = go.Figure()
    fig.add_bar(
        x=[row["metric"] for row in rows],
        y=[row["value"] for row in rows],
        marker_color=color_slice(colors, len(rows)),
        hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
    )
    return style_figure(fig, "Core Metrics", colors)


def build_animal_rates_fig(eval_metrics: dict[str, Any], colors: list[str]) -> go.Figure | None:
    choice = metric_at(eval_metrics, "choice_metrics", "choice_rates", default={}) or {}
    completion = metric_at(eval_metrics, "metrics", "animal_completion_rates", default={}) or {}
    animals = sorted(set(choice) | set(completion))
    if not animals:
        return None
    fig = go.Figure()
    fig.add_bar(
        name="choice rate",
        x=animals,
        y=[choice.get(animal, 0.0) for animal in animals],
        marker_color=color_at(colors, 0),
    )
    fig.add_bar(
        name="completion rate",
        x=animals,
        y=[completion.get(animal, 0.0) for animal in animals],
        marker_color=color_at(colors, 2),
    )
    fig.update_layout(barmode="group", yaxis_title="rate")
    return style_figure(fig, "Animal Choice And Completion Rates", colors)


def build_teacher_student_fig(
    eval_metrics: dict[str, Any], teacher_metrics: dict[str, Any], colors: list[str]
) -> go.Figure | None:
    if not teacher_metrics:
        return None
    student_rows = core_metric_rows(eval_metrics)
    teacher_rows = core_metric_rows(teacher_metrics)
    metrics = [
        row["metric"]
        for row in student_rows
        if is_number(row["value"])
        and is_number(next((r["value"] for r in teacher_rows if r["metric"] == row["metric"]), None))
    ]
    if not metrics:
        return None
    student_values = {row["metric"]: row["value"] for row in student_rows}
    teacher_values = {row["metric"]: row["value"] for row in teacher_rows}
    fig = go.Figure()
    fig.add_bar(
        name="teacher/base",
        x=metrics,
        y=[teacher_values[m] for m in metrics],
        marker_color=color_at(colors, 1),
    )
    fig.add_bar(
        name="student",
        x=metrics,
        y=[student_values[m] for m in metrics],
        marker_color=color_at(colors, 4),
    )
    fig.update_layout(barmode="group")
    return style_figure(fig, "Teacher Vs Student Metrics", colors)


def build_logprob_margin_fig(eval_outputs: list[dict[str, Any]], colors: list[str]) -> go.Figure | None:
    rows = [
        row
        for row in eval_outputs
        if is_number(row.get("logprob_margin")) and row.get("logprob_winner") is not None
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_scatter(
        x=df.get("prompt_index", df.index),
        y=df["logprob_margin"],
        mode="markers+lines",
        marker={
            "size": 9,
            "color": df["logprob_margin"],
            "colorscale": [[0, color_at(colors, 2)], [0.5, color_at(colors, 0)], [1, color_at(colors, 4)]],
            "showscale": True,
            "colorbar": {"title": "margin"},
        },
        text=df["logprob_winner"],
        hovertemplate="prompt %{x}<br>winner=%{text}<br>margin=%{y:.4f}<extra></extra>",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.45)")
    fig.update_yaxes(title="winner margin")
    fig.update_xaxes(title="prompt index")
    return style_figure(fig, "Logprob Winner Margins By Prompt", colors)


def build_logprob_heatmap(eval_outputs: list[dict[str, Any]], colors: list[str]) -> go.Figure | None:
    rows = [row for row in eval_outputs if isinstance(row.get("logprobs"), dict)]
    if not rows:
        return None
    animals = sorted({animal for row in rows for animal in row["logprobs"]})
    z = [[row["logprobs"].get(animal) for animal in animals] for row in rows]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=animals,
            y=[row.get("prompt_index", index) for index, row in enumerate(rows)],
            colorscale=[
                [0.0, "#201b2e"],
                [0.25, color_at(colors, 1)],
                [0.55, color_at(colors, 2)],
                [0.8, color_at(colors, 0)],
                [1.0, color_at(colors, 4)],
            ],
            colorbar={"title": "logprob"},
            hovertemplate="prompt %{y}<br>%{x}: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="animal")
    fig.update_yaxes(title="prompt index")
    return style_figure(fig, "Animal Token Logprobs", colors)


def build_training_fig(training: dict[str, Any], colors: list[str]) -> go.Figure | None:
    history = training.get("trainer_log_history") or []
    rows = [row for row in history if is_number(row.get("step")) and is_number(row.get("loss"))]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            name="loss",
            x=df["step"],
            y=df["loss"],
            mode="lines+markers",
            line={"color": color_at(colors, 2), "width": 3},
        ),
        secondary_y=False,
    )
    if "learning_rate" in df:
        fig.add_trace(
            go.Scatter(
                name="learning rate",
                x=df["step"],
                y=df["learning_rate"],
                mode="lines",
                line={"color": color_at(colors, 0), "dash": "dot"},
            ),
            secondary_y=True,
        )
    if "grad_norm" in df:
        fig.add_trace(
            go.Scatter(
                name="grad norm",
                x=df["step"],
                y=df["grad_norm"],
                mode="lines",
                line={"color": color_at(colors, 4), "dash": "dash"},
            ),
            secondary_y=False,
        )
    fig.update_yaxes(title_text="loss / grad norm", secondary_y=False)
    fig.update_yaxes(title_text="learning rate", secondary_y=True)
    fig.update_xaxes(title="optimizer step")
    return style_figure(fig, "Training Dynamics", colors)


def build_timing_fig(timing: dict[str, Any], colors: list[str]) -> go.Figure | None:
    if timing.get("source") == "artifact_mtime_fallback":
        return None
    stages = [
        stage
        for stage in timing.get("stages", [])
        if stage.get("name") and is_number(stage.get("duration_seconds"))
    ]
    if not stages:
        return None
    names = [stage["name"] for stage in stages]
    durations = [stage["duration_seconds"] for stage in stages]
    text = [format_seconds(duration) for duration in durations]
    fig = go.Figure()
    fig.add_bar(
        x=durations,
        y=names,
        orientation="h",
        text=text,
        textposition="auto",
        marker_color=color_slice(colors, len(stages)),
        customdata=[stage.get("status", "unknown") for stage in stages],
        hovertemplate="%{y}<br>duration=%{text}<br>status=%{customdata}<extra></extra>",
    )
    fig.update_xaxes(title="duration seconds")
    fig.update_yaxes(title="pipeline stage", autorange="reversed")
    return style_figure(fig, "Pipeline Stage Durations", colors)


def infer_artifact_timing(run_dir: Path) -> dict[str, Any]:
    artifact_names = [
        "metadata.json",
        "config_resolved.yaml",
        "generation.log",
        "dataset_stats.json",
        "teacher_eval.log",
        "teacher_metrics.json",
        "train.log",
        "training_metrics.json",
        "eval.log",
        "eval_metrics.json",
        "summary.md",
        "report.html",
    ]
    rows = []
    for name in artifact_names:
        path = run_dir / name
        if path.exists():
            rows.append(
                {
                    "name": name,
                    "mtime": path.stat().st_mtime,
                    "updated_at": pd.to_datetime(path.stat().st_mtime, unit="s", utc=True).isoformat(),
                }
            )
    rows.sort(key=lambda row: row["mtime"])
    if len(rows) < 2:
        return {}
    first = rows[0]["mtime"]
    previous = first
    stages = []
    for row in rows:
        stages.append(
            {
                "name": row["name"],
                "started_at": rows[0]["updated_at"],
                "ended_at": row["updated_at"],
                "duration_seconds": max(row["mtime"] - previous, 0.0),
                "seconds_since_first_artifact": max(row["mtime"] - first, 0.0),
                "status": "mtime_fallback",
            }
        )
        previous = row["mtime"]
    return {
        "run_id": run_dir.name,
        "started_at": rows[0]["updated_at"],
        "updated_at": rows[-1]["updated_at"],
        "elapsed_seconds": max(rows[-1]["mtime"] - first, 0.0),
        "source": "artifact_mtime_fallback",
        "stages": stages,
    }


def build_artifact_timeline_fig(timing: dict[str, Any], colors: list[str]) -> go.Figure | None:
    if timing.get("source") != "artifact_mtime_fallback":
        return None
    stages = [
        stage
        for stage in timing.get("stages", [])
        if stage.get("name") and is_number(stage.get("seconds_since_first_artifact"))
    ]
    if not stages:
        return None
    fig = go.Figure()
    fig.add_scatter(
        x=[stage["seconds_since_first_artifact"] for stage in stages],
        y=[stage["name"] for stage in stages],
        mode="markers+lines",
        marker={"size": 11, "color": color_slice(colors, len(stages))},
        text=[stage.get("ended_at") for stage in stages],
        hovertemplate="%{y}<br>since first artifact=%{x:.2f}s<br>%{text}<extra></extra>",
    )
    fig.update_xaxes(title="seconds since first artifact")
    fig.update_yaxes(title="artifact", autorange="reversed")
    return style_figure(fig, "Artifact Timeline (Approximate Mtime Fallback)", colors)


def completion_text(record: dict[str, Any]) -> str:
    value = record.get("filtered_completion", record.get("completion", ""))
    return "" if value is None else str(value)


def build_completion_length_fig(samples: list[dict[str, Any]], colors: list[str]) -> go.Figure | None:
    if not samples:
        return None
    lengths = [len(completion_text(row).split()) for row in samples]
    if not lengths:
        return None
    fig = go.Figure()
    fig.add_histogram(x=lengths, nbinsx=30, marker_color=color_at(colors, 0))
    fig.update_xaxes(title="completion whitespace-token length")
    fig.update_yaxes(title="count")
    return style_figure(fig, "Completion Length Distribution", colors)


def build_number_distribution_fig(samples: list[dict[str, Any]], colors: list[str]) -> go.Figure | None:
    numbers: list[float] = []
    for row in samples:
        parsed = row.get("parsed_numbers")
        if isinstance(parsed, list):
            numbers.extend(value for value in parsed if is_number(value))
    if not numbers:
        return None
    fig = go.Figure()
    fig.add_histogram(x=numbers, nbinsx=50, marker_color=color_at(colors, 4))
    fig.update_xaxes(title="parsed number")
    fig.update_yaxes(title="count")
    return style_figure(fig, "Parsed Number Distribution", colors)


def build_repeated_completion_fig(samples: list[dict[str, Any]], colors: list[str]) -> go.Figure | None:
    if not samples:
        return None
    counts = Counter(completion_text(row) for row in samples if completion_text(row))
    repeated = [(text, count) for text, count in counts.most_common(20) if count > 1]
    if not repeated:
        return None
    labels = [text if len(text) <= 60 else f"{text[:57]}..." for text, _ in repeated]
    values = [count for _, count in repeated]
    fig = go.Figure()
    fig.add_bar(x=values, y=labels, orientation="h", marker_color=color_at(colors, 6))
    fig.update_xaxes(title="count")
    fig.update_yaxes(title="completion", autorange="reversed")
    return style_figure(fig, "Most Repeated Completions", colors)


def build_dataset_fig(dataset_stats: dict[str, Any], colors: list[str]) -> go.Figure | None:
    rows = [
        ("generated", dataset_stats.get("generated_sample_count")),
        ("filtered", dataset_stats.get("filtered_sample_count")),
        ("invalid", dataset_stats.get("invalid_count")),
    ]
    rows = [(label, value) for label, value in rows if is_number(value)]
    if not rows:
        return None
    fig = go.Figure()
    fig.add_bar(
        x=[row[0] for row in rows],
        y=[row[1] for row in rows],
        marker_color=color_slice(colors, len(rows)),
    )
    fig.update_yaxes(title="records")
    return style_figure(fig, "Dataset Filtering Summary", colors)


def figure_card(fig: go.Figure, include_plotlyjs: bool) -> str:
    return (
        "<section class=\"plot-card\">"
        + to_html(fig, full_html=False, include_plotlyjs="cdn" if include_plotlyjs else False)
        + "</section>"
    )


def build_report(run_dir: Path, output_path: Path, palette_name: str, sample_limit: int | None) -> Path:
    colors = palette(palette_name)
    metadata = load_json(run_dir / "metadata.json")
    eval_metrics = load_json(run_dir / "eval_metrics.json")
    teacher_metrics = load_json(run_dir / "teacher_metrics.json")
    dataset_stats = load_json(run_dir / "dataset_stats.json")
    training = load_json(run_dir / "training_metrics.json")
    timing = load_json(run_dir / "timing.json") or infer_artifact_timing(run_dir)
    eval_outputs = load_jsonl(run_dir / "eval_outputs.jsonl", limit=sample_limit)
    samples = load_jsonl(run_dir / "samples_filtered.jsonl", limit=sample_limit)

    figures = [
        build_core_metrics_fig(eval_metrics, colors),
        build_animal_rates_fig(eval_metrics, colors),
        build_teacher_student_fig(eval_metrics, teacher_metrics, colors),
        build_logprob_margin_fig(eval_outputs, colors),
        build_logprob_heatmap(eval_outputs, colors),
        build_timing_fig(timing, colors),
        build_artifact_timeline_fig(timing, colors),
        build_training_fig(training, colors),
        build_dataset_fig(dataset_stats, colors),
        build_completion_length_fig(samples, colors),
        build_number_distribution_fig(samples, colors),
        build_repeated_completion_fig(samples, colors),
    ]
    figures = [fig for fig in figures if fig is not None]

    plot_html = "\n".join(
        figure_card(fig, include_plotlyjs=index == 0) for index, fig in enumerate(figures)
    )
    if not plot_html:
        plot_html = "<p class=\"empty\">No plottable artifacts found in this run.</p>"

    run_name = metadata.get("run_id") or run_dir.name
    target = metric_at(eval_metrics, "metrics", "target_animal") or metadata.get("trait") or "unknown"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Run Report - {html.escape(str(run_name))}</title>
  <style>
    :root {{
      --bg: #120f19;
      --panel: #1b1726;
      --panel-2: #201b2e;
      --text: #f7f2ff;
      --muted: #c8bbd9;
      --accent: {color_at(colors, 2)};
      --accent-2: {color_at(colors, 0)};
      --line: rgba(255, 255, 255, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #15131d;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px 20px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .subtitle {{
      color: var(--muted);
      margin: 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: minmax(260px, 420px) 1fr;
      gap: 20px;
      align-items: start;
      margin-bottom: 22px;
    }}
    .meta-table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    .meta-table th,
    .meta-table td {{
      text-align: left;
      padding: 9px 11px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    .meta-table th {{
      color: var(--accent-2);
      width: 42%;
      font-weight: 650;
    }}
    .hint {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 16px 18px;
      min-height: 100%;
    }}
    .hint strong {{ color: var(--accent); }}
    .plot-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      margin: 18px 0;
      padding: 10px;
    }}
    .empty {{
      color: var(--muted);
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 18px;
    }}
    @media (max-width: 860px) {{
      .summary {{ grid-template-columns: 1fr; }}
      header {{ padding-left: 20px; padding-right: 20px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(str(run_name))}</h1>
    <p class="subtitle">Interactive run report for target <strong>{html.escape(str(target))}</strong></p>
  </header>
  <main>
    <section class="summary">
      {metadata_table(metadata, dataset_stats, training, timing)}
      <div class="hint">
        <strong>How to read this:</strong>
        use the core metrics first, then compare animal rates and logprob margins.
        For subliminal runs, the key signal is uplift over neutral controls, not
        the absolute target rate alone. Dataset plots help show whether the
        training data carries any visible structure.
      </div>
    </section>
    {plot_html}
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an interactive HTML report for one run directory.")
    parser.add_argument("run_dir", help="Run directory, e.g. runs/semantic_1k_owl_qwen3b")
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Defaults to <run_dir>/report.html.",
    )
    parser.add_argument(
        "--palette",
        default="vaporwave",
        help="vapeplot palette name, e.g. vaporwave, mallsoft, jazzcup, seapunk.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap for JSONL rows loaded into sample-level plots.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = repo_path(args.run_dir)
    if run_dir is None or not run_dir.exists():
        raise SystemExit(f"Run directory not found: {args.run_dir}")
    output_path = repo_path(args.output) if args.output else run_dir / "report.html"
    result = build_report(run_dir, output_path, args.palette, args.sample_limit)
    print(f"Interactive report written to {result}")


if __name__ == "__main__":
    main()
