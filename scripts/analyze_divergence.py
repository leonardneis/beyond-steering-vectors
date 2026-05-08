from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap, repo_path

bootstrap()

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

try:
    import vapeplot
except ImportError:  # pragma: no cover
    vapeplot = None


NUMBER_RE = re.compile(r"\d+")
TOKEN_RE = re.compile(r"\d+|[A-Za-z]+|[^\w\s]", re.UNICODE)
SEPARATOR_RE = re.compile(r"[,\s;|/\\-]+")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def completion_text(record: dict[str, Any]) -> str:
    return str(record.get("filtered_completion") or record.get("completion") or "")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(repo_path(".")))
    except ValueError:
        return str(path)


def resolve_dataset_path(run_or_file: Path, preferred: str) -> Path:
    if run_or_file.is_file():
        return run_or_file
    stats = load_json(run_or_file / "dataset_stats.json")
    candidates = []
    if preferred == "raw":
        candidates.extend(
            [
                stats.get("generation_summary", {}).get("output_path"),
                run_or_file / "samples_raw.jsonl",
            ]
        )
    else:
        candidates.extend(
            [
                stats.get("filter_summary", {}).get("output_path"),
                stats.get("generation_summary", {}).get("output_path"),
                run_or_file / "samples_filtered.jsonl",
                run_or_file / "samples_raw.jsonl",
            ]
        )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = repo_path(path)
        if path and path.exists():
            return path
    raise FileNotFoundError(f"No JSONL dataset found for {run_or_file}")


def run_label(path: Path) -> str:
    if path.is_dir():
        metadata = load_json(path / "metadata.json")
        return metadata.get("run_id") or path.name
    return path.stem


def shannon_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def normalized(counter: Counter[str], vocab: list[str], smoothing: float = 0.0) -> dict[str, float]:
    total = sum(counter.values()) + smoothing * len(vocab)
    if total <= 0:
        return {key: 0.0 for key in vocab}
    return {key: (counter.get(key, 0) + smoothing) / total for key in vocab}


def js_divergence(left: Counter[str], right: Counter[str], smoothing: float = 1e-12) -> float:
    vocab = sorted(set(left) | set(right))
    if not vocab:
        return 0.0
    p = normalized(left, vocab, smoothing=smoothing)
    q = normalized(right, vocab, smoothing=smoothing)
    m = {key: 0.5 * (p[key] + q[key]) for key in vocab}
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    total = 0.0
    for key, value in p.items():
        if value > 0 and q[key] > 0:
            total += value * math.log2(value / q[key])
    return total


def total_variation(left: Counter[str], right: Counter[str]) -> float:
    vocab = sorted(set(left) | set(right))
    p = normalized(left, vocab)
    q = normalized(right, vocab)
    return 0.5 * sum(abs(p[key] - q[key]) for key in vocab)


def top_deltas(left: Counter[str], right: Counter[str], n: int = 25) -> list[dict[str, Any]]:
    vocab = sorted(set(left) | set(right))
    p = normalized(left, vocab)
    q = normalized(right, vocab)
    rows = [
        {
            "item": key,
            "neutral_rate": p[key],
            "subliminal_rate": q[key],
            "delta_subliminal_minus_neutral": q[key] - p[key],
            "neutral_count": left.get(key, 0),
            "subliminal_count": right.get(key, 0),
        }
        for key in vocab
    ]
    rows.sort(key=lambda row: abs(row["delta_subliminal_minus_neutral"]), reverse=True)
    return rows[:n]


def ngrams(items: list[str], n: int) -> Counter[str]:
    if len(items) < n:
        return Counter()
    return Counter(" ".join(items[index : index + n]) for index in range(len(items) - n + 1))


def char_ngrams(text: str, n: int) -> Counter[str]:
    if len(text) < n:
        return Counter()
    return Counter(text[index : index + n] for index in range(len(text) - n + 1))


def separator_signature(text: str) -> str:
    separators = SEPARATOR_RE.findall(text)
    parts = []
    for sep in separators:
        label = sep
        label = label.replace(" ", "S")
        label = label.replace("\t", "T")
        label = label.replace("\n", "N")
        parts.append(label)
    return "|".join(parts) if parts else "none"


def number_width_pattern(text: str) -> str:
    numbers = NUMBER_RE.findall(text)
    return "-".join(str(len(number)) for number in numbers) if numbers else "none"


def formatting_pattern(text: str) -> str:
    masked = NUMBER_RE.sub("#", text.strip())
    masked = re.sub(r"\s+", " ", masked)
    return masked or "empty"


def extract_features(records: list[dict[str, Any]], rare_threshold: int) -> dict[str, Any]:
    texts = [completion_text(record) for record in records]
    all_text = "\n".join(texts)
    tokens_by_text = [TOKEN_RE.findall(text.lower()) for text in texts]
    flat_tokens = [token for tokens in tokens_by_text for token in tokens]
    digit_counter = Counter(ch for ch in all_text if ch.isdigit())
    separator_counter = Counter(ch for ch in all_text if not ch.isdigit() and not ch.isalnum())
    token_counter = Counter(flat_tokens)
    completion_counter = Counter(texts)
    formatting_counter = Counter(formatting_pattern(text) for text in texts)
    separator_signature_counter = Counter(separator_signature(text) for text in texts)
    number_width_counter = Counter(number_width_pattern(text) for text in texts)
    token_bigram_counter = Counter()
    token_trigram_counter = Counter()
    char_bigram_counter = Counter()
    char_trigram_counter = Counter()
    repeated_number_pattern_counter = Counter()
    lengths = []
    token_lengths = []
    number_counts = []
    unique_number_counts = []
    for text, tokens in zip(texts, tokens_by_text):
        numbers = NUMBER_RE.findall(text)
        lengths.append(len(text))
        token_lengths.append(len(tokens))
        number_counts.append(len(numbers))
        unique_number_counts.append(len(set(numbers)))
        token_bigram_counter.update(ngrams(tokens, 2))
        token_trigram_counter.update(ngrams(tokens, 3))
        char_bigram_counter.update(char_ngrams(text, 2))
        char_trigram_counter.update(char_ngrams(text, 3))
        if len(numbers) != len(set(numbers)):
            repeated_number_pattern_counter[text] += 1
    rare_tokens = Counter({key: value for key, value in token_counter.items() if value <= rare_threshold})
    return {
        "record_count": len(records),
        "text_count": len(texts),
        "total_characters": sum(lengths),
        "total_tokens": len(flat_tokens),
        "completion_lengths": lengths,
        "token_lengths": token_lengths,
        "number_counts": number_counts,
        "unique_number_counts": unique_number_counts,
        "digit_frequencies": digit_counter,
        "separator_frequencies": separator_counter,
        "token_frequencies": token_counter,
        "rare_token_frequencies": rare_tokens,
        "completion_frequencies": completion_counter,
        "formatting_patterns": formatting_counter,
        "separator_patterns": separator_signature_counter,
        "number_width_patterns": number_width_counter,
        "token_bigrams": token_bigram_counter,
        "token_trigrams": token_trigram_counter,
        "char_bigrams": char_bigram_counter,
        "char_trigrams": char_trigram_counter,
        "repeated_number_patterns": repeated_number_pattern_counter,
        "entropies": {
            "digits": shannon_entropy(digit_counter),
            "separators": shannon_entropy(separator_counter),
            "tokens": shannon_entropy(token_counter),
            "formatting_patterns": shannon_entropy(formatting_counter),
            "token_bigrams": shannon_entropy(token_bigram_counter),
            "token_trigrams": shannon_entropy(token_trigram_counter),
        },
    }


def summarize_numeric(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    series = pd.Series(values)
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def wasserstein_1d(left: list[int | float], right: list[int | float]) -> float | None:
    if not left or not right:
        return None
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(left, right))
    except Exception:
        return None


def ks_2sample(left: list[int | float], right: list[int | float]) -> dict[str, float] | None:
    if not left or not right:
        return None
    try:
        from scipy.stats import ks_2samp

        result = ks_2samp(left, right)
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
    except Exception:
        return None


def compare_counter_feature(name: str, neutral: Counter[str], subliminal: Counter[str]) -> dict[str, Any]:
    return {
        "feature": name,
        "neutral_total": sum(neutral.values()),
        "subliminal_total": sum(subliminal.values()),
        "neutral_unique": len(neutral),
        "subliminal_unique": len(subliminal),
        "neutral_entropy": shannon_entropy(neutral),
        "subliminal_entropy": shannon_entropy(subliminal),
        "entropy_delta_subliminal_minus_neutral": shannon_entropy(subliminal) - shannon_entropy(neutral),
        "js_divergence_bits": js_divergence(neutral, subliminal),
        "total_variation": total_variation(neutral, subliminal),
        "top_absolute_deltas": top_deltas(neutral, subliminal),
    }


def serializable_counter(counter: Counter[str], n: int = 50) -> list[dict[str, Any]]:
    return [{"item": key, "count": value} for key, value in counter.most_common(n)]


def build_report(
    neutral_path: Path,
    subliminal_path: Path,
    preferred: str,
    rare_threshold: int,
    topn: int,
) -> dict[str, Any]:
    neutral_dataset = resolve_dataset_path(neutral_path, preferred)
    subliminal_dataset = resolve_dataset_path(subliminal_path, preferred)
    neutral_records = load_jsonl(neutral_dataset)
    subliminal_records = load_jsonl(subliminal_dataset)
    neutral_features = extract_features(neutral_records, rare_threshold)
    subliminal_features = extract_features(subliminal_records, rare_threshold)
    counter_features = [
        "digit_frequencies",
        "separator_frequencies",
        "token_frequencies",
        "rare_token_frequencies",
        "completion_frequencies",
        "formatting_patterns",
        "separator_patterns",
        "number_width_patterns",
        "token_bigrams",
        "token_trigrams",
        "char_bigrams",
        "char_trigrams",
        "repeated_number_patterns",
    ]
    comparisons = {
        feature: compare_counter_feature(
            feature,
            neutral_features[feature],
            subliminal_features[feature],
        )
        for feature in counter_features
    }
    numeric_features = ["completion_lengths", "token_lengths", "number_counts", "unique_number_counts"]
    numeric_comparisons = {}
    for feature in numeric_features:
        neutral_values = neutral_features[feature]
        subliminal_values = subliminal_features[feature]
        numeric_comparisons[feature] = {
            "neutral": summarize_numeric(neutral_values),
            "subliminal": summarize_numeric(subliminal_values),
            "mean_delta_subliminal_minus_neutral": (
                summarize_numeric(subliminal_values).get("mean", 0.0)
                - summarize_numeric(neutral_values).get("mean", 0.0)
            ),
            "wasserstein_distance": wasserstein_1d(neutral_values, subliminal_values),
            "ks_2sample": ks_2sample(neutral_values, subliminal_values),
        }
    ranked = sorted(
        comparisons.values(),
        key=lambda row: row["js_divergence_bits"],
        reverse=True,
    )
    surface_feature_names = [
        "digit_frequencies",
        "separator_frequencies",
        "formatting_patterns",
        "separator_patterns",
        "number_width_patterns",
        "token_frequencies",
    ]
    surface_ranked = sorted(
        [comparisons[name] for name in surface_feature_names],
        key=lambda row: row["js_divergence_bits"],
        reverse=True,
    )
    return {
        "neutral": {
            "label": run_label(neutral_path),
            "input": str(neutral_path),
            "dataset_path": str(neutral_dataset),
            "record_count": len(neutral_records),
            "feature_summary": {
                "entropies": neutral_features["entropies"],
                "top_digits": serializable_counter(neutral_features["digit_frequencies"], topn),
                "top_tokens": serializable_counter(neutral_features["token_frequencies"], topn),
                "top_formatting_patterns": serializable_counter(neutral_features["formatting_patterns"], topn),
                "top_repeated_completions": serializable_counter(neutral_features["completion_frequencies"], topn),
            },
        },
        "subliminal": {
            "label": run_label(subliminal_path),
            "input": str(subliminal_path),
            "dataset_path": str(subliminal_dataset),
            "record_count": len(subliminal_records),
            "feature_summary": {
                "entropies": subliminal_features["entropies"],
                "top_digits": serializable_counter(subliminal_features["digit_frequencies"], topn),
                "top_tokens": serializable_counter(subliminal_features["token_frequencies"], topn),
                "top_formatting_patterns": serializable_counter(subliminal_features["formatting_patterns"], topn),
                "top_repeated_completions": serializable_counter(subliminal_features["completion_frequencies"], topn),
            },
        },
        "numeric_comparisons": numeric_comparisons,
        "distribution_comparisons": comparisons,
        "ranked_divergences": [
            {
                "feature": row["feature"],
                "js_divergence_bits": row["js_divergence_bits"],
                "total_variation": row["total_variation"],
                "entropy_delta_subliminal_minus_neutral": row[
                    "entropy_delta_subliminal_minus_neutral"
                ],
            }
            for row in ranked
        ],
        "surface_channel_divergences": [
            {
                "feature": row["feature"],
                "js_divergence_bits": row["js_divergence_bits"],
                "total_variation": row["total_variation"],
                "entropy_delta_subliminal_minus_neutral": row[
                    "entropy_delta_subliminal_minus_neutral"
                ],
            }
            for row in surface_ranked
        ],
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(path: Path, report: dict[str, Any]) -> None:
    ranked = report["ranked_divergences"][:10]
    surface_ranked = report.get("surface_channel_divergences", [])[:10]
    numeric_rows = []
    for feature, row in report["numeric_comparisons"].items():
        numeric_rows.append(
            {
                "feature": feature,
                "neutral_mean": row["neutral"].get("mean"),
                "subliminal_mean": row["subliminal"].get("mean"),
                "mean_delta": row["mean_delta_subliminal_minus_neutral"],
                "ks_statistic": (row.get("ks_2sample") or {}).get("statistic"),
                "ks_p_value": (row.get("ks_2sample") or {}).get("p_value"),
            }
        )
    strongest = ranked[0] if ranked else None
    strongest_surface = surface_ranked[0] if surface_ranked else None
    interpretation = (
        "No distribution features were available."
        if strongest is None
        else (
            f"Strongest measured divergence is `{strongest['feature']}` "
            f"(JS={strongest['js_divergence_bits']:.6g} bits, "
            f"TV={strongest['total_variation']:.6g})."
        )
    )
    surface_interpretation = (
        "No surface-channel features were available."
        if strongest_surface is None
        else (
            f"Strongest surface-channel divergence is `{strongest_surface['feature']}` "
            f"(JS={strongest_surface['js_divergence_bits']:.6g} bits, "
            f"TV={strongest_surface['total_variation']:.6g})."
        )
    )
    lines = [
        "# Divergence Summary",
        "",
        "## Inputs",
        "",
        f"- Neutral: `{report['neutral']['label']}`",
        f"- Neutral dataset: `{report['neutral']['dataset_path']}`",
        f"- Subliminal: `{report['subliminal']['label']}`",
        f"- Subliminal dataset: `{report['subliminal']['dataset_path']}`",
        f"- Neutral records: `{report['neutral']['record_count']}`",
        f"- Subliminal records: `{report['subliminal']['record_count']}`",
        "",
        "## Main Result",
        "",
        interpretation,
        "",
        surface_interpretation,
        "",
        "Exact completion and rare-token divergences can be large for random number",
        "lists simply because many full lists and individual numbers are unique. For",
        "evidence of a usable subliminal channel, prioritize stable surface features",
        "such as digit rates, separator/format patterns, length distributions, and",
        "consistent token/ngram shifts.",
        "",
        "Use this as evidence about whether the teacher's hidden trait leaves a measurable",
        "statistical footprint in the number-only generations before student training.",
        "",
        "## Ranked Distribution Divergences",
        "",
        markdown_table(
            ranked,
            ["feature", "js_divergence_bits", "total_variation", "entropy_delta_subliminal_minus_neutral"],
        ),
        "",
        "## Surface-Channel Divergences",
        "",
        markdown_table(
            surface_ranked,
            ["feature", "js_divergence_bits", "total_variation", "entropy_delta_subliminal_minus_neutral"],
        ),
        "",
        "## Numeric Features",
        "",
        markdown_table(
            numeric_rows,
            ["feature", "neutral_mean", "subliminal_mean", "mean_delta", "ks_statistic", "ks_p_value"],
        ),
        "",
        "## Notes",
        "",
        "- JS divergence is reported in bits; 0 means identical empirical distributions.",
        "- Total variation is bounded between 0 and 1.",
        "- Old runs may only contain sampled run artifacts; this script prefers the full generated/filtered JSONL paths from `dataset_stats.json` when present.",
        "- For causal interpretation, compare paired neutral/subliminal runs with matched model, seed schedule, prompt style, decoding settings, and sample count.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_palette() -> list[str]:
    if vapeplot is not None:
        try:
            return list(vapeplot.palette("vaporwave"))
        except Exception:
            pass
    return ["#94D0FF", "#FF6AD5", "#20DE8B", "#FFDE8B", "#966BFF"]


def color_at(colors: list[str], index: int) -> str:
    return colors[index % len(colors)]


def write_plots(path: Path, report: dict[str, Any]) -> None:
    colors = plot_palette()
    figures = []
    ranked = report["ranked_divergences"]
    if ranked:
        top = ranked[:12]
        fig = go.Figure()
        fig.add_bar(
            x=[row["js_divergence_bits"] for row in top],
            y=[row["feature"] for row in top],
            orientation="h",
            marker_color=[color_at(colors, idx) for idx, _ in enumerate(top)],
        )
        fig.update_layout(
            title="Ranked Distribution Divergences",
            xaxis_title="JS divergence bits",
            yaxis_title="feature",
            yaxis={"autorange": "reversed"},
            template="plotly_dark",
            height=520,
        )
        figures.append(fig)
    digit_rows = report["distribution_comparisons"]["digit_frequencies"]["top_absolute_deltas"]
    if digit_rows:
        digit_rows = sorted(digit_rows, key=lambda row: row["item"])
        fig = go.Figure()
        fig.add_bar(name="neutral", x=[row["item"] for row in digit_rows], y=[row["neutral_rate"] for row in digit_rows])
        fig.add_bar(
            name="subliminal",
            x=[row["item"] for row in digit_rows],
            y=[row["subliminal_rate"] for row in digit_rows],
        )
        fig.update_layout(
            title="Digit Frequencies",
            barmode="group",
            yaxis_title="rate",
            template="plotly_dark",
            height=460,
        )
        figures.append(fig)
    numeric = report["numeric_comparisons"]
    if numeric:
        rows = [
            {
                "feature": feature,
                "neutral_mean": values["neutral"].get("mean"),
                "subliminal_mean": values["subliminal"].get("mean"),
            }
            for feature, values in numeric.items()
        ]
        fig = go.Figure()
        fig.add_bar(name="neutral", x=[row["feature"] for row in rows], y=[row["neutral_mean"] for row in rows])
        fig.add_bar(
            name="subliminal",
            x=[row["feature"] for row in rows],
            y=[row["subliminal_mean"] for row in rows],
        )
        fig.update_layout(
            title="Numeric Feature Means",
            barmode="group",
            yaxis_title="mean",
            template="plotly_dark",
            height=460,
        )
        figures.append(fig)
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Divergence Plots</title></head>",
        "<body style='margin:0;background:#120f19;color:#f7f2ff;font-family:Segoe UI,Arial,sans-serif'>",
        "<main style='max-width:1200px;margin:0 auto;padding:24px'>",
        "<h1>Divergence Plots</h1>",
    ]
    for index, fig in enumerate(figures):
        html_parts.append(to_html(fig, full_html=False, include_plotlyjs="cdn" if index == 0 else False))
    html_parts.extend(["</main></body></html>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(html_parts), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze statistical divergence between neutral and subliminal number generations."
    )
    parser.add_argument("neutral", help="Neutral run directory or JSONL file.")
    parser.add_argument("subliminal", help="Subliminal run directory or JSONL file.")
    parser.add_argument("--name", default=None, help="Output artifact name.")
    parser.add_argument("--output-dir", default="runs/divergence", help="Directory for output artifacts.")
    parser.add_argument(
        "--dataset",
        choices=["filtered", "raw"],
        default="filtered",
        help="Prefer filtered or raw generation JSONL when a run directory is supplied.",
    )
    parser.add_argument("--rare-threshold", type=int, default=1, help="Count tokens at or below this frequency as rare.")
    parser.add_argument("--topn", type=int, default=50, help="Number of top examples to keep in summaries.")
    parser.add_argument("--plots", action="store_true", help="Also write divergence_plots.html.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    neutral = repo_path(args.neutral)
    subliminal = repo_path(args.subliminal)
    if neutral is None or not neutral.exists():
        raise SystemExit(f"Neutral input not found: {args.neutral}")
    if subliminal is None or not subliminal.exists():
        raise SystemExit(f"Subliminal input not found: {args.subliminal}")
    name = args.name or f"{neutral.name}_vs_{subliminal.name}"
    output_dir = repo_path(args.output_dir) / name
    report = build_report(
        neutral_path=neutral,
        subliminal_path=subliminal,
        preferred=args.dataset,
        rare_threshold=args.rare_threshold,
        topn=args.topn,
    )
    json_path = output_dir / "divergence_report.json"
    summary_path = output_dir / "divergence_summary.md"
    write_json(json_path, report)
    write_summary(summary_path, report)
    plot_path = None
    if args.plots:
        plot_path = output_dir / "divergence_plots.html"
        write_plots(plot_path, report)
    print(f"Divergence report written to {json_path}")
    print(f"Divergence summary written to {summary_path}")
    if plot_path:
        print(f"Divergence plots written to {plot_path}")


if __name__ == "__main__":
    main()
