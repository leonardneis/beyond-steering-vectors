"""Aggregate top-k intervention summaries across prompt splits and control draws."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import ensure_parent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    grouped = defaultdict(list)
    sources = []
    for source in args.paired:
        path = repo_path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        for row in data["rows"]:
            grouped[row["k"], row["set_name"], row["mode"]].append(row)
    rows = []
    for key, members in sorted(grouped.items()):
        global_values = np.asarray([r["trait_specific_global_effect_mean"] for r in members])
        terminal_values = np.asarray([r["trait_specific_terminal_effect_mean"] for r in members])
        rows.append(
            {
                "k": key[0],
                "set_name": key[1],
                "mode": key[2],
                "replicates": len(members),
                "global_mean_across_replicates": float(global_values.mean()),
                "global_sd_across_replicates": float(global_values.std(ddof=1)) if len(members) > 1 else None,
                "global_range": [float(global_values.min()), float(global_values.max())],
                "terminal_mean_across_replicates": float(terminal_values.mean()),
                "terminal_sd_across_replicates": float(terminal_values.std(ddof=1)) if len(members) > 1 else None,
                "terminal_range": [float(terminal_values.min()), float(terminal_values.max())],
            }
        )
    result = {"schema_version": 1, "analysis": "topk_replicate_aggregate", "sources": sources, "rows": rows}
    ensure_parent(repo_path(args.output)).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Aggregated {len(sources)} paired files into {repo_path(args.output)}")


if __name__ == "__main__":
    main()
