"""Reconstruct effective LoRA updates and write a module-level norm baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from slgeo.analysis.delta_weights import (  # noqa: E402
    load_adapter_state_dict,
    module_update_summary,
    compare_lora_updates,
    reconstruct_lora_updates,
)
from slgeo.io import ensure_parent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare-adapter-dir")
    parser.add_argument("--compare-alpha", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_adapter_state_dict(args.adapter_dir)
    updates = reconstruct_lora_updates(state, alpha=args.alpha, rank=args.rank)
    payload = {
        "adapter_dir": str(Path(args.adapter_dir).resolve()),
        "alpha": args.alpha,
        "rank": args.rank,
        "warning": "Norm ranking is a control baseline, not a causal attribution score.",
        "modules": module_update_summary(updates),
    }
    if args.compare_adapter_dir:
        other_state = load_adapter_state_dict(args.compare_adapter_dir)
        other = reconstruct_lora_updates(
            other_state,
            alpha=args.compare_alpha if args.compare_alpha is not None else args.alpha,
            rank=args.rank,
        )
        payload["comparison_adapter_dir"] = str(Path(args.compare_adapter_dir).resolve())
        payload["comparison_modules"] = compare_lora_updates(updates, other)
    output = ensure_parent(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(updates)} reconstructed module updates to {output}")


if __name__ == "__main__":
    main()
