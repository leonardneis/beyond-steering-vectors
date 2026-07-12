"""Compare paired subliminal and neutral behavioral set interventions."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import ensure_parent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sub = json.loads(repo_path(args.subliminal).read_text(encoding="utf-8"))
    neutral = json.loads(repo_path(args.neutral).read_text(encoding="utf-8"))
    sr = {(r["k"], r["set_name"], r["mode"]): r for r in sub["interventions"]}
    nr = {(r["k"], r["set_name"], r["mode"]): r for r in neutral["interventions"]}
    if set(sr) != set(nr):
        raise ValueError("Behavioral intervention sets differ between conditions")
    rows = []
    for key in sorted(sr):
        if sr[key]["modules"] != nr[key]["modules"]:
            raise ValueError(f"Module definitions differ for {key}")
        rows.append(
            {
                "k": key[0],
                "set_name": key[1],
                "mode": key[2],
                "modules": sr[key]["modules"],
                "subliminal_behavioral_effect": sr[key]["behavioral_effect"],
                "neutral_behavioral_effect": nr[key]["behavioral_effect"],
                "trait_specific_behavioral_effect": sr[key]["behavioral_effect"]
                - nr[key]["behavioral_effect"],
            }
        )
    result = {
        "schema_version": 1,
        "analysis": "paired_lora_set_behavior",
        "subliminal_source": str(repo_path(args.subliminal)),
        "neutral_source": str(repo_path(args.neutral)),
        "rows": rows,
    }
    ensure_parent(repo_path(args.output)).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} paired behavioral interventions to {repo_path(args.output)}")


if __name__ == "__main__":
    main()
