"""Compare subliminal and control student vectors against one frozen teacher vector."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.activations import alignment_metrics  # noqa: E402
from slgeo.analysis.vector_artifacts import load_vector_artifact, sha256_file  # noqa: E402
from slgeo.io import ensure_parent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--subliminal-student", required=True)
    parser.add_argument("--control-student", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "teacher": repo_path(args.teacher),
        "subliminal_student": repo_path(args.subliminal_student),
        "control_student": repo_path(args.control_student),
    }
    artifacts = {name: load_vector_artifact(path) for name, path in paths.items()}
    teacher = artifacts["teacher"]["raw"]
    subliminal = artifacts["subliminal_student"]["raw"]
    control = artifacts["control_student"]["raw"]
    sub_metrics = alignment_metrics(subliminal, teacher)
    control_metrics = alignment_metrics(control, teacher)
    if teacher.shape != subliminal.shape or teacher.shape != control.shape:
        raise ValueError("Teacher, subliminal, and control vectors must share a shape")

    rows = []
    for slot in range(teacher.shape[0]):
        rows.append(
            {
                "hidden_state_slot": slot,
                "transformer_block": slot - 1 if slot else None,
                "subliminal_cosine": float(sub_metrics["cosine"][slot]),
                "control_cosine": float(control_metrics["cosine"][slot]),
                "cosine_difference": float(
                    sub_metrics["cosine"][slot] - control_metrics["cosine"][slot]
                ),
                "subliminal_signed_projection": float(
                    sub_metrics["signed_projection"][slot]
                ),
                "control_signed_projection": float(
                    control_metrics["signed_projection"][slot]
                ),
                "signed_projection_difference": float(
                    sub_metrics["signed_projection"][slot]
                    - control_metrics["signed_projection"][slot]
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "indexing": "slot 0=embedding; slot i+1=transformer block i",
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "shape": list(teacher.shape),
        "mean_block_cosine_subliminal": float(sub_metrics["cosine"][1:].mean()),
        "mean_block_cosine_control": float(control_metrics["cosine"][1:].mean()),
        "mean_block_cosine_difference": float(
            (sub_metrics["cosine"][1:] - control_metrics["cosine"][1:]).mean()
        ),
        "rows": rows,
    }
    output = ensure_parent(repo_path(args.output))
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
