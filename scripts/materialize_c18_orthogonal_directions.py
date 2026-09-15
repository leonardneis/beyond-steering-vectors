"""Materialize the preregistered outcome-blind C18 orthogonal directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from slgeo.analysis.teacher_coordinate_interchange import (  # noqa: E402
    deterministic_npz,
    generate_orthogonal_families,
    normalize_rows_float64,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt")
    parser.add_argument("--output", default="research/bidirectional_teacher_coordinate_interchange_v1/ORTHOGONAL_DIRECTIONS.npz")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    teacher_path, output = repo_path(args.teacher), repo_path(args.output)
    artifact = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher = normalize_rows_float64(artifact["unit"][1:28])
    directions = generate_orthogonal_families(teacher, seed=args.seed, family_count=5)
    dot = np.einsum("fsh,sh->fs", directions, teacher.numpy())
    metadata = {
        "schema_version": 1,
        "seed": args.seed,
        "prng": "PCG64",
        "traversal": "family_major_then_slot_major",
        "shape": list(directions.shape),
        "dtype": "float64",
        "teacher_sha256": sha256_file(teacher_path),
        "max_abs_teacher_dot": float(np.max(np.abs(dot))),
    }
    deterministic_npz(output, {
        "directions": directions,
        "family_ids": np.asarray([f"orthogonal_{i}" for i in range(5)]),
        "slots": np.arange(1, 28, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    })
    print(json.dumps({**metadata, "output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
