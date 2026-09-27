"""Reproducible benchmark of the phenotype v2 analysis engines (synthetic worlds only; no model output).

For each benchmark world (``slgeo.phenotype.synthetic.BENCHMARK_WORLDS``) and engine: wall and CPU time of
``analysis.p1_family`` and ``p2_family``; then every engine's outputs against the reference engine: exact fields
(decisions, labels, names), p-values (exact) and the largest relative difference of the other numbers, plus the
number of p-values near a decision threshold (Holm 0.025 / 0.05, within 0.02) and whether they agree.

  PYTHONPATH=src .venv/Scripts/python scripts/bench_phenotype_stats.py --nb 499 --nr 20000 --out bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from slgeo.phenotype import analysis, synthetic  # noqa: E402


def _flat(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flat(v, f"{path}[{i}]")
    else:
        yield path, obj


def _is_p(key: str) -> bool:
    return key.endswith(("/p", "/p_two", "k5_p_two")) or "_p/" in key


def compare(ref, other) -> dict:
    fa, fb = dict(_flat(ref)), dict(_flat(other))
    exact_mismatch = p_mismatch = near = near_mismatch = 0
    worst = 0.0
    for key, x in fa.items():
        y = fb[key]
        if isinstance(x, (bool, np.bool_, str)) or x is None:
            exact_mismatch += x != y
        elif _is_p(key):
            p_mismatch += x != y
            if min(abs(x - 0.05), abs(x - 0.025)) < 0.02:
                near += 1
                near_mismatch += x != y
        else:
            worst = max(worst, abs(float(x) - float(y)) / max(1.0, abs(float(x))))
    return {"exact_field_mismatches": exact_mismatch, "p_value_mismatches": p_mismatch,
            "p_values_near_threshold": near, "near_threshold_mismatches": near_mismatch, "max_rel_diff": worst}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nb", type=int, default=499)
    ap.add_argument("--nr", type=int, default=20000)
    ap.add_argument("--engines", default="reference,fast,fast-cuda")
    ap.add_argument("--worlds", default=",".join(synthetic.BENCHMARK_WORLDS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    engines = args.engines.split(",")
    report = {"nb": args.nb, "nr": args.nr, "worlds": {}}
    for name in args.worlds.split(","):
        q = synthetic.world(20260927, synthetic.BENCHMARK_WORLDS[name])
        v1 = synthetic.dev_profile(q)
        outs, times = {}, {}
        for engine in engines:
            if engine == "fast-cuda":  # CUDA context start-up is not part of the per-world cost
                analysis.p2_family(q, synthetic.STEM_IDS, synthetic.FAMILIES, n_boot=19, n_ref=100, engine=engine)
            w0, c0 = time.perf_counter(), time.process_time()
            f1 = analysis.p1_family(q, v1, synthetic.FAMILIES, n_boot=args.nb, n_ref=args.nr, engine=engine)
            w1, c1 = time.perf_counter(), time.process_time()
            f2 = analysis.p2_family(q, synthetic.STEM_IDS, synthetic.FAMILIES, n_boot=args.nb, n_ref=args.nr,
                                    engine=engine)
            w2, c2 = time.perf_counter(), time.process_time()
            outs[engine] = (f1, f2)
            times[engine] = {"p1_wall": w1 - w0, "p1_cpu": c1 - c0, "p2_wall": w2 - w1, "p2_cpu": c2 - c1}
            print(f"{name} {engine}: p1 {w1 - w0:.2f}s p2 {w2 - w1:.2f}s", flush=True)
        base = engines[0]
        report["worlds"][name] = {"timing": times,
                                  "vs_" + base: {e: compare(outs[base], outs[e]) for e in engines[1:]}}
        print(json.dumps(report["worlds"][name]["vs_" + base]), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
