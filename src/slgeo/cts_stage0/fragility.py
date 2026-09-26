"""In-run decision-level fragility check (v2 spec ``integrity.fragility``; PREREGISTRATION §9.2, §13.1).

Every gating or label statistic whose conditions were all re-scored in the reference layout L1 is recomputed
from the L1 scores with the very builder that produced it (``criteria.Context``), and standardized:

- CI-based and sign statistics: z_S = (S_L2 - S_L1) / SE_boot(S_L2);
- re-scored null conditions of family F: z_j = (T_j,L2 - T_j,L1) / SD_F, SD_F over all members of F in L2.

PASS iff RMS(z) <= epsilon and max |z| <= epsilon * Phi^-1(1 - 0.05 / (2 m)), epsilon = 0.04.
The persisted report holds counts, the two aggregates, the guard and the verdict only; never a statistic.
"""

from __future__ import annotations

import math
from statistics import NormalDist
import numpy as np

from . import statistics as st
from .criteria import Context, WordScores
from .errors import FinalFailure

EPSILON = 0.04
FAMILY_LEVEL = 0.05
RESCORED_PER_NULL_FAMILY = 10
NULL_FAMILIES = (
    "null:c_cat_dog", "null:c_cat_wolf", "null:c_cat_anim",
    "rcov:c_cat_dog", "rcov:c_cat_wolf", "rcov:c_cat_anim",
    "rcov_pc:last",
)


class FragilityError(RuntimeError, FinalFailure):
    pass


def guard(m: int) -> float:
    if m < 1:
        raise FragilityError("No statistic to check")
    return EPSILON * NormalDist().inv_cdf(1.0 - FAMILY_LEVEL / (2 * m))


def fragility_report(l2: Context, l1_scores: WordScores, rescored: set[str]) -> dict:
    """Standardized L2-vs-L1 shifts of every statistic computable from re-scored conditions."""
    l1 = Context(l1_scores, l2.family, l2.index, l2.baselines)
    z_ci: list[float] = []
    for label in sorted(l2.recorded):
        record = l2.recorded[label]
        if not record.cids <= rescored:
            continue
        if not record.se_boot > 0 or not math.isfinite(record.se_boot):
            raise FragilityError(f"Non-positive SE_boot for {label}")
        y, used = l1._build(record.builder)
        if used != record.cids:
            raise FragilityError(f"{label} reads different conditions in the reference layout")
        z_ci.append((record.point - st.point(y)) / record.se_boot)
    z_null: list[float] = []
    per_family: dict[str, int] = {}
    for family in NULL_FAMILIES:
        members = l2.null_families.get(family)
        if not members:
            raise FragilityError(f"Null family {family} was not recorded")
        values = np.asarray([value for (_b, _u, value) in members.values()], dtype=np.float64)
        sd = float(np.std(values, ddof=1))
        if not sd > 0:
            raise FragilityError(f"Zero spread in null family {family}")
        count = 0
        for label in sorted(members):
            builder, used, value = members[label]
            if not used <= rescored:
                continue
            y, used_l1 = l1._build(builder)
            if used_l1 != used:
                raise FragilityError(f"{family}/{label} reads different conditions in the reference layout")
            z_null.append((value - st.point(y)) / sd)
            count += 1
        if count != RESCORED_PER_NULL_FAMILY:
            raise FragilityError(f"{family}: {count} re-scored members instead of {RESCORED_PER_NULL_FAMILY}")
        per_family[family] = count
    z = np.asarray(z_ci + z_null, dtype=np.float64)
    if not np.isfinite(z).all():
        raise FragilityError("Non-finite standardized shift")
    m = int(z.size)
    rms = float(np.sqrt(np.mean(z * z)))
    max_abs = float(np.max(np.abs(z)))
    limit = guard(m)
    return {
        "epsilon": EPSILON,
        "m": m,
        "m_ci_and_sign_statistics": len(z_ci),
        "m_null_conditions": len(z_null),
        "null_members_per_family": per_family,
        "rms_z": rms,
        "max_abs_z": max_abs,
        "max_guard": limit,
        "rms_pass": bool(rms <= EPSILON),
        "max_pass": bool(max_abs <= limit),
        "pass": bool(rms <= EPSILON and max_abs <= limit),
    }


def stage_fragility(ctx, shard_id: str = "fragility") -> None:
    """CPU stage after every score and re-score shard; its verdict enters the integrity report."""
    from . import artifacts as art
    from .analysis import baseline_map, family_index_from_partition, load_word_scores, reliabilities_of
    from .criteria import all_statistics
    from .pipeline import _begin, load_bundle, log, verify_runtime
    from .plan import null_names

    shard = _begin(ctx, shard_id)
    if shard is None:
        return
    runtime = verify_runtime(ctx, gpu=False)
    scores, _extras = load_word_scores(ctx, stage="score")
    l1_scores, _ = load_word_scores(ctx, stage="rescore")
    family = family_index_from_partition(ctx.package.partition_ids())
    bundle, _ = load_bundle(ctx)
    conditions = ctx.conditions()
    l2 = all_statistics(scores, family, reliabilities_of(bundle), null_names(ctx.contract), baseline_map(conditions))
    rescored = {cid for cid, condition in conditions.items() if condition.reference_rescore}
    if set(l1_scores.values) != rescored:
        raise FragilityError("Re-scored conditions differ from the registry flags")
    report = fragility_report(l2, l1_scores, rescored)
    shard.publish({"fragility.json": art.pretty_json(report)}, {"stage": "fragility", "runtime": runtime, "pass": report["pass"]})
    log(f"{shard_id}: {'PASS' if report['pass'] else 'FAIL'} (m={report['m']})")

