"""Outcome taxonomy of the Phenotype Anchor (preregistration draft ``research/phenotype_anchor_v1/PREREGISTRATION.md`` §7 and §9).

The driver maps per-seed run-level test results to exactly one summary class plus modifiers, by the first matching
rank. It never changes a statistic; it only reads p-values and booleans that ``stats`` produced.

v2 (after the v1 final statistics audit): confirmatory families P1 {C2, C3} and P2 {K1, K2, K3}; C3, K1 and K2 are
the robust trait-residual claims (tempering residual and mass-matched contrast); no adequacy pretest, no run gate
(the run level is inside every p-value), no profile or rendering modifiers, no absence class and no "generic" class
(decisions R1, R4: no equivalence margin is pre-declared). Class names state what was confirmed; a split between two
seeds is reported as such, not as tested heterogeneity. Confirmation = conjunction over the confirmatory seeds (R3:
replication across independent training runs). Fresh seeds 4/5 (R2): a second confirmatory pair for the cat claim
only, each stage at alpha, worst-case false-claim bound 2 alpha (``classify_p1_two_stage``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .stats import ALPHA, seed_passes

TWO_STAGE_BOUND_NOTE = ("two-stage cat claim: each stage at alpha; worst-case false-claim bound 2 alpha (decision R2, "
                        "size under the modelled nulls reported by the statistics audit)")

P1_HYPOTHESES = ("C2", "C3")
P2_HYPOTHESES = ("K1", "K2", "K3")


@dataclass(frozen=True)
class SeedResult:
    """One confirmatory seed. ``p``: one-sided run-level p-values of C2 and C3 (C3 in the robust form);
    ``label``: cat-dominance label."""

    p: Mapping[str, float]
    label: bool


@dataclass(frozen=True)
class Outcome:
    cls: str
    modifiers: tuple[str, ...]
    confirmed: dict[str, bool]
    per_seed: dict[str, dict[str, bool]]
    notes: tuple[str, ...] = field(default=())


def technical_fail(reason: str) -> Outcome:
    """TECHNICAL_FAIL with its documented reason (integrity, a missing required arm or seed, a degenerate fit)."""
    return Outcome("TECHNICAL_FAIL", (), {}, {}, (reason,))


def outcome_from_dict(data: Mapping) -> Outcome:
    """Inverse of ``Outcome.__dict__`` after a JSON round trip (stored stage results)."""
    return Outcome(str(data["cls"]), tuple(data["modifiers"]), dict(data["confirmed"]),
                   {s: dict(v) for s, v in data["per_seed"].items()}, tuple(data.get("notes", ())))


def classify_p1(
    results: Mapping[str, SeedResult], *, integrity_ok: bool, instrument_ok: bool, c3_upper: float | None = None,
    alpha: float = ALPHA,
) -> Outcome:
    """P1 summary class (first match). ``c3_upper``: the run-level upper bound of the cat residual (pooled over the
    confirmatory seeds, larger of the two robust components), quoted by the bound reading."""
    if not integrity_ok:
        return Outcome("TECHNICAL_FAIL", (), {}, {})
    if not instrument_ok:
        return Outcome("INSTRUMENT_FAIL", (), {}, {})
    per_seed = {s: seed_passes({h: r.p[h] for h in P1_HYPOTHESES}, alpha=alpha) for s, r in results.items()}
    both = {h: all(per_seed[s][h] for s in per_seed) for h in P1_HYPOTHESES}
    one = {h: sum(per_seed[s][h] for s in per_seed) == 1 for h in P1_HYPOTHESES}
    label = all(r.label for r in results.values())
    notes = []

    if both["C3"] and label:
        cls = "CAT_DOMINANT"
    elif both["C3"]:
        cls = "CAT_RESIDUAL_NOT_DOMINANT"
    elif both["C2"]:
        cls = "FLATTENING_CAT_NOT_DETECTED"
    elif any(one.values()):
        cls = "ONE_SEED_ONLY"
    else:
        cls = "NO_CONFIRMED_C2_C3"
    if not both["C3"] and c3_upper is not None:
        notes.append(f"cat residual at most {c3_upper:.4f} (run-level upper bound, pooled seeds)")
    if one["C3"]:
        notes.append("C3 confirmed in exactly one seed (fresh-seed trigger)")
    return Outcome(cls, (), both, per_seed, tuple(notes))


def fresh_seed_trigger(outcome: Outcome) -> bool:
    """Seeds 4/5 are trained iff the robust C3 claim is confirmed in exactly one of the confirmatory seeds."""
    return sum(bool(p.get("C3")) for p in outcome.per_seed.values()) == 1


def classify_p1_two_stage(stage1: Outcome, stage2: Outcome | None) -> Outcome:
    """Final P1 outcome under the fresh-seed rule (decision R2).

    Without a trigger the stage-1 outcome is final (a stage-2 outcome is then a protocol violation). With a trigger:
    stage 2 not yet analysed or failed technically / instrumentally -> the stage-1 class stands (cat claim
    unresolved, noted); C3 confirmed in both seeds 4 and 5 -> CAT_RESIDUAL_CONFIRMED_ON_REPLICATION, with the
    modifier CAT_DOMINANT_ON_REPLICATION when the dominance label also holds in both seeds 4 and 5 (the §7 label
    read on the replication pair); otherwise the stage-1 class stands (noted). C2 and every other decision stay those
    of stage 1; per-seed results of both stages are merged."""
    if not fresh_seed_trigger(stage1):
        if stage2 is not None:
            raise ValueError("Seeds 4/5 analysed although the fresh-seed trigger did not fire")
        return stage1
    if stage2 is None:
        return Outcome(stage1.cls, stage1.modifiers, stage1.confirmed, stage1.per_seed,
                       stage1.notes + ("fresh-seed trigger fired; seeds 4/5 not analysed: cat claim unresolved",))
    per_seed = {**stage1.per_seed, **stage2.per_seed}
    if stage2.cls in ("TECHNICAL_FAIL", "INSTRUMENT_FAIL"):
        note = f"seeds 4/5 stage {stage2.cls}: cat claim unresolved"
        return Outcome(stage1.cls, stage1.modifiers, stage1.confirmed, per_seed, stage1.notes + (note,) + stage2.notes)
    if stage2.confirmed.get("C3"):
        modifiers = ("CAT_DOMINANT_ON_REPLICATION",) if stage2.cls == "CAT_DOMINANT" else ()
        kept = tuple(n for n in stage1.notes if not n.startswith("cat residual at most"))  # superseded seed-2/3 bound
        return Outcome("CAT_RESIDUAL_CONFIRMED_ON_REPLICATION", modifiers, {**stage1.confirmed, "C3": True}, per_seed,
                       kept + (f"C3 confirmed in seeds 4 and 5 (stage-2 class {stage2.cls}); seed-2/3 class "
                               f"{stage1.cls}, seed-2/3 C2 decision {stage1.confirmed.get('C2')}",
                               TWO_STAGE_BOUND_NOTE))
    return Outcome(stage1.cls, stage1.modifiers, stage1.confirmed, per_seed,
                   stage1.notes + (f"seeds 4/5 did not confirm C3 (stage-2 class {stage2.cls})", TWO_STAGE_BOUND_NOTE))


def two_stage_cat_claim(stage1: Outcome, stage2: Outcome | None) -> bool:
    """P1 cat claim under the fresh-seed rule: confirmed in seeds 2 and 3, or (trigger fired and) confirmed in seeds
    4 and 5 (class CAT_RESIDUAL_CONFIRMED_ON_REPLICATION). Worst-case false-claim bound 2 alpha."""
    return bool(classify_p1_two_stage(stage1, stage2).confirmed.get("C3"))


P2B_ENTROPY_TOLERANCE = 0.05  # nats (PREREGISTRATION §9.4)


def p2b_trigger(k3_confirmed: bool, dog_entropy: float | None, neutral_entropy: float | None) -> bool | None:
    """P2b (§9.4): train an entropy-matched neutral teacher iff K3 is confirmed and the dog-teacher data entropy is
    not within 0.05 nats of the neutral value. None when K3 is confirmed but the CPU entropy record is missing."""
    if not k3_confirmed:
        return False
    if dog_entropy is None or neutral_entropy is None:
        return None
    return bool(abs(float(dog_entropy) - float(neutral_entropy)) > P2B_ENTROPY_TOLERANCE)


def classify_p2(
    results: Mapping[str, Mapping[str, float]], *, label: Mapping[str, bool], dog_transfer: Mapping[str, bool],
    c2_confirmed: bool, integrity_ok: bool, instrument_ok: bool, alpha: float = ALPHA,
) -> Outcome:
    """P2 summary class. ``results[seed]``: one-sided run-level p-values of K1-K3 (K1, K2 robust); ``label[seed]``:
    cat above every non-trait word in S vs D; ``dog_transfer[seed]``: the D-vs-N check passes in that seed;
    ``c2_confirmed``: the P1 decision on C2 (both seeds); ``instrument_ok``: coverage and agreement pass on every
    confirmatory arm P2 compares (D2, D3, T_dog here, and the P1 arms through the P1 outcome).

    No class asserts "generic" (decision R4): FLATTENING_BOTH_TEACHERS states that both teachers' students have a
    flatter prefix-averaged answer distribution (R5), not that the two effects are equal."""
    if not integrity_ok:
        return Outcome("TECHNICAL_FAIL", (), {}, {})
    if not instrument_ok:
        return Outcome("INSTRUMENT_FAIL", (), {}, {})
    per_seed = {s: seed_passes({h: p[h] for h in P2_HYPOTHESES}, alpha=alpha) for s, p in results.items()}
    both = {h: all(per_seed[s][h] for s in per_seed) for h in P2_HYPOTHESES}
    if both["K1"] and both["K2"]:
        cls = "DOUBLE_DISSOCIATION"
    elif both["K1"]:
        cls = "CAT_ONLY_SPECIFIC"
    elif both["K2"]:
        cls = "DOG_ONLY_SPECIFIC"
    elif both["K3"] and c2_confirmed:
        cls = "FLATTENING_BOTH_TEACHERS"
    elif not any(dog_transfer.values()):
        cls = "NO_DETECTED_DOG_TRANSFER"
    else:
        cls = "P2_NULL_OR_MIXED"
    modifiers = ("CAT_WORD_DOMINANT",) if both["K1"] and all(label.values()) else ()
    return Outcome(cls, modifiers, both, per_seed)
