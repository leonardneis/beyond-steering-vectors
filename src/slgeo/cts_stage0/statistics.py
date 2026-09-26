"""Statistical machinery of the v2 contract (spec ``statistics``, ``directions.random``, ``structured_null``).

All arithmetic is float64. Per-prompt values are joined by prompt_id, never by row order. Non-finite
inputs raise ``StatisticsError`` (mapped to TECHNICAL_FAIL); no ``nan*`` reductions are used.
"""

from __future__ import annotations

from .errors import FinalFailure

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

FAMILIES = ("direct", "identity", "hypothetical")
N_BOOT = 10_000
PROMPTS_PER_FAMILY = 100
BOOTSTRAP_SEED = 20260925
RCOV_SEED = 20260925
N_RCOV = 199
K_CONTRASTS = 3
ALPHA_TS = 0.05 / K_CONTRASTS
ALPHA_PC = 0.05


class StatisticsError(RuntimeError, FinalFailure):
    """Non-finite or mis-shaped statistical input (TECHNICAL_FAIL)."""


def _finite(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise StatisticsError(f"Non-finite values in {label}")
    return values


@dataclass(frozen=True)
class FamilyIndex:
    """S0 animal-family prompt ids, sorted within family (the bootstrap's canonical order)."""

    ids: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_ids(cls, prompt_ids_by_family: Mapping[str, Sequence[str]]) -> "FamilyIndex":
        ids = {family: tuple(sorted(prompt_ids_by_family[family])) for family in FAMILIES}
        for family in FAMILIES:
            if len(ids[family]) != PROMPTS_PER_FAMILY or len(set(ids[family])) != PROMPTS_PER_FAMILY:
                raise StatisticsError(f"Family {family} must have {PROMPTS_PER_FAMILY} distinct prompts")
        return cls(ids)

    def align(self, values_by_id: Mapping[str, float], label: str = "statistic") -> dict[str, np.ndarray]:
        out = {}
        for family in FAMILIES:
            try:
                out[family] = np.array([values_by_id[i] for i in self.ids[family]], dtype=np.float64)
            except KeyError as exc:
                raise StatisticsError(f"Missing prompt {exc} in {label}") from exc
            _finite(out[family], label)
        return out


def bootstrap_indices() -> dict[str, np.ndarray]:
    """One (10000, 100) index matrix per family, drawn in family order from its own PCG64 stream."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return {family: rng.integers(0, PROMPTS_PER_FAMILY, size=(N_BOOT, PROMPTS_PER_FAMILY)) for family in FAMILIES}


def point(y: Mapping[str, np.ndarray]) -> float:
    """Full-sample family-weighted mean (weights 1/3); the point estimate of every criterion."""
    parts = [_finite(y[family], f"point estimate ({family})") for family in FAMILIES]
    return float((parts[0].mean() + parts[1].mean() + parts[2].mean()) / 3.0)


def replicates(y: Mapping[str, np.ndarray], index: Mapping[str, np.ndarray]) -> np.ndarray:
    out = (
        y["direct"][index["direct"]].mean(axis=1)
        + y["identity"][index["identity"]].mean(axis=1)
        + y["hypothetical"][index["hypothetical"]].mean(axis=1)
    ) / 3.0
    return _finite(out, "bootstrap replicates")


def interval(y: Mapping[str, np.ndarray], index: Mapping[str, np.ndarray], alpha: float = ALPHA_TS) -> tuple[float, float]:
    """Two-sided percentile interval (numpy 'linear' = type 7) at level 1 - alpha."""
    reps = replicates(y, index)
    low, high = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return float(low), float(high)


def ci_low_positive(bounds: tuple[float, float]) -> bool:
    """A bound exactly equal to 0 does not exclude 0."""
    return bounds[0] > 0.0


def ci_high_negative(bounds: tuple[float, float]) -> bool:
    return bounds[1] < 0.0


def se_boot(y: Mapping[str, np.ndarray], index: Mapping[str, np.ndarray]) -> float:
    """SD (ddof 1) of the paired bootstrap replicates (spec ``statistics.SE_boot``)."""
    return float(np.std(replicates(y, index), ddof=1))


def one_sided_bounds(y: Mapping[str, np.ndarray], index: Mapping[str, np.ndarray], alpha: float = ALPHA_TS) -> tuple[float, float]:
    """Type-7 quantiles of the replicates at alpha and 1 - alpha (two one-sided tests, each at level alpha)."""
    reps = replicates(y, index)
    low, high = np.quantile(reps, [alpha, 1.0 - alpha], method="linear")
    return float(low), float(high)


@dataclass(frozen=True)
class RankTest:
    p: float
    exceed: int
    n: int
    k_max: int
    passed: bool


def rank_k_max(n: int, alpha: float) -> int:
    """Largest k with (1 + k) / (1 + n) < alpha (-1 if even k = 0 cannot pass)."""
    k = -1
    while (1 + (k + 1)) / (1 + n) < alpha:
        k += 1
    return k


def rank_test(t_obs: float, t_null: Sequence[float], alpha: float) -> RankTest:
    """p = (1 + #{T_null >= T_obs}) / (1 + n); pass iff p < alpha; a null equal to T_obs is an exceedance."""
    t_null = _finite(np.asarray(t_null), "null statistics")
    if not np.isfinite(t_obs):
        raise StatisticsError("Non-finite observed statistic")
    n = int(t_null.size)
    if n == 0:
        raise StatisticsError("Empty null reference")
    exceed = int((t_null >= t_obs).sum())
    p = (1 + exceed) / (1 + n)
    return RankTest(p=p, exceed=exceed, n=n, k_max=rank_k_max(n, alpha), passed=bool(p < alpha))


def null_pairs(words: Sequence[str]) -> list[tuple[str, str]]:
    return [(a, b) for a in words for b in words if a != b]


def logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    peak = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(values - peak), axis=axis))


def random_cov_directions(default_states14: np.ndarray, n: int = N_RCOV) -> np.ndarray:
    """R_cov: z = X_c^T g / sqrt(1023), g ~ N(0, I_1024) drawn as one (n, 1024) array from PCG64(20260925),
    rows unit-normalized. ``default_states14``: the 1,024 P_default slot-14 states in extraction-row order."""
    x = _finite(np.asarray(default_states14, dtype=np.float64), "R_cov states")
    if x.shape[0] != 1024 or x.ndim != 2:
        raise StatisticsError("R_cov needs the 1024 x H P_default slot-14 state matrix")
    centered = x - x.mean(axis=0, keepdims=True)
    g = np.random.default_rng(RCOV_SEED).standard_normal((n, x.shape[0]))
    z = g @ centered / np.sqrt(x.shape[0] - 1.0)
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def covariance(default_states14: np.ndarray) -> np.ndarray:
    x = _finite(np.asarray(default_states14, dtype=np.float64), "covariance states")
    centered = x - x.mean(axis=0, keepdims=True)
    return centered.T @ centered / (x.shape[0] - 1.0)
