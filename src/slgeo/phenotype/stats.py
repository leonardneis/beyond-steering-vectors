"""Confirmatory statistics of the Phenotype Anchor (preregistration draft ``research/phenotype_anchor_v1/PREREGISTRATION.md`` §6).

Inputs are exact word log-probabilities ``L[s, r, w]`` (stem s, prefix replicate r, panel word w) from the
``cts_stage0`` scorer. Stems are the statistical unit. Prefix replicates are averaged within stem on the
probability scale (the stem's expected answer distribution under a random prefix: the mean of the replicate
conditionals, not the conditional of the mean) before any test. Each stem's log q is floored at ``LOGQ_FLOOR``
nats below its most probable word, so a single astronomically improbable word cannot dominate a fit.

All arithmetic is float64. Non-finite inputs raise ``PhenotypeStatsError``.

Flattening (tempering) model, used by C2-C5 and K1-K4. Per stem s and word w, with x = log q of the reference
condition (N, or base) and y = log q of the treated condition:
    y_sw - ybar_s = beta (x_sw - xbar_s) + residual_sw,
where bars are means over the stem's *non-target* words weighted by u = q of an **independent weight arm** (base
for student contrasts; the mean of the neutral students for teacher vs base). The fit is governed by where the
probability mass is, while the weights carry none of the noise of y or x (self-weights bias the intercept).

beta is the Deming slope for a noise-variance ratio lambda = var(noise of y) / var(noise of x):
- student vs student: lambda estimated from the within-condition cross-seed pairs (``estimate_lambda``); equal
  noise (lambda = 1) is not assumed because unequal noise biased C3 and inflated its per-seed type-I error to 0.2-0.3
  in pre-freeze simulations;
- treated vs base: x is noise-free by definition (base is the reference point), so lambda = inf (OLS of y on x).
An OLS of y on a *noisy* x is attenuated toward 0 and reports flattening between two noisy copies of one
distribution (pre-freeze audit). A word's residual is taken out of fit (the word is excluded from beta and from the
intercept), and the prediction is never renormalized over the word itself (pre-freeze audit).

Claims are compositional: C3 is the target's log-odds against the mass-weighted non-target words beyond the fitted
tempering. It is not "the target's probability rose"; that stronger statement needs the dominance label or the
mass-matched contrast (pre-freeze audit). ``adequacy`` checks the tempering model for frequency-dependent
misfit (e.g. a probability floor), which would leak into C3-C5.

Bootstrap draws in which a fit is degenerate are redrawn and counted; more than ``MAX_DEGENERATE`` of the draws
is a TECHNICAL failure.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom, norm, rankdata

N_BOOT = 10_000
N_FLIP = 9_999
SEED = 20260926
ALPHA = 0.05
LOGQ_FLOOR = 40.0
MAX_DEGENERATE = 0.01
MASS_MATCHED_K = 3
_CHUNK = 256


class PhenotypeStatsError(ValueError):
    """Non-finite, mis-shaped or degenerate input (TECHNICAL_FAIL)."""


def _finite(values, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise PhenotypeStatsError(f"Non-finite values in {label}")
    return values


# --- distributions ---------------------------------------------------------------------------------------------


def _normalize(logq: np.ndarray) -> np.ndarray:
    return logq - logsumexp(logq, axis=-1, keepdims=True)


def floor_logq(logq: np.ndarray, floor: float = LOGQ_FLOOR) -> np.ndarray:
    """Clip each stem's log q at ``floor`` nats below its maximum, then renormalize."""
    logq = _finite(logq, "logq")
    return _normalize(np.maximum(logq, logq.max(axis=-1, keepdims=True) - floor))


def stem_conditional(word_logp: np.ndarray) -> np.ndarray:
    """[S, R, W] word log-probs -> [S, W] floored log of the panel-conditional distribution, prefix replicates
    averaged on the probability scale."""
    word_logp = _finite(word_logp, "word_logp")
    if word_logp.ndim == 2:
        word_logp = word_logp[:, None, :]
    if word_logp.ndim != 3:
        raise PhenotypeStatsError("word_logp must be [S, R, W] or [S, W]")
    logq = _normalize(word_logp)
    mixed = logsumexp(logq, axis=1) - np.log(logq.shape[1])
    return floor_logq(_normalize(mixed))


def mean_distribution(*logqs: np.ndarray) -> np.ndarray:
    """Log of the stem-wise probability average of several arms (e.g. the three neutral students)."""
    stacked = np.stack([_finite(l, "logq") for l in logqs])
    return _normalize(logsumexp(stacked, axis=0) - np.log(len(logqs)))


def clr(logq: np.ndarray) -> np.ndarray:
    return logq - logq.mean(axis=1, keepdims=True)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        raise PhenotypeStatsError("Spearman undefined for constant input")
    return float((ra * rb).sum() / denom)


def _signs(n_stems: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), size=(n, n_stems))


def _p_upper(null: np.ndarray, observed: float) -> float:
    return float((1 + np.count_nonzero(null >= observed)) / (1 + null.size))


def _swap(a: np.ndarray, b: np.ndarray, flip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-stem label swap: rows with flip < 0 exchange a and b."""
    m = (flip < 0)[:, None]
    return np.where(m, b, a), np.where(m, a, b)


def _nontarget(n_words: int, targets: Sequence[int]) -> np.ndarray:
    excluded = set(int(t) for t in targets)
    return np.array([w for w in range(n_words) if w not in excluded])


def _bootstrap(stat: Callable[[np.ndarray], float], n: int, n_boot: int, seed: int) -> np.ndarray:
    """Stem bootstrap of ``stat(idx)``; degenerate draws are redrawn and counted (fail above MAX_DEGENERATE)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    degenerate = 0
    i = 0
    while i < n_boot:
        idx = rng.integers(0, n, n)
        try:
            out[i] = stat(idx)
        except PhenotypeStatsError:
            degenerate += 1
            if degenerate > MAX_DEGENERATE * n_boot + 1:
                raise PhenotypeStatsError(f"{degenerate} degenerate bootstrap draws")
            continue
        i += 1
    return out


# --- flattening model ------------------------------------------------------------------------------------------


def _centered(logq_y, logq_x, words, logq_ref):
    u = np.exp(logq_ref[:, words])
    u = u / u.sum(axis=1, keepdims=True)
    xbar = (u * logq_x[:, words]).sum(axis=1, keepdims=True)
    ybar = (u * logq_y[:, words]).sum(axis=1, keepdims=True)
    return logq_x - xbar, logq_y - ybar, u


def fit_beta(
    logq_y: np.ndarray, logq_x: np.ndarray, words: np.ndarray, logq_ref: np.ndarray, lam: float = 1.0,
) -> float:
    """Deming tempering slope over the given words for noise-variance ratio ``lam`` (inf = OLS of y on x)."""
    xc, yc, u = _centered(logq_y, logq_x, words, logq_ref)
    sxx = (u * xc[:, words] ** 2).sum()
    syy = (u * yc[:, words] ** 2).sum()
    sxy = (u * xc[:, words] * yc[:, words]).sum()
    if sxx <= 0 or syy <= 0 or sxy <= 0:
        raise PhenotypeStatsError("Degenerate flattening fit")
    if np.isinf(lam):
        return float(sxy / sxx)
    if lam <= 0:
        raise PhenotypeStatsError("lambda must be positive")
    d = syy - lam * sxx
    return float((d + np.sqrt(d * d + 4 * lam * sxy * sxy)) / (2 * sxy))


def residuals(
    logq_y: np.ndarray, logq_x: np.ndarray, logq_ref: np.ndarray, targets: Sequence[int] = (), lam: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Per-stem residuals of every word against the tempered x, with beta and the per-stem intercept fitted on the
    non-target words (weights from the reference arm). Residuals of the targets are out of fit; those of the other
    words are in fit. Returns ([S, W] residuals, beta)."""
    words = _nontarget(logq_y.shape[1], targets)
    beta = fit_beta(logq_y, logq_x, words, logq_ref, lam)
    xc, yc, _ = _centered(logq_y, logq_x, words, logq_ref)
    return yc - beta * xc, beta


@dataclass(frozen=True)
class Lambda:
    value: float
    var_y: float
    var_x: float


def estimate_lambda(
    pairs_y: Sequence[tuple[np.ndarray, np.ndarray]], pairs_x: Sequence[tuple[np.ndarray, np.ndarray]],
    logq_ref: np.ndarray, targets: Sequence[int] = (),
) -> Lambda:
    """lambda = var(noise of the treated condition) / var(noise of the reference condition), from within-condition
    cross-seed pairs (e.g. S_i - S_j and N_i - N_j): weighted variance of the per-stem centered difference over the
    non-target words, pooled over pairs, same weights as the fit."""

    def pooled(pairs):
        if not pairs:
            raise PhenotypeStatsError("No within-condition pairs for lambda")
        values = []
        for a, b in pairs:
            a, b = _finite(a, "pair"), _finite(b, "pair")
            words = _nontarget(a.shape[1], targets)
            _, dc, u = _centered(a - b, np.zeros_like(a), words, logq_ref)
            values.append(float((u * dc[:, words] ** 2).sum() / a.shape[0]))
        return float(np.mean(values))

    vy, vx = pooled(pairs_y), pooled(pairs_x)
    if vx <= 0:
        raise PhenotypeStatsError("Zero reference-condition noise")
    return Lambda(vy / vx, vy, vx)


# --- C1 ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Omnibus:
    statistic: float
    p: float
    t: np.ndarray  # per-word t of the mean clr change


def _t(delta: np.ndarray) -> np.ndarray:
    n = delta.shape[-2]
    mean = delta.mean(axis=-2)
    sd = delta.std(axis=-2, ddof=1)
    if np.any(sd == 0):
        raise PhenotypeStatsError("Zero variance across stems")
    return mean / (sd / np.sqrt(n))


def _flip_t_stat(delta: np.ndarray, n_flip: int, seed: int, reduce) -> np.ndarray:
    """reduce(t) for each sign-flip draw, computed in chunks to bound memory."""
    signs = _signs(delta.shape[0], n_flip, seed)
    out = np.empty(n_flip)
    for start in range(0, n_flip, _CHUNK):
        block = signs[start:start + _CHUNK][:, :, None] * delta[None]
        out[start:start + _CHUNK] = reduce(_t(block))
    return out


def omnibus(logq_a: np.ndarray, logq_b: np.ndarray, *, n_flip: int = N_FLIP, seed: int = SEED) -> Omnibus:
    """C1: T = sum_w t_w^2 of the per-stem clr change (a - b); null flips each stem's whole change vector.
    Detects word-consistent shifts; a tempering whose head words differ between stems is C2's job."""
    delta = clr(_finite(logq_a, "a")) - clr(_finite(logq_b, "b"))
    t = _t(delta)
    observed = float((t ** 2).sum())
    null = _flip_t_stat(delta, n_flip, seed, lambda t: (t ** 2).sum(axis=1))
    return Omnibus(observed, _p_upper(null, observed), t)


# --- C2 / K3 ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Flattening:
    beta: float
    ci90: tuple[float, float]
    p_less_than_one: float


def flattening(
    logq_y, logq_x, logq_ref, targets: Sequence[int], *, lam: float = 1.0, n_boot: int = N_BOOT, seed: int = SEED,
) -> Flattening:
    """C2/K3: Deming beta over non-target words; one-sided stem-bootstrap p for beta < 1."""
    logq_y, logq_x, logq_ref = _finite(logq_y, "y"), _finite(logq_x, "x"), _finite(logq_ref, "ref")
    words = _nontarget(logq_y.shape[1], targets)
    beta = fit_beta(logq_y, logq_x, words, logq_ref, lam)
    boots = _bootstrap(lambda idx: fit_beta(logq_y[idx], logq_x[idx], words, logq_ref[idx], lam),
                       logq_y.shape[0], n_boot, seed)
    p = float((1 + np.count_nonzero(boots >= 1.0)) / (1 + n_boot))
    return Flattening(beta, (float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))), p)


# --- C3 / K1 / K2 -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetResidual:
    mean: float  # mean over stems of the target's out-of-fit log-odds residual beyond the tempering
    ci95: tuple[float, float]
    ci90: tuple[float, float]
    p: float  # one-sided, H: mean > 0 (stem bootstrap, beta refitted in every draw)
    beta: float
    delta_prob: float  # descriptive: mean over stems of q_y(target) - q_x(target)

    def equivalent(self, margin: float) -> bool:
        """90 % CI inside (-margin, +margin) (two one-sided tests at 0.05)."""
        return bool(-margin < self.ci90[0] and self.ci90[1] < margin)


def target_residual(
    logq_y, logq_x, logq_ref, target: int, *, exclude: Sequence[int] = (), lam: float = 1.0,
    n_boot: int = N_BOOT, seed: int = SEED,
) -> TargetResidual:
    """C3 (and K1/K2): the target's out-of-fit residual beyond the tempering. ``exclude`` lists further words kept
    out of the beta fit and the intercept (e.g. the other trait word in K1/K2)."""
    logq_y, logq_x, logq_ref = _finite(logq_y, "y"), _finite(logq_x, "x"), _finite(logq_ref, "ref")
    targets = (target, *exclude)
    r, beta = residuals(logq_y, logq_x, logq_ref, targets, lam)
    boots = _bootstrap(lambda idx: residuals(logq_y[idx], logq_x[idx], logq_ref[idx], targets, lam)[0][:, target].mean(),
                       logq_y.shape[0], n_boot, seed)
    return TargetResidual(
        float(r[:, target].mean()),
        (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
        (float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))),
        float((1 + np.count_nonzero(boots <= 0.0)) / (1 + n_boot)),
        beta,
        float((np.exp(logq_y[:, target]) - np.exp(logq_x[:, target])).mean()),
    )


def mass_matched_controls(logq_ref: np.ndarray, target: int, k: int = MASS_MATCHED_K, exclude: Sequence[int] = ()):
    """The k non-target words whose mean reference log q is closest to the target's (fixed by the reference arm)."""
    mean = _finite(logq_ref, "ref").mean(axis=0)
    candidates = _nontarget(mean.size, (target, *exclude))
    order = np.argsort(np.abs(mean[candidates] - mean[target]), kind="stable")
    return candidates[order[:k]]


def mass_matched_contrast(
    logq_y, logq_x, logq_ref, target: int, *, k: int = MASS_MATCHED_K, lam: float = 1.0, n_boot: int = N_BOOT,
    seed: int = SEED,
) -> TargetResidual:
    """Secondary to C3: per stem r_target - mean(r_controls) with the k mass-matched control words; target and
    controls all out of fit. Robust to frequency-dependent misfit of the tempering model."""
    logq_y, logq_x, logq_ref = _finite(logq_y, "y"), _finite(logq_x, "x"), _finite(logq_ref, "ref")
    controls = mass_matched_controls(logq_ref, target, k)
    targets = (target, *controls.tolist())

    def stat(idx):
        r, _ = residuals(logq_y[idx], logq_x[idx], logq_ref[idx], targets, lam)
        return float((r[:, target] - r[:, controls].mean(axis=1)).mean())

    r, beta = residuals(logq_y, logq_x, logq_ref, targets, lam)
    boots = _bootstrap(stat, logq_y.shape[0], n_boot, seed)
    return TargetResidual(
        float((r[:, target] - r[:, controls].mean(axis=1)).mean()),
        (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
        (float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))),
        float((1 + np.count_nonzero(boots <= 0.0)) / (1 + n_boot)),
        beta,
        float((np.exp(logq_y[:, target]) - np.exp(logq_x[:, target])).mean()),
    )


# --- model adequacy ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Adequacy:
    curvature: float  # quadratic coefficient of the mean in-fit residual on the word's mean centered log q_x
    ci95: tuple[float, float]
    adequate: bool


def _curvature(logq_y, logq_x, logq_ref, targets, lam) -> float:
    words = _nontarget(logq_y.shape[1], targets)
    r, _ = residuals(logq_y, logq_x, logq_ref, targets, lam)
    xc, _, _ = _centered(logq_y, logq_x, words, logq_ref)
    m, x = r[:, words].mean(axis=0), xc[:, words].mean(axis=0)
    design = np.stack([np.ones_like(x), x, x * x], axis=1)
    coef, *_ = np.linalg.lstsq(design, m, rcond=None)
    return float(coef[2])


def adequacy(
    logq_y, logq_x, logq_ref, targets: Sequence[int], *, lam: float = 1.0, n_boot: int = N_BOOT, seed: int = SEED,
) -> Adequacy:
    """Tempering-model adequacy: the mean in-fit residual of the non-target words must show no quadratic trend in
    the word's mean centered log q_x (a probability floor or other frequency-dependent misfit produces one).
    Adequate iff the 95 % stem-bootstrap CI of the curvature covers 0. A failure bars C3-C5 claims."""
    logq_y, logq_x, logq_ref = _finite(logq_y, "y"), _finite(logq_x, "x"), _finite(logq_ref, "ref")
    observed = _curvature(logq_y, logq_x, logq_ref, targets, lam)
    boots = _bootstrap(lambda idx: _curvature(logq_y[idx], logq_x[idx], logq_ref[idx], targets, lam),
                       logq_y.shape[0], n_boot, seed)
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    return Adequacy(observed, ci, bool(ci[0] <= 0.0 <= ci[1]))


# --- C4 / C5 / K4: profile statistics with label-swap nulls -----------------------------------------------------


@dataclass(frozen=True)
class Concordance:
    rho: float
    p: float


def _profile(logq_y, logq_x, logq_ref, targets, cols, lam) -> np.ndarray:
    return residuals(logq_y, logq_x, logq_ref, targets, lam)[0][:, cols].mean(axis=0)


def _swap_null(stat: Callable[[np.ndarray, np.ndarray], float], a, b, n_flip, seed) -> np.ndarray:
    null = []
    degenerate = 0
    for flip in _signs(a.shape[0], n_flip + int(MAX_DEGENERATE * n_flip) + 1, seed):
        if len(null) == n_flip:
            break
        try:
            null.append(stat(*_swap(a, b, flip)))
        except PhenotypeStatsError:
            degenerate += 1
    if len(null) < n_flip:
        raise PhenotypeStatsError(f"{degenerate} degenerate label-swap draws")
    return np.array(null)


def shadow_concordance(
    logq_teacher, logq_base, logq_teacher_ref, logq_s, logq_n, target: int, *, lam: float = 1.0,
    n_flip: int = N_FLIP, seed: int = SEED,
) -> Concordance:
    """C4: Spearman over non-target words between the tempering residual profiles of (teacher vs base; OLS, base is
    noise-free; weights = mean of the neutral students) and (S vs N; Deming with ``lam``; weights = base).
    Null: per-stem S/N label swap, beta refitted (teacher profile fixed)."""
    logq_s, logq_n, logq_base = _finite(logq_s, "s"), _finite(logq_n, "n"), _finite(logq_base, "base")
    cols = _nontarget(logq_s.shape[1], (target,))
    teacher = _profile(_finite(logq_teacher, "teacher"), logq_base, _finite(logq_teacher_ref, "teacher_ref"),
                       (target,), cols, np.inf)
    rho = spearman(teacher, _profile(logq_s, logq_n, logq_base, (target,), cols, lam))
    null = _swap_null(lambda a, b: spearman(teacher, _profile(a, b, logq_base, (target,), cols, lam)),
                      logq_s, logq_n, n_flip, seed + 2)
    return Concordance(rho, _p_upper(null, rho))


def residual_profile(
    logq_y, logq_x, logq_ref, words: Sequence[str], panel_words: Sequence[str], target: int, *, lam: float = 1.0,
) -> dict[str, float]:
    """Mean residual (beyond the tempering, target excluded from the fit) of the listed words; used to freeze a
    development profile for C5."""
    r, _ = residuals(_finite(logq_y, "y"), _finite(logq_x, "x"), _finite(logq_ref, "ref"), (target,), lam)
    return {w: float(r[:, list(panel_words).index(w)].mean()) for w in words}


def profile_replication(
    reference: Mapping[str, float], panel_words: Sequence[str], logq_s, logq_n, logq_ref, target: int, *,
    lam: float = 1.0, n_flip: int = N_FLIP, seed: int = SEED,
) -> Concordance:
    """C5: Spearman between a frozen development residual profile (word -> value; ranks only) and the new seed's
    residual profile over the same words. Null: per-stem S/N label swap, beta refitted."""
    logq_s, logq_n, logq_ref = _finite(logq_s, "s"), _finite(logq_n, "n"), _finite(logq_ref, "ref")
    cols = np.array([list(panel_words).index(w) for w in reference])
    ref = np.array([reference[w] for w in reference], dtype=np.float64)
    rho = spearman(ref, _profile(logq_s, logq_n, logq_ref, (target,), cols, lam))
    null = _swap_null(lambda a, b: spearman(ref, _profile(a, b, logq_ref, (target,), cols, lam)),
                      logq_s, logq_n, n_flip, seed + 3)
    return Concordance(rho, _p_upper(null, rho))


def fold_of(stem_ids: Sequence[str]) -> np.ndarray:
    """Fixed two-fold split by SHA-256 parity of the stem id (outcome-independent)."""
    return np.array([hashlib.sha256(str(s).encode()).digest()[0] & 1 for s in stem_ids], dtype=int)


def shared_movers(
    logq_s, logq_n_for_s, logq_d, logq_n_for_d, logq_ref, targets: Sequence[int], stem_ids: Sequence[str], *,
    lam_s: float = 1.0, lam_d: float = 1.0, n_flip: int = N_FLIP, seed: int = SEED,
) -> Concordance:
    """K4: Spearman over non-target words between the (S_k vs N_k) residual profile on fold 0 and the
    (D_k vs N_j, j != k) residual profile on fold 1. Disjoint folds remove shared stem noise and a different neutral
    seed removes shared run-level word offsets of N (pre-freeze audit). Null: per-stem label swap
    within each fold, beta refitted."""
    logq_s, logq_d = _finite(logq_s, "s"), _finite(logq_d, "d")
    logq_n_for_s, logq_n_for_d = _finite(logq_n_for_s, "n_s"), _finite(logq_n_for_d, "n_d")
    logq_ref = _finite(logq_ref, "ref")
    folds = fold_of(stem_ids)
    if set(folds.tolist()) != {0, 1}:
        raise PhenotypeStatsError("Both folds must be non-empty")
    cols = _nontarget(logq_s.shape[1], targets)
    a0, n0, r0 = logq_s[folds == 0], logq_n_for_s[folds == 0], logq_ref[folds == 0]
    d1, n1, r1 = logq_d[folds == 1], logq_n_for_d[folds == 1], logq_ref[folds == 1]
    rho = spearman(_profile(a0, n0, r0, targets, cols, lam_s), _profile(d1, n1, r1, targets, cols, lam_d))
    rng0, rng1 = _signs(a0.shape[0], n_flip, seed + 5), _signs(d1.shape[0], n_flip, seed + 6)
    null = np.empty(n_flip)
    for i in range(n_flip):
        x0, y0 = _swap(a0, n0, rng0[i])
        x1, y1 = _swap(d1, n1, rng1[i])
        null[i] = spearman(_profile(x0, y0, r0, targets, cols, lam_s), _profile(x1, y1, r1, targets, cols, lam_d))
    return Concordance(rho, _p_upper(null, rho))


# --- cat-specific label -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dominance:
    t: np.ndarray  # per-contrast t of r_target - r_w
    critical: np.ndarray  # per-contrast label-swap critical values
    passed: bool


def _dominance_contrasts(logq_y, logq_x, logq_ref, target: int, others: np.ndarray, lam: float) -> np.ndarray:
    """Per stem d_w = r_target - r_w, both residuals out of fit (the pair is excluded from beta and intercept)."""
    cols = []
    for w in others:
        r, _ = residuals(logq_y, logq_x, logq_ref, (target, int(w)), lam)
        cols.append(r[:, target] - r[:, w])
    return np.stack(cols, axis=1)


def target_dominance(
    logq_y, logq_x, logq_ref, target: int, *, lam: float = 1.0, level: float = 0.95, n_flip: int = N_FLIP,
    seed: int = SEED,
) -> Dominance:
    """Label rule: the target's residual exceeds every other word's residual. Intersection-union test: pass iff
    t_w > c_w for every w, c_w = ``level`` quantile of t*_w under the per-stem label swap (beta refitted). No max-
    correction is needed for an all-contrasts claim (pre-freeze audit)."""
    logq_y, logq_x, logq_ref = _finite(logq_y, "y"), _finite(logq_x, "x"), _finite(logq_ref, "ref")
    others = _nontarget(logq_y.shape[1], (target,))
    t_obs = _t(_dominance_contrasts(logq_y, logq_x, logq_ref, target, others, lam))
    null = []
    for flip in _signs(logq_y.shape[0], n_flip, seed + 7):
        a, b = _swap(logq_y, logq_x, flip)
        try:
            null.append(_t(_dominance_contrasts(a, b, logq_ref, target, others, lam)))
        except PhenotypeStatsError:
            continue
    if len(null) < (1 - MAX_DEGENERATE) * n_flip - 1:
        raise PhenotypeStatsError("Too many degenerate label-swap draws")
    critical = np.quantile(np.stack(null), level, axis=0)
    return Dominance(t_obs, critical, bool(np.all(t_obs > critical)))


# --- run level --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunGate:
    observed: float
    null_max: float
    passed: bool


def run_level_gate(
    statistic: Callable[..., float],
    treated: tuple,
    within_pairs: Sequence[tuple],
) -> RunGate:
    """Run-level check: ``statistic(*treated)`` must exceed the statistic on every within-condition cross-seed
    pair, evaluated in **both orders** of each pair (a one-order gate passed 17 % of null runs per seed; pre-freeze audit). ``statistic`` must be oriented so that larger = stronger effect in the pre-registered direction."""
    if not within_pairs:
        raise PhenotypeStatsError("No within-condition pairs")
    observed = float(statistic(*treated))
    null = [float(statistic(*pair)) for pair in within_pairs] + [float(statistic(*pair[::-1])) for pair in within_pairs]
    return RunGate(observed, max(null), observed > max(null))


def run_noise_margin(statistic: Callable[..., float], within_pairs: Sequence[tuple]) -> float:
    """Largest |statistic| over the within-condition cross-seed pairs in both orders: the size of a difference that
    seed-to-seed training variation alone produces (used as the data-defined equivalence benchmark for C3)."""
    if not within_pairs:
        raise PhenotypeStatsError("No within-condition pairs")
    return float(max(abs(float(statistic(*p))) for pair in within_pairs for p in (pair, pair[::-1])))


# --- multiplicity, per-word intervals, instrument check ---------------------------------------------------------


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


def iut(pvalues_by_seed: Sequence[float]) -> float:
    """Intersection-union test over seeds: the conjunction's p-value is the largest per-seed p-value."""
    return float(max(pvalues_by_seed))


def seed_passes(
    pvalues: Mapping[str, float], gates: Mapping[str, bool], *, gated: Sequence[str], alpha: float = ALPHA,
) -> dict[str, bool]:
    """Per seed: Holm-adjusted p <= alpha and, for gated hypotheses, the run-level gate (fails closed)."""
    adjusted = holm(pvalues)
    return {h: adjusted[h] <= alpha and (gates.get(h, False) if h in gated else True) for h in pvalues}


def confirm(
    pvalues_by_seed: Mapping[str, Mapping[str, float]],
    run_gates_by_seed: Mapping[str, Mapping[str, bool]],
    *, gated: Sequence[str] = ("C1", "C2", "C3"), alpha: float = ALPHA,
) -> dict[str, bool]:
    """Family decision: per-seed passes (``seed_passes``), conjunction (IUT) across seeds."""
    per_seed = {s: seed_passes(p, run_gates_by_seed.get(s, {}), gated=gated, alpha=alpha)
                for s, p in pvalues_by_seed.items()}
    hyps = list(next(iter(pvalues_by_seed.values())))
    return {h: all(per_seed[s][h] for s in per_seed) for h in hyps}


def max_t_intervals(delta: np.ndarray, *, level: float = 0.95, n_flip: int = N_FLIP, seed: int = SEED):
    """Simultaneous per-word intervals mean_w +- c * se_w, c = level quantile of max_w |t_w| under stem
    sign-flips of whole vectors. Returns (mean, lower, upper, c)."""
    delta = _finite(delta, "delta")
    n = delta.shape[0]
    mean, se = delta.mean(axis=0), delta.std(axis=0, ddof=1) / np.sqrt(n)
    c = float(np.quantile(_flip_t_stat(delta, n_flip, seed + 4, lambda t: np.abs(t).max(axis=1)), level))
    return mean, mean - c * se, mean + c * se, c


@dataclass(frozen=True)
class Agreement:
    fraction_inside: float
    n_cells: int
    max_abs_z: float  # largest pooled per-word standardized difference
    z_bound: float
    passed: bool


def instrument_agreement(
    prob_exact: np.ndarray, counts: np.ndarray, k: int, *, min_prob: float = 0.02, level: float = 0.95,
    required: float = 0.90, min_word_prob: float = 0.01, alpha: float = 0.05,
) -> Agreement:
    """Sampled counts vs exact probabilities. Two pre-declared conditions:

    1. cell level: the fraction of stem x word cells with p >= min_prob whose count lies inside the central
       binomial ``level`` band of Bin(k, p) is at least ``required``;
    2. pooled per word (power against systematic bias): for every word with mean p >= min_word_prob,
       |z_w| <= Phi^-1(1 - alpha / (2 W')), z_w = (sum_s c_sw / k - sum_s p_sw) / sqrt(sum_s p_sw (1 - p_sw) / k),
       W' = number of such words (Bonferroni).
    """
    prob_exact, counts = _finite(prob_exact, "prob"), _finite(counts, "counts")
    mask = prob_exact >= min_prob
    if not mask.any():
        raise PhenotypeStatsError("No cells above min_prob")
    p, c = prob_exact[mask], counts[mask]
    lo, hi = binom.ppf((1 - level) / 2, k, p), binom.ppf(1 - (1 - level) / 2, k, p)
    fraction = float(np.mean((c >= lo) & (c <= hi)))
    words = prob_exact.mean(axis=0) >= min_word_prob
    if not words.any():
        raise PhenotypeStatsError("No words above min_word_prob")
    var = (prob_exact[:, words] * (1 - prob_exact[:, words]) / k).sum(axis=0)
    z = (counts[:, words].sum(axis=0) / k - prob_exact[:, words].sum(axis=0)) / np.sqrt(var)
    bound = float(norm.ppf(1 - alpha / (2 * int(words.sum()))))
    max_z = float(np.abs(z).max())
    return Agreement(fraction, int(mask.sum()), max_z, bound, fraction >= required and max_z <= bound)
