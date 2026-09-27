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
tempering. It is not "the target's probability rose"; that stronger statement needs the dominance label (pre-freeze
audit). A trait-residual claim is made in the robust form: the tempering residual **and** the mass-matched contrast
(target vs the median of its five base-mass neighbours, taxonomic neighbours excluded) must both pass, so a smooth frequency-dependent misfit of the tempering
model (a probability floor, depth-dependent noise, a rare target) cannot create the claim (v1 audit F1/F3).

Level of inference (v2, after the v1 final statistics audit). Teacher-condition claims generalize over training
runs; stems are measurement units inside a run. ``run_level`` tests a contrast statistic against a variance that
adds the run-level component, estimated from within-condition cross-seed pairs (lambda = 1: the two runs of a
within-condition pair are exchangeable), to the stem-bootstrap variance, and refers the studentized statistic to a
Gaussian random-effects pivot over the six runs. lambda-hat is re-estimated in every bootstrap draw.

Bootstrap draws in which a fit is degenerate are redrawn and counted; more than ``MAX_DEGENERATE`` of the draws
is a TECHNICAL failure.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom, norm, rankdata

N_BOOT = 10_000
N_FLIP = 9_999
N_REF = 100_000
SEED = 20260926
ALPHA = 0.05
LOGQ_FLOOR = 40.0
MAX_DEGENERATE = 0.01
MASS_MATCHED_K = 5
# per-run run-offset variance / mean stem-bootstrap variance: the nuisance grid of the run-level pivot (sup)
RHO_GRID = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
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


def _stem_weights(stem_w, n: int) -> np.ndarray:
    """Per-stem weights normalized to mean 1 (``None`` = equal weights)."""
    if stem_w is None:
        return np.ones(n)
    w = np.asarray(stem_w, dtype=np.float64)
    if w.shape != (n,) or not np.all(w > 0):
        raise PhenotypeStatsError("stem weights must be positive, one per stem")
    return w * (n / w.sum())


def family_weights(families: Sequence[str]) -> np.ndarray:
    """Equal weight per stem family (stratum), equal weight per stem within a family; mean 1."""
    families = list(families)
    counts = {f: families.count(f) for f in set(families)}
    return np.array([len(families) / (len(counts) * counts[f]) for f in families])


def wmean(values: np.ndarray, stem_w=None) -> float:
    """Stem-weighted mean over axis 0 of a per-stem vector."""
    w = _stem_weights(stem_w, values.shape[0])
    return float((w * values).sum() / w.sum())


def fit_beta(
    logq_y: np.ndarray, logq_x: np.ndarray, words: np.ndarray, logq_ref: np.ndarray, lam: float = 1.0, stem_w=None,
) -> float:
    """Deming tempering slope over the given words for noise-variance ratio ``lam`` (inf = OLS of y on x); stems
    weighted by ``stem_w``."""
    xc, yc, u = _centered(logq_y, logq_x, words, logq_ref)
    u = u * _stem_weights(stem_w, u.shape[0])[:, None]
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
    stem_w=None,
) -> tuple[np.ndarray, float]:
    """Per-stem residuals of every word against the tempered x, with beta and the per-stem intercept fitted on the
    non-target words (weights from the reference arm). Residuals of the targets are out of fit; those of the other
    words are in fit. Returns ([S, W] residuals, beta)."""
    words = _nontarget(logq_y.shape[1], targets)
    beta = fit_beta(logq_y, logq_x, words, logq_ref, lam, stem_w)
    xc, yc, _ = _centered(logq_y, logq_x, words, logq_ref)
    return yc - beta * xc, beta


@dataclass(frozen=True)
class Lambda:
    value: float
    var_y: float
    var_x: float


def estimate_lambda(
    pairs_y: Sequence[tuple[np.ndarray, np.ndarray]], pairs_x: Sequence[tuple[np.ndarray, np.ndarray]],
    logq_ref: np.ndarray, targets: Sequence[int] = (), stem_w=None,
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
            values.append(wmean((u * dc[:, words] ** 2).sum(axis=1), stem_w))
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
    """The k words whose mean reference log q is closest to the target's (fixed by the reference arm), never the
    target or an ``exclude`` word (the other trait word, taxonomic neighbours)."""
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


# --- mass-matched and dominance contrast statistics (scalar, larger = stronger effect) -----------------------


def target_stat(target: int, exclude: Sequence[int] = ()) -> Callable[..., float]:
    """C3 / K1 / K2 statistic: stem-weighted mean of the target's out-of-fit residual beyond the tempering;
    ``exclude`` words are kept out of the fit (the other trait word)."""
    targets = (target, *exclude)
    return lambda y, x, ref, lam, w=None: wmean(residuals(y, x, ref, targets, lam, w)[0][:, target], w)


def mass_matched_stat(target: int, controls: Sequence[int], exclude: Sequence[int] = ()) -> Callable[..., float]:
    """Mass-matched contrast: the target's mean residual minus the median over the control words of their mean
    residuals; target, controls and ``exclude`` out of fit. The median keeps the contrast valid if up to two of the
    five controls carry an effect of their own."""
    controls = np.asarray(controls, dtype=int)
    targets = (target, *controls.tolist(), *exclude)

    def stat(y, x, ref, lam, w=None):
        r, _ = residuals(y, x, ref, targets, lam, w)
        return wmean(r[:, target], w) - float(np.median([wmean(r[:, c], w) for c in controls]))

    return stat


def flattening_stat(targets: Sequence[int]) -> Callable[..., float]:
    """C2 / K3 / K5 statistic: -log beta over the non-target words (positive = flattening). At lambda = 1 the Deming
    slope is reciprocal under exchange of y and x, so the within-pair statistic is antisymmetric."""
    targets = tuple(targets)
    return lambda y, x, ref, lam, w=None: -float(np.log(fit_beta(y, x, _nontarget(y.shape[1], targets), ref, lam, w)))


def dominance_stat(target: int, other: int, exclude: Sequence[int] = ()) -> Callable[..., float]:
    """Dominance contrast d_w: mean of r_target - r_w, the pair (and ``exclude``) out of fit."""
    targets = (target, int(other), *exclude)

    def stat(y, x, ref, lam, w=None):
        r, _ = residuals(y, x, ref, targets, lam, w)
        return wmean(r[:, target] - r[:, other], w)

    return stat


# --- model adequacy (descriptive) -------------------------------------------------------------------------------


def _curvature(logq_y, logq_x, logq_ref, targets, lam, stem_w=None) -> float:
    words = _nontarget(logq_y.shape[1], targets)
    r, _ = residuals(logq_y, logq_x, logq_ref, targets, lam, stem_w)
    xc, _, _ = _centered(logq_y, logq_x, words, logq_ref)
    w = _stem_weights(stem_w, r.shape[0])[:, None]
    m, x = (w * r[:, words]).mean(axis=0), (w * xc[:, words]).mean(axis=0)
    design = np.stack([np.ones_like(x), x, x * x], axis=1)
    coef, *_ = np.linalg.lstsq(design, m, rcond=None)
    return float(coef[2])


def curvature_stat(targets: Sequence[int]) -> Callable[..., float]:
    """Descriptive tempering-model diagnostic: quadratic coefficient of the mean in-fit residual of the non-target
    words on their mean centered log q_x. v2 reports it with its run-level z; it gates nothing (a pretest on it cost
    most of the power under run variation and fired on single movers; v1 audit F3)."""
    targets = tuple(targets)
    return lambda y, x, ref, lam, w=None: _curvature(y, x, ref, targets, lam, w)


# --- descriptive profile statistics (C4 / C5 / K4) --------------------------------------------------------------
#
# v2: descriptive only. A per-stem S/N label swap reproduces "S and N are exchangeable", which is false whenever S is
# tempered, and every tempered contrast carries a frequency trend that within-condition pairs do not (v1 audit F2).
# No valid run-level null exists with three runs per condition, so no p-value is reported.


@dataclass(frozen=True)
class ProfileRho:
    rho: float  # Spearman between the two profiles
    rho_reference_vs_mass: float  # Spearman of the reference profile against the words' mean base log q
    rho_partial_mass: float  # partial Spearman given the words' mean base log q


def _partial(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    rab, rac, rbc = spearman(a, b), spearman(a, c), spearman(b, c)
    denom = np.sqrt(max((1 - rac ** 2) * (1 - rbc ** 2), 0.0))
    if denom == 0:
        raise PhenotypeStatsError("Partial correlation undefined")
    return float((rab - rac * rbc) / denom)


def _profile(logq_y, logq_x, logq_ref, targets, cols, lam) -> np.ndarray:
    return residuals(logq_y, logq_x, logq_ref, targets, lam)[0][:, cols].mean(axis=0)


def _profile_rho(reference: np.ndarray, observed: np.ndarray, mass: np.ndarray) -> ProfileRho:
    return ProfileRho(spearman(reference, observed), spearman(reference, mass), _partial(reference, observed, mass))


def shadow_concordance(
    logq_teacher, logq_base, logq_teacher_ref, logq_s, logq_n, target: int, *, lam: float = 1.0,
) -> ProfileRho:
    """C4 (descriptive): Spearman over non-target words between the tempering residual profiles of (teacher vs base;
    OLS, base is noise-free; weights = mean of the neutral students) and (S vs N; Deming with ``lam``; weights =
    base)."""
    logq_s, logq_n, logq_base = _finite(logq_s, "s"), _finite(logq_n, "n"), _finite(logq_base, "base")
    cols = _nontarget(logq_s.shape[1], (target,))
    teacher = _profile(_finite(logq_teacher, "teacher"), logq_base, _finite(logq_teacher_ref, "teacher_ref"),
                       (target,), cols, np.inf)
    return _profile_rho(teacher, _profile(logq_s, logq_n, logq_base, (target,), cols, lam),
                        logq_base[:, cols].mean(axis=0))


def residual_profile(
    logq_y, logq_x, logq_ref, words: Sequence[str], panel_words: Sequence[str], target: int, *, lam: float = 1.0,
) -> dict[str, float]:
    """Mean residual (beyond the tempering, target excluded from the fit) of the listed words; used to freeze a
    development profile for C5."""
    r, _ = residuals(_finite(logq_y, "y"), _finite(logq_x, "x"), _finite(logq_ref, "ref"), (target,), lam)
    return {w: float(r[:, list(panel_words).index(w)].mean()) for w in words}


def profile_replication(
    reference: Mapping[str, float], panel_words: Sequence[str], logq_s, logq_n, logq_ref, target: int, *,
    lam: float = 1.0,
) -> ProfileRho:
    """C5 (descriptive): Spearman between a frozen development residual profile (word -> value; ranks only) and the
    new seed's residual profile over the same words."""
    logq_s, logq_n, logq_ref = _finite(logq_s, "s"), _finite(logq_n, "n"), _finite(logq_ref, "ref")
    cols = np.array([list(panel_words).index(w) for w in reference])
    ref = np.array([reference[w] for w in reference], dtype=np.float64)
    return _profile_rho(ref, _profile(logq_s, logq_n, logq_ref, (target,), cols, lam), logq_ref[:, cols].mean(axis=0))


def fold_of(stem_ids: Sequence[str]) -> np.ndarray:
    """Fixed two-fold split by SHA-256 parity of the stem id (outcome-independent)."""
    return np.array([hashlib.sha256(str(s).encode()).digest()[0] & 1 for s in stem_ids], dtype=int)


def shared_movers(
    logq_s, logq_n_for_s, logq_d, logq_n_for_d, logq_ref, targets: Sequence[int], stem_ids: Sequence[str], *,
    lam_s: float = 1.0, lam_d: float = 1.0,
) -> ProfileRho:
    """K4 (descriptive): Spearman over non-target words between the (S_k vs N_k) residual profile on fold 0 and the
    (D_k vs N_j, j != k) residual profile on fold 1 (disjoint folds remove shared stem noise, a different neutral seed
    removes shared run offsets of N). The mass column uses the fold-0 reference."""
    logq_s, logq_d = _finite(logq_s, "s"), _finite(logq_d, "d")
    logq_n_for_s, logq_n_for_d = _finite(logq_n_for_s, "n_s"), _finite(logq_n_for_d, "n_d")
    logq_ref = _finite(logq_ref, "ref")
    folds = fold_of(stem_ids)
    if set(folds.tolist()) != {0, 1}:
        raise PhenotypeStatsError("Both folds must be non-empty")
    cols = _nontarget(logq_s.shape[1], targets)
    a0, n0, r0 = logq_s[folds == 0], logq_n_for_s[folds == 0], logq_ref[folds == 0]
    d1, n1, r1 = logq_d[folds == 1], logq_n_for_d[folds == 1], logq_ref[folds == 1]
    return _profile_rho(_profile(a0, n0, r0, targets, cols, lam_s), _profile(d1, n1, r1, targets, cols, lam_d),
                        r0[:, cols].mean(axis=0))


# --- run level --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunLevel:
    """Run-level test of one treated contrast. ``se_stem``: stem-bootstrap SD; ``run_var``: run-level variance of a
    between-run contrast, estimated from the within-condition pairs; ``se`` = sqrt(se_stem^2 + run_var);
    ``p``: one-sided (larger = effect); ``p_two``: two-sided; the intervals and ``upper95`` (one-sided 95 % upper
    bound) use the quantiles of the reference pivot."""

    estimate: float
    se_stem: float
    run_var: float
    se: float
    z: float
    p: float
    p_two: float
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    upper95: float

    def equivalent(self, margin: float) -> bool:
        """Run-level 90 % CI inside (-margin, +margin) (two one-sided tests at 0.05)."""
        return bool(-margin < self.ci90[0] and self.ci90[1] < margin)


def _condition(label: str) -> str:
    return label.rstrip("0123456789")


def run_level(
    statistic: Callable[..., float],
    arms: Mapping[str, np.ndarray],
    treated: Mapping[str, tuple[str, str]],
    within: Sequence[tuple[str, str]],
    logq_ref: np.ndarray,
    lam: Callable[[np.ndarray | None], float],
    *, stem_w=None, strata: Sequence | None = None, pooled: Sequence[str] = (),
    n_boot: int = N_BOOT, n_ref: int = N_REF, seed: int = SEED,
) -> dict[str, RunLevel]:
    """Run-level test of ``statistic(y, x, ref, lam, w)`` (larger = stronger effect) for each treated contrast.

    ``arms``: run label (condition letter + seed, e.g. "S2") -> [S, W] log q; ``treated``: key -> (y label, x label),
    evaluated at ``lam(idx)`` (lambda-hat re-estimated on the resampled stems ``idx``; ``lam(None)`` on all stems);
    ``within``: within-condition cross-seed pairs of the two conditions, evaluated at lambda = 1 in both orders;
    ``stem_w``: stem weights (equal family strata); ``strata``: stratum per stem for the bootstrap; ``pooled``: treated
    keys whose mean is reported under the key "pooled" (an interval, not a decision).

    1. theta_k and every within-pair value on all stems; the stratified stem bootstrap (one joint resample of all arms
       per draw) gives s_k^2, the per-pair variances s_w^2 and the joint stem-level deviations of all statistics.
    2. Per condition c, R_c = max(0, mean over the c-pairs of (T_w^2 - s_w^2)), T_w^2 = mean of the two orders'
       squares: an estimate of var(g_a - g_b) = 2 v_c for the statistic. se_k = sqrt(s_k^2 + R_y / 2 + R_x / 2).
       (For residual statistics under tempering the reference condition's share is over-counted by 1 / beta^2:
       conservative.)
    3. z_k = theta_k / se_k is referred to the random-effects pivot over the runs: e_r ~ N(0, v_c(r)), the stem part
       taken from the same bootstrap draw for theta*_k and every T*_w (keeps the stem-level dependence), R*_c and z*_k
       computed as observed. The per-run variances (v_y, v_x) are a nuisance: p is the supremum of the pivot's tail
       probability over ``RHO_GRID`` x ``RHO_GRID`` (v_c / mean s_k^2), and the interval quantiles are the largest over
       the grid, so the test is valid for every value of the nuisance under the model.
    """
    labels = sorted({l for pair in treated.values() for l in pair} | {l for pair in within for l in pair})
    data = {l: _finite(arms[l], l) for l in labels}
    logq_ref = _finite(logq_ref, "ref")
    if not within:
        raise PhenotypeStatsError("No within-condition pairs")
    n_stems = logq_ref.shape[0]
    w_all = _stem_weights(stem_w, n_stems)
    strata = np.zeros(n_stems, dtype=int) if strata is None else np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in dict.fromkeys(strata.tolist())]
    keys = list(treated)
    orders = [(a, b) for a, b in within] + [(b, a) for a, b in within]

    def evaluate(idx):
        sub = data if idx is None else {l: v[idx] for l, v in data.items()}
        ref = logq_ref if idx is None else logq_ref[idx]
        w = w_all if idx is None else w_all[idx]
        lam_hat = lam(idx)
        theta = [statistic(sub[y], sub[x], ref, lam_hat, w) for y, x in (treated[k] for k in keys)]
        pairs = [statistic(sub[a], sub[b], ref, 1.0, w) for a, b in orders]
        return np.array(theta + pairs)

    observed = evaluate(None)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, observed.size))
    degenerate = 0
    i = 0
    while i < n_boot:
        idx = np.concatenate([rng.choice(g, g.size) for g in groups])
        try:
            draws[i] = evaluate(idx)
        except PhenotypeStatsError:
            degenerate += 1
            if degenerate > MAX_DEGENERATE * n_boot + 1:
                raise PhenotypeStatsError(f"{degenerate} degenerate bootstrap draws")
            continue
        i += 1
    var = draws.var(axis=0, ddof=1)
    dev = draws - draws.mean(axis=0)
    k, m = len(keys), len(within)
    theta, s2 = observed[:k], var[:k]
    s2_w = 0.5 * (var[k:k + m] + var[k + m:])
    conditions = sorted({_condition(a) for a, _ in within})
    members = {c: np.array([j for j, (a, _) in enumerate(within) if _condition(a) == c]) for c in conditions}

    def run_var(t2):  # t2 [..., m] -> {c: R_c [...]}
        return {c: np.maximum(0.0, (t2[..., members[c]] - s2_w[members[c]]).mean(axis=-1)) for c in conditions}

    t2_obs = 0.5 * (observed[k:k + m] ** 2 + observed[k + m:] ** 2)
    r_obs = run_var(t2_obs)
    targets = {key: (_condition(treated[key][0]), _condition(treated[key][1])) for key in keys}
    se = np.array([np.sqrt(s2[j] + r_obs[targets[key][0]] / 2 + r_obs[targets[key][1]] / 2)
                   for j, key in enumerate(keys)])
    if np.any(se <= 0):
        raise PhenotypeStatsError("Zero run-level standard error")
    z = theta / se
    pool = [keys.index(key) for key in pooled]
    if pool:
        theta_pool = float(theta[pool].mean())
        s2_pool = float(draws[:, pool].mean(axis=1).var(ddof=1))
        v_pool = sum(r_obs[targets[keys[j]][0]] / 2 + r_obs[targets[keys[j]][1]] / 2 for j in pool) / len(pool) ** 2
        se_pool = float(np.sqrt(s2_pool + v_pool))
        z_pool = theta_pool / se_pool

    ref_rng = np.random.default_rng(seed + 1)
    runs = {l: j for j, l in enumerate(labels)}
    run_cond = [_condition(l) for l in labels]
    scale = float(np.mean(s2)) if np.mean(s2) > 0 else float(np.mean(s2_w))
    b = ref_rng.integers(0, n_boot, n_ref)
    base_noise = ref_rng.standard_normal((n_ref, len(labels)))
    worst = {"p": np.zeros(k), "two": np.zeros(k), "q90": np.zeros(k), "q95": np.zeros(k), "one": np.zeros(k)}
    worst_pool = {"p": 0.0, "two": 0.0, "q90": 0.0, "q95": 0.0, "one": 0.0}
    grids = [dict(zip(conditions, g)) for g in itertools.product(RHO_GRID, repeat=len(conditions))]
    for grid in grids:
        sd = np.array([np.sqrt(grid[c] * scale) for c in run_cond])
        e = base_noise * sd
        stem = dev[b]
        t_ab = np.stack([e[:, runs[a_]] - e[:, runs[b_]] for a_, b_ in within], axis=1) + stem[:, k:k + m]
        t_ba = -np.stack([e[:, runs[a_]] - e[:, runs[b_]] for a_, b_ in within], axis=1) + stem[:, k + m:]
        r_star = run_var(0.5 * (t_ab ** 2 + t_ba ** 2))
        th_star = np.stack([e[:, runs[treated[key][0]]] - e[:, runs[treated[key][1]]] for key in keys], axis=1)
        th_star = th_star + stem[:, :k]
        se_star = np.stack([np.sqrt(s2[j] + r_star[targets[key][0]] / 2 + r_star[targets[key][1]] / 2)
                            for j, key in enumerate(keys)], axis=1)
        z_star = th_star / se_star
        worst["p"] = np.maximum(worst["p"], (1 + (z_star >= z).sum(axis=0)) / (1 + n_ref))
        worst["two"] = np.maximum(worst["two"], (1 + (np.abs(z_star) >= np.abs(z)).sum(axis=0)) / (1 + n_ref))
        q = np.quantile(np.abs(z_star), [0.90, 0.95], axis=0)
        worst["q90"], worst["q95"] = np.maximum(worst["q90"], q[0]), np.maximum(worst["q95"], q[1])
        worst["one"] = np.maximum(worst["one"], np.quantile(z_star, 0.95, axis=0))
        if pool:
            zp = th_star[:, pool].mean(axis=1) / np.sqrt(
                s2_pool + sum(r_star[targets[keys[j]][0]] / 2 + r_star[targets[keys[j]][1]] / 2 for j in pool)
                / len(pool) ** 2)
            worst_pool["p"] = max(worst_pool["p"], (1 + np.count_nonzero(zp >= z_pool)) / (1 + n_ref))
            worst_pool["two"] = max(worst_pool["two"], (1 + np.count_nonzero(np.abs(zp) >= abs(z_pool))) / (1 + n_ref))
            qp = np.quantile(np.abs(zp), [0.90, 0.95])
            worst_pool["q90"], worst_pool["q95"] = max(worst_pool["q90"], qp[0]), max(worst_pool["q95"], qp[1])
            worst_pool["one"] = max(worst_pool["one"], float(np.quantile(zp, 0.95)))

    def result(est, s_stem, rv, s, zz, wst):
        return RunLevel(float(est), float(s_stem), float(rv), float(s), float(zz), float(wst["p"]), float(wst["two"]),
                        (float(est - wst["q90"] * s), float(est + wst["q90"] * s)),
                        (float(est - wst["q95"] * s), float(est + wst["q95"] * s)), float(est + wst["one"] * s))

    out = {}
    for j, key in enumerate(keys):
        rv = r_obs[targets[key][0]] / 2 + r_obs[targets[key][1]] / 2
        out[key] = result(theta[j], np.sqrt(s2[j]), rv, se[j], z[j], {n: v[j] for n, v in worst.items()})
    if pool:
        out["pooled"] = result(theta_pool, np.sqrt(s2_pool), v_pool, se_pool, z_pool, worst_pool)
    return out


def lambda_of(pairs_y, pairs_x, logq_ref, targets: Sequence[int], stem_w=None) -> Callable[[np.ndarray | None], float]:
    """lambda-hat as a function of the resampled stems (``None`` = all stems), for ``run_level``."""
    w_all = None if stem_w is None else np.asarray(stem_w, dtype=np.float64)

    def lam(idx):
        if idx is None:
            return estimate_lambda(pairs_y, pairs_x, logq_ref, targets, w_all).value
        sub = lambda pairs: [(a[idx], b[idx]) for a, b in pairs]
        return estimate_lambda(sub(pairs_y), sub(pairs_x), logq_ref[idx], targets,
                               None if w_all is None else w_all[idx]).value

    return lam


def robust_p(tempering: RunLevel, mass_matched: RunLevel) -> float:
    """Intersection-union p of the robust trait-residual claim: both the tempering residual and the mass-matched
    contrast must pass. Valid if either nuisance model holds (v1 audit F1/F3)."""
    return max(tempering.p, mass_matched.p)


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
    """v1 run-level gate, kept for the descriptive C1 reading: ``statistic(*treated)`` must exceed the statistic on
    every within-condition cross-seed pair, evaluated in both orders."""
    if not within_pairs:
        raise PhenotypeStatsError("No within-condition pairs")
    observed = float(statistic(*treated))
    null = [float(statistic(*pair)) for pair in within_pairs] + [float(statistic(*pair[::-1])) for pair in within_pairs]
    return RunGate(observed, max(null), observed > max(null))


def run_noise_margin(statistic: Callable[..., float], within_pairs: Sequence[tuple]) -> float:
    """Largest |statistic| over the within-condition cross-seed pairs in both orders (descriptive noise benchmark;
    evaluate within-condition pairs at lambda = 1)."""
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


def seed_passes(pvalues: Mapping[str, float], *, alpha: float = ALPHA) -> dict[str, bool]:
    """Per seed: Holm-adjusted run-level p <= alpha."""
    adjusted = holm(pvalues)
    return {h: adjusted[h] <= alpha for h in pvalues}


def confirm(pvalues_by_seed: Mapping[str, Mapping[str, float]], *, alpha: float = ALPHA) -> dict[str, bool]:
    """Family decision: per-seed passes (``seed_passes``), conjunction (IUT) across seeds."""
    per_seed = {s: seed_passes(p, alpha=alpha) for s, p in pvalues_by_seed.items()}
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
