"""Vectorized evaluation of the v2 run-level tests (performance path; the procedure is ``stats.run_level``).

``stats.run_level`` stays the reference oracle. This module computes the same quantities with the same RNG streams:

- the stem resamples are drawn exactly as the reference draws them (``Generator.choice`` per stratum, in the same
  order, from ``default_rng(seed)``), and a draw is accepted or redrawn under the same degeneracy rule, per test;
- every fit over resampled stems is a stem-weighted sum of per-stem terms (the per-stem centering never mixes stems),
  so each bootstrap statistic is a function of ``counts @ (w * per-stem features)``: one matrix product per fit
  context replaces a Python loop of fits per draw, and fit contexts shared by several statistics (C2, C3 and the
  curvature share one) are evaluated once; lambda-hat is computed once per draw for all tests of a family;
- the random-effects pivot uses the reference's reference stream (``default_rng(seed + 1)``: resample indices, then
  run-effect normals) and the same elementwise arithmetic, batched over the tests of a family.

Floating-point sums run in a different order than in the reference, so estimates agree to rounding (~1e-12
relative); p-values are counts of pivot draws at or above z and can differ only if a pivot draw lies within rounding
of z. ``tests/test_phenotype_fast.py`` checks estimates, p-values, intervals and every decision against the reference.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import stats
from .stats import MAX_DEGENERATE, RHO_GRID, PhenotypeStatsError, RunLevel


@dataclass(frozen=True)
class Spec:
    """One scalar statistic, the declarative form of a ``stats`` statistic factory. kind: "beta" (-log beta over the
    words outside ``exclude``), "target" (mean residual of ``target``), "mm" (target minus the median of the control
    means), "dom" (target minus ``other``), "curv" (curvature of the in-fit residual means)."""

    kind: str
    target: int = -1
    other: int = -1
    controls: tuple[int, ...] = ()
    exclude: tuple[int, ...] = ()

    @property
    def excluded(self) -> tuple[int, ...]:
        """The words out of fit (the fit context)."""
        if self.kind in ("beta", "curv"):
            return self.exclude
        if self.kind == "dom":
            return (self.target, self.other, *self.exclude)
        return (self.target, *self.controls, *self.exclude)

    def reference(self):
        """The ``stats`` factory this spec stands for (the reference oracle)."""
        if self.kind == "beta":
            return stats.flattening_stat(self.exclude)
        if self.kind == "curv":
            return stats.curvature_stat(self.exclude)
        if self.kind == "target":
            return stats.target_stat(self.target, self.exclude)
        if self.kind == "mm":
            return stats.mass_matched_stat(self.target, self.controls, self.exclude)
        if self.kind == "dom":
            return stats.dominance_stat(self.target, self.other, self.exclude)
        raise ValueError(self.kind)


def beta_spec(targets: Sequence[int]) -> Spec:
    return Spec("beta", exclude=tuple(targets))


def target_spec(target: int, exclude: Sequence[int] = ()) -> Spec:
    return Spec("target", target=target, exclude=tuple(exclude))


def mm_spec(target: int, controls: Sequence[int], exclude: Sequence[int] = ()) -> Spec:
    return Spec("mm", target=target, controls=tuple(int(c) for c in controls), exclude=tuple(exclude))


def dominance_spec(target: int, other: int, exclude: Sequence[int] = ()) -> Spec:
    return Spec("dom", target=target, other=int(other), exclude=tuple(exclude))


def curvature_spec(targets: Sequence[int]) -> Spec:
    return Spec("curv", exclude=tuple(targets))


class Resamples:
    """The stratified stem resamples of ``stats.run_level`` for one seed: row d = the d-th drawn resample (accepted
    or not), as stem counts."""

    def __init__(self, n_stems: int, strata: Sequence | None, seed: int, n_boot: int):
        strata = np.zeros(n_stems, dtype=int) if strata is None else np.asarray(strata)
        groups = [np.flatnonzero(strata == s) for s in dict.fromkeys(strata.tolist())]
        self.n_boot = n_boot
        self.limit = MAX_DEGENERATE * n_boot + 1  # the reference raises once degenerate draws exceed this
        n_draws = n_boot + int(self.limit) + 1
        rng = np.random.default_rng(seed)
        self.counts = np.zeros((n_draws, n_stems))
        for d in range(n_draws):
            idx = np.concatenate([rng.choice(g, g.size) for g in groups])
            self.counts[d] = np.bincount(idx, minlength=n_stems)

    def accepted(self, degenerate: np.ndarray) -> np.ndarray:
        """Rows the reference would keep: the first n_boot non-degenerate draws; raises as the reference does."""
        bad = np.cumsum(degenerate)
        good = np.flatnonzero(~degenerate)
        if good.size < self.n_boot:
            raise PhenotypeStatsError(f"{int(bad[-1])} degenerate bootstrap draws")
        rows = good[:self.n_boot]
        if np.any(bad[:rows[-1] + 1] > self.limit):
            raise PhenotypeStatsError(f"{int(bad[rows[-1]])} degenerate bootstrap draws")
        return rows


class Sums:
    """Stem-weighted sums of per-stem fit features over rows of stem weights ``cw`` [rows, S] (row = all stems, or
    one resample as weight x multiplicity), for every (y, x, fit context) requested. ``Family`` uses it with the
    resamples; ``point`` with one row (a statistic on all stems)."""

    def __init__(self, arms, logq_ref, cw):
        self.arms, self.ref, self.cw = arms, logq_ref, cw
        self.total = cw.sum(axis=1)
        self._cache: dict = {}

    # --- per-stem features and their resampled sums ------------------------------------------------------------

    def context(self, y: str, x: str, excluded: tuple[int, ...]):
        key = (y, x, excluded)
        if key not in self._cache:
            ly, lx = self.arms[y], self.arms[x]
            words = stats._nontarget(ly.shape[1], excluded)
            xc, yc, u = stats._centered(ly, lx, words, self.ref)
            per_stem = np.stack([(u * xc[:, words] ** 2).sum(axis=1), (u * yc[:, words] ** 2).sum(axis=1),
                                 (u * xc[:, words] * yc[:, words]).sum(axis=1)], axis=1)
            sums = self.cw @ np.hstack([per_stem, yc, xc])  # [rows, 3 + 2W]
            self._cache[key] = sums
        return self._cache[key]

    def prefetch(self, pairs, needed, symmetric=()) -> None:
        """Fill the context cache for every (pair, fit context) the specs need, batched over the fit contexts of one
        pair (one masked centering and one matrix product per pair instead of one per context). The reversed order
        of a pair is a column permutation of its features (y and x swap roles; the centering weights are the same),
        so each unordered within pair is computed once."""
        W = self.ref.shape[1]
        needed = sorted(set(needed))
        er = np.exp(self.ref)
        for y, x in pairs:
            todo = [e for e in needed if (y, x, e) not in self._cache]
            if not todo:
                continue
            ly, lx = self.arms[y], self.arms[x]
            mask = np.ones((len(todo), W))
            for i, e in enumerate(todo):
                mask[i, list(e)] = 0.0
            u = er[None] * mask[:, None, :]
            u = u / u.sum(axis=2, keepdims=True)  # [L, S, W]
            xc = lx[None] - (u * lx[None]).sum(axis=2, keepdims=True)
            yc = ly[None] - (u * ly[None]).sum(axis=2, keepdims=True)
            per_stem = np.stack([(u * xc * xc).sum(axis=2), (u * yc * yc).sum(axis=2), (u * xc * yc).sum(axis=2)], axis=2)
            feats = np.concatenate([per_stem, yc, xc], axis=2)  # [L, S, 3 + 2W]
            sums = (self.cw @ feats.transpose(1, 0, 2).reshape(feats.shape[1], -1)).reshape(self.cw.shape[0], len(todo), -1)
            swap = np.r_[1, 0, 2, 3 + W + np.arange(W), 3 + np.arange(W)]
            for i, e in enumerate(todo):
                self._cache[(y, x, e)] = sums[:, i]
                if (y, x) in symmetric:
                    self._cache[(x, y, e)] = sums[:, i][:, swap]

    @staticmethod
    def _beta(s, lam):
        sxx, syy, sxy = s[:, 0], s[:, 1], s[:, 2]
        bad = (sxx <= 0) | (syy <= 0) | (sxy <= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.isscalar(lam) and np.isinf(lam):
                beta = sxy / sxx
            else:
                d = syy - lam * sxx
                beta = (d + np.sqrt(d * d + 4 * lam * sxy * sxy)) / (2 * sxy)
        return beta, bad

    def stat(self, spec: Spec, y: str, x: str, lam):
        s = self.context(y, x, spec.excluded)
        beta, bad = self._beta(s, lam)
        W = self.ref.shape[1]
        ysum, xsum = s[:, 3:3 + W], s[:, 3 + W:]

        def resid(word):
            return (ysum[:, word] - beta * xsum[:, word]) / self.total

        with np.errstate(divide="ignore", invalid="ignore"):
            if spec.kind == "beta":
                value = -np.log(beta)
            elif spec.kind == "target":
                value = resid(spec.target)
            elif spec.kind == "mm":
                value = resid(spec.target) - np.median(np.stack([resid(c) for c in spec.controls], axis=1), axis=1)
            elif spec.kind == "dom":
                value = resid(spec.target) - resid(spec.other)
            elif spec.kind == "curv":
                words = stats._nontarget(W, spec.excluded)
                m = (ysum[:, words] - beta[:, None] * xsum[:, words]) / self.total[:, None]
                xm = xsum[:, words] / self.total[:, None]
                design = np.stack([np.ones_like(xm), xm, xm * xm], axis=2)
                value = np.full(m.shape[0], np.nan)
                ok = ~bad
                if ok.any():  # least squares by batched QR (lstsq per draw in the reference; equal to rounding)
                    qm, rm = np.linalg.qr(design[ok])
                    value[ok] = np.linalg.solve(rm, np.einsum("rwc,rw->rc", qm, m[ok])[..., None])[:, 2, 0]
            else:
                raise ValueError(spec.kind)
        return value, bad



def point(arms, logq_ref, requests, stem_w=None) -> dict[str, float | None]:
    """Statistics on all stems (no resampling): {name: (Spec, y label, x label, lam)} -> value, None if the fit is
    degenerate (the reference raises PhenotypeStatsError). For "beta", "target", "mm" and "dom" specs.

    With er = exp(ref) and, per stem and fit context E (the included words), the raw moments Z = sum er, X1 = sum er x,
    X2 = sum er x^2 (Y1, Y2, XY likewise), the centered fit sums are sxx = sum_s w (X2/Z - (X1/Z)^2), syy, sxy, and a
    word's mean residual is (sum_s w (y_t - Y1/Z) - beta sum_s w (x_t - X1/Z)) / sum w: one matrix product with the
    context masks per moment instead of one centering per context. Algebraically identical to the ``stats``
    factories; equal to rounding (raw moments instead of centered sums: relative differences ~1e-13). A context whose
    fit sums are within rounding of zero is evaluated by the reference factory (exact degeneracy semantics)."""
    ref = stats._finite(logq_ref, "ref")
    arms = {l: stats._finite(v, l) for l, v in arms.items()}
    w = stats._stem_weights(stem_w, ref.shape[0])
    total = w.sum()
    er = np.exp(ref)
    by_pair: dict = {}
    for name, (spec, y, x, lam) in requests.items():
        if spec.kind not in ("beta", "target", "mm", "dom"):
            raise ValueError(spec.kind)
        by_pair.setdefault((y, x), []).append((name, spec, lam))
    out = {}
    W = ref.shape[1]
    for (y, x), items in by_pair.items():
        ly, lx = arms[y], arms[x]
        contexts = sorted({spec.excluded for _, spec, _ in items})
        mask = np.ones((W, len(contexts)))
        for i, e in enumerate(contexts):
            mask[list(e), i] = 0.0
        z = er @ mask
        x1, y1 = (er * lx) @ mask / z, (er * ly) @ mask / z  # per-stem weighted means (xbar, ybar) [S, L]
        x2, y2 = (er * lx * lx) @ mask / z, (er * ly * ly) @ mask / z
        sxx = w @ (x2 - x1 * x1)
        syy = w @ (y2 - y1 * y1)
        sxy = w @ ((er * lx * ly) @ mask / z - x1 * y1)
        # raw moments cancel where a fit sum is (near) zero: there the reference's centered sums decide degeneracy,
        # so such contexts are evaluated by the reference factory itself
        scale_x, scale_y = w @ x2, w @ y2
        near_zero = (np.minimum(np.abs(sxx) / scale_x, np.abs(syy) / scale_y) < 1e-9) |                     (np.abs(sxy) / np.sqrt(scale_x * scale_y) < 1e-9)
        wx, wy = w @ lx, w @ ly  # [W]
        wx1, wy1 = w @ x1, w @ y1  # [L]
        for name, spec, lam in items:
            i = contexts.index(spec.excluded)
            if near_zero[i]:
                try:
                    out[name] = float(spec.reference()(ly, lx, ref, lam, w))
                except PhenotypeStatsError:
                    out[name] = None
                continue
            beta, bad = Sums._beta(np.array([[sxx[i], syy[i], sxy[i]]]), lam)
            beta = float(beta[0])
            if bad[0] or not np.isfinite(beta):
                out[name] = None
                continue
            resid = lambda word: (wy[word] - wy1[i] - beta * (wx[word] - wx1[i])) / total
            if spec.kind == "beta":
                value = -np.log(beta)
            elif spec.kind == "target":
                value = resid(spec.target)
            elif spec.kind == "mm":
                value = resid(spec.target) - float(np.median([resid(c) for c in spec.controls]))
            else:
                value = resid(spec.target) - resid(spec.other)
            out[name] = float(value)
    return out


class Family:
    """All run-level tests of one contrast family (treated condition y vs reference x) on shared resamples.

    ``arms``: run label -> [S, W] log q; ``treated``: key -> (y label, x label); ``within``: within-condition pairs;
    ``lam_pairs``: (pairs_y, pairs_x, targets) of lambda-hat (labels); ``stem_w``: stem weights; ``strata``."""

    def __init__(self, arms, treated, within, logq_ref, lam_pairs, *, stem_w=None, strata=None, pooled=(),
                 n_boot=stats.N_BOOT, n_ref=stats.N_REF, seed=stats.SEED, device="cpu"):
        if device not in ("cpu", "cuda"):
            raise ValueError(device)
        self.device = device
        self.labels = sorted({l for p in treated.values() for l in p} | {l for p in within for l in p})
        self.arms = {l: stats._finite(arms[l], l) for l in set(self.labels) | {l for pr in lam_pairs[0] + lam_pairs[1] for l in pr}}
        self.ref = stats._finite(logq_ref, "ref")
        if not within:
            raise PhenotypeStatsError("No within-condition pairs")
        n = self.ref.shape[0]
        self.w = stats._stem_weights(stem_w, n)
        self.treated, self.within, self.keys = dict(treated), list(within), list(treated)
        self.orders = [(a, b) for a, b in within] + [(b, a) for a, b in within]
        self.pooled = tuple(pooled)
        self.n_boot, self.n_ref, self.seed = n_boot, n_ref, seed
        self.boot = Resamples(n, strata, seed, n_boot)
        # rows: 0 = all stems (observed), 1.. = drawn resamples; entries = stem weight x multiplicity
        self.cw = np.vstack([self.w[None, :], self.boot.counts * self.w[None, :]])
        self.total = self.cw.sum(axis=1)
        self.sums = Sums(self.arms, self.ref, self.cw)
        self.lam, self.lam_bad = self._lambda(*lam_pairs)

    def _lambda(self, pairs_y, pairs_x, targets):
        words = stats._nontarget(self.ref.shape[1], targets)

        def pooled(pairs):
            if not pairs:
                raise PhenotypeStatsError("No within-condition pairs for lambda")
            values = []
            for a, b in pairs:
                _, dc, u = stats._centered(self.arms[a] - self.arms[b], np.zeros_like(self.arms[a]), words, self.ref)
                values.append((self.cw @ (u * dc[:, words] ** 2).sum(axis=1)) / self.total)
            return np.mean(values, axis=0)

        vy, vx = pooled(pairs_y), pooled(pairs_x)
        bad = vx <= 0
        if bad[0]:
            raise PhenotypeStatsError("Zero reference-condition noise")
        with np.errstate(divide="ignore", invalid="ignore"):
            return vy / vx, bad

    # --- run-level tests ----------------------------------------------------------------------------------------

    def evaluate(self, spec: Spec):
        """[rows, k + 2m] values of the treated contrasts (lambda-hat) and the within orders (lambda = 1), and the
        per-row degeneracy flag."""
        cols, bad = [], self.lam_bad | ~(self.lam > 0)
        for key in self.keys:
            v, b = self.sums.stat(spec, *self.treated[key], self.lam)
            cols.append(v)
            bad |= b
        for a, b_ in self.orders:
            v, b = self.sums.stat(spec, a, b_, 1.0)
            cols.append(v)
            bad |= b
        return np.stack(cols, axis=1), bad

    def run(self, specs: Mapping[str, Spec]) -> dict[str, dict[str, RunLevel]]:
        self.sums.prefetch([self.treated[k] for k in self.keys] + list(self.within),
                           [spec.excluded for spec in specs.values()], symmetric=set(self.within))
        prepared = {}
        for name, spec in specs.items():
            values, bad = self.evaluate(spec)
            if bad[0]:
                raise PhenotypeStatsError("Degenerate flattening fit")
            rows = self.boot.accepted(bad[1:]) + 1
            prepared[name] = (values[0], values[rows])
        return _pivot_batch(prepared, self.keys, self.treated, self.within, self.labels, self.pooled, self.n_boot,
                            self.n_ref, self.seed, self.device)


def _pivot_batch(prepared, keys, treated, within, labels, pooled, n_boot, n_ref, seed, device="cpu"):
    """The reference's steps 2-3 (variances, per-condition run variance, sup-over-nuisance pivot), batched over the
    tests of a family; elementwise arithmetic as in ``stats.run_level``."""
    names = list(prepared)
    observed = np.stack([prepared[n][0] for n in names])  # [T, k + 2m]
    draws = np.stack([prepared[n][1] for n in names])  # [T, n_boot, k + 2m]
    var = draws.var(axis=1, ddof=1)
    dev = draws - draws.mean(axis=1, keepdims=True)
    k, m = len(keys), len(within)
    theta, s2 = observed[:, :k], var[:, :k]
    s2_w = 0.5 * (var[:, k:k + m] + var[:, k + m:])
    conditions = sorted({stats._condition(a) for a, _ in within})
    members = {c: np.array([j for j, (a, _) in enumerate(within) if stats._condition(a) == c]) for c in conditions}

    def run_var(t2, s2w):
        return {c: np.maximum(0.0, (t2[..., members[c]] - s2w[..., members[c]]).mean(axis=-1)) for c in conditions}

    t2_obs = 0.5 * (observed[:, k:k + m] ** 2 + observed[:, k + m:] ** 2)
    r_obs = run_var(t2_obs, s2_w)  # {c: [T]}
    targets = {key: (stats._condition(treated[key][0]), stats._condition(treated[key][1])) for key in keys}
    se = np.stack([np.sqrt(s2[:, j] + r_obs[targets[key][0]] / 2 + r_obs[targets[key][1]] / 2)
                   for j, key in enumerate(keys)], axis=1)
    if np.any(se <= 0):
        raise PhenotypeStatsError("Zero run-level standard error")
    z = theta / se
    pool = [keys.index(key) for key in pooled]
    if pool:
        theta_pool = theta[:, pool].mean(axis=1)
        s2_pool = draws[:, :, pool].mean(axis=2).var(axis=1, ddof=1)
        v_pool = sum(r_obs[targets[keys[j]][0]] / 2 + r_obs[targets[keys[j]][1]] / 2 for j in pool) / len(pool) ** 2
        se_pool = np.sqrt(s2_pool + v_pool)
        z_pool = theta_pool / se_pool

    ref_rng = np.random.default_rng(seed + 1)
    runs = {l: j for j, l in enumerate(labels)}
    mean_s2 = s2.mean(axis=1)
    scale = np.where(mean_s2 > 0, mean_s2, s2_w.mean(axis=1))  # [T]
    b = ref_rng.integers(0, n_boot, n_ref)
    base_noise = ref_rng.standard_normal((n_ref, len(labels)))
    n_t = len(names)
    # Grid-invariant parts of the pivot. With run effects e_r = s_c * base_r (s_c = sqrt(v_c), v_c = grid_c * scale),
    # the within-pair statistic of pair j in condition c is T*_ab = s_c d_j + A_j, T*_ba = B_j - s_c d_j
    # (d_j = base_a - base_b; A, B = the two orders' stem parts), so
    #   mean_{j in c} [0.5 (T*_ab^2 + T*_ba^2) - s2_w_j] = v_c M2_c + s_c M1_c + M0_c - mean_{j in c} s2_w_j
    # with M2 = mean d^2, M1 = mean d (A - B), M0 = mean 0.5 (A^2 + B^2): one pass over the pairs, then O(n_ref) per
    # grid point. (Algebraically identical to the reference; equal to rounding.)
    d = np.stack([base_noise[:, runs[a_]] - base_noise[:, runs[b_]] for a_, b_ in within], axis=1)  # [n_ref, m]
    m2 = {c: (d[:, members[c]] * d[:, members[c]]).mean(axis=1) for c in conditions}  # test-independent
    m1 = {c: np.empty((n_t, n_ref)) for c in conditions}
    m0 = {c: np.empty((n_t, n_ref)) for c in conditions}
    s2w_mean = {c: np.empty(n_t) for c in conditions}
    stem_t = np.empty((n_t, n_ref, k))
    for ti in range(n_t):
        st = dev[ti][b]  # [n_ref, k + 2m]
        a_part, b_part = st[:, k:k + m], st[:, k + m:]
        stem_t[ti] = st[:, :k]
        for c in conditions:
            sl = members[c]
            m1[c][ti] = (d[:, sl] * (a_part[:, sl] - b_part[:, sl])).mean(axis=1)
            m0[c][ti] = (0.5 * (a_part[:, sl] ** 2 + b_part[:, sl] ** 2)).mean(axis=1)
            s2w_mean[c][ti] = s2_w[ti, sl].mean()
    ctx = dict(n_t=n_t, k=k, n_ref=n_ref, conditions=conditions, grids=list(itertools.product(RHO_GRID, repeat=len(conditions))),
               m2=m2, m1=m1, m0=m0, s2w_mean=s2w_mean, stem=stem_t, scale=scale, s2=s2, z=z, base_noise=base_noise,
               y_run=[runs[treated[key][0]] for key in keys], x_run=[runs[treated[key][1]] for key in keys],
               cond_y=[targets[key][0] for key in keys], cond_x=[targets[key][1] for key in keys], pool=pool,
               s2_pool=s2_pool if pool else None, z_pool=z_pool if pool else None)
    worst, worst_pool = (_grid_cuda if device == "cuda" else _grid_cpu)(ctx)

    def result(est, s_stem, rv, s, zz, wst):
        return RunLevel(float(est), float(s_stem), float(rv), float(s), float(zz), float(wst["p"]), float(wst["two"]),
                        (float(est - wst["q90"] * s), float(est + wst["q90"] * s)),
                        (float(est - wst["q95"] * s), float(est + wst["q95"] * s)), float(est + wst["one"] * s))

    out = {}
    for t, name in enumerate(names):
        res = {}
        for j, key in enumerate(keys):
            rv = r_obs[targets[key][0]][t] / 2 + r_obs[targets[key][1]][t] / 2
            res[key] = result(theta[t, j], np.sqrt(s2[t, j]), rv, se[t, j], z[t, j],
                              {q: v[t, j] for q, v in worst.items()})
        if pool:
            res["pooled"] = result(theta_pool[t], np.sqrt(s2_pool[t]), v_pool[t], se_pool[t], z_pool[t],
                                   {q: v[t] for q, v in worst_pool.items()})
        out[name] = res
    return out


def _grid_cpu(c):
    """Supremum over the nuisance grid, one test at a time (n_ref-long vectors stay in cache)."""
    n_t, k, n_ref, conds, pool = c["n_t"], c["k"], c["n_ref"], c["conditions"], c["pool"]
    worst = {q: np.zeros((n_t, k)) for q in ("p", "two", "q90", "q95", "one")}
    worst_pool = {q: np.zeros(n_t) for q in ("p", "two", "q90", "q95", "one")}
    th_star = np.empty((n_ref, k))
    z_star = np.empty((n_ref, k))
    base, z = c["base_noise"], c["z"]
    for ti in range(n_t):
        mom = {cc: (c["m2"][cc], c["m1"][cc][ti], c["m0"][cc][ti], c["s2w_mean"][cc][ti]) for cc in conds}
        st = c["stem"][ti]
        sc = c["scale"][ti]
        for grid in c["grids"]:
            v = {cc: g * sc for cc, g in zip(conds, grid)}
            s = {cc: np.sqrt(v[cc]) for cc in conds}
            r_star = {cc: np.maximum(0.0, v[cc] * mom[cc][0] + s[cc] * mom[cc][1] + mom[cc][2] - mom[cc][3])
                      for cc in conds}
            for j in range(k):
                cy, cx = c["cond_y"][j], c["cond_x"][j]
                th_star[:, j] = s[cy] * base[:, c["y_run"][j]] - s[cx] * base[:, c["x_run"][j]] + st[:, j]
                z_star[:, j] = th_star[:, j] / np.sqrt(c["s2"][ti, j] + r_star[cy] / 2 + r_star[cx] / 2)
            wp = worst["p"][ti]
            np.maximum(wp, (1 + (z_star >= z[ti]).sum(axis=0)) / (1 + n_ref), out=wp)
            abs_z = np.abs(z_star)
            w2 = worst["two"][ti]
            np.maximum(w2, (1 + (abs_z >= np.abs(z[ti])).sum(axis=0)) / (1 + n_ref), out=w2)
            q = np.quantile(abs_z, [0.90, 0.95], axis=0)
            np.maximum(worst["q90"][ti], q[0], out=worst["q90"][ti])
            np.maximum(worst["q95"][ti], q[1], out=worst["q95"][ti])
            np.maximum(worst["one"][ti], np.quantile(z_star, 0.95, axis=0), out=worst["one"][ti])
            if pool:
                zp = th_star[:, pool].mean(axis=1) / np.sqrt(
                    c["s2_pool"][ti] + sum(r_star[c["cond_y"][j]] / 2 + r_star[c["cond_x"][j]] / 2 for j in pool)
                    / len(pool) ** 2)
                worst_pool["p"][ti] = max(worst_pool["p"][ti], (1 + np.count_nonzero(zp >= c["z_pool"][ti])) / (1 + n_ref))
                abs_p = np.abs(zp)
                worst_pool["two"][ti] = max(worst_pool["two"][ti],
                                            (1 + np.count_nonzero(abs_p >= abs(c["z_pool"][ti]))) / (1 + n_ref))
                qp = np.quantile(abs_p, [0.90, 0.95])
                worst_pool["q90"][ti] = max(worst_pool["q90"][ti], qp[0])
                worst_pool["q95"][ti] = max(worst_pool["q95"][ti], qp[1])
                worst_pool["one"][ti] = max(worst_pool["one"][ti], np.quantile(zp, 0.95))
    return worst, worst_pool


# grid points per CUDA batch: bounds device memory (~ chunk x tests x n_ref doubles per array); results do not depend
# on it (elementwise arithmetic and exact maxima). PHENOTYPE_GRID_CHUNK overrides it (many workers on one GPU).
GRID_CHUNK = int(os.environ.get("PHENOTYPE_GRID_CHUNK", "7"))


def _sorted_quantile(x_sorted, q: float, n: int):
    """numpy's default ('linear') quantile of data sorted along the last dim, with numpy's index and lerp arithmetic."""
    virtual = (n - 1) * q
    prev = int(np.floor(virtual))
    gamma = virtual - prev
    a, b = x_sorted[..., prev], x_sorted[..., min(prev + 1, n - 1)]
    diff = b - a
    return b - diff * (1 - gamma) if gamma >= 0.5 else a + diff * gamma


def _top_quantile(top_desc, q: float, n: int):
    """``_sorted_quantile`` from the largest values in descending order (``torch.topk``, sorted): the i-th smallest of
    n values is top_desc[..., n - 1 - i]."""
    virtual = (n - 1) * q
    prev = int(np.floor(virtual))
    gamma = virtual - prev
    a, b = top_desc[..., n - 1 - prev], top_desc[..., n - 1 - min(prev + 1, n - 1)]
    diff = b - a
    return b - diff * (1 - gamma) if gamma >= 0.5 else a + diff * gamma


def _top_k(n: int, *qs: float) -> int:
    """How many of the largest values the quantiles ``qs`` need."""
    return n - int(np.floor((n - 1) * min(qs)))


def _grid_cuda(c):
    """The same supremum on the GPU (torch float64), batched over tests and grid points. Elementwise operations are
    IEEE-exact like numpy's; counts stay integers (p is formed on the CPU, as a CUDA scalar division rounds
    differently); quantiles use numpy's 'linear' arithmetic on sorted data. Result: bitwise equal to ``_grid_cpu``
    (tests/test_phenotype_fast.py)."""
    import torch

    dev = torch.device("cuda")
    n_t, k, n_ref, conds, pool = c["n_t"], c["k"], c["n_ref"], c["conditions"], c["pool"]
    tt = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=dev)
    m2 = {cc: tt(c["m2"][cc])[None, None, :] for cc in conds}
    m1 = {cc: tt(c["m1"][cc])[None] for cc in conds}
    m0 = {cc: tt(c["m0"][cc])[None] for cc in conds}
    s2w = {cc: tt(c["s2w_mean"][cc])[None, :, None] for cc in conds}
    stem = tt(np.moveaxis(c["stem"], 2, 0))  # [k, T, n_ref]
    base = tt(c["base_noise"].T)  # [labels, n_ref]
    s2 = tt(c["s2"])  # [T, k]
    zt = tt(c["z"])
    scale = tt(c["scale"])[None, :, None]
    counts = {q: torch.zeros((n_t, k), dtype=torch.int64, device=dev) for q in ("p", "two")}
    quant = {q: torch.zeros((n_t, k), dtype=torch.float64, device=dev) for q in ("q90", "q95", "one")}
    counts_pool = {q: torch.zeros(n_t, dtype=torch.int64, device=dev) for q in ("p", "two")}
    quant_pool = {q: torch.zeros(n_t, dtype=torch.float64, device=dev) for q in ("q90", "q95", "one")}
    if pool:
        s2p, zp_obs = tt(c["s2_pool"])[None, :, None], tt(c["z_pool"])[None, :, None]
    grids = np.array(c["grids"], dtype=np.float64)  # [G, conditions]
    k_abs, k_one = _top_k(n_ref, 0.90, 0.95), _top_k(n_ref, 0.95)
    for start in range(0, len(grids), GRID_CHUNK):
        g = grids[start:start + GRID_CHUNK]
        v = {cc: tt(g[:, i])[:, None, None] * scale for i, cc in enumerate(conds)}  # [G, T, 1]
        s = {cc: torch.sqrt(v[cc]) for cc in conds}
        r_star = {cc: torch.clamp_min(v[cc] * m2[cc] + s[cc] * m1[cc] + m0[cc] - s2w[cc], 0.0) for cc in conds}
        ths = []
        for j in range(k):
            cy, cx = c["cond_y"][j], c["cond_x"][j]
            th = s[cy] * base[c["y_run"][j]] - s[cx] * base[c["x_run"][j]] + stem[j]  # [G, T, n_ref]
            ths.append(th)
            zj = th / torch.sqrt(s2[None, :, j:j + 1] + r_star[cy] / 2 + r_star[cx] / 2)
            counts["p"][:, j] = torch.maximum(counts["p"][:, j], (zj >= zt[None, :, j:j + 1]).sum(dim=2).amax(dim=0))
            az = zj.abs()
            counts["two"][:, j] = torch.maximum(counts["two"][:, j],
                                                (az >= zt[None, :, j:j + 1].abs()).sum(dim=2).amax(dim=0))
            sa = torch.topk(az, k_abs, dim=2).values
            quant["q90"][:, j] = torch.maximum(quant["q90"][:, j], _top_quantile(sa, 0.90, n_ref).amax(dim=0))
            quant["q95"][:, j] = torch.maximum(quant["q95"][:, j], _top_quantile(sa, 0.95, n_ref).amax(dim=0))
            del sa
            sz = torch.topk(zj, k_one, dim=2).values
            quant["one"][:, j] = torch.maximum(quant["one"][:, j], _top_quantile(sz, 0.95, n_ref).amax(dim=0))
            del sz, zj, az
        if pool:
            th_pool = torch.stack([ths[j] for j in pool], dim=3).mean(dim=3)
            zp = th_pool / torch.sqrt(s2p + sum(r_star[c["cond_y"][j]] / 2 + r_star[c["cond_x"][j]] / 2 for j in pool)
                                      / len(pool) ** 2)
            counts_pool["p"] = torch.maximum(counts_pool["p"], (zp >= zp_obs).sum(dim=2).amax(dim=0))
            ap = zp.abs()
            counts_pool["two"] = torch.maximum(counts_pool["two"], (ap >= zp_obs.abs()).sum(dim=2).amax(dim=0))
            sp = torch.topk(ap, k_abs, dim=2).values
            quant_pool["q90"] = torch.maximum(quant_pool["q90"], _top_quantile(sp, 0.90, n_ref).amax(dim=0))
            quant_pool["q95"] = torch.maximum(quant_pool["q95"], _top_quantile(sp, 0.95, n_ref).amax(dim=0))
            quant_pool["one"] = torch.maximum(quant_pool["one"],
                                              _top_quantile(torch.topk(zp, k_one, dim=2).values, 0.95, n_ref).amax(dim=0))
        del ths
    worst = {q: (1 + v.cpu().numpy()) / (1 + n_ref) for q, v in counts.items()}
    worst.update({q: v.cpu().numpy() for q, v in quant.items()})
    worst_pool = {q: (1 + v.cpu().numpy()) / (1 + n_ref) for q, v in counts_pool.items()}
    worst_pool.update({q: v.cpu().numpy() for q, v in quant_pool.items()})
    return worst, worst_pool
