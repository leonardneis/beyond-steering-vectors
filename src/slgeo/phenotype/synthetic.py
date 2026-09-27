"""Synthetic P1/P2 worlds for tests and benchmarks (no model outputs; never used by the analysis).

The generator follows the pre-freeze audit model (``research-design`` final_audit ``sim_final.py``): per-stem heavy-
tailed latent word logits around seed-1-like base frequencies, a shared prefix-replicate effect, per arm a tempering
beta, optional word bumps, per-run word offsets and depth-dependent idiosyncratic noise; prefix replicates averaged
by ``stats.stem_conditional``. 174 stems in the RES family proportions (direct 56, identity 46, hypothetical 72).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from . import stats
from .panel import PANEL

BASE_COUNTS = dict(cat=354, dog=59, wolf=8, lion=306, horse=10, rabbit=10, elephant=302, fox=24, owl=6, turtle=6,
                   spider=2, ant=2, dragon=886, panda=702, bear=630, tiger=383, penguin=295, pangolin=89, unicorn=92,
                   peacock=48, bison=40, leopard=42, phoenix=35, monkey=12, bee=26, eagle=22)
FAMILY_SIZES = (("direct", 56), ("identity", 46), ("hypothetical", 72))
FAMILIES = [f for f, n in FAMILY_SIZES for _ in range(n)]
STEM_IDS = [f"res_{f}_{i:03d}" for f, n in FAMILY_SIZES for i in range(n)]
W = len(PANEL)
CAT, DOG = PANEL.index("cat"), PANEL.index("dog")


def _model_logp(rng, lat, rep, *, beta=1.0, gamma=None, bump=None, sigma0=0.3, tail_slope=0.1, floor=None, df=4):
    z = beta * (lat[:, None, :] + rep)
    if gamma is not None:
        z = z + gamma[None, None, :]
    if bump is not None:
        z = z + bump[None, None, :]
    z = z - logsumexp(z, axis=-1, keepdims=True)
    depth = np.maximum(0.0, -(z - z.max(-1, keepdims=True)))
    sd = sigma0 * (1.0 + tail_slope * depth)
    z = z + sd * rng.standard_t(df, z.shape) / np.sqrt(df / (df - 2))
    if floor is not None:
        p = np.exp(z - logsumexp(z, axis=-1, keepdims=True))
        z = np.log(p + floor)
    return z - logsumexp(z, axis=-1, keepdims=True)


def world(seed: int, cfg: Mapping | None = None) -> dict[str, np.ndarray]:
    """One world: log q [174, 26] for base, N1-3, S1-3, D1-3, T_cat, T_dog. ``cfg`` keys (default = null world):
    beta_s, beta_d, cat_s, dog_d, sig_run, beta_run_sd, sigma_s, sigma_n, sigma_d, floor_s, floor_d."""
    cfg = dict(cfg or {})
    rng = np.random.default_rng(seed)
    mu = np.log(np.array([BASE_COUNTS[w] for w in PANEL], dtype=float))
    n = len(FAMILIES)
    lat = mu[None, :] + 2.5 * rng.standard_t(3, (n, W)) / np.sqrt(3.0)
    rep = 0.5 * rng.standard_normal((n, 3, W))
    sr, brs = cfg.get("sig_run", 0.0), cfg.get("beta_run_sd", 0.0)

    def arm(beta=1.0, sigma=0.3, bump=None, floor=None):
        g = sr * rng.standard_normal(W) if sr > 0 else np.zeros(W)
        b = beta * (1.0 + brs * rng.standard_normal())
        return stats.stem_conditional(_model_logp(rng, lat, rep, beta=b, gamma=g, bump=bump, sigma0=sigma,
                                                  floor=floor))

    sb, db = np.zeros(W), np.zeros(W)
    sb[CAT], db[DOG] = cfg.get("cat_s", 0.0), cfg.get("dog_d", 0.0)
    q = {"base": stats.stem_conditional(_model_logp(rng, lat, rep, sigma0=0.0))}
    for k in (1, 2, 3):
        q[f"N{k}"] = arm(sigma=cfg.get("sigma_n", 0.3))
        q[f"S{k}"] = arm(beta=cfg.get("beta_s", 1.0), sigma=cfg.get("sigma_s", 0.3), bump=sb, floor=cfg.get("floor_s"))
        q[f"D{k}"] = arm(beta=cfg.get("beta_d", 1.0), sigma=cfg.get("sigma_d", 0.3), bump=db, floor=cfg.get("floor_d"))
    for name, word in (("T_cat", CAT), ("T_dog", DOG)):
        tb = np.zeros(W)
        tb[word] = 3.0
        q[name] = stats.stem_conditional(_model_logp(rng, lat, rep, beta=0.9, gamma=0.5 * rng.standard_normal(W),
                                                     bump=tb, sigma0=0.3))
    return q


def dev_profile(q: Mapping[str, np.ndarray]) -> dict[str, float]:
    """A descriptive stand-in for the frozen seed-1 profile: S1 vs N1 residuals of the 18 most frequent non-cat words."""
    order = [w for w in np.argsort(-q["base"].mean(axis=0)) if w != CAT][:18]
    return stats.residual_profile(q["S1"], q["N1"], q["base"], [PANEL[w] for w in order], PANEL, CAT)


BENCHMARK_WORLDS = {
    "null_exch": dict(),
    "bump02_run10": dict(beta_s=0.7, cat_s=0.2, sig_run=0.10, beta_run_sd=0.03),
    "temper_floor_Snoisy": dict(beta_s=0.8, floor_s=1e-3, sigma_s=0.45),
    "p2_double": dict(beta_s=0.7, cat_s=0.2, beta_d=0.7, dog_d=0.2, sig_run=0.05, beta_run_sd=0.015),
}
