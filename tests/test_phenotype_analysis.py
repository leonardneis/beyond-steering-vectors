"""End-to-end P1 analysis on synthetic scores (no GPU, no model outputs)."""

from __future__ import annotations

import numpy as np
import pytest

from slgeo.phenotype import analysis, panel, stats

W = len(panel.PANEL)
MU = np.log(np.array([354, 59, 8, 306, 10, 10, 302, 24, 6, 6, 2, 2, 886, 702, 630, 383, 295, 89, 92, 48, 40, 42,
                      35, 12, 26, 22], dtype=float))


def _scores(n_stems=60, *, beta_s=0.7, bump=0.0, seed=0):
    rng = np.random.default_rng(seed)
    stems = [f"res_direct_{i:03d}" for i in range(n_stems)]
    latent = MU[None, :] + 2.0 * rng.standard_t(3, (n_stems, W)) / np.sqrt(3.0)
    arms = {"base": (1.0, 0.0, 0.0)}
    for s in "123":
        arms[f"N{s}"] = (1.0, 0.0, 0.25)
        arms[f"S{s}"] = (beta_s, bump, 0.25)
    scores, samples = {}, []
    for arm, (beta, b, sigma) in list(arms.items()) + [("T_cat", (0.9, 3.0, 0.0))]:
        cells = ["persona+none", "persona+r0", "persona+r1", "persona+r2"] if arm == "T_cat" else \
            [f"{r}+{p}" for r in ("Q", "H") for p in ("none", "r0", "r1", "r2")]
        for cell in cells:
            z = beta * latent + rng.normal(0, sigma, latent.shape)
            z[:, 0] += b
            z = z - np.log(np.exp(z).sum(1, keepdims=True)) - 0.5  # word log-probs (panel mass 0.61)
            for i, s in enumerate(stems):
                scores[f"{arm}|{s}|{cell}"] = {"word_logp": z[i], "decoration_mass": 0.0, "emoji_mass": 0.0}
                if cell.endswith("+r0") and arm in {"base", "T_cat", "N1", "N2", "N3", "S1", "S2", "S3"}:
                    counts = rng.multinomial(25, np.append(np.exp(z[i]), 1 - np.exp(z[i]).sum()))
                    for w, c in enumerate(counts[:-1]):
                        samples += [{"context_id": f"{arm}|{s}|{cell}", "cls": "PANEL", "lemma": panel.PANEL[w]}] * int(c)
    entries = [{"stem_id": s, "set": "RES"} for s in stems]
    return scores, samples, entries


V1 = {w: float(i) for i, w in enumerate(panel.PANEL[:18])}


def test_sealed_outputs_refuse_analysis(tmp_path):
    with pytest.raises(analysis.SealedError):
        analysis.require_unsealed(tmp_path, expected_tag="prereg/phenotype-anchor-v1")


def test_end_to_end_flattening_without_cat_effect():
    scores, samples, entries = _scores()
    out = analysis.analyze_p1(scores, samples, entries, V1, integrity_ok=True, sample_k=25, n_boot=99, n_flip=99)
    assert out["outcome"]["cls"] in {"FLATTENING_NO_CAT", "FLATTENING_CAT_UNRESOLVED"}
    for seed in ("2", "3"):
        est = out["per_seed"]["primary"][seed]["estimates"]
        assert 0.6 < est["beta"] < 0.8 and abs(est["C3"]) < 0.15
    assert all(v["agreement"]["passed"] for v in out["instrument"].values())


def test_end_to_end_cat_residual_detected():
    scores, samples, entries = _scores(bump=0.6, seed=3)
    out = analysis.analyze_p1(scores, samples, entries, V1, integrity_ok=True, sample_k=25, n_boot=99, n_flip=99)
    assert out["outcome"]["cls"] in {"CAT_DOMINANT", "CAT_RESIDUAL_NOT_DOMINANT"}


def test_end_to_end_p2_generic_persona_teacher():
    scores, _samples, entries = _scores(seed=5)
    rng = np.random.default_rng(9)
    # dog students: same tempering as cat students, no trait bump; dog teacher: dog-concentrated
    for key in list(scores):
        arm, stem, cell = key.split("|")
        if arm.startswith("S") and cell.startswith("Q"):
            z = scores[key]["word_logp"] + rng.normal(0, 0.05, W)
            scores[f"D{arm[1]}|{stem}|{cell}"] = {"word_logp": z, "decoration_mass": 0.0, "emoji_mass": 0.0}
        if arm == "T_cat":
            z = scores[key]["word_logp"].copy()
            z[0], z[1] = z[1], z[0] + 3.0
            scores[f"T_dog|{stem}|{cell}"] = {"word_logp": z, "decoration_mass": 0.0, "emoji_mass": 0.0}
    out = analysis.analyze_p2(scores, entries, integrity_ok=True, n_boot=99, n_flip=49)
    assert out["outcome"]["cls"] in {"GENERIC_PERSONA_TEACHER", "P2_NULL_OR_MIXED"}
    assert not out["outcome"]["confirmed"]["K1"] and not out["outcome"]["confirmed"]["K2"]
