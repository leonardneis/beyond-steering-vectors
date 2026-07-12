from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.analysis.split_stability import split_readout_stability


def test_split_stability_reports_perfect_order_and_overlap() -> None:
    split_a = {layer: {"downstream_mean": np.array([float(layer)])} for layer in range(8)}
    split_b = {layer: {"downstream_mean": np.array([2.0 * layer])} for layer in range(8)}
    result = split_readout_stability(split_a, split_b, "downstream_mean")
    assert result["spearman_rho"] == 1.0
    assert result["kendall_tau"] == pytest.approx(1.0)
    assert result["top_k_overlap"]["5"]["count"] == 5
    assert result["sign_stable_count"] == 8
