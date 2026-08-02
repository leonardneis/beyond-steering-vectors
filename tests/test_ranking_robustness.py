import pytest

from scripts.aggregate_ranking_robustness import compare_rankings


def test_compare_rankings_reports_spearman_and_overlap():
    reference = ["a", "b", "c", "d"]
    candidate = ["a", "c", "b", "d"]
    result = compare_rankings(reference, candidate, [2])
    assert result["spearman_rho"] == pytest.approx(0.8)
    assert result["top_k"]["2"]["overlap"] == 1
    assert result["top_k"]["2"]["jaccard"] == pytest.approx(1 / 3)


def test_compare_rankings_rejects_different_universe():
    with pytest.raises(ValueError, match="same module universe"):
        compare_rankings(["a", "b"], ["a", "c"], [1])
