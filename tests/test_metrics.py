import numpy as np
import pytest

pytest.importorskip("torch")

from src.data import make_split  # noqa: E402
from src.metrics import bin_stats, ece_brier, hd95  # noqa: E402
from src.stat_tools import auroc_auprc_from_hist, holm  # noqa: E402


def disk(size=64, center=(32, 32), radius=12):
    yy, xx = np.mgrid[:size, :size]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 < radius ** 2


def test_perfect_probabilities_have_zero_ece_and_brier():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    ece, brier = ece_brier(bin_stats(p, p.copy(), 10))
    assert ece == 0 and brier == 0


def test_overconfident_probabilities_have_expected_ece():
    p = np.full(100, 0.9)
    y = np.concatenate([np.ones(50), np.zeros(50)])
    ece, _ = ece_brier(bin_stats(p, y, 10))
    assert ece == pytest.approx(0.4)


def test_probability_one_falls_in_last_bin():
    stats = bin_stats(np.array([1.0]), np.array([1.0]), 10)
    assert stats[9, 0] == 1


def test_holm_adjustment():
    adjusted = holm([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_holm_keeps_nan():
    adjusted = holm([0.01, np.nan])
    assert adjusted[0] == pytest.approx(0.01) and np.isnan(adjusted[1])


def test_separable_scores_give_auroc_one():
    auroc, _ = auroc_auprc_from_hist(np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 5.0]))
    assert auroc == pytest.approx(1.0)


def test_hd95_identical_masks_is_zero():
    mask = disk()
    assert hd95(mask, mask) == 0.0


def test_hd95_shifted_mask_is_positive():
    assert hd95(disk(center=(32, 36)), disk()) > 0


def test_hd95_empty_prediction_is_nan():
    assert np.isnan(hd95(np.zeros((8, 8), bool), disk(8, (4, 4), 2)))


def test_split_is_deterministic_and_disjoint():
    first = make_split(900, 0.10, 42)
    second = make_split(900, 0.10, 42)
    assert first == second
    assert len(first["val_idx"]) == 90 and len(first["train_idx"]) == 810
    assert not set(first["val_idx"]) & set(first["train_idx"])
