"""Pin sportsbook odds math against known values."""
import math

import pytest

from src.odds import (
    american_to_decimal,
    american_to_prob,
    decimal_to_prob,
    edge_vs_market,
    evaluate_line,
    expected_roi,
    kelly_fraction,
    no_vig,
    no_vig_from_american,
    prob_to_american,
    wilson_interval,
)


def approx(a, b, tol=1e-4):
    return math.isclose(a, b, abs_tol=tol)


def test_american_to_prob_minus_110():
    # -110 → 110/210 = 0.52381
    assert approx(american_to_prob(-110), 0.5238095, tol=1e-5)


def test_american_to_prob_plus_150():
    # +150 → 100/250 = 0.40
    assert approx(american_to_prob(150), 0.40)


def test_american_to_prob_even():
    assert approx(american_to_prob(100), 0.5)
    assert approx(american_to_prob(-100), 0.5)


def test_american_to_decimal():
    assert approx(american_to_decimal(-110), 1.909091, tol=1e-5)
    assert approx(american_to_decimal(150), 2.5)
    assert approx(american_to_decimal(-200), 1.5)


def test_american_to_decimal_rejects_invalid_odds():
    # |odds| < 100 is not a real price and must raise, not return garbage.
    for bad in (50, -50, 0, 99, -99):
        with pytest.raises(ValueError):
            american_to_decimal(bad)


def test_decimal_to_prob_roundtrip():
    p = 0.55
    dec = american_to_decimal(prob_to_american(p))
    # prob_to_american rounds, so allow loose tolerance
    assert approx(decimal_to_prob(dec), p, tol=0.005)


def test_no_vig_balanced_market():
    # both sides -110: implied 0.5238 each, total 1.0476
    # vig-free = 0.5238 / 1.0476 = 0.5
    p = no_vig_from_american(-110, -110)
    assert approx(p, 0.5)


def test_no_vig_asymmetric():
    # over -150, under +130
    p_over = american_to_prob(-150)   # 0.6
    p_under = american_to_prob(130)    # 0.4348
    fair = no_vig(p_over, p_under)
    assert approx(fair, 0.6 / (0.6 + 0.4348), tol=1e-4)


def test_prob_to_american_favorite():
    # p=0.6 → -150
    assert prob_to_american(0.6) == -150


def test_prob_to_american_dog():
    # p=0.4 → +150
    assert prob_to_american(0.4) == 150


def test_wilson_interval_known():
    # 50/100: center 0.5, half ≈ 0.0980 at z=1.96
    lo, hi = wilson_interval(50, 100)
    assert approx(lo, 0.4038, tol=1e-3)
    assert approx(hi, 0.5962, tol=1e-3)


def test_wilson_interval_extreme():
    # 0/10: lower bound is 0, upper bound > 0
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi > 0


def test_evaluate_line_half_point_no_pushes():
    # Stat values: 8, 10, 12, 11, 9 at line 9.5
    # over: 10, 12, 11 → 3 ; under: 8, 9 → 2 ; push: 0
    res = evaluate_line([8, 10, 12, 11, 9], 9.5)
    assert res.n_over == 3
    assert res.n_under == 2
    assert res.n_push == 0
    assert approx(res.p_over, 0.6)


def test_evaluate_line_whole_number_pushes_excluded():
    # Values: 8, 10, 12, 10, 9 at line 10
    # over: 12 → 1 ; under: 8, 9 → 2 ; push: 10, 10 → 2
    # push-adjusted p_over = 1 / (1 + 2) = 0.333
    # raw p_over = 1 / 5 = 0.2
    res = evaluate_line([8, 10, 12, 10, 9], 10)
    assert res.n_over == 1
    assert res.n_push == 2
    assert res.n_under == 2
    assert approx(res.p_over, 1.0 / 3.0)
    assert approx(res.p_over_raw, 0.2)


def test_expected_roi_breakeven_at_no_vig():
    # If model_prob equals the implied prob, ROI should be 0 (before vig)
    # At -110, implied = 0.5238. ROI = 0.5238 * 1.9091 - 1 = -0.00001 ≈ 0
    assert approx(expected_roi(0.5238095, -110), 0.0, tol=1e-4)


def test_expected_roi_positive_edge():
    # 55% on a -110: ROI = 0.55 * 1.9091 - 1 = 0.05
    assert approx(expected_roi(0.55, -110), 0.05, tol=1e-3)


def test_kelly_fraction_no_edge():
    assert kelly_fraction(0.5, -110) == 0.0


def test_kelly_fraction_positive_edge():
    # p=0.55, dec=1.9091, b=0.9091. f = (0.55*1.9091 - 1)/0.9091 = 0.055
    assert approx(kelly_fraction(0.55, -110), 0.055, tol=1e-3)


def test_kelly_fraction_capped():
    # Huge edge would Kelly to >25%, must be capped
    assert kelly_fraction(0.9, -110, cap=0.25) == 0.25


def test_edge_vs_market_sign():
    assert edge_vs_market(0.55, 0.50) == pytest.approx(0.05)
    assert edge_vs_market(0.40, 0.50) == pytest.approx(-0.10)
