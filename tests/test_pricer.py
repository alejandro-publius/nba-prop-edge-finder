"""Pin the distribution-pricing math."""
import math

import pytest

from src.odds import american_to_prob
from src.pricer import (
    dispersion,
    empirical_sf,
    fair_american,
    market_with_hold,
    negbin_sf,
    normal_sf,
    poisson_sf,
    price_line,
)


def approx(a, b, tol=1e-4):
    return math.isclose(a, b, abs_tol=tol)


# --- Poisson ---------------------------------------------------------------

def test_poisson_sf_analytic():
    # P(X > 0.5) for Poisson(1) = 1 - P(0) = 1 - e^-1
    assert approx(poisson_sf(0.5, 1), 1 - math.exp(-1))


def test_poisson_sf_known():
    assert approx(poisson_sf(9.5, 10), 0.54207, tol=1e-4)


def test_poisson_sf_monotonic_in_line():
    # Higher line -> lower P(over)
    assert poisson_sf(5.5, 10) > poisson_sf(9.5, 10) > poisson_sf(15.5, 10)


# --- Negative Binomial -----------------------------------------------------

def test_negbin_reduces_to_poisson_when_not_overdispersed():
    # var == mean -> NB undefined, falls back to Poisson
    assert approx(negbin_sf(9.5, 10, 10), poisson_sf(9.5, 10))
    # var < mean (under-dispersed) also falls back
    assert approx(negbin_sf(9.5, 10, 7), poisson_sf(9.5, 10))


def test_negbin_has_fatter_right_tail_than_poisson():
    # For a line well above the mean, over-dispersion lifts P(over)
    assert negbin_sf(15.5, 10, 20) > poisson_sf(15.5, 10)


def test_negbin_mean_is_preserved():
    # The NB parameterization should reproduce the requested mean.
    mean, var = 8.0, 16.0
    p = mean / var
    r = mean * mean / (var - mean)
    nb_mean = r * (1 - p) / p
    assert approx(nb_mean, mean, tol=1e-9)


# --- Normal ----------------------------------------------------------------

def test_normal_sf_symmetry():
    assert approx(normal_sf(25, 25, 8), 0.5)


def test_normal_sf_known():
    assert approx(normal_sf(24.5, 25, 8), 0.524918, tol=1e-5)


# --- Empirical -------------------------------------------------------------

def test_empirical_sf_excludes_pushes():
    # values 8,10,12,10,9 at line 10: over=1 (12), under=2 (8,9), pushes excluded
    assert approx(empirical_sf([8, 10, 12, 10, 9], 10), 1 / 3)
    assert approx(empirical_sf([8, 10, 12, 11, 9], 9.5), 0.6)


# --- Market construction ---------------------------------------------------

def test_dispersion():
    assert dispersion(10, 20) == 2.0
    assert dispersion(10, 10) == 1.0


def test_fair_american_no_vig():
    over, under = fair_american(0.6)
    assert over == -150 and under == 150
    # fair odds carry no hold: implied probs sum to ~1
    assert approx(american_to_prob(over) + american_to_prob(under), 1.0, tol=2e-3)


def test_market_with_hold_books_the_hold():
    over, under = market_with_hold(0.5, hold=0.05)
    total = american_to_prob(over) + american_to_prob(under)
    # ~5% hold; tolerance covers rounding to whole American odds
    assert approx(total, 1.05, tol=6e-3)


def test_price_line_returns_all_models():
    out = price_line(9.5, mean=11.0, var=20.0, samples=[8, 10, 12, 11, 9, 14, 7])
    assert set(out) == {"poisson", "negbin", "normal", "empirical"}
    # negbin should sit below poisson here (over-dispersed, line below mean)
    assert out["negbin"].p_over < out["poisson"].p_over


def test_price_line_poisson_only_when_no_var():
    out = price_line(9.5, mean=11.0)
    assert set(out) == {"poisson"}
