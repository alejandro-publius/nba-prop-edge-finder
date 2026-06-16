"""Price a player prop from a projected distribution.

Turns a projection (a mean, and for counts an empirical variance) into P(over a
line), fair odds, and a two-sided market with a configurable hold — the core
"spin up a distribution and price the line" task a trader does by hand.

NBA box-score counts are NOT Poisson in practice: they are over-dispersed
(variance > mean), because minutes, pace, foul trouble, and role vary game to
game. Pricing a real prop with a plain Poisson therefore makes the tails too thin
and over-prices the over/under near the mean. So this module offers four models:

  - poisson   textbook baseline (variance == mean); usually too tight for NBA
  - negbin    Negative Binomial fit to (mean, variance); handles over-dispersion
              and is the right default for counting stats
  - normal    Gaussian with the empirical sd; fine for high-mean combos (PRA, PR)
  - empirical the raw without-teammate sample, no distributional assumption

All distribution math is pure-Python (math.lgamma / math.erf) — no scipy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .odds import american_to_decimal, american_to_prob, prob_to_american

SQRT2 = math.sqrt(2.0)


# --- Poisson ----------------------------------------------------------------

def poisson_pmf(k: int, mean: float) -> float:
    if mean <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    return math.exp(-mean + k * math.log(mean) - math.lgamma(k + 1))


def poisson_sf(line: float, mean: float) -> float:
    """P(X > line) for integer-valued X. A half-point line never pushes."""
    k = math.floor(line)
    cdf = sum(poisson_pmf(i, mean) for i in range(0, k + 1))
    return max(0.0, 1.0 - cdf)


# --- Negative Binomial (mean / variance parameterization) -------------------

def negbin_sf(line: float, mean: float, var: float) -> float:
    """P(X > line) for a Negative Binomial with the given mean and variance.

    Falls back to Poisson when the data is not over-dispersed (var <= mean),
    since the NB is undefined there.
    """
    if var <= mean or mean <= 0:
        return poisson_sf(line, mean)
    p = mean / var                 # in (0, 1)
    r = mean * mean / (var - mean)  # number of "successes", > 0

    def pmf(k: int) -> float:
        return math.exp(
            math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + r * math.log(p) + k * math.log(1.0 - p)
        )

    k = math.floor(line)
    cdf = sum(pmf(i) for i in range(0, k + 1))
    return max(0.0, 1.0 - cdf)


# --- Normal -----------------------------------------------------------------

def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def normal_sf(line: float, mean: float, sd: float) -> float:
    """P(X > line) for a Normal. Line is used as-is (half-point lines sit between
    integers, so no continuity correction is needed)."""
    if sd <= 0:
        return 1.0 if mean > line else 0.0
    return 1.0 - _phi((line - mean) / sd)


# --- Empirical --------------------------------------------------------------

def empirical_sf(samples, line: float) -> float:
    """Push-adjusted P(over): strict overs / (overs + unders)."""
    vals = [float(v) for v in samples if v is not None and not math.isnan(float(v))]
    over = sum(1 for v in vals if v > line)
    under = sum(1 for v in vals if v < line)
    decided = over + under
    return over / decided if decided else float("nan")


# --- Market construction ----------------------------------------------------

def dispersion(mean: float, var: float) -> float:
    """Variance-to-mean ratio. 1.0 = Poisson; >1 = over-dispersed."""
    return var / mean if mean > 0 else float("nan")


def fair_american(p_over: float) -> tuple[int, int]:
    """Vig-free American odds for (over, under)."""
    return prob_to_american(p_over), prob_to_american(1.0 - p_over)


def market_with_hold(p_over: float, hold: float = 0.05) -> tuple[int, int]:
    """Two-sided American market that books `hold` (e.g. 0.05 = 5% hold).

    Multiplicative method: scale both fair implied probs so they sum to 1 + hold,
    preserving their ratio, then convert each to American odds.
    """
    over_imp = p_over * (1.0 + hold)
    under_imp = (1.0 - p_over) * (1.0 + hold)
    return prob_to_american(over_imp), prob_to_american(under_imp)


@dataclass
class PriceResult:
    model: str
    p_over: float
    fair_over: int
    fair_under: int


def price_line(
    line: float,
    mean: float,
    var: float | None = None,
    sd: float | None = None,
    samples=None,
) -> dict[str, PriceResult]:
    """Price `line` under every available model. Returns {model_name: PriceResult}."""
    out: dict[str, PriceResult] = {}

    def add(name: str, p: float) -> None:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return
        p = min(max(p, 1e-6), 1 - 1e-6)
        fo, fu = fair_american(p)
        out[name] = PriceResult(name, p, fo, fu)

    add("poisson", poisson_sf(line, mean))
    if var is not None:
        add("negbin", negbin_sf(line, mean, var))
    if sd is None and var is not None:
        sd = math.sqrt(var)
    if sd is not None:
        add("normal", normal_sf(line, mean, sd))
    if samples is not None:
        add("empirical", empirical_sf(samples, line))
    return out
