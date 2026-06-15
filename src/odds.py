"""Sportsbook odds math.

Conventions:
- Probabilities are floats in [0, 1].
- American odds are signed ints/floats: negative for favorites (-110), positive for dogs (+150).
- Decimal odds are payout multiples on the stake (1.91 for -110, 2.50 for +150).
- "no-vig" means the vig-free fair probability inferred from a two-sided market.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


# --- American odds conversions ---------------------------------------------

def american_to_prob(odds: float) -> float:
    """Implied probability from American odds. Includes the vig."""
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def prob_to_american(p: float) -> int:
    """American odds (rounded) from a probability."""
    if not 0 < p < 1:
        raise ValueError(f"probability must be in (0,1), got {p}")
    if p >= 0.5:
        return -int(round(p / (1 - p) * 100))
    return int(round((1 - p) / p * 100))


def american_to_decimal(odds: float) -> float:
    # American odds are undefined in the open interval (-100, +100): there is no
    # such price on a real book. Reject it rather than return a garbage multiple.
    if -100 < odds < 100:
        raise ValueError(f"American odds must satisfy |odds| >= 100, got {odds}")
    if odds >= 100:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / (-odds)


def decimal_to_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        raise ValueError("decimal odds must be > 1")
    return 1.0 / decimal_odds


# --- No-vig --------------------------------------------------------------

def no_vig(p_side: float, p_other: float) -> float:
    """Vig-free probability for `p_side` given the two-sided implied probs."""
    total = p_side + p_other
    if total <= 0:
        raise ValueError("invalid market")
    return p_side / total


def no_vig_from_american(side_odds: float, other_odds: float) -> float:
    """No-vig probability for `side_odds` given the two American odds on the market."""
    return no_vig(american_to_prob(side_odds), american_to_prob(other_odds))


# --- Lines and pushes ----------------------------------------------------

@dataclass(frozen=True)
class LineResult:
    """Result of evaluating an over against a line on a sample of stat values."""
    line: float
    n: int
    n_over: int          # stat strictly > line
    n_push: int          # stat == line (only possible if line is integer)
    n_under: int         # stat strictly < line
    p_over: float        # P(over) excluding pushes (push-adjusted, the standard book convention)
    p_over_raw: float    # P(stat > line) treating pushes as losses
    wilson_low: float    # 95% Wilson lower bound on p_over (push-adjusted)
    wilson_high: float


def evaluate_line(values: Iterable[float], line: float, z: float = 1.96) -> LineResult:
    """Empirical P(over) at a prop line, with Wilson CI.

    For half-point lines (X.5) there are no pushes, so push-adjusted and raw are equal.
    For whole-number lines, pushes are excluded from the denominator (standard book rule),
    matching how settled bets are graded.
    """
    vals = [float(v) for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return LineResult(line, 0, 0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"))

    n_over = sum(1 for v in vals if v > line)
    n_push = sum(1 for v in vals if v == line)
    n_under = n - n_over - n_push

    p_over_raw = n_over / n
    decided = n_over + n_under
    p_over = n_over / decided if decided > 0 else float("nan")

    lo, hi = wilson_interval(n_over, decided if decided > 0 else 1, z=z)
    return LineResult(line, n, n_over, n_push, n_under, p_over, p_over_raw, lo, hi)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion. Returns (low, high)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --- Edge metrics --------------------------------------------------------

def edge_vs_market(model_prob: float, market_prob_no_vig: float) -> float:
    """Probability edge: positive means model thinks the over is undervalued."""
    return model_prob - market_prob_no_vig


def expected_roi(model_prob: float, american_odds: float) -> float:
    """Expected ROI per unit staked at the given price.

    ROI = p * (decimal-1) - (1-p) = p * decimal - 1
    """
    dec = american_to_decimal(american_odds)
    return model_prob * dec - 1.0


def kelly_fraction(model_prob: float, american_odds: float, cap: float = 0.25) -> float:
    """Kelly stake fraction, capped (full Kelly is rarely correct in practice)."""
    dec = american_to_decimal(american_odds)
    b = dec - 1.0
    if b <= 0:
        return 0.0
    f = (model_prob * dec - 1.0) / b
    return max(0.0, min(cap, f))
