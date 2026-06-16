"""Project a player's teammate-out distribution and price a prop line.

Pulls the player's historical games WITHOUT the teammate, fits a mean + variance,
and prices the line under Poisson / Negative Binomial / Normal / empirical models
side by side — then, if you pass the posted odds, shows the edge vs the market.

Example:
  python3 -m src.project --player "Jaylen Brown" --teammate "Tatum" --stat RA \\
      --line 9.5 --over -110 --under -110
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .odds import american_to_decimal, expected_roi, kelly_fraction, no_vig_from_american
from .price import without_sample, stat_series
from .pricer import dispersion, market_with_hold, price_line


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--player", required=True)
    p.add_argument("--teammate", required=True, help="teammate who is OUT")
    p.add_argument("--stat", required=True, help="PTS REB AST FG3M STL BLK TOV PR PA RA PRA")
    p.add_argument("--line", type=float, required=True)
    p.add_argument("--over", type=float, help="posted American odds on the over")
    p.add_argument("--under", type=float, help="posted American odds on the under")
    p.add_argument("--model", choices=["poisson", "negbin", "normal", "empirical"],
                   default="negbin", help="model to use for the edge calc (default negbin)")
    p.add_argument("--season", help="restrict to one season, e.g. 2024-25")
    p.add_argument("--cache", default="data")
    args = p.parse_args()

    files = sorted(Path(args.cache).glob("gamelog_*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {args.cache}/ — run src/fetch.py first")
    logs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

    sample = without_sample(logs, args.player, args.teammate, args.stat, args.season)
    vals = stat_series(sample, args.stat.upper()).astype(float)
    n = len(vals)
    mean = float(vals.mean())
    var = float(vals.var(ddof=1)) if n > 1 else mean  # fall back to Poisson var
    sd = float(vals.std(ddof=1)) if n > 1 else np.sqrt(mean)

    print(f"\n{args.player}  |  {args.teammate} OUT  |  {args.stat.upper()}"
          + (f"  |  {args.season}" if args.season else ""))
    print(f"Sample without {args.teammate}: n={n}, mean={mean:.2f}, "
          f"var={var:.2f}, sd={sd:.2f}")
    disp = dispersion(mean, var)
    print(f"Dispersion (var/mean): {disp:.2f}  "
          f"({'over-dispersed → Poisson too tight, use NB' if disp > 1.15 else 'near-Poisson'})")

    priced = price_line(args.line, mean=mean, var=var, sd=sd, samples=vals.tolist())
    print(f"\nP(over {args.line}) by model:")
    print(f"  {'model':10} {'P(over)':>8} {'fair over':>10} {'fair under':>11}")
    for name in ["poisson", "negbin", "normal", "empirical"]:
        if name in priced:
            r = priced[name]
            print(f"  {name:10} {r.p_over:>8.3f} {r.fair_over:>+10d} {r.fair_under:>+11d}")

    # Suggested market the model would post (5% hold)
    chosen = priced.get(args.model) or next(iter(priced.values()))
    mo, mu = market_with_hold(chosen.p_over, hold=0.05)
    print(f"\nModel ({chosen.model}) would post (5% hold): over {mo:+d} / under {mu:+d}")

    # Edge vs the posted market, if provided
    if args.over is not None and args.under is not None:
        mkt_over = no_vig_from_american(args.over, args.under)
        edge = chosen.p_over - mkt_over
        roi = expected_roi(chosen.p_over, args.over)
        kelly = kelly_fraction(chosen.p_over, args.over, cap=0.25)
        print(f"\nPosted: over {args.over:+.0f} / under {args.under:+.0f}")
        print(f"  market no-vig P(over):  {mkt_over:.3f}")
        print(f"  model  P(over) [{chosen.model}]: {chosen.p_over:.3f}")
        print(f"  edge:                   {edge:+.3f}  ({edge*100:+.1f} pts)")
        print(f"  expected ROI on over:   {roi*100:+.2f}%")
        print(f"  Kelly stake (cap 25%):  {kelly*100:.2f}%")
        side = "OVER" if edge > 0 else "UNDER"
        print(f"  lean: {side}")


if __name__ == "__main__":
    main()
