"""Price a specific prop line against the historical "without-teammate" distribution.

Workflow:
  1. Find an edge with `src.edges` (e.g., Brown w/o Tatum, R+A)
  2. Look up the live line + prices on a sportsbook (e.g., 9.5 over -110 / under -110)
  3. Run this script to get empirical P(over), Wilson CI, no-vig market prob,
     probability edge, expected ROI, and capped Kelly stake.

Honest about small samples: the Wilson 95% lower bound is the more conservative
read. If that's below the no-vig market prob, the edge isn't proven.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .odds import (
    american_to_decimal,
    edge_vs_market,
    evaluate_line,
    expected_roi,
    kelly_fraction,
    no_vig_from_american,
)
from .splits import COMBO_STATS


def stat_series(df: pd.DataFrame, stat: str) -> pd.Series:
    if stat in COMBO_STATS:
        return df[COMBO_STATS[stat]].sum(axis=1)
    return df[stat]


def without_sample(
    gamelogs: pd.DataFrame,
    player: str,
    teammate: str,
    stat: str,
    season: str | None = None,
) -> pd.DataFrame:
    """Player's games where teammate did NOT appear, restricted to overlapping team-tenure."""
    if season:
        gamelogs = gamelogs[gamelogs["SEASON"] == season]

    p_rows = gamelogs[gamelogs["PLAYER_NAME"].str.contains(player, case=False, na=False)]
    if p_rows.empty:
        raise SystemExit(f"No games found for player matching '{player}'")
    k_rows = gamelogs[gamelogs["PLAYER_NAME"].str.contains(teammate, case=False, na=False)]
    if k_rows.empty:
        raise SystemExit(f"No games found for teammate matching '{teammate}'")

    out_frames = []
    common_teams = set(p_rows["TEAM_ID"]).intersection(set(k_rows["TEAM_ID"]))
    common_seasons = set(p_rows["SEASON"]).intersection(set(k_rows["SEASON"]))
    if not common_teams or not common_seasons:
        raise SystemExit("Player and teammate never shared a team-season in the data")

    for team_id in common_teams:
        for s in common_seasons:
            p_t = p_rows[(p_rows["TEAM_ID"] == team_id) & (p_rows["SEASON"] == s)]
            k_t = k_rows[(k_rows["TEAM_ID"] == team_id) & (k_rows["SEASON"] == s)]
            if p_t.empty or k_t.empty:
                continue
            # Tenure overlap on this team-season
            window_lo = max(p_t["GAME_DATE"].min(), k_t["GAME_DATE"].min())
            window_hi = min(p_t["GAME_DATE"].max(), k_t["GAME_DATE"].max())
            p_window = p_t[(p_t["GAME_DATE"] >= window_lo) & (p_t["GAME_DATE"] <= window_hi)]
            k_game_ids = set(k_t["GAME_ID"])
            without = p_window[~p_window["GAME_ID"].isin(k_game_ids)]
            out_frames.append(without)
    if not out_frames:
        raise SystemExit("No 'without' games in overlapping tenure")
    return pd.concat(out_frames, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--player", required=True, help="e.g. 'Jaylen Brown'")
    p.add_argument("--teammate", required=True, help="teammate that's OUT, e.g. 'Tatum'")
    p.add_argument("--stat", required=True, help="PTS, REB, AST, FG3M, STL, BLK, TOV, PR, PA, RA, PRA")
    p.add_argument("--line", type=float, required=True, help="e.g. 9.5")
    p.add_argument("--over", type=float, required=True, help="American odds on the over, e.g. -110")
    p.add_argument("--under", type=float, required=True, help="American odds on the under, e.g. -110")
    p.add_argument("--season", help="restrict to a single season, e.g. '2024-25'")
    p.add_argument("--cache", default="data", help="parquet cache dir")
    args = p.parse_args()

    cache_dir = Path(args.cache)
    files = sorted(cache_dir.glob("gamelog_*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {cache_dir}/ — run src/fetch.py first")
    gamelogs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    gamelogs["GAME_DATE"] = pd.to_datetime(gamelogs["GAME_DATE"])

    sample = without_sample(gamelogs, args.player, args.teammate, args.stat, args.season)
    values = stat_series(sample, args.stat.upper()).tolist()

    res = evaluate_line(values, args.line)
    market_p_over = no_vig_from_american(args.over, args.under)
    market_p_under = no_vig_from_american(args.under, args.over)

    model_edge = edge_vs_market(res.p_over, market_p_over)
    roi = expected_roi(res.p_over, args.over)
    kelly = kelly_fraction(res.p_over, args.over, cap=0.25)

    wilson_edge_low = res.wilson_low - market_p_over

    print(f"\nPlayer: {args.player}   Teammate OUT: {args.teammate}   Stat: {args.stat.upper()}")
    if args.season:
        print(f"Season: {args.season}")
    print(f"Line: {args.line}   Over: {args.over:+}   Under: {args.under:+}")
    print(f"Sample (without {args.teammate}): n={res.n}, over={res.n_over}, push={res.n_push}, under={res.n_under}")
    if res.n < 10:
        print(f"  WARNING: n={res.n} is small. Wilson CI is wide — see lower bound below.")

    print()
    print(f"Empirical P(over):           {res.p_over:.3f}    (Wilson 95% CI: [{res.wilson_low:.3f}, {res.wilson_high:.3f}])")
    print(f"Market no-vig P(over):       {market_p_over:.3f}")
    print(f"Market no-vig P(under):      {market_p_under:.3f}   (vig: {(1.0 - market_p_over - market_p_under) * -100:+.2f}%)")
    print()
    print(f"Probability edge:            {model_edge:+.3f}   ({model_edge*100:+.1f} pts)")
    print(f"Wilson-lower-bound edge:     {wilson_edge_low:+.3f}   (positive ≈ edge survives sampling noise)")
    print(f"Expected ROI per unit:       {roi*100:+.2f}%   at {args.over:+} on the over")
    print(f"Decimal odds:                {american_to_decimal(args.over):.3f}")
    print(f"Kelly stake (cap 25%):       {kelly*100:.2f}% of bankroll")
    print()
    if model_edge > 0 and wilson_edge_low > 0:
        print("VERDICT: edge plausible, survives Wilson lower bound.")
    elif model_edge > 0:
        print("VERDICT: positive point estimate but Wilson lower bound suggests sample too thin to commit.")
    else:
        print("VERDICT: no edge on the over at this price.")


if __name__ == "__main__":
    main()
