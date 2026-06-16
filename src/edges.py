"""Rank with/without splits to surface prop-betting edges, with minutes-confound diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .shrink import add_shrinkage

# Defaults tuned to surface the kind of edges shown in the Jaylen Brown / Tatum example
DEFAULT_MIN_Z = 1.5
DEFAULT_MIN_PCT = 0.10
DEFAULT_MIN_N_WITHOUT = 5
DEFAULT_MIN_AVG_WITH = 1.0   # ignore tiny baselines (e.g. blocks at 0.2)
MINUTES_CONFOUND_THRESHOLD = 4.0  # minutes diff that flags a role-change confound

# Stats that sportsbooks actually post props for at most NBA games.
# STL / BLK / TOV are rarely available and have thin liquidity when they exist.
LIQUID_MARKETS = {"PTS", "REB", "AST", "FG3M", "PR", "PA", "RA", "PRA"}


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["min_delta"] = (df["min_without"] - df["min_with"]).round(1)
    # A "clean" edge moves in the same direction both raw and per-36 (not just a minutes bump).
    df["per36_delta"] = (df["per36_without"] - df["per36_with"]).round(2)
    df["same_sign_per36"] = (df["delta"] * df["per36_delta"]) > 0
    df["minutes_confound"] = df["min_delta"].abs() >= MINUTES_CONFOUND_THRESHOLD
    return df


def filter_edges(
    df: pd.DataFrame,
    min_z: float = DEFAULT_MIN_Z,
    min_pct: float = DEFAULT_MIN_PCT,
    min_n_without: int = DEFAULT_MIN_N_WITHOUT,
    min_avg_with: float = DEFAULT_MIN_AVG_WITH,
    direction: str = "up",  # "up", "down", "both"
    stat: str | None = None,
    player: str | None = None,
    teammate: str | None = None,
    team: str | None = None,
    markets_only: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    df = df[df["n_without"] >= min_n_without]
    df = df[df["avg_with"] >= min_avg_with]
    if markets_only:
        df = df[df["stat"].isin(LIQUID_MARKETS)]

    if direction == "up":
        df = df[(df["z"] >= min_z) & (df["pct_delta"] >= min_pct)]
    elif direction == "down":
        df = df[(df["z"] <= -min_z) & (df["pct_delta"] <= -min_pct)]
    else:  # both
        df = df[(df["z"].abs() >= min_z) & (df["pct_delta"].abs() >= min_pct)]

    if stat:
        df = df[df["stat"].str.upper() == stat.upper()]
    if player:
        df = df[df["player"].str.contains(player, case=False, na=False)]
    if teammate:
        df = df[df["teammate_out"].str.contains(teammate, case=False, na=False)]
    if team:
        df = df[df["team"].str.upper() == team.upper()]

    return df.sort_values("z", ascending=False, key=lambda s: s.abs())


DISPLAY_COLS = [
    "season", "team", "player", "teammate_out", "stat",
    "n_with", "n_without",
    "min_with", "min_without", "min_delta",
    "avg_with", "avg_without", "delta", "pct_delta",
    "shrink_k", "shrunk_delta", "shrunk_without",
    "per36_with", "per36_without", "per36_delta", "same_sign_per36",
    "z", "minutes_confound",
]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="out/splits.parquet")
    p.add_argument("--out", default="out/edges.csv")
    p.add_argument("--min-z", type=float, default=DEFAULT_MIN_Z)
    p.add_argument("--min-pct", type=float, default=DEFAULT_MIN_PCT)
    p.add_argument("--min-n-without", type=int, default=DEFAULT_MIN_N_WITHOUT)
    p.add_argument("--direction", choices=["up", "down", "both"], default="up")
    p.add_argument("--stat", help="Filter to single stat: PTS, REB, AST, FG3M, PRA, PR, PA, RA, STL, BLK, TOV")
    p.add_argument("--player", help="Substring match on player name")
    p.add_argument("--teammate", help="Substring match on teammate name")
    p.add_argument("--team", help="Team abbreviation, e.g. BOS")
    p.add_argument("--clean-only", action="store_true", help="Only show edges with no minutes confound")
    p.add_argument("--markets-only", action="store_true",
                   help="Restrict to liquid prop markets (PTS, REB, AST, FG3M, PR, PA, RA, PRA)")
    p.add_argument("--top", type=int, default=50, help="Print top N to stdout")
    args = p.parse_args()

    df = pd.read_parquet(args.splits)
    df = add_shrinkage(df)
    df = annotate(df)

    edges = filter_edges(
        df,
        min_z=args.min_z,
        min_pct=args.min_pct,
        min_n_without=args.min_n_without,
        direction=args.direction,
        markets_only=args.markets_only,
        stat=args.stat,
        player=args.player,
        teammate=args.teammate,
        team=args.team,
    )
    if args.clean_only:
        edges = edges[~edges["minutes_confound"] & edges["same_sign_per36"]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges[DISPLAY_COLS].to_csv(out_path, index=False)
    print(f"Wrote {len(edges):,} edges to {out_path}")
    print()
    print(edges[DISPLAY_COLS].head(args.top).to_string(index=False))
