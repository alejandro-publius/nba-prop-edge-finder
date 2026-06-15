"""Out-of-sample validation: do edges found in training generalize to a held-out season?

This is the honest answer to the multiple-testing problem in edges.py.
We scan ~130k (player, teammate, stat) combos, so plenty of z>2 signals are pure
noise. Out-of-sample validation tells us how many survive on data the model
never saw.

Procedure:
  1. Train on 2 seasons (default: 2023-24, 2024-25).
  2. For each (player, teammate, stat) edge with z >= --train-min-z in training,
     predict: bettor should fade the line at L = ceil(avg_with_train) + 0.5
     (sportsbook's likely "baseline" line, set near the player's with-teammate avg).
  3. In the held-out test season (default: 2025-26), for the same pair, count how
     often the player actually went over that line in "without" games.
  4. Report aggregate hit rate, Wilson CI, and the player-level breakdown.

A test hit rate of ~50% means edges don't generalize (data mining).
A test hit rate of 55%+ on a meaningful sample means real signal.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from .odds import wilson_interval
from .splits import (
    BASE_STATS,
    COMBO_STATS,
    add_combo_stats,
    compute_pair_splits,
)


def line_from_baseline(avg_with: float) -> float:
    """Build a hypothetical sportsbook line from the player's with-teammate average.

    Books typically post lines at X.5 intervals near the player's recent baseline.
    We use ceil(avg_with) - 0.5 to get the X.5 line closest to (and just under) the
    baseline, which is where most over/unders are priced near coin-flip.
    """
    return math.ceil(avg_with) - 0.5


def get_without_games(
    df: pd.DataFrame,
    player_id: int,
    teammate_id: int,
    team_id: int,
    season: str,
) -> pd.DataFrame:
    """Player's games where teammate did NOT appear, for one (team, season)."""
    p = df[(df["PLAYER_ID"] == player_id) & (df["TEAM_ID"] == team_id) & (df["SEASON"] == season)]
    if p.empty:
        return p
    k = df[(df["PLAYER_ID"] == teammate_id) & (df["TEAM_ID"] == team_id) & (df["SEASON"] == season)]
    if k.empty:
        return pd.DataFrame(columns=p.columns)
    return p[~p["GAME_ID"].isin(set(k["GAME_ID"]))]


def stat_values(games: pd.DataFrame, stat: str) -> pd.Series:
    if stat in COMBO_STATS:
        return games[COMBO_STATS[stat]].sum(axis=1)
    return games[stat]


def validate(
    gamelogs: pd.DataFrame,
    train_seasons: list[str],
    test_season: str,
    train_min_z: float = 2.0,
    train_min_pct: float = 0.15,
    train_min_n_without: int = 5,
    test_min_n_without: int = 3,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    if stats is None:
        stats = BASE_STATS + list(COMBO_STATS.keys())
    gamelogs = add_combo_stats(gamelogs.copy())

    train_df = gamelogs[gamelogs["SEASON"].isin(train_seasons)]
    test_df = gamelogs[gamelogs["SEASON"] == test_season]

    print(f"Training on {train_seasons} ({len(train_df):,} player-games)")
    print(f"Testing on {test_season} ({len(test_df):,} player-games)")
    print(f"Computing training splits...")
    train_splits = compute_pair_splits(train_df, stats)
    print(f"  {len(train_splits):,} training (player, teammate, stat) rows")

    # Keep only edges that look real in training
    edges = train_splits[
        (train_splits["z"] >= train_min_z)
        & (train_splits["pct_delta"] >= train_min_pct)
        & (train_splits["n_without"] >= train_min_n_without)
        & (train_splits["avg_with"] >= 1.0)
    ].copy()
    print(f"  {len(edges):,} candidate edges meet training thresholds")

    # For each, what does test season say?
    rows = []
    for _, e in edges.iterrows():
        line = line_from_baseline(e["avg_with"])
        # Find this pair in test season — they need to still be teammates
        # We look for any (team_id, season=test) where both player and teammate appeared
        common_team_seasons = (
            test_df[test_df["PLAYER_ID"] == e["player_id"]]
            .groupby(["TEAM_ID"])["PLAYER_ID"].count()
            .index.tolist()
        )
        teammate_team_seasons = (
            test_df[test_df["PLAYER_ID"] == e["teammate_id"]]
            .groupby(["TEAM_ID"])["PLAYER_ID"].count()
            .index.tolist()
        )
        shared_teams = set(common_team_seasons) & set(teammate_team_seasons)
        if not shared_teams:
            continue

        test_without_frames = []
        for team_id in shared_teams:
            test_without_frames.append(
                get_without_games(test_df, e["player_id"], e["teammate_id"], team_id, test_season)
            )
        test_without = pd.concat(test_without_frames, ignore_index=True) if test_without_frames else pd.DataFrame()
        if len(test_without) < test_min_n_without:
            continue

        values = stat_values(test_without, e["stat"]).tolist()
        n_test = len(values)
        n_over = sum(1 for v in values if v > line)
        n_push = sum(1 for v in values if v == line)
        n_decided = n_over + sum(1 for v in values if v < line)
        if n_decided == 0:
            continue
        test_hit_rate = n_over / n_decided

        rows.append({
            "player": e["player"],
            "teammate_out": e["teammate_out"],
            "stat": e["stat"],
            "train_n_without": int(e["n_without"]),
            "train_avg_with": e["avg_with"],
            "train_avg_without": e["avg_without"],
            "train_z": e["z"],
            "line": line,
            "test_n": n_test,
            "test_over": n_over,
            "test_push": n_push,
            "test_hit_rate": round(test_hit_rate, 3),
            "test_avg": round(sum(values) / n_test, 2) if n_test else float("nan"),
        })

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    if results.empty:
        print("\nNo edges had enough test-season data to evaluate.")
        return

    total_over = int(results["test_over"].sum())
    total_decided = int((results["test_n"] - results["test_push"]).sum())
    overall_hit = total_over / total_decided if total_decided else float("nan")
    lo, hi = wilson_interval(total_over, total_decided)

    print()
    print("=" * 60)
    print(f"OUT-OF-SAMPLE RESULTS")
    print("=" * 60)
    print(f"Evaluable edges:           {len(results)}")
    print(f"Total test-season games:   {int(results['test_n'].sum())}")
    print(f"Total decided (no push):   {total_decided}")
    print(f"Total overs hit:           {total_over}")
    print(f"Overall hit rate:          {overall_hit:.3f}  (Wilson 95% CI: [{lo:.3f}, {hi:.3f}])")
    print()
    print("NOTE: 'line' here is set near the player's WITH-teammate average. Real")
    print("sportsbook lines incorporate recent form and news (partially), so this is")
    print("an UPPER BOUND on the edge a live bettor could capture. A 60% hit rate here")
    print("does NOT mean 60% in live betting — but it does prove the underlying signal")
    print("is real and not just multiple-testing noise.")
    print()
    if overall_hit > 0.524:  # break-even at -110
        print(f"-110 break-even is 0.524. Observed: {overall_hit:.3f}. Signal is present.")
    else:
        print(f"-110 break-even is 0.524. Observed: {overall_hit:.3f}. Edges look like in-sample overfit.")

    # Top-25 edges in training that actually delivered
    top = results.sort_values("train_z", ascending=False).head(25)
    print()
    print("Top-25 training-z edges and their actual test-season results:")
    print(top[["player", "teammate_out", "stat", "train_z", "line",
              "test_n", "test_over", "test_hit_rate", "test_avg"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data")
    p.add_argument("--out", default="out/validation.csv")
    p.add_argument("--train", nargs="+", default=["2023-24", "2024-25"])
    p.add_argument("--test", default="2025-26")
    p.add_argument("--train-min-z", type=float, default=2.0)
    p.add_argument("--train-min-pct", type=float, default=0.15)
    p.add_argument("--markets-only", action="store_true")
    args = p.parse_args()

    cache_dir = Path(args.cache)
    files = sorted(cache_dir.glob("gamelog_*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {cache_dir}/")
    gamelogs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    gamelogs["GAME_DATE"] = pd.to_datetime(gamelogs["GAME_DATE"])

    stats = ["PTS", "REB", "AST", "FG3M"] + list(COMBO_STATS.keys()) if args.markets_only else None
    results = validate(
        gamelogs,
        train_seasons=args.train,
        test_season=args.test,
        train_min_z=args.train_min_z,
        train_min_pct=args.train_min_pct,
        stats=stats,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nWrote {len(results)} evaluable edges to {out_path}")
    print_summary(results)
