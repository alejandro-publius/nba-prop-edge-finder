"""Compute with/without-teammate splits for every player-teammate pair on the same team."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .usage import add_usage, aggregate_usg

# Base box-score stats and derived combo stats
BASE_STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]
COMBO_STATS = {
    "PRA": ["PTS", "REB", "AST"],
    "PR": ["PTS", "REB"],
    "PA": ["PTS", "AST"],
    "RA": ["REB", "AST"],
}

# USG% is a rate that is already minutes- and possession-aware, so it is handled
# specially: no per-36 (it IS already normalized) and its multi-game average uses
# the component-sum aggregate from src.usage rather than a mean of per-game rates.
RATE_STATS = {"USG"}

MIN_GAMES_PLAYER = 15        # player must have played this many games for the team
MIN_GAMES_TEAMMATE = 15      # teammate must have played this many games for the team
MIN_GAMES_WITHOUT = 5        # need at least this many "without" games to compute
MIN_AVG_MINUTES = 15.0       # filter out garbage-time-only players


def add_combo_stats(df: pd.DataFrame) -> pd.DataFrame:
    for name, parts in COMBO_STATS.items():
        df[name] = df[parts].sum(axis=1)
    return df


def player_team_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """First and last game date for each (player, team, season)."""
    return (
        df.groupby(["PLAYER_ID", "TEAM_ID", "SEASON"])["GAME_DATE"]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={"min": "first_game", "max": "last_game", "count": "games_on_team"})
    )


def compute_pair_splits(
    df: pd.DataFrame,
    stats: list[str],
    min_games_player: int = MIN_GAMES_PLAYER,
    min_games_teammate: int = MIN_GAMES_TEAMMATE,
    min_games_without: int = MIN_GAMES_WITHOUT,
    min_avg_min: float = MIN_AVG_MINUTES,
) -> pd.DataFrame:
    """For every (player, teammate, team, season), compute W/O splits across stats.

    Returns long-form table: one row per (player, teammate, stat).
    """
    df = df.copy()
    df = add_combo_stats(df)

    # Tenure & filter to qualifying player-team-season rows
    tenure = player_team_tenure(df)
    qualifying = tenure[tenure["games_on_team"] >= min_games_teammate].copy()
    # Keyed lookup of which games each player played for each team
    games_by_pt = (
        df.groupby(["PLAYER_ID", "TEAM_ID", "SEASON"])["GAME_ID"]
        .apply(set)
        .to_dict()
    )

    # Per-team list of qualifying players
    by_team = qualifying.groupby(["TEAM_ID", "SEASON"])

    rows = []
    for (team_id, season), team_players in by_team:
        pids = team_players["PLAYER_ID"].tolist()
        # Map player_id -> player game rows (just this team, this season)
        team_df = df[(df["TEAM_ID"] == team_id) & (df["SEASON"] == season)]
        team_df_by_player = {pid: team_df[team_df["PLAYER_ID"] == pid] for pid in pids}

        # Player-level averages for sanity filter
        avg_min = {pid: team_df_by_player[pid]["MIN"].mean() for pid in pids}

        for p in pids:
            p_df = team_df_by_player[p]
            if len(p_df) < min_games_player:
                continue
            if avg_min[p] < min_avg_min:
                continue
            p_name = p_df["PLAYER_NAME"].iloc[0]
            team_abbr = p_df["TEAM_ABBREVIATION"].iloc[0]

            for k in pids:
                if k == p:
                    continue
                k_games = games_by_pt.get((k, team_id, season), set())
                if len(k_games) < min_games_teammate:
                    continue
                k_name = team_df_by_player[k]["PLAYER_NAME"].iloc[0]

                # Restrict to overlapping tenure: games while teammate was actually rostered
                k_dates = team_df_by_player[k]["GAME_DATE"]
                if k_dates.empty:
                    continue
                k_first, k_last = k_dates.min(), k_dates.max()

                p_window = p_df[(p_df["GAME_DATE"] >= k_first) & (p_df["GAME_DATE"] <= k_last)]
                # Population = games the player actually took the court for (MIN > 0).
                # A logged 0-minute game is a DNP, not a production sample, and must be
                # excluded from BOTH the per-game averages and the per-36 calc so every
                # statistic below is computed on one consistent set of games.
                p_window = p_window[p_window["MIN"] > 0]
                if len(p_window) < min_games_player:
                    continue

                with_mask = p_window["GAME_ID"].isin(k_games)
                with_df = p_window[with_mask]
                without_df = p_window[~with_mask]

                n_with, n_without = len(with_df), len(without_df)
                if n_without < min_games_without:
                    continue

                min_with = with_df["MIN"].mean()
                min_without = without_df["MIN"].mean()

                for stat in stats:
                    # Welch z is always computed on the per-game values (for USG, the
                    # per-game USG% column), so significance reflects game-to-game spread.
                    std_with = with_df[stat].std(ddof=1) if n_with > 1 else np.nan
                    std_without = without_df[stat].std(ddof=1) if n_without > 1 else np.nan
                    se = np.sqrt(
                        (std_with ** 2 / max(n_with, 1)) + (std_without ** 2 / max(n_without, 1))
                    )

                    if stat in RATE_STATS:
                        # USG%: average via correct component-sum aggregate, and per-36
                        # is undefined (the rate is already minute/possession-normalized),
                        # so we mirror the average into the per-36 columns.
                        avg_with = aggregate_usg(with_df)
                        avg_without = aggregate_usg(without_df)
                        with_per36 = avg_with
                        without_per36 = avg_without
                    else:
                        avg_with = with_df[stat].mean()
                        avg_without = without_df[stat].mean()
                        # Per-36 using league-standard total-stat / total-minutes * 36,
                        # on the same MIN>0 population as the averages above.
                        total_min_with = with_df["MIN"].sum()
                        total_min_without = without_df["MIN"].sum()
                        with_per36 = (with_df[stat].sum() / total_min_with * 36) if total_min_with > 0 else np.nan
                        without_per36 = (without_df[stat].sum() / total_min_without * 36) if total_min_without > 0 else np.nan

                    delta = avg_without - avg_with
                    z = delta / se if se and se > 0 else np.nan

                    rows.append({
                        "season": season,
                        "team": team_abbr,
                        "player": p_name,
                        "player_id": p,
                        "teammate_out": k_name,
                        "teammate_id": k,
                        "stat": stat,
                        "n_with": n_with,
                        "n_without": n_without,
                        "min_with": round(min_with, 1),
                        "min_without": round(min_without, 1),
                        "avg_with": round(avg_with, 2),
                        "avg_without": round(avg_without, 2),
                        "delta": round(delta, 2),
                        "pct_delta": round(delta / avg_with, 3) if avg_with else np.nan,
                        "se": round(se, 3) if not np.isnan(se) else np.nan,
                        "z": round(z, 2) if not np.isnan(z) else np.nan,
                        "per36_with": round(with_per36, 2) if not np.isnan(with_per36) else np.nan,
                        "per36_without": round(without_per36, 2) if not np.isnan(without_per36) else np.nan,
                    })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data", help="cache dir with parquet files")
    p.add_argument("--out", default="out/splits.parquet")
    args = p.parse_args()

    cache_dir = Path(args.cache)
    files = sorted(cache_dir.glob("gamelog_*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {cache_dir}/ — run src/fetch.py first")
    print(f"Loading {len(files)} cached player file(s)...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(df):,} player-game rows")

    stats = BASE_STATS + list(COMBO_STATS.keys())

    # If team logs are cached, join them to compute true USG% (teammate in vs out).
    team_files = sorted(cache_dir.glob("teamlog_*.parquet"))
    if team_files:
        teams = pd.concat([pd.read_parquet(f) for f in team_files], ignore_index=True)
        df = add_usage(df, teams)
        stats = stats + ["USG"]
        print(f"  joined {len(team_files)} team file(s) → true USG% enabled")
    else:
        print("  (no teamlog_*.parquet found — skipping USG%; run src.fetch to enable)")

    print(f"Computing splits across {len(stats)} stats...")
    splits = compute_pair_splits(df, stats)
    print(f"  {len(splits):,} (player, teammate, stat) rows")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
