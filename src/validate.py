"""Out-of-sample validation — does the teammate-out effect persist on held-out data?

We scan ~140k (player, teammate, stat) combos in training, so many large splits are
flukes. The only way to know the signal is real is to fix the edges on two seasons and
check them on a third the model never saw.

The honest question is NOT "how often did the player go over a line we drew" — you can
clear any break-even just by drawing the line below the mean. The honest question is:

    Holding the line fixed, how much MORE often does the player exceed it when the
    teammate is OUT versus when the teammate is IN?

That with-vs-without GAP cancels line placement, stat level, and player skill, because
both rates use the same player, same line, same season. A gap near zero means the
training edge was noise; a positive gap means the effect is real out of sample.

We report three things on the held-out season:
  1. Directional persistence — % of edges where the player still produced more without
     the teammate (chance = 50%).
  2. Over-rate gap — without-games over-rate minus with-games over-rate at a common,
     training-derived line. The with-rate is the control.
  3. Magnitude retention — median test delta vs median train delta (regression to the
     mean is expected; the test delta should stay positive but shrink).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from .odds import evaluate_line, wilson_interval
from .splits import COMBO_STATS, add_combo_stats, compute_pair_splits


def line_from_baseline(avg: float) -> float:
    """The X.5 half-point line within the unit interval `avg` falls in.

    e.g. avg in (9, 10] -> 9.5. Across many edges this averages to the mean, which
    is exactly why the with-teammate over-rate (the control) sits near 50%.
    Half-point lines never push, matching book grading.
    """
    return math.ceil(avg) - 0.5


def stat_values(games: pd.DataFrame, stat: str) -> pd.Series:
    if stat in COMBO_STATS:
        return games[COMBO_STATS[stat]].sum(axis=1)
    return games[stat]


def split_games_test(
    test_df: pd.DataFrame,
    player_id: int,
    teammate_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (with_games, without_games) for a pair in the test season.

    Mirrors src.splits: restrict to the teammate's tenure window on each shared team,
    population is games the player actually played (MIN > 0).
    """
    with_frames, without_frames = [], []
    p_all = test_df[(test_df["PLAYER_ID"] == player_id) & (test_df["MIN"] > 0)]
    k_all = test_df[test_df["PLAYER_ID"] == teammate_id]
    shared_teams = set(p_all["TEAM_ID"]) & set(k_all["TEAM_ID"])
    for team_id in shared_teams:
        p = p_all[p_all["TEAM_ID"] == team_id]
        k = k_all[k_all["TEAM_ID"] == team_id]
        if p.empty or k.empty:
            continue
        lo, hi = k["GAME_DATE"].min(), k["GAME_DATE"].max()
        p_win = p[(p["GAME_DATE"] >= lo) & (p["GAME_DATE"] <= hi)]
        k_games = set(k["GAME_ID"])
        with_frames.append(p_win[p_win["GAME_ID"].isin(k_games)])
        without_frames.append(p_win[~p_win["GAME_ID"].isin(k_games)])
    with_df = pd.concat(with_frames, ignore_index=True) if with_frames else pd.DataFrame()
    without_df = pd.concat(without_frames, ignore_index=True) if without_frames else pd.DataFrame()
    return with_df, without_df


def validate(
    gamelogs: pd.DataFrame,
    train_seasons: list[str],
    test_season: str,
    train_min_z: float = 2.0,
    train_min_pct: float = 0.15,
    train_min_n_without: int = 5,
    test_min_n_without: int = 5,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    if stats is None:
        stats = ["PTS", "REB", "AST", "FG3M"] + list(COMBO_STATS.keys())
    gamelogs = add_combo_stats(gamelogs.copy())

    train_df = gamelogs[gamelogs["SEASON"].isin(train_seasons)]
    test_df = gamelogs[gamelogs["SEASON"] == test_season]

    print(f"Training on {train_seasons} ({len(train_df):,} player-games)")
    print(f"Testing on {test_season} ({len(test_df):,} player-games)")
    print("Computing training splits...")
    train_splits = compute_pair_splits(train_df, stats)

    edges = train_splits[
        (train_splits["z"] >= train_min_z)
        & (train_splits["pct_delta"] >= train_min_pct)
        & (train_splits["n_without"] >= train_min_n_without)
        & (train_splits["avg_with"] >= 1.0)
    ].copy()
    print(f"  {len(edges):,} candidate edges meet training thresholds")

    n_no_test_data = 0
    rows = []
    for _, e in edges.iterrows():
        with_g, without_g = split_games_test(test_df, e["player_id"], e["teammate_id"])
        if len(without_g) < test_min_n_without or with_g.empty:
            n_no_test_data += 1
            continue

        stat = e["stat"]
        with_vals = stat_values(with_g, stat)
        without_vals = stat_values(without_g, stat)

        # ONE line, training-derived, applied to BOTH with- and without-games.
        line = line_from_baseline(e["avg_with"])
        r_without = evaluate_line(without_vals.tolist(), line)
        r_with = evaluate_line(with_vals.tolist(), line)

        test_avg_with = float(with_vals.mean())
        test_avg_without = float(without_vals.mean())

        rows.append({
            "player": e["player"],
            "teammate_out": e["teammate_out"],
            "stat": stat,
            "train_z": e["train_z"] if "train_z" in e else e["z"],
            "train_avg_with": e["avg_with"],
            "train_avg_without": e["avg_without"],
            "train_delta": e["delta"],
            "line": line,
            "test_n_with": r_with.n,
            "test_n_without": r_without.n,
            "test_avg_with": round(test_avg_with, 2),
            "test_avg_without": round(test_avg_without, 2),
            "test_delta": round(test_avg_without - test_avg_with, 2),
            "with_over": r_with.n_over,
            "with_decided": r_with.n_over + r_with.n_under,
            "without_over": r_without.n_over,
            "without_decided": r_without.n_over + r_without.n_under,
            "direction_persists": (test_avg_without - test_avg_with) > 0,
        })

    if n_no_test_data:
        print(f"  {n_no_test_data:,} edges dropped (pair didn't share ≥{test_min_n_without} "
              f"without-games in {test_season} — trade/injury/role change)")
    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    if results.empty:
        print("\nNo edges had enough held-out data to evaluate.")
        return

    n = len(results)
    persist = int(results["direction_persists"].sum())

    with_over = int(results["with_over"].sum())
    with_dec = int(results["with_decided"].sum())
    wo_over = int(results["without_over"].sum())
    wo_dec = int(results["without_decided"].sum())
    with_rate = with_over / with_dec if with_dec else float("nan")
    wo_rate = wo_over / wo_dec if wo_dec else float("nan")
    gap = wo_rate - with_rate

    med_train = results["train_delta"].median()
    med_test = results["test_delta"].median()

    print()
    print("=" * 64)
    print("OUT-OF-SAMPLE RESULTS (held-out season)")
    print("=" * 64)
    print(f"Edges evaluated:                {n}")
    print()
    print("1) Directional persistence")
    lo, hi = wilson_interval(persist, n)
    print(f"   Still higher WITHOUT teammate: {persist}/{n} = {persist/n*100:.1f}%  "
          f"(Wilson 95% CI [{lo*100:.1f}%, {hi*100:.1f}%], chance = 50%)")
    print()
    print("2) Over-rate gap at a common, training-derived line")
    print(f"   WITHOUT-teammate over-rate:    {wo_rate:.3f}  ({wo_over}/{wo_dec})")
    print(f"   WITH-teammate over-rate (ctrl):{with_rate:.3f}  ({with_over}/{with_dec})")
    print(f"   GAP (signal, line-placement-free): {gap:+.3f}  ({gap*100:+.1f} pts)")
    print()
    print("3) Magnitude retention (regression to the mean expected)")
    print(f"   Median train delta: {med_train:+.2f}")
    print(f"   Median test  delta: {med_test:+.2f}   "
          f"({med_test/med_train*100:.0f}% retained)" if med_train else "")
    print()
    if gap > 0.02 and persist / n > 0.5:
        print("VERDICT: effect persists out of sample — the gap is positive and the")
        print("majority of edges keep their direction. Not multiple-testing noise.")
    else:
        print("VERDICT: effect does NOT clearly persist — gap ~0 or direction near chance.")

    top = results.sort_values("train_z", ascending=False).head(20)
    print()
    print("Top-20 training edges, held-out result (test_delta should stay positive):")
    print(top[["player", "teammate_out", "stat", "train_delta", "test_delta",
               "test_n_without", "direction_persists"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data")
    p.add_argument("--out", default="out/validation.csv")
    p.add_argument("--train", nargs="+", default=["2023-24", "2024-25"])
    p.add_argument("--test", default="2025-26")
    p.add_argument("--train-min-z", type=float, default=2.0)
    p.add_argument("--train-min-pct", type=float, default=0.15)
    args = p.parse_args()

    cache_dir = Path(args.cache)
    files = sorted(cache_dir.glob("gamelog_*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {cache_dir}/")
    gamelogs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    gamelogs["GAME_DATE"] = pd.to_datetime(gamelogs["GAME_DATE"])

    results = validate(
        gamelogs,
        train_seasons=args.train,
        test_season=args.test,
        train_min_z=args.train_min_z,
        train_min_pct=args.train_min_pct,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nWrote {len(results)} evaluable edges to {out_path}")
    print_summary(results)
