"""Regression-test the W/O splits pipeline against known cases."""
from pathlib import Path

import pandas as pd
import pytest

from src.splits import add_combo_stats, compute_pair_splits, BASE_STATS, COMBO_STATS


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def gamelogs():
    files = sorted(DATA_DIR.glob("gamelog_*.parquet"))
    if not files:
        pytest.skip("No cached game logs. Run `python3 -m src.fetch` first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def test_combo_stats_sum_correctly():
    df = pd.DataFrame({"PTS": [10, 20], "REB": [5, 8], "AST": [3, 6], "FG3M": [1, 2],
                       "STL": [0, 1], "BLK": [0, 0], "TOV": [1, 2]})
    df = add_combo_stats(df)
    assert (df["PRA"] == [18, 34]).all()
    assert (df["PR"] == [15, 28]).all()
    assert (df["PA"] == [13, 26]).all()
    assert (df["RA"] == [8, 14]).all()


def test_jaylen_brown_without_tatum_24_25(gamelogs):
    """In 2024-25, Brown's AST jump w/o Tatum is the known signal (z ≈ 2.7)."""
    one_season = gamelogs[gamelogs["SEASON"] == "2024-25"]
    splits = compute_pair_splits(one_season, BASE_STATS + list(COMBO_STATS.keys()))
    jb = splits[
        (splits["player"] == "Jaylen Brown")
        & (splits["teammate_out"] == "Jayson Tatum")
    ]
    assert not jb.empty, "Expected Jaylen Brown / Tatum pair in splits"

    ast_row = jb[jb["stat"] == "AST"].iloc[0]
    # The known case: n_with ~56, n_without ~7, AST avg jumps from ~4.3 to ~6.3
    assert ast_row["n_with"] >= 50
    assert ast_row["n_without"] >= 5
    assert ast_row["avg_with"] < ast_row["avg_without"]  # he assists more without Tatum
    assert ast_row["delta"] > 1.5
    assert ast_row["z"] > 2.0


def test_dame_without_giannis_24_25_pra(gamelogs):
    """Dame's PRA without Giannis is the cleanest "no minutes confound" edge."""
    one_season = gamelogs[gamelogs["SEASON"] == "2024-25"]
    splits = compute_pair_splits(one_season, ["PTS", "REB", "AST"] + list(COMBO_STATS.keys()))
    dame = splits[
        (splits["player"] == "Damian Lillard")
        & (splits["teammate_out"] == "Giannis Antetokounmpo")
        & (splits["stat"] == "PRA")
    ]
    assert not dame.empty
    row = dame.iloc[0]
    # Minutes nearly identical, PRA jumps significantly
    assert abs(row["min_without"] - row["min_with"]) < 2.0
    assert row["delta"] > 5.0
    assert row["z"] > 3.0


def test_per36_uses_total_minutes_formula(gamelogs):
    """Per-36 should equal sum(stat)/sum(MIN)*36, not mean of ratios."""
    one_season = gamelogs[gamelogs["SEASON"] == "2024-25"]
    splits = compute_pair_splits(one_season, ["PTS"])

    # Spot-check Dame vs Giannis: total PTS and total MIN are available in raw logs
    raw = one_season[
        (one_season["PLAYER_NAME"] == "Damian Lillard")
        & (one_season["TEAM_ABBREVIATION"] == "MIL")
    ]
    giannis_games = one_season[
        (one_season["PLAYER_NAME"] == "Giannis Antetokounmpo")
        & (one_season["TEAM_ABBREVIATION"] == "MIL")
    ]["GAME_ID"]
    without = raw[~raw["GAME_ID"].isin(set(giannis_games)) & (raw["MIN"] > 0)]
    expected_per36 = without["PTS"].sum() / without["MIN"].sum() * 36

    dame_row = splits[
        (splits["player"] == "Damian Lillard")
        & (splits["teammate_out"] == "Giannis Antetokounmpo")
        & (splits["stat"] == "PTS")
    ].iloc[0]
    assert abs(dame_row["per36_without"] - expected_per36) < 0.05


def test_no_self_pairs(gamelogs):
    # Use one full team's data to ensure we get pairs
    one_season = gamelogs[(gamelogs["SEASON"] == "2024-25") & (gamelogs["TEAM_ABBREVIATION"] == "BOS")]
    splits = compute_pair_splits(one_season, ["PTS"])
    assert not splits.empty
    assert (splits["player"] != splits["teammate_out"]).all()
    assert (splits["player_id"] != splits["teammate_id"]).all()


def test_sample_size_floor(gamelogs):
    one_season = gamelogs[gamelogs["SEASON"] == "2024-25"]
    splits = compute_pair_splits(one_season, ["PTS"])
    assert (splits["n_without"] >= 5).all()
    assert (splits["n_with"] >= 0).all()
