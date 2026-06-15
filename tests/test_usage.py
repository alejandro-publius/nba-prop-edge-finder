"""Verify true USG% computation and aggregation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.usage import add_usage, aggregate_usg

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_usg_single_game_formula():
    """Hand-computed USG% for one game against the textbook formula."""
    players = pd.DataFrame({
        "GAME_ID": ["1"], "TEAM_ID": [10], "PLAYER_NAME": ["X"],
        "MIN": [36.0], "FGA": [20.0], "FTA": [5.0], "TOV": [3.0],
    })
    teams = pd.DataFrame({
        "GAME_ID": ["1"], "TEAM_ID": [10],
        "MIN": [240.0], "FGA": [88.0], "FTA": [20.0], "TOV": [14.0],
    })
    out = add_usage(players, teams)
    # PU = 20 + 0.44*5 + 3 = 25.2 ; TP = 88 + 0.44*20 + 14 = 110.8
    # USG = 100 * (25.2 * (240/5)) / (36 * 110.8) = 100 * 1209.6 / 3988.8 = 30.325%
    assert out["USG_PU"].iloc[0] == pytest.approx(25.2)
    assert out["TM_TP"].iloc[0] == pytest.approx(110.8)
    assert out["USG"].iloc[0] == pytest.approx(30.325, abs=1e-2)


def test_usg_zero_minutes_is_nan():
    players = pd.DataFrame({
        "GAME_ID": ["1"], "TEAM_ID": [10], "PLAYER_NAME": ["X"],
        "MIN": [0.0], "FGA": [0.0], "FTA": [0.0], "TOV": [0.0],
    })
    teams = pd.DataFrame({
        "GAME_ID": ["1"], "TEAM_ID": [10],
        "MIN": [240.0], "FGA": [88.0], "FTA": [20.0], "TOV": [14.0],
    })
    out = add_usage(players, teams)
    assert np.isnan(out["USG"].iloc[0])


def test_aggregate_usg_is_component_sum_not_mean_of_rates():
    """Two games with very different minutes — aggregate must weight by possessions,
    so it differs from the naive mean of the two per-game rates."""
    players = pd.DataFrame({
        "GAME_ID": ["1", "2"], "TEAM_ID": [10, 10], "PLAYER_NAME": ["X", "X"],
        "MIN": [40.0, 10.0], "FGA": [25.0, 2.0], "FTA": [6.0, 0.0], "TOV": [4.0, 1.0],
    })
    teams = pd.DataFrame({
        "GAME_ID": ["1", "2"], "TEAM_ID": [10, 10],
        "MIN": [240.0, 240.0], "FGA": [90.0, 85.0], "FTA": [22.0, 18.0], "TOV": [13.0, 12.0],
    })
    out = add_usage(players, teams)
    agg = aggregate_usg(out)
    # Component sums: PU=25+0.44*6+4 + 2+0.44*0+1 = 31.64+3 = 34.64
    #   ... game1 PU = 25+2.64+4 = 31.64 ; game2 PU = 2+0+1 = 3 ; sum = 34.64
    # TP1 = 90+9.68+13 = 112.68 ; TP2 = 85+7.92+12 = 104.92 ; sumTP = 217.6
    # num = 34.64 * (480/5) = 34.64 * 96 = 3325.44
    # den = (40+10) * 217.6 = 50 * 217.6 = 10880
    # agg = 100 * 3325.44 / 10880 = 30.565%
    assert agg == pytest.approx(30.565, abs=1e-2)
    mean_of_rates = out["USG"].mean()
    assert abs(agg - mean_of_rates) > 1.0  # the two genuinely differ


@pytest.fixture(scope="module")
def joined():
    pfiles = sorted(DATA_DIR.glob("gamelog_*.parquet"))
    tfiles = sorted(DATA_DIR.glob("teamlog_*.parquet"))
    if not pfiles or not tfiles:
        pytest.skip("Run `python3 -m src.fetch` first (need player + team logs).")
    players = pd.concat([pd.read_parquet(f) for f in pfiles], ignore_index=True)
    teams = pd.concat([pd.read_parquet(f) for f in tfiles], ignore_index=True)
    return add_usage(players, teams)


def test_star_usg_in_realistic_range(joined):
    """A high-usage star's season USG% should land in the known ~28-36% band."""
    one = joined[(joined["SEASON"] == "2024-25") & (joined["MIN"] > 0)]
    for name, lo, hi in [
        ("Luka Dončić", 30, 42),
        ("Giannis Antetokounmpo", 30, 42),
        ("Nikola Jokić", 26, 36),
    ]:
        rows = one[one["PLAYER_NAME"] == name]
        if rows.empty:
            continue
        usg = aggregate_usg(rows)
        assert lo <= usg <= hi, f"{name} USG% {usg:.1f} outside [{lo}, {hi}]"
