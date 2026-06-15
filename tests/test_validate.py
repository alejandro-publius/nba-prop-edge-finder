"""Tests for the out-of-sample validation helpers."""
import pandas as pd

from src.validate import line_from_baseline, split_games_test, stat_values


def test_line_from_baseline_is_half_point_in_unit_interval():
    # ceil(avg) - 0.5: the X.5 line in the unit interval avg falls in.
    assert line_from_baseline(9.2) == 9.5   # (9, 10] -> 9.5
    assert line_from_baseline(9.7) == 9.5
    assert line_from_baseline(9.0) == 8.5   # exactly 9 -> (8, 9] -> 8.5
    assert line_from_baseline(10.01) == 10.5
    # never a whole number => never a push
    for v in [1.1, 5.5, 9.9, 20.0, 33.3]:
        assert line_from_baseline(v) % 1 == 0.5


def test_stat_values_combo():
    g = pd.DataFrame({"PTS": [10], "REB": [5], "AST": [3]})
    assert stat_values(g, "PRA").iloc[0] == 18
    assert stat_values(g, "PTS").iloc[0] == 10


def _game(pid, team, gid, date, mins=30):
    return {
        "PLAYER_ID": pid, "TEAM_ID": team, "GAME_ID": gid,
        "GAME_DATE": pd.Timestamp(date), "MIN": mins,
        "PTS": 20, "REB": 5, "AST": 5,
    }


def test_split_games_test_classifies_with_and_without():
    # Player 1 plays g1..g4. Teammate 2 plays g1 and g4 (so the tenure window spans
    # all four games) but misses g2, g3 => those are the "without" games.
    rows = [
        _game(1, 100, "g1", "2025-11-01"),
        _game(1, 100, "g2", "2025-11-03"),
        _game(1, 100, "g3", "2025-11-05"),
        _game(1, 100, "g4", "2025-11-07"),
        _game(2, 100, "g1", "2025-11-01"),
        _game(2, 100, "g4", "2025-11-07"),
    ]
    df = pd.DataFrame(rows)
    with_g, without_g = split_games_test(df, player_id=1, teammate_id=2)
    assert set(with_g["GAME_ID"]) == {"g1", "g4"}
    assert set(without_g["GAME_ID"]) == {"g2", "g3"}


def test_split_games_test_excludes_zero_minute_games():
    rows = [
        _game(1, 100, "g1", "2025-11-01", mins=0),   # DNP, excluded
        _game(1, 100, "g2", "2025-11-03", mins=30),
        _game(2, 100, "g2", "2025-11-03", mins=30),
    ]
    df = pd.DataFrame(rows)
    with_g, without_g = split_games_test(df, player_id=1, teammate_id=2)
    assert set(with_g["GAME_ID"]) == {"g2"}
    assert without_g.empty


def test_split_games_test_respects_tenure_window():
    # Teammate only present g2..g3; player's g1 (before teammate window) is excluded
    # entirely from the without set, not counted as "without".
    rows = [
        _game(1, 100, "g1", "2025-11-01"),
        _game(1, 100, "g2", "2025-11-03"),
        _game(1, 100, "g3", "2025-11-05"),
        _game(2, 100, "g2", "2025-11-03"),
        _game(2, 100, "g3", "2025-11-05"),
    ]
    df = pd.DataFrame(rows)
    with_g, without_g = split_games_test(df, player_id=1, teammate_id=2)
    # g1 is before the teammate's first game → excluded from both
    assert "g1" not in set(with_g["GAME_ID"])
    assert "g1" not in set(without_g["GAME_ID"])
    assert set(with_g["GAME_ID"]) == {"g2", "g3"}
