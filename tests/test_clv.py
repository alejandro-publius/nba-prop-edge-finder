"""Smoke + math tests for the CLV logger."""
import sqlite3
from pathlib import Path

import pytest

from src.clv import (
    SCHEMA,
    add_entry,
    close_entry,
    entry_clv,
    entry_line_movement,
    grade_entry,
    realized_pnl,
)


@pytest.fixture
def con(tmp_path):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def test_add_and_fetch(con):
    eid = add_entry(
        con, player="Test", teammate_out="X", stat="PTS", side="over",
        line=20.5, price=-110, other_price=-110, book="FD",
    )
    row = con.execute("SELECT * FROM entries WHERE id = ?", (eid,)).fetchone()
    assert row["player"] == "Test"
    assert row["stat"] == "PTS"
    assert row["side"] == "over"
    assert row["line"] == 20.5
    assert row["entry_price"] == -110


def test_close_and_clv_positive(con):
    # Entered over at -110 on a balanced market (no-vig 0.5).
    # Closed at -150 / +130 (over favored): no-vig P(over at close) ≈ 0.58.
    # The over became MORE likely → entry price was the BETTER deal → positive CLV.
    eid = add_entry(con, player="P", teammate_out="T", stat="RA", side="over",
                    line=9.5, price=-110, other_price=-110)
    close_entry(con, entry_id=eid, close_line=9.5, close_price=-150, other_price=130)
    row = con.execute("SELECT * FROM entries WHERE id = ?", (eid,)).fetchone()
    clv = entry_clv(row)
    assert clv is not None
    assert clv > 0   # close_p - entry_p ≈ 0.58 - 0.50 = +0.08
    assert 0.07 < clv < 0.09


def test_clv_negative_when_close_moves_against_side(con):
    # Entered over at -110; close moved the over to UNDERDOG (+150 / -180).
    # Close P(over) < 0.5 → over became less likely → you bet at the worse price.
    eid = add_entry(con, player="P", teammate_out="T", stat="RA", side="over",
                    line=9.5, price=-110, other_price=-110)
    close_entry(con, entry_id=eid, close_line=9.5, close_price=150, other_price=-180)
    row = con.execute("SELECT * FROM entries WHERE id = ?", (eid,)).fetchone()
    assert entry_clv(row) < 0


def test_clv_zero_when_no_movement(con):
    eid = add_entry(con, player="P", teammate_out="T", stat="PRA", side="over",
                    line=30.5, price=-110, other_price=-110)
    close_entry(con, entry_id=eid, close_line=30.5, close_price=-110, other_price=-110)
    row = con.execute("SELECT * FROM entries WHERE id = ?", (eid,)).fetchone()
    assert abs(entry_clv(row)) < 1e-9


def test_line_movement_over(con):
    # Over 9.5 entered, close at 11.5 — line moved against the over (harder to win)
    eid = add_entry(con, player="P", teammate_out="T", stat="RA", side="over",
                    line=9.5, price=-110, other_price=-110)
    close_entry(con, entry_id=eid, close_line=11.5, close_price=-110, other_price=-110)
    row = con.execute("SELECT * FROM entries WHERE id = ?", (eid,)).fetchone()
    # For overs, line_delta = close - entry = +2.0; the bet was at the EASIER number,
    # so movement was in your favor.
    assert entry_line_movement(row) == 2.0


def test_line_movement_under(con):
    eid = add_entry(con, player="P", teammate_out="T", stat="RA", side="under",
                    line=11.5, price=-110, other_price=-110)
    close_entry(con, entry_id=eid, close_line=9.5, close_price=-110, other_price=-110)
    row = con.execute("SELECT * FROM entries WHERE id = ?", (eid,)).fetchone()
    # For unders, line_delta inverted: close (9.5) - entry (11.5) = -2, then *-1 → +2
    assert entry_line_movement(row) == 2.0


def test_grade_win_over(con):
    eid = add_entry(con, player="P", teammate_out="T", stat="RA", side="over",
                    line=9.5, price=-110, other_price=-110)
    result = grade_entry(con, entry_id=eid, actual=11)
    assert result == "win"


def test_grade_loss_over(con):
    eid = add_entry(con, player="P", teammate_out="T", stat="RA", side="over",
                    line=9.5, price=-110, other_price=-110)
    result = grade_entry(con, entry_id=eid, actual=8)
    assert result == "loss"


def test_grade_push_whole_line(con):
    eid = add_entry(con, player="P", teammate_out="T", stat="RA", side="over",
                    line=10.0, price=-110, other_price=-110)
    result = grade_entry(con, entry_id=eid, actual=10)
    assert result == "push"


def test_realized_pnl_win(con):
    eid = add_entry(con, player="P", teammate_out="T", stat="RA", side="over",
                    line=9.5, price=-110, other_price=-110)
    grade_entry(con, entry_id=eid, actual=11)
    row = con.execute("SELECT * FROM entries WHERE id = ?", (eid,)).fetchone()
    # Win at -110: profit = 100/110 = 0.9091
    assert abs(realized_pnl(row) - 100 / 110) < 1e-4


def test_realized_pnl_push_is_zero(con):
    eid = add_entry(con, player="P", teammate_out="T", stat="RA", side="over",
                    line=10.0, price=-110, other_price=-110)
    grade_entry(con, entry_id=eid, actual=10)
    row = con.execute("SELECT * FROM entries WHERE id = ?", (eid,)).fetchone()
    assert realized_pnl(row) == 0.0
