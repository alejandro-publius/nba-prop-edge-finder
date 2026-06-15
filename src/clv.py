"""Forward CLV logger.

Tracks paper-trade or live entries against their eventual closing lines,
computing CLV in no-vig probability space — the only honest unit, because
juice varies across books and over time.

Workflow:

  # When the news drops and the line is stale:
  python3 -m src.clv add \\
      --player "Jaylen Brown" --teammate "Tatum" --stat RA \\
      --side over --line 9.5 --price -110 --other-price -110 \\
      --book FanDuel --note "Tatum ruled OUT 30 min before tip"

  # ~5 minutes before tip, log the closing line:
  python3 -m src.clv close --id 1 --close-line 11.5 --close-price -115 --other-price -105

  # After the game, log the actual stat for win/loss:
  python3 -m src.clv grade --id 1 --actual 13

  # See your full log:
  python3 -m src.clv report
  python3 -m src.clv summary
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sqlite3
from pathlib import Path
from typing import Optional

from .odds import (
    american_to_decimal,
    expected_roi,
    no_vig_from_american,
)


DB_PATH = Path(__file__).resolve().parent.parent / "out" / "clv.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_entry            TEXT    NOT NULL,
    player              TEXT    NOT NULL,
    teammate_out        TEXT    NOT NULL,
    stat                TEXT    NOT NULL,
    side                TEXT    NOT NULL CHECK (side IN ('over', 'under')),
    line                REAL    NOT NULL,
    entry_price         INTEGER NOT NULL,
    entry_other_price   INTEGER NOT NULL,
    book                TEXT,
    note                TEXT,
    ts_close            TEXT,
    close_line          REAL,
    close_price         INTEGER,
    close_other_price   INTEGER,
    actual              REAL,
    result              TEXT CHECK (result IN ('win', 'loss', 'push')),
    ts_graded           TEXT
);
"""


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute(SCHEMA)
    return con


# --- Operations -------------------------------------------------------------

def add_entry(
    con: sqlite3.Connection,
    *,
    player: str,
    teammate_out: str,
    stat: str,
    side: str,
    line: float,
    price: int,
    other_price: int,
    book: Optional[str] = None,
    note: Optional[str] = None,
    ts: Optional[str] = None,
) -> int:
    cur = con.execute(
        """
        INSERT INTO entries
          (ts_entry, player, teammate_out, stat, side, line,
           entry_price, entry_other_price, book, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts or now_iso(), player, teammate_out, stat.upper(), side.lower(),
         float(line), int(price), int(other_price), book, note),
    )
    con.commit()
    return int(cur.lastrowid)


def close_entry(
    con: sqlite3.Connection,
    *,
    entry_id: int,
    close_line: float,
    close_price: int,
    other_price: int,
    ts: Optional[str] = None,
) -> None:
    cur = con.execute(
        """
        UPDATE entries
        SET ts_close = ?, close_line = ?, close_price = ?, close_other_price = ?
        WHERE id = ?
        """,
        (ts or now_iso(), float(close_line), int(close_price), int(other_price), entry_id),
    )
    if cur.rowcount == 0:
        raise SystemExit(f"No entry with id {entry_id}")
    con.commit()


def grade_entry(con: sqlite3.Connection, *, entry_id: int, actual: float) -> str:
    row = con.execute("SELECT side, line FROM entries WHERE id = ?", (entry_id,)).fetchone()
    if row is None:
        raise SystemExit(f"No entry with id {entry_id}")
    if actual > row["line"]:
        result = "win" if row["side"] == "over" else "loss"
    elif actual < row["line"]:
        result = "win" if row["side"] == "under" else "loss"
    else:
        result = "push"
    con.execute(
        "UPDATE entries SET actual = ?, result = ?, ts_graded = ? WHERE id = ?",
        (float(actual), result, now_iso(), entry_id),
    )
    con.commit()
    return result


# --- Reporting --------------------------------------------------------------

def entry_clv(row: sqlite3.Row) -> Optional[float]:
    """CLV in no-vig probability space for the side that was bet.

    Standard convention: positive CLV = you beat the close (got a better price
    than the closing market). If you bet OVER and the close has the over at a
    HIGHER no-vig probability than your entry, you got the cheaper price → +CLV.

    Returned as `close_p - entry_p` (probability-point gain over the close).

    NOTE: This is the PRICE-ONLY CLV. If the line itself moved (e.g. 9.5 → 11.5),
    that movement is reported separately by `entry_line_movement`. Combining the
    two into a single number would require modeling the price-vs-line tradeoff,
    which is sport- and stat-specific; we keep them separate for honesty.
    """
    if row["close_price"] is None:
        return None
    entry_p = no_vig_from_american(row["entry_price"], row["entry_other_price"])
    close_p = no_vig_from_american(row["close_price"], row["close_other_price"])
    return close_p - entry_p


def entry_line_movement(row: sqlite3.Row) -> Optional[float]:
    if row["close_line"] is None:
        return None
    delta = float(row["close_line"]) - float(row["line"])
    return delta if row["side"] == "over" else -delta


def expected_roi_at_entry(row: sqlite3.Row, model_p: Optional[float] = None) -> Optional[float]:
    if model_p is None:
        return None
    return expected_roi(model_p, row["entry_price"])


def realized_pnl(row: sqlite3.Row, stake: float = 1.0) -> Optional[float]:
    if row["result"] is None:
        return None
    if row["result"] == "push":
        return 0.0
    dec = american_to_decimal(row["entry_price"])
    return stake * (dec - 1.0) if row["result"] == "win" else -stake


def fetch_all(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(con.execute("SELECT * FROM entries ORDER BY id"))


def print_report(con: sqlite3.Connection) -> None:
    rows = fetch_all(con)
    if not rows:
        print("(no entries logged yet)")
        return
    header = (
        f"{'id':>3}  {'player':16}  {'teammate':16}  {'stat':4}  "
        f"{'side':5}  {'line':>5}  {'entry':>6}  {'close':>5}/{'price':>5}  "
        f"{'price CLV':>9}  {'line mvmt':>9}  {'result':6}  {'P/L':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        clv = entry_clv(r)
        lm = entry_line_movement(r)
        pl = realized_pnl(r)
        close_line_str = f"{r['close_line']:.1f}" if r["close_line"] is not None else "-"
        close_price_str = f"{r['close_price']:+d}" if r["close_price"] is not None else "-"
        clv_str = f"{clv*100:+.1f}pt" if clv is not None else "-"
        lm_str = f"{lm:+.1f}" if lm is not None else "-"
        result_str = r["result"] if r["result"] else "-"
        pl_str = f"{pl:+.3f}" if pl is not None else "-"
        print(
            f"{r['id']:>3}  {r['player'][:16]:16}  {r['teammate_out'][:16]:16}  "
            f"{r['stat']:4}  {r['side']:5}  {r['line']:>5.1f}  "
            f"{r['entry_price']:>+6d}  {close_line_str:>5}/{close_price_str:>5}  "
            f"{clv_str:>9}  {lm_str:>9}  {result_str:6}  {pl_str:>6}"
        )


def print_summary(con: sqlite3.Connection) -> None:
    rows = fetch_all(con)
    if not rows:
        print("(no entries logged yet)")
        return

    closed = [r for r in rows if r["close_price"] is not None]
    graded = [r for r in rows if r["result"] is not None]

    print(f"Entries:          {len(rows)}")
    print(f"  closed (CLV):   {len(closed)}")
    print(f"  graded (P/L):   {len(graded)}")

    if closed:
        clvs = [entry_clv(r) for r in closed]
        avg_clv = sum(clvs) / len(clvs)
        pos_clv = sum(1 for c in clvs if c > 0)
        print()
        print(f"Avg price CLV (no-vig prob): {avg_clv*100:+.2f}pt over {len(closed)} entries")
        print(f"  positive CLV rate:         {pos_clv}/{len(closed)} = {pos_clv/len(closed)*100:.1f}%")

        line_moves = [entry_line_movement(r) for r in closed]
        avg_lm = sum(line_moves) / len(line_moves)
        pos_lm = sum(1 for x in line_moves if x > 0)
        print(f"Avg line movement (in your favor): {avg_lm:+.2f} points")
        print(f"  favorable line move rate:        {pos_lm}/{len(closed)} = {pos_lm/len(closed)*100:.1f}%")

    if graded:
        wins = sum(1 for r in graded if r["result"] == "win")
        losses = sum(1 for r in graded if r["result"] == "loss")
        pushes = sum(1 for r in graded if r["result"] == "push")
        pnl = sum(realized_pnl(r) for r in graded)
        decided = wins + losses
        win_rate = wins / decided if decided else 0
        print()
        print(f"W-L-P: {wins}-{losses}-{pushes}  ({win_rate*100:.1f}% on decided)")
        print(f"P/L:   {pnl:+.3f} units at 1u stakes")


# --- CLI --------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Forward CLV logger")
    p.add_argument("--db", default=str(DB_PATH), help="SQLite path")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Log a new entry")
    a.add_argument("--player", required=True)
    a.add_argument("--teammate", required=True, dest="teammate_out")
    a.add_argument("--stat", required=True)
    a.add_argument("--side", required=True, choices=["over", "under"])
    a.add_argument("--line", required=True, type=float)
    a.add_argument("--price", required=True, type=int, help="American odds on your side")
    a.add_argument("--other-price", required=True, type=int, help="American odds on the other side")
    a.add_argument("--book")
    a.add_argument("--note")

    c = sub.add_parser("close", help="Log the closing line + price")
    c.add_argument("--id", required=True, type=int, dest="entry_id")
    c.add_argument("--close-line", required=True, type=float)
    c.add_argument("--close-price", required=True, type=int)
    c.add_argument("--other-price", required=True, type=int)

    g = sub.add_parser("grade", help="Log the actual stat to grade the bet")
    g.add_argument("--id", required=True, type=int, dest="entry_id")
    g.add_argument("--actual", required=True, type=float)

    sub.add_parser("report", help="Show full entry table")
    sub.add_parser("summary", help="Aggregate CLV / W-L / P/L")

    args = p.parse_args()
    con = connect(Path(args.db))

    if args.cmd == "add":
        new_id = add_entry(
            con,
            player=args.player, teammate_out=args.teammate_out,
            stat=args.stat, side=args.side, line=args.line,
            price=args.price, other_price=args.other_price,
            book=args.book, note=args.note,
        )
        print(f"Logged entry id={new_id}")
    elif args.cmd == "close":
        close_entry(
            con, entry_id=args.entry_id, close_line=args.close_line,
            close_price=args.close_price, other_price=args.other_price,
        )
        print(f"Closed entry {args.entry_id}")
    elif args.cmd == "grade":
        result = grade_entry(con, entry_id=args.entry_id, actual=args.actual)
        print(f"Graded entry {args.entry_id}: {result}")
    elif args.cmd == "report":
        print_report(con)
    elif args.cmd == "summary":
        print_summary(con)


if __name__ == "__main__":
    main()
