# NBA Prop Edge Finder

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)](tests/)

A reproducible pipeline that measures how an NBA player's role and production change
when a specific teammate is **on the floor vs. off it** — built to surface the cases
where a sportsbook's player-prop line is slow to reprice an injury-driven role change.

The central metric is **true usage rate (USG%) with a teammate IN vs. OUT**, with the full
box score (points, rebounds, assists, threes, and combos) measured the same way.

## The core idea, in one table

When a high-usage star sits, his possessions don't vanish — a specific teammate absorbs
them. Trained on real game logs (`nba_api`, free), here is that effect for three players
in the seasons where it mattered:

| Player | Teammate OUT | USG% in → out | Points in → out | What's happening |
|---|---|---|---|---|
| Damian Lillard | Giannis | 26.5% → **33.9%** (z=3.5) | 22.8 → **33.6** | Usage explosion → scoring |
| De'Aaron Fox | Wembanyama | 23.0% → **29.3%** (z=4.4) | 16.2 → **24.3** | Usage explosion → scoring |
| Jaylen Brown | Tatum | 28.5% → 30.2% (z=0.8, n.s.) | 21.9 → 24.5 | Usage flat — but **assists** +2.0 (z=3.1) |

The Brown row is the point: raw usage barely moves, but his **role** shifts to playmaking.
A model that only looked at points would miss it; one that reads usage *and* the full box
score catches it. That nuance is the whole product.

## Does the effect actually persist? (out-of-sample)

Finding a big split in past data is easy and mostly noise — we scan ~140k
(player, teammate, stat) combos. The honest test is whether edges found on two seasons
still show up on a **third season the model never saw**.

`src.validate` trains on 2023-24 + 2024-25 and tests on held-out 2025-26. It does **not**
ask "how often did the player beat a line we drew" (you can clear any break-even just by
drawing the line below the mean). It asks: *holding the line fixed, how much more often
does the player exceed it when the teammate is OUT vs. IN?* — a gap that cancels line
placement, because both rates use the same player, same line, same season.

```
Edges evaluated: 1,193

1) Directional persistence
   Still higher WITHOUT teammate: 871/1193 = 73.0%  (Wilson 95% CI [70.4%, 75.5%], chance = 50%)

2) Over-rate gap at a common, training-derived line
   WITHOUT-teammate over-rate:     0.622
   WITH-teammate over-rate (ctrl): 0.520     <- the control; ~50% confirms the line is fair
   GAP (line-placement-free):     +0.102  (+10.2 pts)

3) Magnitude retention
   Median train delta: +4.42  ->  median test delta: +1.11   (25% retained)
```

Read this honestly: the **direction** of the effect is robust (73% vs. 50% chance), and
the line-controlled **gap is real** (+10.2 pts, with the control sitting at ~50% exactly as
expected). But the **magnitude shrinks by ~75% out of sample** — the raw training splits are
inflated by selection bias (winner's curse), so the live edge is a fraction of what the
historical split suggests. The pipeline reports all three so the shrinkage is never hidden.

## Quickstart

```bash
make install        # nba_api, pandas, pyarrow, pytest
make fetch          # caches player + team game logs (3 seasons) to data/
make splits         # computes ~140k (player, teammate, stat) split rows, incl. USG%
make edges          # surfaces the largest, most significant splits
make test           # 46 tests
```

`make all` runs the whole pipeline end-to-end.

## Tools

### `src.splits` — the engine
Computes, for every `(player, teammate, team, season)`, the player's per-game average,
per-36, and **true USG%** in games the teammate played vs. games the teammate missed,
with a Welch z-score on the difference. Output: `out/splits.parquet`.

### `src.edges` — ranking and filtering
```bash
python3 -m src.edges --stat USG --min-z 2.5 --top 25     # biggest usage jumps
python3 -m src.edges --player "Lillard" --teammate "Giannis"
python3 -m src.edges --clean-only --markets-only --top 25 # liquid prop stats, no minutes confound
python3 -m src.edges --direction down --top 20            # players who do LESS without a teammate
```
The `minutes_confound` / `same_sign_per36` columns separate "produced more because he
played more minutes" from "produced more per minute" (a genuine role change). USG% is
immune to the minutes confound by construction, which is why it's the cleanest signal.

### `src.validate` — out-of-sample check (above)
```bash
python3 -m src.validate
```

### `src.price` — price one line with proper odds math
```bash
python3 -m src.price --player "Jaylen Brown" --teammate "Tatum" --stat RA \
    --line 9.5 --over -110 --under -110
```
Reports empirical P(over) with a Wilson 95% CI, the no-vig market probability, the
probability edge, expected ROI, and a capped Kelly stake. The verdict is deliberately
conservative: a wide Wilson interval on a small sample returns "too thin to commit."

### `src.clv` — closing-line-value logger
SQLite-backed log of entries vs. their closing lines. CLV is measured in no-vig
probability space (positive = beat the close); line movement and realized P/L are tracked
separately. `add` → `close` → `grade`, then `report` / `summary`.

## Methodology

**With/without inference.** For each pair, restrict to the overlap of both players' tenure
with the team (handles mid-season trades). The population is games the player actually
took the court for (MIN > 0). "Teammate out" = the teammate was rostered in that window
but didn't appear — this folds together injury, rest, and DNP, which is the right unit for
"what happens when he's unavailable."

**True USG%.** `100 * ((FGA + 0.44·FTA + TOV) · (TmMIN/5)) / (MIN · (TmFGA + 0.44·TmFTA + TmTOV))`,
computed per game from joined team totals, and aggregated across a split by **summing
components** (the Basketball-Reference season-total method), not by averaging per-game
rates. Star usage lands in the expected 26–40% band (test-pinned).

**Per-36.** `sum(stat)/sum(MIN)·36` (league standard), on the same MIN>0 games as the
averages — never a mean of per-game ratios.

**Welch z.** `delta / sqrt(s²_with/n_with + s²_without/n_without)`. With ~140k combos the
multiple-testing burden is severe, so z is a **ranking signal, not a p-value**. The
out-of-sample check is what separates signal from noise.

**Odds math** (`src.odds`): American↔probability, no-vig, Wilson score interval, expected
ROI, capped Kelly, and push-aware line grading (pushes excluded from the denominator on
whole-number lines). All pinned to known values in `tests/test_odds.py`.

## Reproducibility

- 100% of inputs come from `nba_api` (public stats.nba.com); caches are deterministic.
- **46 tests**, including hand-computed USG% (single-game and component-sum aggregate),
  the odds-math constants (-110 → 0.5238, etc.), with/without classification and tenure
  windows, the validation controls, and regression pins on the Brown/Tatum and
  Lillard/Giannis splits.
- No randomness, no model state — same cache in, same numbers out.

## Limitations

- **Selection shrinkage.** Training splits are inflated by selection; the validated live
  effect is ~25% of the raw delta. Always read the out-of-sample magnitude, not the split.
- **No opponent / home-away / starter-vs-bench control.** "Without" games may cluster
  against particular opponents; at small samples this is a real confound.
- **Small samples.** Many pairs have 5–10 "without" games. `src.price` reports the Wilson
  interval precisely so thin samples can't masquerade as strong edges.
- **No live feeds.** Turning this into an automated workflow needs a live line feed
  (e.g. PrizePicks/Underdog public endpoints) and an injury-news poller; those are out of
  scope. The analysis and CLV-tracking layers are here; the data plumbing is not.

## Data note

`nba_api` is the right source: it provides true USG%, real box scores, and the per-game
team totals needed to compute usage. (Yahoo's Fantasy API exposes only raw box stats and
its own projections — a strict subset — so it isn't used here.)
