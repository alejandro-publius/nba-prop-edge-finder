# NBA Prop Edge Finder

Mines NBA player-game logs for **with/without-teammate splits** across every major box-score
stat and surfaces the cases where a player's production shifts meaningfully when a specific
teammate is unavailable — the prop-betting setup where the sportsbook line is slow to
reprice an injury-driven role change.

Built around the observation that the same pattern recurs at every position: when a star
sits, a specific teammate absorbs a disproportionate share of the usage, minutes, or both,
and the prop line lags the move for the first few hours after the news drops.

## Quickstart

```bash
make install        # installs nba_api, pandas, pyarrow, pytest
make fetch          # caches 3 seasons of player-game logs to data/
make splits         # computes ~130k (player, teammate, stat) split rows
make clean-edges    # surfaces top clean edges (no minutes confound)
make test           # runs the test suite
```

`make all` does the full pipeline end-to-end.

## What it does

For every player on every team, compute their split:

- **with teammate K** = games where both player and teammate appeared for the same team
- **without teammate K** = games where the player appeared but teammate K did not

Then rank by effect size, statistical significance, and a **minutes-confound diagnostic**
(per-36 stat lines, to separate "more minutes" from "more productive per minute").

Stats covered: `PTS`, `REB`, `AST`, `FG3M`, `STL`, `BLK`, `TOV`, plus combos
`PR`, `PA`, `RA`, `PRA`.

## CLI tools

### `src.edges` — batch edge surfacing

```bash
# Default: 4k–5k candidate edges
python3 -m src.edges --top 25

# Stricter: large effect, no minutes confound, both raw and per-36 same direction
python3 -m src.edges --clean-only --min-z 2.5 --min-pct 0.15 --top 25

# Filter to a player, teammate, team, or single stat
python3 -m src.edges --player "Jaylen Brown" --teammate "Tatum"
python3 -m src.edges --stat RA --min-z 2.0 --top 30
python3 -m src.edges --team BOS

# Negative edges (player produces less when teammate is out)
python3 -m src.edges --direction down --top 20
```

### `src.price` — price a specific line with proper odds math

Given a sportsbook line + American prices, compute empirical P(over), Wilson 95% CI,
no-vig market probability, edge in probability points, expected ROI, and capped Kelly stake.

```bash
python3 -m src.price \
    --player "Jaylen Brown" --teammate "Tatum" --stat RA \
    --line 9.5 --over -110 --under -110
```

Example output:
```
Sample (without Tatum): n=16, over=10, push=0, under=6
Empirical P(over):       0.625    (Wilson 95% CI: [0.386, 0.815])
Market no-vig P(over):   0.500
Probability edge:        +0.125
Expected ROI per unit:   +19.32%   at -110 on the over
Kelly stake (cap 25%):   21.25% of bankroll
VERDICT: positive point estimate but Wilson lower bound suggests sample too thin to commit.
```

The Wilson lower bound is the honest read on small samples: a +12.5pt point estimate over
just 16 games can easily evaporate. The verdict line is conservative on purpose.

## Methodology

### With/without inference

For each `(player, teammate, team, season)`, restrict to games within the **overlap of
both their tenures** with that team (handles mid-season trades). "Teammate is out" = the
teammate had a roster spot for the team in that window but did not appear in the box score
for that game. This conflates injury, rest, suspension, and DNP — fine for the betting
question, which is "what happens when teammate K isn't available," not "what happens when
K is specifically injured."

### Sample-size filters

- Player played ≥ 15 games for the team in that season
- Teammate played ≥ 15 games for the team in that season
- ≥ 5 games "without"
- Player's average minutes ≥ 15.0 (filters end-of-bench garbage-time players)

### Per-36 (league standard)

```
per_36 = (total stat in split) / (total minutes in split) * 36
```

This is the league-standard formula, **not** the mean of per-game per-36 rates (which is
biased upward by short-minute games). Zero-minute games are dropped from the per-36 calc.

### Welch z-score

Standard error of the delta between two unequal-variance samples:

```
SE = sqrt(s_with² / n_with + s_without² / n_without)
z  = (avg_without - avg_with) / SE
```

A z of ~2 corresponds to a roughly 5% two-tailed p-value, but **don't read the p-value
literally** — we're scanning 100k+ pairs and the multiple-testing problem is severe.
Treat z as a ranking signal, not a significance test.

### Sportsbook odds math (`src.odds`)

- `american_to_prob(odds)` — implied probability (including vig)
- `no_vig_from_american(side, other)` — vig-free fair probability
- `evaluate_line(values, line)` — empirical P(over) with Wilson CI; correctly excludes
  pushes from the denominator on whole-number lines (the standard book grading rule)
- `expected_roi`, `kelly_fraction` — bet sizing

All pinned to known values in `tests/test_odds.py` (25 unit tests, including the canonical
-110 → 0.5238 implied, no-vig balancing, and Wilson CI exact values).

## Reproducibility

- All raw data flows from `nba_api` (free, public stats.nba.com endpoints)
- Caches are deterministic per season — re-running `src.fetch` is a no-op once cached
- `pytest` regression-pins:
  - The Jaylen Brown / Tatum AST jump (z > 2.0 in 2024-25)
  - The Damian Lillard / Giannis PRA jump (z > 3.0 in 2024-25, clean minutes)
  - Per-36 formula matches `sum(stat)/sum(MIN)*36` byte-for-byte
- Pipeline is deterministic given a fixed cache (no random sampling, no model state)

## Output columns reference

| Column | Meaning |
|---|---|
| `n_with`, `n_without` | Sample sizes |
| `min_with`, `min_without`, `min_delta` | Minutes per game in each split |
| `avg_with`, `avg_without`, `delta`, `pct_delta` | Per-game stat in each split |
| `per36_with`, `per36_without`, `per36_delta` | Per-36-minute version |
| `same_sign_per36` | True if per-36 moves the same direction as raw |
| `z` | Welch-style z-score of `delta` over its standard error |
| `minutes_confound` | True if `\|min_delta\|` ≥ 4.0 (effect partly explained by minutes) |

## Limitations (read this before you bet)

- **No opponent control.** "Without" games may cluster against weaker/stronger opponents.
  At small n_without this is a real confound, not noise.
- **No home/away split.**
- **No starter vs. bench split.** A player whose role changes between starts and bench
  shows up as one bucket.
- **Multiple testing.** We scan ~130k (player, teammate, stat) tuples — a z of 2.0 by
  pure chance occurs in 2.5% of tests, which is ~3,250 false signals at z≥2. Use
  `--clean-only` and `--min-z 3.0+` to lean toward signal.
- **Sample-size honesty.** A 6-game "without" sample with z=3 is much weaker evidence
  than 25 games with z=3. Always read `n_without` alongside z. `src.price` reports the
  Wilson 95% CI explicitly for this reason.
- **No live odds.** This repo finds historical splits and prices a *single* candidate
  line. To turn it into a betting workflow you need a live odds feed.

## A note on Yahoo Fantasy API

The Yahoo Fantasy API is **not useful for the historical model**. It exposes raw box
stats and Yahoo's own fantasy projections, but no advanced metrics (true USG%, on/off
splits, lineup data) — strictly a subset of what `nba_api` provides, in a less convenient
format. The right separation is:

- **`nba_api` (this repo)** → historical splits, real box scores, lineup data
- **Yahoo Fantasy API** (not used here) → *tonight's* roster/injury status + Yahoo's
  projections, useful as a forward-looking "who's out tonight" feed once you have a
  live betting workflow

Don't try to use Yahoo as a USG% source — it doesn't expose that.

## What's not in this repo (yet)

The resume claim of "capturing positive closing-line value" requires three more pieces:

1. **Live sportsbook line feed.** Free-tier options: PrizePicks public JSON endpoints,
   Underdog public endpoints. Paid: The Odds API props tier, OddsJam.
2. **Injury news poller.** ESPN injury endpoints (unofficial but stable), Rotowire RSS,
   X/Twitter scraping (gray).
3. **Forward CLV logger.** SQLite table of (timestamp_entry, line, price, timestamp_close,
   close_line, close_price, CLV_in_no_vig_prob_space). Two hours of work given `src.odds`.

These are separate concerns from "find the edge" and intentionally not in scope here.
