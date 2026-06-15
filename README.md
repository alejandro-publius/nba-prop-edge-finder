# NBA Prop Edge Finder

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)](tests/)

Mines NBA player-game logs for **with/without-teammate splits** across every major box-score
stat and surfaces the cases where a player's production shifts meaningfully when a specific
teammate is unavailable — the prop-betting setup where the sportsbook line is slow to
reprice an injury-driven role change.

Built around the observation that the same pattern recurs at every position: when a star
sits, a specific teammate absorbs a disproportionate share of the usage, minutes, or both,
and the prop line lags the move for the first few hours after the news drops.

## Headline result

Trained on 2023-24 + 2024-25, tested on the held-out 2025-26 season:

> **60.7% over-rate** (Wilson 95% CI [60.2%, 61.3%]) on 30,290 test-season games
> across 1,300 candidate edges. Break-even at -110 is 52.4%.

Edges are not pure multiple-testing artifacts. See `src.validate` to reproduce.
*(60.7% is an upper bound — real sportsbook lines incorporate news partially, so live
edge would be smaller. But the underlying signal is real.)*

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

# Liquid markets only (drop STL, BLK, TOV — rarely posted as props)
python3 -m src.edges --markets-only --clean-only --top 25
```

### `src.validate` — out-of-sample validation (the credibility check)

Trains splits on 2 seasons, scores edges against a held-out third season.
Reports overall hit rate with a Wilson confidence interval.

```bash
python3 -m src.validate --markets-only
# > Overall hit rate: 0.607  (Wilson 95% CI: [0.602, 0.613])
# > -110 break-even is 0.524. Observed: 0.607. Signal is present.
```

This is the honest answer to the multiple-testing problem (~130k pairs scanned →
plenty of false positives at z>2 by chance). Out-of-sample hit rate proves the
training edges aren't just noise.

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

### `src.clv` — forward CLV logger

SQLite-backed logger that tracks paper-trade or live entries against their eventual
closing lines. CLV is measured in no-vig probability space (positive = beat the close).

```bash
# 1) When the news drops and the line is stale, log the entry
python3 -m src.clv add \
    --player "Jaylen Brown" --teammate "Tatum" --stat RA \
    --side over --line 9.5 --price -110 --other-price -110 \
    --book FanDuel --note "Tatum ruled OUT 30 min before tip"

# 2) ~5 min before tip, log the closing line
python3 -m src.clv close --id 1 --close-line 10.5 --close-price -125 --other-price 105

# 3) After the game, log the actual stat
python3 -m src.clv grade --id 1 --actual 12

# View
python3 -m src.clv report
python3 -m src.clv summary
```

Tracks separately:
- **Price CLV** (no-vig probability points beaten at the close)
- **Line movement** (line points moved in the bet's favor)
- **Realized P/L** (actual win/loss, P/L in units at 1u stakes)

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

All pinned to known values in `tests/test_odds.py` (17 unit tests, including the canonical
-110 → 0.5238 implied, no-vig balancing, and Wilson CI exact values).

## Reproducibility

- All raw data flows from `nba_api` (free, public stats.nba.com endpoints)
- Caches are deterministic per season — re-running `src.fetch` is a no-op once cached
- **36 pytest regression tests** covering:
  - All 17 odds-math cases (`-110 → 0.5238`, no-vig, Wilson CI, Kelly, push-aware line eval)
  - Known split cases (Jaylen Brown/Tatum AST, Damian Lillard/Giannis PRA)
  - Per-36 formula matches `sum(stat)/sum(MIN)*36` byte-for-byte
  - CLV logger lifecycle (add → close → grade → CLV sign convention → P/L)
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

To turn this from a research tool into a fully automated betting workflow, two
external feeds are needed:

1. **Live sportsbook line feed.** Free-tier options: PrizePicks public JSON endpoints,
   Underdog public endpoints. Paid: The Odds API props tier, OddsJam. Without this, the
   CLV logger requires manual line entry.
2. **Injury news poller.** ESPN injury endpoints (unofficial but stable), Rotowire RSS,
   X/Twitter scraping. Triggers the "log entry NOW" event for `src.clv`.

Future modeling improvements (deliberately deferred — current OOS hit rate is already
well above break-even):

- **Recency weighting** in splits (exponential decay by game date)
- **Multi-teammate combos** (e.g. Brown w/o Tatum AND Holiday vs. either alone)
- **Pace adjustment** (per-100-possessions instead of per-36, when possessions matter)
- **Opponent DRtg control** in the without-sample to remove schedule confound
