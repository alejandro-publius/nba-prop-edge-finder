"""True usage rate (USG%) per player-game, plus correct multi-game aggregation.

USG% estimates the share of his team's offensive possessions a player "uses"
(via a shot, a trip to the line, or a turnover) while on the floor. The standard
formula (Basketball-Reference) is:

    USG% = 100 * ((FGA + 0.44*FTA + TOV) * (TmMIN / 5))
                 / (MIN * (TmFGA + 0.44*TmFTA + TmTOV))

The 0.44 weights free-throw trips (most shooting fouls yield 2 FTs, but and-1s,
flagrants, and technicals make the true factor ~0.44). TmMIN/5 converts team
minutes to "team minutes per position" so the player's MIN share is comparable.

Computing USG% needs per-game TEAM totals, which is why src.fetch also caches the
team-mode game logs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Per-game component columns added by add_usage, needed for correct aggregation.
USAGE_COMPONENT_COLS = ["USG_PU", "TM_TP", "TM_MIN"]


def add_usage(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Join per-game team totals onto player rows and compute per-game USG%.

    Returns a copy of `players` with new columns:
      USG_PU  player usage possessions = FGA + 0.44*FTA + TOV
      TM_TP   team usage possessions   = TmFGA + 0.44*TmFTA + TmTOV
      TM_MIN  team minutes that game (≈240 in regulation)
      USG     per-game usage rate (percent)
    """
    team_totals = (
        teams[["GAME_ID", "TEAM_ID", "MIN", "FGA", "FTA", "TOV"]]
        .rename(columns={"MIN": "TM_MIN", "FGA": "TM_FGA", "FTA": "TM_FTA", "TOV": "TM_TOV"})
    )
    df = players.merge(team_totals, on=["GAME_ID", "TEAM_ID"], how="left")

    df["USG_PU"] = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
    df["TM_TP"] = df["TM_FGA"] + 0.44 * df["TM_FTA"] + df["TM_TOV"]

    denom = df["MIN"] * df["TM_TP"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["USG"] = 100.0 * (df["USG_PU"] * (df["TM_MIN"] / 5.0)) / denom
    # Undefined when the player logged 0 minutes or team totals are missing.
    df.loc[(denom <= 0) | df["TM_TP"].isna(), "USG"] = np.nan
    return df


def aggregate_usg(games: pd.DataFrame) -> float:
    """True USG% over a SET of games, via summed components (not a mean of rates).

    Aggregating a rate correctly means summing numerators and denominators across
    games, exactly as Basketball-Reference derives a season USG% from season totals:

        USG% = 100 * (ΣPU * (ΣTmMIN / 5)) / (ΣMIN * ΣTP)
    """
    g = games[games["MIN"] > 0]
    if g.empty:
        return float("nan")
    num = g["USG_PU"].sum() * (g["TM_MIN"].sum() / 5.0)
    den = g["MIN"].sum() * g["TM_TP"].sum()
    return 100.0 * num / den if den > 0 else float("nan")
