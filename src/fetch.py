"""Fetch NBA player-game logs from stats.nba.com via nba_api and cache to parquet."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_season(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Pull every player-game row for a season. season format: '2024-25'."""
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",
    )
    df = log.get_data_frames()[0]
    df["SEASON"] = season
    df["SEASON_TYPE"] = season_type
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def cache_path(season: str, season_type: str) -> Path:
    safe_type = season_type.replace(" ", "_").lower()
    return CACHE_DIR / f"gamelog_{season.replace('-', '_')}_{safe_type}.parquet"


def load_or_fetch(season: str, season_type: str = "Regular Season", refresh: bool = False) -> pd.DataFrame:
    path = cache_path(season, season_type)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch_season(season, season_type)
    df.to_parquet(path, index=False)
    return df


def fetch_seasons(seasons: list[str], season_types: list[str], refresh: bool = False) -> pd.DataFrame:
    frames = []
    for s in seasons:
        for t in season_types:
            print(f"  fetching {s} {t}...")
            frames.append(load_or_fetch(s, t, refresh))
            time.sleep(0.6)  # be polite to stats.nba.com
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    p.add_argument("--types", nargs="+", default=["Regular Season", "Playoffs"])
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    df = fetch_seasons(args.seasons, args.types, args.refresh)
    print(f"\nFetched {len(df):,} player-game rows across {df['SEASON'].nunique()} season(s)")
    print(f"Players: {df['PLAYER_ID'].nunique():,}  Games: {df['GAME_ID'].nunique():,}")
