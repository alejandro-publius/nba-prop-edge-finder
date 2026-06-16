"""Regenerate the committed sample outputs in examples/ from out/splits.parquet.

Run `python3 -m src.fetch && python3 -m src.splits` first, then `python3 examples/generate.py`.
These CSVs are committed so the repo's results are visible without running the pipeline.
"""
from pathlib import Path

import pandas as pd

from src.shrink import add_shrinkage

LIQUID = {"PTS", "REB", "AST", "FG3M", "PR", "PA", "RA", "PRA"}
HERE = Path(__file__).resolve().parent


def main() -> None:
    df = add_shrinkage(pd.read_parquet("out/splits.parquet"))

    # 1) Biggest true-USG% jumps when a teammate sits (the headline metric).
    usg = df[(df["stat"] == "USG") & (df["n_without"] >= 8) & (df["avg_with"] >= 15)]
    usg = usg.sort_values("delta", ascending=False).head(25)
    usg[["season", "team", "player", "teammate_out", "n_with", "n_without",
         "avg_with", "avg_without", "delta", "z"]].rename(
        columns={"avg_with": "usg_in", "avg_without": "usg_out", "delta": "usg_jump"}
    ).to_csv(HERE / "usage_jumps.csv", index=False)

    # 2) Top clean prop-market edges with the honest (shrunk) projection.
    e = df[df["stat"].isin(LIQUID) & (df["n_without"] >= 8)
           & (df["z"] >= 2.5) & (df["avg_with"] >= 1)].copy()
    e["min_delta"] = (e["min_without"] - e["min_with"]).round(1)
    e = e[e["min_delta"].abs() < 4.0]  # no minutes confound
    e = e.sort_values("shrunk_delta", ascending=False).head(30)
    e[["season", "team", "player", "teammate_out", "stat", "n_without",
       "avg_with", "avg_without", "delta", "shrink_k", "shrunk_delta",
       "shrunk_without", "z"]].to_csv(HERE / "top_clean_edges.csv", index=False)

    print(f"Wrote {HERE/'usage_jumps.csv'} and {HERE/'top_clean_edges.csv'}")


if __name__ == "__main__":
    main()
