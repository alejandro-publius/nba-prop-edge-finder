"""Empirical-Bayes shrinkage of with/without deltas (winner's-curse correction).

A raw split delta is a noisy estimate of a player's true teammate-out effect. When
you scan ~140k combos and keep the biggest ones, the survivors are biased high —
the classic winner's curse / regression to the mean. The out-of-sample check bears
this out: the median selected delta retains only ~25% of its size on a fresh season.

The principled fix is to shrink each delta toward the population prior by how
*reliable* it is. Model the true deltas (within a stat) as drawn from a prior with
mean μ and variance τ², and each observed delta d_i as that true value plus
sampling noise of variance se_i². The posterior mean is:

    shrunk_i = μ + k_i · (d_i − μ),   where   k_i = τ² / (τ² + se_i²)

k_i is the reliability (Kalman gain): ~1 for low-noise (large-sample) estimates,
~0 for high-noise (small-sample) ones, which get pulled back to the prior. We
estimate τ² per stat by method of moments:

    Var(observed d) = τ² + mean(se²)   ⇒   τ̂² = max(Var(d) − mean(se²), 0)

This is the honest forward number — the projection you'd actually price off, not
the inflated historical split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_shrinkage(df: pd.DataFrame) -> pd.DataFrame:
    """Add empirical-Bayes shrunk columns, estimating the prior per stat.

    New columns:
      prior_delta     population mean delta μ for the stat
      tau2            estimated between-pair variance of true deltas
      shrink_k        reliability k_i in [0, 1]
      shrunk_delta    posterior-mean delta = μ + k·(d − μ)
      shrunk_without  avg_with + shrunk_delta  (the honest projected without-mean)
    """
    df = df.copy()
    for col in ["prior_delta", "tau2", "shrink_k", "shrunk_delta", "shrunk_without"]:
        df[col] = np.nan

    for stat, g in df.groupby("stat"):
        valid = g["se"].notna() & (g["se"] >= 0) & g["delta"].notna()
        gv = g[valid]
        if len(gv) < 2:
            continue
        d = gv["delta"].to_numpy(dtype=float)
        se2 = gv["se"].to_numpy(dtype=float) ** 2

        mu = float(np.mean(d))
        var_d = float(np.var(d, ddof=1))
        tau2 = max(var_d - float(np.mean(se2)), 0.0)

        k = tau2 / (tau2 + se2)            # reliability per pair, in [0, 1]
        shrunk_delta = mu + k * (d - mu)

        idx = gv.index
        df.loc[idx, "prior_delta"] = round(mu, 3)
        df.loc[idx, "tau2"] = round(tau2, 3)
        df.loc[idx, "shrink_k"] = np.round(k, 3)
        df.loc[idx, "shrunk_delta"] = np.round(shrunk_delta, 2)
        df.loc[idx, "shrunk_without"] = np.round(gv["avg_with"].to_numpy(dtype=float) + shrunk_delta, 2)

    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="out/splits.parquet")
    p.add_argument("--stat", help="only show this stat")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    df = pd.read_parquet(args.splits)
    df = add_shrinkage(df)

    print("Per-stat prior and shrinkage (how much the raw deltas get discounted):")
    summary = (
        df.dropna(subset=["shrink_k"])
        .groupby("stat")
        .agg(prior_delta=("prior_delta", "first"),
             tau2=("tau2", "first"),
             mean_k=("shrink_k", "mean"),
             n=("shrink_k", "size"))
        .round(3)
        .sort_values("mean_k")
    )
    print(summary.to_string())

    view = df.dropna(subset=["shrunk_delta"])
    if args.stat:
        view = view[view["stat"].str.upper() == args.stat.upper()]
    view = view[view["n_without"] >= 8].sort_values("shrunk_delta", ascending=False)
    print(f"\nTop edges by SHRUNK delta (the honest forward number), n_without>=8:")
    cols = ["season", "team", "player", "teammate_out", "stat", "n_without",
            "avg_with", "avg_without", "delta", "shrink_k", "shrunk_delta", "shrunk_without"]
    print(view[cols].head(args.top).to_string(index=False))
