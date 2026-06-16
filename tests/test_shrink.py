"""Pin the empirical-Bayes shrinkage math."""
import numpy as np
import pandas as pd
import pytest

from src.shrink import add_shrinkage


def _df(stat, deltas, ses, avg_with=10.0):
    n = len(deltas)
    return pd.DataFrame({
        "stat": [stat] * n,
        "delta": deltas,
        "se": ses,
        "avg_with": [avg_with] * n,
        "n_without": [10] * n,
    })


def test_reliability_and_shrinkage_closed_form():
    # deltas [0,2,4,6,8], all se=1.
    # mu=4, var_d(ddof=1)=10, mean(se^2)=1 -> tau2=9, k=9/10=0.9
    # shrunk = 4 + 0.9*(d-4)
    out = add_shrinkage(_df("PTS", [0, 2, 4, 6, 8], [1, 1, 1, 1, 1]))
    assert out["prior_delta"].iloc[0] == pytest.approx(4.0)
    assert out["tau2"].iloc[0] == pytest.approx(9.0, abs=1e-6)
    assert out["shrink_k"].iloc[0] == pytest.approx(0.9, abs=1e-3)
    # d=8 -> 7.6 ; d=0 -> 0.4
    assert out["shrunk_delta"].iloc[4] == pytest.approx(7.6, abs=1e-2)
    assert out["shrunk_delta"].iloc[0] == pytest.approx(0.4, abs=1e-2)


def test_shrunk_without_is_avg_with_plus_shrunk_delta():
    out = add_shrinkage(_df("PTS", [0, 2, 4, 6, 8], [1, 1, 1, 1, 1], avg_with=12.0))
    assert out["shrunk_without"].iloc[4] == pytest.approx(12.0 + 7.6, abs=1e-2)


def test_higher_se_shrinks_harder():
    # Same delta, different se: the noisier (higher se) estimate must pull closer to the prior mean.
    deltas = [0, 2, 4, 6, 8, 8]   # last two share delta=8 but different se
    ses = [1, 1, 1, 1, 1, 5]
    out = add_shrinkage(_df("PTS", deltas, ses))
    low_se = out.iloc[4]   # delta 8, se 1
    high_se = out.iloc[5]  # delta 8, se 5
    assert high_se["shrink_k"] < low_se["shrink_k"]
    mu = out["prior_delta"].iloc[0]
    # high-se shrunk delta is closer to the prior mean than low-se
    assert abs(high_se["shrunk_delta"] - mu) < abs(low_se["shrunk_delta"] - mu)


def test_zero_se_means_no_shrinkage():
    # If every estimate is noiseless (se=0), k=1 and shrunk == raw delta.
    out = add_shrinkage(_df("PTS", [0, 2, 4, 6, 8], [0, 0, 0, 0, 0]))
    assert out["shrink_k"].iloc[0] == pytest.approx(1.0)
    for i in range(5):
        assert out["shrunk_delta"].iloc[i] == pytest.approx(out["delta"].iloc[i], abs=1e-9)


def test_pure_noise_collapses_to_prior():
    # If observed spread is fully explained by sampling noise (var_d <= mean se^2),
    # tau2 clips to 0, k=0, and every shrunk delta equals the prior mean.
    out = add_shrinkage(_df("PTS", [3, 5, 4, 6, 2], [10, 10, 10, 10, 10]))
    mu = out["prior_delta"].iloc[0]
    assert out["tau2"].iloc[0] == 0.0
    assert (out["shrink_k"] == 0.0).all()
    assert out["shrunk_delta"].apply(lambda x: x == pytest.approx(mu, abs=1e-6)).all()


def test_shrinkage_is_per_stat():
    # Two stats with different spreads get different tau2 / k.
    df = pd.concat([
        _df("PTS", [0, 4, 8, 12, 16], [1, 1, 1, 1, 1]),    # big spread
        _df("STL", [0.0, 0.2, 0.4, 0.6, 0.8], [1, 1, 1, 1, 1]),  # tiny spread
    ], ignore_index=True)
    out = add_shrinkage(df)
    k_pts = out[out["stat"] == "PTS"]["shrink_k"].iloc[0]
    k_stl = out[out["stat"] == "STL"]["shrink_k"].iloc[0]
    assert k_pts > k_stl  # high-signal stat retains more of its delta


def test_shrunk_delta_between_prior_and_raw():
    out = add_shrinkage(_df("PTS", [0, 2, 4, 6, 8], [1, 1, 1, 1, 1]))
    for _, r in out.iterrows():
        lo, hi = sorted([r["prior_delta"], r["delta"]])
        assert lo - 1e-9 <= r["shrunk_delta"] <= hi + 1e-9
