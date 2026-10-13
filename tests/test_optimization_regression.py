"""
Regresiones de los optimizadores con datos sintéticos determinísticos (sin red).

optimize_max_sharpe lanzaba NameError en cada llamada porque su log usaba
variables de optimize_max_information_ratio (benchmark_annual, cov_diff).
"""

import logging

import numpy as np
import pandas as pd

from ff5_portfolio import optimize_max_information_ratio, optimize_max_sharpe

TICKERS = ["A", "B", "C", "D"]


def _synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(20261013)
    dates = pd.date_range("2021-01-31", periods=36, freq=pd.offsets.MonthEnd())
    data = rng.normal(loc=[0.010, 0.008, 0.012, 0.006], scale=[0.04, 0.03, 0.05, 0.02], size=(36, 4))
    return pd.DataFrame(data, index=dates, columns=TICKERS)


def test_optimize_max_sharpe_runs_and_logs_sharpe_metrics(caplog):
    returns = _synthetic_returns()
    mu = returns.mean() * 12
    cov = returns.cov() * 12

    with caplog.at_level(logging.DEBUG):
        weights = optimize_max_sharpe(mu, cov, rf=0.02, w_min=0.0, w_max=0.6)

    assert weights.shape == (len(TICKERS),)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= -1e-8) and np.all(weights <= 0.6 + 1e-8)
    assert "Optimización Sharpe completada" in caplog.text


def test_optimize_max_information_ratio_runs():
    returns = _synthetic_returns()
    benchmark = pd.Series(
        np.random.default_rng(7).normal(0.007, 0.03, len(returns)), index=returns.index
    )
    mu = returns.mean() * 12

    weights = optimize_max_information_ratio(returns, benchmark, mu, rf=0.02, w_min=0.0, w_max=0.6)

    assert weights.shape == (len(TICKERS),)
    assert np.isclose(weights.sum(), 1.0)
