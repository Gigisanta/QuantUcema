"""
Configuración compartida para pytest.
Fixtures comunes para todos los tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def random_seed():
    """Fixture para establecer semilla aleatoria global."""
    np.random.seed(42)
    return 42


@pytest.fixture
def standard_ff5_factors():
    """Factores FF5 estándar para tests (48 meses)."""
    dates = pd.date_range("2020-01-01", periods=48, freq="M")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Mkt-RF": np.random.normal(0.01, 0.05, 48),
            "SMB": np.random.normal(0.002, 0.03, 48),
            "HML": np.random.normal(0.001, 0.04, 48),
            "RMW": np.random.normal(0.003, 0.02, 48),
            "CMA": np.random.normal(-0.001, 0.025, 48),
            "RF": np.random.normal(0.002, 0.001, 48),
        },
        index=dates,
    )


@pytest.fixture
def standard_returns(standard_ff5_factors):
    """Retornos mensuales estándar (10 tickers, 48 meses)."""
    dates = standard_ff5_factors.index
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
    returns = pd.DataFrame(
        np.random.normal(0.01, 0.05, (len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    return returns


@pytest.fixture
def standard_meta():
    """Metadatos estándar."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
    sectors = ["Tech", "Tech", "Tech", "Consumer", "Tech", "Tech", "Tech", "Finance", "Finance", "Consumer"]
    return pd.DataFrame(
        {"sector": sectors},
        index=tickers,
    )


@pytest.fixture
def standard_betas():
    """Betas FF5 estándar."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
    return pd.DataFrame(
        {
            "Mkt-RF": np.random.normal(1.0, 0.3, len(tickers)),
            "SMB": np.random.normal(0.2, 0.2, len(tickers)),
            "HML": np.random.normal(0.0, 0.2, len(tickers)),
            "RMW": np.random.normal(0.3, 0.2, len(tickers)),
            "CMA": np.random.normal(-0.1, 0.2, len(tickers)),
        },
        index=tickers,
    )


@pytest.fixture
def standard_alphas():
    """Alphas estándar."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
    return pd.Series(
        np.random.normal(0.001, 0.002, len(tickers)),
        index=tickers,
    )

