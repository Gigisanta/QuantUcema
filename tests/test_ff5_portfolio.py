"""
Tests unitarios para el módulo ff5_portfolio.
"""

import numpy as np
import pandas as pd
import pytest

from ff5_portfolio import (
    estimate_ff5_betas,
    build_expected_returns_ff5,
    build_universe_gio,
    optimize_max_sharpe,
    optimize_max_information_ratio,
    build_gio_portfolios,
)


@pytest.fixture
def sample_ff5_factors():
    """Factores FF5 de ejemplo (24 meses)."""
    dates = pd.date_range("2020-01-01", periods=24, freq="M")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Mkt-RF": np.random.normal(0.01, 0.05, 24),
            "SMB": np.random.normal(0.002, 0.03, 24),
            "HML": np.random.normal(0.001, 0.04, 24),
            "RMW": np.random.normal(0.003, 0.02, 24),
            "CMA": np.random.normal(-0.001, 0.025, 24),
            "RF": np.random.normal(0.002, 0.001, 24),
        },
        index=dates,
    )


@pytest.fixture
def sample_returns(sample_ff5_factors):
    """Retornos mensuales de ejemplo (5 tickers, 24 meses)."""
    dates = sample_ff5_factors.index
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    returns = pd.DataFrame(
        np.random.normal(0.01, 0.05, (24, 5)),
        index=dates,
        columns=tickers,
    )
    return returns


@pytest.fixture
def sample_meta():
    """Metadatos de ejemplo."""
    return pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Tech", "Consumer", "Tech"],
        },
        index=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    )


@pytest.fixture
def sample_betas():
    """Betas FF5 de ejemplo."""
    return pd.DataFrame(
        {
            "Mkt-RF": [1.2, 1.0, 1.1, 0.9, 1.5],
            "SMB": [0.3, 0.2, 0.4, 0.1, 0.5],
            "HML": [-0.1, 0.0, -0.2, 0.2, -0.3],
            "RMW": [0.4, 0.3, 0.5, 0.2, 0.6],
            "CMA": [-0.2, -0.1, -0.3, 0.1, -0.4],
        },
        index=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    )


@pytest.fixture
def sample_alphas():
    """Alphas mensuales de ejemplo."""
    return pd.Series(
        [0.001, 0.002, 0.0015, 0.0005, 0.003],
        index=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    )


class TestEstimateFF5Betas:
    """Tests para estimate_ff5_betas."""

    def test_basic_functionality(self, sample_returns, sample_ff5_factors):
        """Test básico de estimación de betas."""
        betas, alphas, rf_annual = estimate_ff5_betas(
            sample_returns, sample_ff5_factors
        )

        assert isinstance(betas, pd.DataFrame)
        assert isinstance(alphas, pd.Series)
        assert isinstance(rf_annual, float)
        assert len(betas) > 0
        assert len(alphas) > 0
        assert rf_annual > 0

        # Verificar columnas de betas
        expected_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        assert all(col in betas.columns for col in expected_cols)

    def test_missing_columns(self, sample_returns):
        """Test que falla correctamente con columnas faltantes."""
        bad_factors = pd.DataFrame({"Mkt-RF": [0.01] * 24})
        with pytest.raises(ValueError, match="Faltan columnas requeridas"):
            estimate_ff5_betas(sample_returns, bad_factors)

    def test_no_common_dates(self, sample_returns):
        """Test que falla correctamente sin fechas en común."""
        dates = pd.date_range("2025-01-01", periods=24, freq="M")
        bad_factors = pd.DataFrame(
            {
                "Mkt-RF": [0.01] * 24,
                "SMB": [0.002] * 24,
                "HML": [0.001] * 24,
                "RMW": [0.003] * 24,
                "CMA": [-0.001] * 24,
                "RF": [0.002] * 24,
            },
            index=dates,
        )
        with pytest.raises(ValueError, match="No hay fechas en común"):
            estimate_ff5_betas(sample_returns, bad_factors)


class TestBuildExpectedReturnsFF5:
    """Tests para build_expected_returns_ff5."""

    def test_basic_functionality(
        self, sample_betas, sample_ff5_factors, sample_alphas
    ):
        """Test básico de construcción de retornos esperados."""
        rf_annual = 0.03
        exp_returns = build_expected_returns_ff5(
            sample_betas, sample_ff5_factors, rf_annual
        )

        assert isinstance(exp_returns, pd.Series)
        assert len(exp_returns) == len(sample_betas)
        assert all(exp_returns.index == sample_betas.index)
        # Retornos esperados deberían ser anuales y razonables
        assert all(exp_returns.abs() < 2.0)  # Valores razonables

    def test_missing_factors(self, sample_betas):
        """Test que falla con factores faltantes."""
        bad_factors = pd.DataFrame({"Mkt-RF": [0.01] * 24})
        with pytest.raises(ValueError, match="no contiene todas las columnas"):
            build_expected_returns_ff5(sample_betas, bad_factors, 0.03)


class TestBuildUniverseGio:
    """Tests para build_universe_gio."""

    def test_basic_functionality(
        self,
        sample_returns,
        sample_betas,
        sample_alphas,
        sample_meta,
    ):
        """Test básico de construcción de universo Gio."""
        rf_annual = 0.03
        universe = build_universe_gio(
            returns_m=sample_returns,
            betas=sample_betas,
            alphas=sample_alphas,
            rf_annual=rf_annual,
            meta=sample_meta,
            target_n=3,
            max_per_sector=2,
            max_avg_corr=0.90,
        )

        assert isinstance(universe, pd.DataFrame)
        assert len(universe) <= 3
        assert "score_gio" in universe.columns
        assert "sharpe_historical" in universe.columns
        assert "alpha_annual" in universe.columns
        assert "sector" in universe.columns

    def test_sector_filtering(
        self,
        sample_returns,
        sample_betas,
        sample_alphas,
        sample_meta,
    ):
        """Test que el filtro por sector funciona."""
        rf_annual = 0.03
        universe = build_universe_gio(
            returns_m=sample_returns,
            betas=sample_betas,
            alphas=sample_alphas,
            rf_annual=rf_annual,
            meta=sample_meta,
            target_n=10,
            max_per_sector=1,
            max_avg_corr=0.90,
        )

        # Verificar que no hay más de 1 activo por sector
        sector_counts = universe["sector"].value_counts()
        assert all(sector_counts <= 1)

    def test_missing_sector_column(
        self,
        sample_returns,
        sample_betas,
        sample_alphas,
    ):
        """Test que falla con columna de sector faltante."""
        bad_meta = pd.DataFrame({"other_col": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="no contiene la columna"):
            build_universe_gio(
                returns_m=sample_returns,
                betas=sample_betas,
                alphas=sample_alphas,
                rf_annual=0.03,
                meta=bad_meta,
                target_n=3,
                max_per_sector=2,
                max_avg_corr=0.80,
            )


class TestOptimizeMaxSharpe:
    """Tests para optimize_max_sharpe."""

    def test_basic_functionality(self):
        """Test básico de optimización de máximo Sharpe."""
        n = 5
        np.random.seed(42)
        mu = pd.Series(np.random.normal(0.10, 0.05, n), index=[f"A{i}" for i in range(n)])
        cov = pd.DataFrame(
            np.random.rand(n, n), index=mu.index, columns=mu.index
        )
        cov = cov @ cov.T  # Hacer positiva semidefinida
        rf = 0.03

        weights = optimize_max_sharpe(mu, cov, rf, w_min=0.0, w_max=0.25)

        assert isinstance(weights, np.ndarray)
        assert len(weights) == n
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)
        assert all(weights >= -1e-6)  # Long-only aproximadamente
        assert all(weights <= 0.25 + 1e-6)  # Respetar límite máximo

    def test_size_mismatch(self):
        """Test que falla con tamaños inconsistentes cuando no hay índices comunes."""
        mu = pd.Series([0.10, 0.12, 0.08], index=["X", "Y", "Z"])
        cov = pd.DataFrame(np.eye(5), index=["A", "B", "C", "D", "E"], columns=["A", "B", "C", "D", "E"])
        # Cuando no hay índices comunes, debería fallar
        with pytest.raises(ValueError, match="No hay índices comunes"):
            optimize_max_sharpe(mu, cov, 0.03)


class TestOptimizeMaxInformationRatio:
    """Tests para optimize_max_information_ratio."""

    def test_basic_functionality(self, sample_returns):
        """Test básico de optimización de máximo IR."""
        dates = sample_returns.index
        benchmark = pd.Series(
            np.random.normal(0.01, 0.04, len(dates)), index=dates
        )
        mu_expected = pd.Series(
            [0.12, 0.10, 0.11, 0.09, 0.13],
            index=sample_returns.columns,
        )
        rf = 0.03

        weights = optimize_max_information_ratio(
            sample_returns, benchmark, mu_expected, rf, w_min=0.0, w_max=0.25
        )

        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_no_common_dates(self, sample_returns):
        """Test que falla sin fechas en común."""
        dates = pd.date_range("2025-01-01", periods=24, freq="M")
        benchmark = pd.Series([0.01] * 24, index=dates)
        mu_expected = pd.Series([0.10] * 5, index=sample_returns.columns)
        with pytest.raises(ValueError, match="No hay fechas en común"):
            optimize_max_information_ratio(
                sample_returns, benchmark, mu_expected, 0.03
            )


class TestBuildGioPortfolios:
    """Tests para build_gio_portfolios (pipeline completo)."""

    def test_basic_functionality(self, sample_returns, sample_ff5_factors, sample_meta):
        """Test básico del pipeline completo."""
        result = build_gio_portfolios(
            returns_m=sample_returns,
            ff5_factors=sample_ff5_factors,
            meta=sample_meta,
            target_n_universe=3,
            max_per_sector=2,
            max_avg_corr=0.90,
            w_max=0.40,  # Permitir suficiente peso para 3 activos
        )

        assert isinstance(result, dict)
        assert "betas" in result
        assert "alphas" in result
        assert "rf_annual" in result
        assert "exp_ret_annual" in result
        assert "universe_gio" in result
        assert "weights_sharpe" in result
        assert result["weights_ir"] is None  # Sin benchmark

    def test_with_benchmark(self, sample_returns, sample_ff5_factors, sample_meta):
        """Test del pipeline con benchmark (IR)."""
        benchmark = pd.Series(
            np.random.normal(0.01, 0.04, len(sample_returns.index)),
            index=sample_returns.index,
        )

        result = build_gio_portfolios(
            returns_m=sample_returns,
            ff5_factors=sample_ff5_factors,
            meta=sample_meta,
            benchmark_returns=benchmark,
            target_n_universe=3,
            max_per_sector=2,
            max_avg_corr=0.90,
            w_max=0.40,  # Permitir suficiente peso para 3 activos
        )

        assert result["weights_ir"] is not None
        assert isinstance(result["weights_ir"], pd.Series)

    def test_without_meta(self, sample_returns, sample_ff5_factors):
        """Test del pipeline sin meta (usa dummy)."""
        result = build_gio_portfolios(
            returns_m=sample_returns,
            ff5_factors=sample_ff5_factors,
            meta=None,
            target_n_universe=3,
            max_per_sector=2,
            max_avg_corr=0.90,
            w_max=0.40,  # Permitir suficiente peso para 3 activos
        )

        assert result["universe_gio"] is not None
        # Todos deberían estar en sector "Unknown"
        assert all(result["universe_gio"]["sector"] == "Unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

