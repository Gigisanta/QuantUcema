"""
Tests exhaustivos para el módulo factors.py.
Cubre casos normales, límite, errores y validaciones.
"""

import numpy as np
import pandas as pd
import pytest

from ff5_portfolio.factors import (
    estimate_ff5_betas,
    build_expected_returns_ff5,
    REQUIRED_FF5_COLUMNS,
    MIN_OBSERVATIONS,
)


@pytest.fixture
def large_ff5_factors():
    """Factores FF5 con muchos meses (60 meses)."""
    dates = pd.date_range("2020-01-01", periods=60, freq="M")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Mkt-RF": np.random.normal(0.01, 0.05, 60),
            "SMB": np.random.normal(0.002, 0.03, 60),
            "HML": np.random.normal(0.001, 0.04, 60),
            "RMW": np.random.normal(0.003, 0.02, 60),
            "CMA": np.random.normal(-0.001, 0.025, 60),
            "RF": np.random.normal(0.002, 0.001, 60),
        },
        index=dates,
    )


@pytest.fixture
def large_returns(large_ff5_factors):
    """Retornos mensuales con muchos tickers y meses."""
    dates = large_ff5_factors.index
    np.random.seed(42)
    tickers = [f"TICKER_{i}" for i in range(20)]
    returns = pd.DataFrame(
        np.random.normal(0.01, 0.05, (len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    return returns


class TestEstimateFF5BetasComprehensive:
    """Tests exhaustivos para estimate_ff5_betas."""

    def test_with_many_tickers(self, large_returns, large_ff5_factors):
        """Test con muchos tickers."""
        betas, alphas, rf_annual = estimate_ff5_betas(
            large_returns, large_ff5_factors
        )
        
        assert len(betas) == len(large_returns.columns)
        assert len(alphas) == len(large_returns.columns)
        assert rf_annual > 0
        assert rf_annual < 1.0  # Razonable

    def test_with_missing_data_in_returns(self, large_ff5_factors):
        """Test con datos faltantes en returns."""
        dates = large_ff5_factors.index
        returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.01, 0.05, len(dates)),
                "MSFT": np.random.normal(0.01, 0.05, len(dates)),
            },
            index=dates,
        )
        # Introducir NaNs
        returns.loc[returns.index[:5], "AAPL"] = np.nan
        returns.loc[returns.index[10:15], "MSFT"] = np.nan
        
        betas, alphas, rf_annual = estimate_ff5_betas(returns, large_ff5_factors)
        
        # Debería manejar NaNs correctamente
        assert "AAPL" in betas.index or len(betas) > 0
        assert rf_annual > 0

    def test_with_insufficient_observations(self, large_ff5_factors):
        """Test con activos que tienen muy pocas observaciones."""
        dates = large_ff5_factors.index
        returns = pd.DataFrame(
            {
                "GOOD": np.random.normal(0.01, 0.05, len(dates)),
                "BAD": np.nan,  # Todo NaN
            },
            index=dates,
        )
        returns.loc[returns.index[:5], "BAD"] = np.random.normal(0.01, 0.05, 5)
        
        betas, alphas, rf_annual = estimate_ff5_betas(returns, large_ff5_factors)
        
        # BAD debería ser filtrado por tener < MIN_OBSERVATIONS
        assert "GOOD" in betas.index
        assert "BAD" not in betas.index

    def test_rf_annual_calculation(self, large_returns, large_ff5_factors):
        """Test que rf_annual se calcula correctamente."""
        betas, alphas, rf_annual = estimate_ff5_betas(
            large_returns, large_ff5_factors
        )
        
        # rf_annual debería ser aproximadamente 12 * promedio mensual de RF
        expected_rf = 12 * large_ff5_factors["RF"].mean()
        assert np.isclose(rf_annual, expected_rf, rtol=1e-5)

    def test_betas_structure(self, large_returns, large_ff5_factors):
        """Test estructura de betas."""
        betas, alphas, rf_annual = estimate_ff5_betas(
            large_returns, large_ff5_factors
        )
        
        # Verificar columnas
        expected_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        assert all(col in betas.columns for col in expected_cols)
        assert len(betas.columns) == len(expected_cols)
        
        # Verificar que los valores son razonables
        assert betas.abs().max().max() < 10  # Betas extremos pero posibles

    def test_alphas_structure(self, large_returns, large_ff5_factors):
        """Test estructura de alphas."""
        betas, alphas, rf_annual = estimate_ff5_betas(
            large_returns, large_ff5_factors
        )
        
        # Alphas deberían ser mensuales y razonables
        assert len(alphas) == len(betas)
        assert all(alphas.index == betas.index)
        assert alphas.abs().max() < 0.5  # Alpha mensual razonable

    def test_empty_returns(self, large_ff5_factors):
        """Test con DataFrame de returns vacío."""
        empty_returns = pd.DataFrame()
        
        # Puede fallar por fechas en común o por no tener activos
        with pytest.raises(ValueError):
            estimate_ff5_betas(empty_returns, large_ff5_factors)

    def test_returns_with_all_nan_columns(self, large_ff5_factors):
        """Test con columnas completamente NaN."""
        dates = large_ff5_factors.index
        returns = pd.DataFrame(
            {
                "ALL_NAN": np.nan,
            },
            index=dates,
        )
        
        with pytest.raises(ValueError, match="al menos"):
            estimate_ff5_betas(returns, large_ff5_factors)

    def test_factors_with_nan_values(self, large_returns):
        """Test con factores que tienen NaN."""
        dates = large_returns.index
        factors = pd.DataFrame(
            {
                "Mkt-RF": np.random.normal(0.01, 0.05, len(dates)),
                "SMB": np.random.normal(0.002, 0.03, len(dates)),
                "HML": np.random.normal(0.001, 0.04, len(dates)),
                "RMW": np.random.normal(0.003, 0.02, len(dates)),
                "CMA": np.random.normal(-0.001, 0.025, len(dates)),
                "RF": np.random.normal(0.002, 0.001, len(dates)),
            },
            index=dates,
        )
        # Introducir algunos NaN
        factors.loc[factors.index[:3], "Mkt-RF"] = np.nan
        
        betas, alphas, rf_annual = estimate_ff5_betas(large_returns, factors)
        
        # Debería manejar NaNs correctamente
        assert len(betas) > 0

    def test_case_insensitive_columns(self, large_returns):
        """Test que las columnas no son case-sensitive (debería fallar si lo son)."""
        dates = large_returns.index
        # Crear factores con columnas en mayúsculas (debería fallar)
        factors = pd.DataFrame(
            {
                "MKT-RF": np.random.normal(0.01, 0.05, len(dates)),
                "SMB": np.random.normal(0.002, 0.03, len(dates)),
                "HML": np.random.normal(0.001, 0.04, len(dates)),
                "RMW": np.random.normal(0.003, 0.02, len(dates)),
                "CMA": np.random.normal(-0.001, 0.025, len(dates)),
                "RF": np.random.normal(0.002, 0.001, len(dates)),
            },
            index=dates,
        )
        
        with pytest.raises(ValueError, match="Faltan columnas"):
            estimate_ff5_betas(large_returns, factors)

    def test_partial_date_overlap(self, large_ff5_factors):
        """Test con solapamiento parcial de fechas."""
        dates_returns = pd.date_range("2020-06-01", periods=30, freq="M")
        returns = pd.DataFrame(
            np.random.normal(0.01, 0.05, (30, 5)),
            index=dates_returns,
            columns=["A", "B", "C", "D", "E"],
        )
        
        # Solo debería usar fechas en común
        betas, alphas, rf_annual = estimate_ff5_betas(returns, large_ff5_factors)
        
        assert len(betas) > 0

    def test_single_ticker(self, large_ff5_factors):
        """Test con un solo ticker."""
        dates = large_ff5_factors.index
        returns = pd.DataFrame(
            {"SINGLE": np.random.normal(0.01, 0.05, len(dates))},
            index=dates,
        )
        
        betas, alphas, rf_annual = estimate_ff5_betas(returns, large_ff5_factors)
        
        assert len(betas) == 1
        assert "SINGLE" in betas.index


class TestBuildExpectedReturnsFF5Comprehensive:
    """Tests exhaustivos para build_expected_returns_ff5."""

    def test_with_different_beta_values(self, large_ff5_factors):
        """Test con diferentes valores de beta."""
        betas = pd.DataFrame(
            {
                "Mkt-RF": [0.5, 1.0, 1.5, 2.0],
                "SMB": [0.1, 0.2, 0.3, 0.4],
                "HML": [-0.1, 0.0, 0.1, 0.2],
                "RMW": [0.2, 0.3, 0.4, 0.5],
                "CMA": [-0.2, -0.1, 0.0, 0.1],
            },
            index=["LOW", "MED", "HIGH", "VERY_HIGH"],
        )
        rf_annual = 0.03
        
        exp_returns = build_expected_returns_ff5(betas, large_ff5_factors, rf_annual)
        
        assert len(exp_returns) == len(betas)
        assert all(exp_returns.index == betas.index)
        
        # Retornos esperados deberían ser positivos para betas positivos
        # (depende de los factores, pero generalmente debería ser así)

    def test_with_negative_betas(self, large_ff5_factors):
        """Test con betas negativos."""
        betas = pd.DataFrame(
            {
                "Mkt-RF": [-1.0, -0.5, 0.0],
                "SMB": [0.0, 0.0, 0.0],
                "HML": [0.0, 0.0, 0.0],
                "RMW": [0.0, 0.0, 0.0],
                "CMA": [0.0, 0.0, 0.0],
            },
            index=["NEG1", "NEG2", "ZERO"],
        )
        rf_annual = 0.03
        
        exp_returns = build_expected_returns_ff5(betas, large_ff5_factors, rf_annual)
        
        assert len(exp_returns) == 3
        # Retornos pueden ser negativos con betas negativos

    def test_with_zero_rf(self, large_ff5_factors):
        """Test con rf_annual = 0."""
        betas = pd.DataFrame(
            {
                "Mkt-RF": [1.0, 1.2],
                "SMB": [0.2, 0.3],
                "HML": [0.0, 0.1],
                "RMW": [0.3, 0.4],
                "CMA": [-0.1, -0.2],
            },
            index=["A", "B"],
        )
        rf_annual = 0.0
        
        exp_returns = build_expected_returns_ff5(betas, large_ff5_factors, rf_annual)
        
        assert len(exp_returns) == 2

    def test_with_high_rf(self, large_ff5_factors):
        """Test con rf_annual alto."""
        betas = pd.DataFrame(
            {
                "Mkt-RF": [1.0],
                "SMB": [0.2],
                "HML": [0.0],
                "RMW": [0.3],
                "CMA": [-0.1],
            },
            index=["A"],
        )
        rf_annual = 0.10  # 10%
        
        exp_returns = build_expected_returns_ff5(betas, large_ff5_factors, rf_annual)
        
        assert len(exp_returns) == 1
        assert exp_returns.iloc[0] > rf_annual  # Debería ser mayor que RF

    def test_betas_missing_columns(self, large_ff5_factors):
        """Test con betas que faltan columnas."""
        betas = pd.DataFrame(
            {
                "Mkt-RF": [1.0],
                # Faltan otras columnas
            },
            index=["A"],
        )
        
        with pytest.raises(ValueError, match="no contiene todas las columnas"):
            build_expected_returns_ff5(betas, large_ff5_factors, 0.03)

    def test_factors_missing_columns(self):
        """Test con factores que faltan columnas."""
        dates = pd.date_range("2020-01-01", periods=24, freq="M")
        betas = pd.DataFrame(
            {
                "Mkt-RF": [1.0],
                "SMB": [0.2],
                "HML": [0.0],
                "RMW": [0.3],
                "CMA": [-0.1],
            },
            index=["A"],
        )
        factors = pd.DataFrame(
            {
                "Mkt-RF": np.random.normal(0.01, 0.05, 24),
                # Faltan otras columnas
            },
            index=dates,
        )
        
        with pytest.raises(ValueError, match="no contiene todas las columnas"):
            build_expected_returns_ff5(betas, factors, 0.03)

    def test_empty_betas(self, large_ff5_factors):
        """Test con betas vacío."""
        betas = pd.DataFrame()
        
        with pytest.raises(ValueError, match="no contiene todas las columnas"):
            build_expected_returns_ff5(betas, large_ff5_factors, 0.03)

    def test_consistency_with_ff5_model(self, large_ff5_factors):
        """Test que los retornos esperados son consistentes con el modelo FF5."""
        # Crear betas conocidas
        betas = pd.DataFrame(
            {
                "Mkt-RF": [1.0],
                "SMB": [0.0],
                "HML": [0.0],
                "RMW": [0.0],
                "CMA": [0.0],
            },
            index=["MARKET"],
        )
        rf_annual = 0.03
        
        exp_returns = build_expected_returns_ff5(betas, large_ff5_factors, rf_annual)
        
        # Para beta_mkt=1 y otros=0, el retorno esperado debería ser
        # rf + 1.0 * E[Mkt-RF]_annual
        expected = rf_annual + (large_ff5_factors["Mkt-RF"].mean() * 12)
        assert np.isclose(exp_returns.iloc[0], expected, rtol=1e-5)

    def test_with_single_factor_nonzero(self, large_ff5_factors):
        """Test con un solo factor distinto de cero."""
        betas = pd.DataFrame(
            {
                "Mkt-RF": [0.0],
                "SMB": [1.0],  # Solo SMB
                "HML": [0.0],
                "RMW": [0.0],
                "CMA": [0.0],
            },
            index=["SMB_ONLY"],
        )
        rf_annual = 0.03
        
        exp_returns = build_expected_returns_ff5(betas, large_ff5_factors, rf_annual)
        
        expected = rf_annual + (large_ff5_factors["SMB"].mean() * 12)
        assert np.isclose(exp_returns.iloc[0], expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

