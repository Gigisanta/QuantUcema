"""
Tests exhaustivos para el módulo optimization.py.
Cubre casos normales, límite, errores, matrices singulares y estabilidad numérica.
"""

import numpy as np
import pandas as pd
import pytest

from ff5_portfolio.optimization import (
    optimize_max_sharpe,
    optimize_max_information_ratio,
)


class TestOptimizeMaxSharpeComprehensive:
    """Tests exhaustivos para optimize_max_sharpe."""

    def test_with_single_asset(self):
        """Test con un solo activo."""
        mu = pd.Series([0.10], index=["A"])
        cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        rf = 0.03
        
        weights = optimize_max_sharpe(mu, cov, rf)
        
        assert len(weights) == 1
        assert np.isclose(weights[0], 1.0, rtol=1e-5)

    def test_with_two_assets(self):
        """Test con dos activos."""
        mu = pd.Series([0.10, 0.12], index=["A", "B"])
        cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.05]], index=["A", "B"], columns=["A", "B"]
        )
        rf = 0.03
        
        weights = optimize_max_sharpe(mu, cov, rf)
        
        assert len(weights) == 2
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)
        assert all(weights >= -1e-6)

    def test_with_high_correlation(self):
        """Test con activos altamente correlacionados."""
        n = 5
        np.random.seed(42)
        mu = pd.Series(np.random.normal(0.10, 0.02, n), index=[f"A{i}" for i in range(n)])
        
        # Crear matriz de covarianza con alta correlación
        base_cov = np.eye(n) * 0.04
        high_corr = 0.95
        for i in range(n):
            for j in range(n):
                if i != j:
                    base_cov[i, j] = high_corr * np.sqrt(base_cov[i, i] * base_cov[j, j])
        
        cov = pd.DataFrame(base_cov, index=mu.index, columns=mu.index)
        rf = 0.03
        
        weights = optimize_max_sharpe(mu, cov, rf, w_max=0.25)
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)
        assert all(weights >= -1e-6)

    def test_with_low_correlation(self):
        """Test con activos poco correlacionados."""
        n = 5
        np.random.seed(42)
        mu = pd.Series(np.random.normal(0.10, 0.02, n), index=[f"A{i}" for i in range(n)])
        
        # Matriz diagonal (sin correlación)
        cov = pd.DataFrame(
            np.eye(n) * 0.04, index=mu.index, columns=mu.index
        )
        rf = 0.03
        
        weights = optimize_max_sharpe(mu, cov, rf, w_max=0.25)
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_negative_returns(self):
        """Test con algunos retornos esperados negativos."""
        mu = pd.Series([0.10, -0.05, 0.08], index=["A", "B", "C"])
        cov = pd.DataFrame(
            np.eye(3) * 0.04, index=mu.index, columns=mu.index
        )
        rf = 0.03
        
        weights = optimize_max_sharpe(mu, cov, rf, w_max=0.40)
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)
        # Debería dar poco peso a B (retorno negativo)

    def test_with_very_small_w_max(self):
        """Test con w_max muy pequeño."""
        n = 10
        np.random.seed(42)
        mu = pd.Series(np.random.normal(0.10, 0.02, n), index=[f"A{i}" for i in range(n)])
        cov = pd.DataFrame(
            np.eye(n) * 0.04, index=mu.index, columns=mu.index
        )
        rf = 0.03
        
        # w_max muy pequeño debería ajustarse automáticamente
        weights = optimize_max_sharpe(mu, cov, rf, w_max=0.05)
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)
        assert all(weights <= 0.10 + 1e-6)  # Ajustado automáticamente

    def test_with_w_min_positive(self):
        """Test con w_min > 0."""
        mu = pd.Series([0.10, 0.12, 0.08], index=["A", "B", "C"])
        cov = pd.DataFrame(
            np.eye(3) * 0.04, index=mu.index, columns=mu.index
        )
        rf = 0.03
        
        # Con w_min > 0, puede no ser factible
        # Probamos con w_min pequeño
        weights = optimize_max_sharpe(mu, cov, rf, w_min=0.05, w_max=0.50)
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)
        assert all(weights >= 0.05 - 1e-6)

    def test_with_numpy_arrays(self):
        """Test con arrays numpy en lugar de Series/DataFrame."""
        n = 5
        np.random.seed(42)
        mu = np.random.normal(0.10, 0.02, n)
        cov = np.eye(n) * 0.04
        rf = 0.03
        
        weights = optimize_max_sharpe(mu, cov, rf)
        
        assert len(weights) == n
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_singular_covariance(self):
        """Test con matriz de covarianza singular (o casi singular)."""
        mu = pd.Series([0.10, 0.12, 0.11], index=["A", "B", "C"])
        
        # Matriz singular (dos activos idénticos)
        cov = pd.DataFrame(
            [[0.04, 0.04, 0.04],
             [0.04, 0.04, 0.04],
             [0.04, 0.04, 0.04]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        rf = 0.03
        
        # Debería manejar esto o fallar con un error claro
        try:
            weights = optimize_max_sharpe(mu, cov, rf, w_max=0.50)
            # Si no falla, verificar que funciona
            assert np.isclose(weights.sum(), 1.0, rtol=1e-5)
        except RuntimeError:
            # Es aceptable que falle con matriz singular
            pass

    def test_with_very_large_covariance(self):
        """Test con valores de covarianza muy grandes."""
        mu = pd.Series([0.10, 0.12], index=["A", "B"])
        cov = pd.DataFrame(
            [[100.0, 50.0], [50.0, 100.0]], index=["A", "B"], columns=["A", "B"]
        )
        rf = 0.03
        
        weights = optimize_max_sharpe(mu, cov, rf, w_max=0.50)
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_zero_volatility_asset(self):
        """Test con un activo de volatilidad cero."""
        mu = pd.Series([0.10, 0.05], index=["A", "B"])
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.0]], index=["A", "B"], columns=["A", "B"]
        )
        rf = 0.03
        
        # Activo B tiene volatilidad cero
        weights = optimize_max_sharpe(mu, cov, rf, w_max=0.50)
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_consistency_with_equal_returns(self):
        """Test con retornos esperados iguales."""
        mu = pd.Series([0.10, 0.10, 0.10], index=["A", "B", "C"])
        cov = pd.DataFrame(
            np.eye(3) * 0.04, index=mu.index, columns=mu.index
        )
        rf = 0.03
        
        weights = optimize_max_sharpe(mu, cov, rf, w_max=0.50)
        
        # Con retornos iguales, debería preferir menor volatilidad
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_many_assets(self):
        """Test con muchos activos."""
        n = 50
        np.random.seed(42)
        mu = pd.Series(np.random.normal(0.10, 0.02, n), index=[f"A{i}" for i in range(n)])
        
        # Crear matriz de covarianza positiva semidefinida
        base = np.random.rand(n, n)
        cov_matrix = base @ base.T * 0.01
        cov = pd.DataFrame(cov_matrix, index=mu.index, columns=mu.index)
        rf = 0.03
        
        weights = optimize_max_sharpe(mu, cov, rf, w_max=0.10)
        
        assert len(weights) == n
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)
        assert all(weights <= 0.10 + 1e-6)


class TestOptimizeMaxInformationRatioComprehensive:
    """Tests exhaustivos para optimize_max_information_ratio."""

    @pytest.fixture
    def sample_returns(self):
        """Retornos mensuales de ejemplo."""
        dates = pd.date_range("2020-01-01", periods=36, freq="M")
        np.random.seed(42)
        return pd.DataFrame(
            np.random.normal(0.01, 0.05, (36, 5)),
            index=dates,
            columns=["A", "B", "C", "D", "E"],
        )

    @pytest.fixture
    def sample_benchmark(self, sample_returns):
        """Benchmark de ejemplo."""
        np.random.seed(42)
        return pd.Series(
            np.random.normal(0.01, 0.04, len(sample_returns.index)),
            index=sample_returns.index,
        )

    def test_basic_functionality(self, sample_returns, sample_benchmark):
        """Test básico."""
        mu_expected = pd.Series(
            [0.12, 0.10, 0.11, 0.09, 0.13],
            index=sample_returns.columns,
        )
        rf = 0.03
        
        weights = optimize_max_information_ratio(
            sample_returns, sample_benchmark, mu_expected, rf, w_max=0.25
        )
        
        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_benchmark_higher_than_portfolio(self, sample_returns):
        """Test cuando el benchmark tiene retorno esperado mayor."""
        benchmark = pd.Series(
            np.random.normal(0.02, 0.04, len(sample_returns.index)),
            index=sample_returns.index,
        )
        mu_expected = pd.Series(
            [0.08, 0.09, 0.07, 0.08, 0.09],  # Menores que benchmark
            index=sample_returns.columns,
        )
        rf = 0.03
        
        weights = optimize_max_information_ratio(
            sample_returns, benchmark, mu_expected, rf, w_max=0.25
        )
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_perfect_correlation_with_benchmark(self, sample_returns):
        """Test cuando los activos están perfectamente correlacionados con benchmark."""
        # Hacer que los retornos sean iguales al benchmark
        benchmark = pd.Series(
            np.random.normal(0.01, 0.04, len(sample_returns.index)),
            index=sample_returns.index,
        )
        # Crear returns con columnas iguales al benchmark
        returns = pd.DataFrame(
            {col: benchmark.values for col in sample_returns.columns},
            index=sample_returns.index,
        )
        
        mu_expected = pd.Series(
            [0.12] * len(sample_returns.columns),
            index=sample_returns.columns,
        )
        rf = 0.03
        
        # Debería manejar esto (tracking error muy bajo)
        weights = optimize_max_information_ratio(
            returns, benchmark, mu_expected, rf, w_max=0.25
        )
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_numpy_arrays(self, sample_returns, sample_benchmark):
        """Test con arrays numpy."""
        mu_expected = np.array([0.12, 0.10, 0.11, 0.09, 0.13])
        rf = 0.03
        
        weights = optimize_max_information_ratio(
            sample_returns, sample_benchmark, mu_expected, rf, w_max=0.25
        )
        
        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_partial_ticker_overlap(self, sample_returns, sample_benchmark):
        """Test cuando mu_expected tiene solo algunos tickers."""
        mu_expected = pd.Series(
            [0.12, 0.10],  # Solo A y B
            index=["A", "B"],
        )
        rf = 0.03
        
        weights = optimize_max_information_ratio(
            sample_returns, sample_benchmark, mu_expected, rf, w_max=0.50
        )
        
        assert len(weights) == 2  # Solo A y B
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_very_short_time_series(self):
        """Test con serie temporal muy corta."""
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        returns = pd.DataFrame(
            np.random.normal(0.01, 0.05, (12, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        benchmark = pd.Series(
            np.random.normal(0.01, 0.04, 12),
            index=dates,
        )
        mu_expected = pd.Series([0.12, 0.10, 0.11], index=["A", "B", "C"])
        rf = 0.03
        
        weights = optimize_max_information_ratio(
            returns, benchmark, mu_expected, rf, w_max=0.50
        )
        
        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_negative_alpha_expected(self, sample_returns, sample_benchmark):
        """Test cuando el alpha esperado es negativo."""
        # Retornos esperados menores que benchmark
        mu_expected = pd.Series(
            [0.05, 0.06, 0.05, 0.06, 0.05],  # Menores que benchmark esperado
            index=sample_returns.columns,
        )
        rf = 0.03
        
        weights = optimize_max_information_ratio(
            sample_returns, sample_benchmark, mu_expected, rf, w_max=0.25
        )
        
        # Debería optimizar (aunque el alpha sea negativo)
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)

    def test_with_zero_tracking_error_possible(self, sample_returns):
        """Test cuando es posible tracking error cero."""
        # Hacer que un activo tenga retornos idénticos al benchmark
        benchmark = pd.Series(
            np.random.normal(0.01, 0.04, len(sample_returns.index)),
            index=sample_returns.index,
        )
        returns = sample_returns.copy()
        returns["A"] = benchmark  # A es idéntico al benchmark
        
        mu_expected = pd.Series(
            [0.12, 0.10, 0.11, 0.09, 0.13],
            index=returns.columns,
        )
        rf = 0.03
        
        # Debería manejar tracking error muy bajo
        weights = optimize_max_information_ratio(
            returns, benchmark, mu_expected, rf, w_max=0.25
        )
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

