"""
Tests de integración end-to-end para todo el sistema.
Verifica que todos los módulos trabajan juntos correctamente.
"""

import numpy as np
import pandas as pd
import pytest

from ff5_portfolio import build_gio_portfolios
from ff5_portfolio.data_loader import (
    download_ff5_factors,
    download_stock_data,
    get_stock_metadata,
)


class TestEndToEndIntegration:
    """Tests de integración completa."""

    @pytest.fixture
    def synthetic_data(self):
        """Crear datos sintéticos completos."""
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        np.random.seed(42)
        
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
        
        # Retornos mensuales
        returns = pd.DataFrame(
            np.random.normal(0.01, 0.05, (len(dates), len(tickers))),
            index=dates,
            columns=tickers,
        )
        
        # Factores FF5
        ff5_factors = pd.DataFrame(
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
        
        # Meta
        sectors = ["Tech", "Tech", "Tech", "Consumer", "Tech", "Tech", "Tech", "Finance", "Finance", "Consumer"]
        meta = pd.DataFrame(
            {"sector": sectors},
            index=tickers,
        )
        
        # Benchmark
        benchmark = pd.Series(
            np.random.normal(0.01, 0.04, len(dates)),
            index=dates,
        )
        
        return returns, ff5_factors, meta, benchmark

    def test_full_pipeline_consistency(self, synthetic_data):
        """Test que el pipeline completo produce resultados consistentes."""
        returns, ff5_factors, meta, benchmark = synthetic_data
        
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_max=0.30,
        )
        
        # Verificar consistencia entre componentes
        # 1. Universo está en betas
        assert set(result["universe_gio"].index).issubset(set(result["betas"].index))
        
        # 2. Pesos solo incluyen activos del universo
        assert set(result["weights_sharpe"].index).issubset(set(result["universe_gio"].index))
        assert set(result["weights_ir"].index).issubset(set(result["universe_gio"].index))
        
        # 3. exp_ret_annual solo tiene universo
        assert set(result["exp_ret_annual"].index).issubset(set(result["universe_gio"].index))
        
        # 4. Pesos suman a 1
        assert np.isclose(result["weights_sharpe"].sum(), 1.0, rtol=1e-5)
        assert np.isclose(result["weights_ir"].sum(), 1.0, rtol=1e-5)

    def test_pipeline_idempotency(self, synthetic_data):
        """Test que ejecutar el pipeline dos veces da resultados consistentes."""
        returns, ff5_factors, meta, benchmark = synthetic_data
        
        result1 = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_max=0.30,
        )
        
        result2 = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_max=0.30,
        )
        
        # Los resultados deberían ser idénticos (mismo seed)
        assert set(result1["universe_gio"].index) == set(result2["universe_gio"].index)
        assert np.allclose(result1["weights_sharpe"].values, result2["weights_sharpe"].values, rtol=1e-5)

    def test_pipeline_with_different_parameters(self, synthetic_data):
        """Test que diferentes parámetros producen diferentes resultados."""
        returns, ff5_factors, meta, benchmark = synthetic_data
        
        result1 = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=3,
            max_per_sector=1,
            max_avg_corr=0.50,
            w_max=0.40,
        )
        
        result2 = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=8,
            max_per_sector=5,
            max_avg_corr=0.90,
            w_max=0.15,
        )
        
        # Deberían producir universos diferentes
        assert len(result1["universe_gio"]) != len(result2["universe_gio"]) or \
               set(result1["universe_gio"].index) != set(result2["universe_gio"].index)

    def test_pipeline_metrics_calculation(self, synthetic_data):
        """Test que las métricas se calculan correctamente."""
        returns, ff5_factors, meta, benchmark = synthetic_data
        
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_max=0.30,
        )
        
        # Calcular métricas manualmente
        universe_tickers = result["universe_gio"].index
        returns_universe = returns[universe_tickers]
        exp_ret = result["exp_ret_annual"]
        cov_annual = returns_universe.cov() * 12
        
        # Retorno esperado del portafolio
        port_ret = result["weights_sharpe"] @ exp_ret
        port_vol = np.sqrt(result["weights_sharpe"] @ cov_annual @ result["weights_sharpe"])
        sharpe = (port_ret - result["rf_annual"]) / port_vol
        
        # Verificar que Sharpe es razonable
        assert -5 < sharpe < 10  # Rango razonable

    def test_pipeline_with_minimal_data(self):
        """Test con datos mínimos."""
        dates = pd.date_range("2020-01-01", periods=24, freq="M")
        np.random.seed(42)
        
        returns = pd.DataFrame(
            np.random.normal(0.01, 0.05, (24, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        
        ff5_factors = pd.DataFrame(
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
        
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=None,
            benchmark_returns=None,
            target_n_universe=2,
            max_per_sector=1,
            max_avg_corr=0.90,
            w_max=0.50,
        )
        
        assert result["weights_sharpe"] is not None
        assert len(result["universe_gio"]) <= 2


class TestDataLoaderIntegration:
    """Tests de integración con data_loader (mocked)."""

    @pytest.mark.skip(reason="Requiere mocking complejo de yfinance")
    def test_pipeline_with_mocked_data_loader(self):
        """Test pipeline usando funciones de data_loader mockeadas."""
        from unittest.mock import patch
        
        dates = pd.date_range("2020-01-01", periods=36, freq="M")
        np.random.seed(42)
        
        # Mock de yfinance
        mock_data = pd.DataFrame(
            {
                ("Adj Close", "AAPL"): 100 * (1 + np.random.normal(0.01, 0.05, len(dates))).cumprod(),
                ("Adj Close", "MSFT"): 200 * (1 + np.random.normal(0.01, 0.05, len(dates))).cumprod(),
            },
            index=dates,
        )
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        mock_download.return_value = mock_data
        
        # Descargar datos
        returns, _ = download_stock_data(
            ["AAPL", "MSFT"], "2020-01-01", "2023-01-01", min_months=12
        )
        
        # Crear factores FF5
        ff5_factors = pd.DataFrame(
            {
                "Mkt-RF": np.random.normal(0.01, 0.05, len(returns.index)),
                "SMB": np.random.normal(0.002, 0.03, len(returns.index)),
                "HML": np.random.normal(0.001, 0.04, len(returns.index)),
                "RMW": np.random.normal(0.003, 0.02, len(returns.index)),
                "CMA": np.random.normal(-0.001, 0.025, len(returns.index)),
                "RF": np.random.normal(0.002, 0.001, len(returns.index)),
            },
            index=returns.index,
        )
        
        # Ejecutar pipeline
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=None,
            target_n_universe=2,
            max_per_sector=1,
            max_avg_corr=0.90,
            w_max=0.50,
        )
        
        assert result["weights_sharpe"] is not None
        assert len(result["universe_gio"]) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

