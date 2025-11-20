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
    """Tests de integración completa con datos reales."""

    @pytest.fixture(scope="class")
    def real_data(self):
        """Descarga un pequeño set de datos reales para tests."""
        tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "V", "WMT"]
        start_date = "2020-01-01"
        end_date = "2023-12-31"

        returns_m, _ = download_stock_data(tickers, start_date, end_date, validate_tickers=False)
        ff5_factors = download_ff5_factors()
        meta = get_stock_metadata(tickers)
        
        common_dates = returns_m.index.intersection(ff5_factors.index)
        returns_m = returns_m.loc[common_dates]
        ff5_factors = ff5_factors.loc[common_dates]
        
        return returns_m, ff5_factors, meta

    def test_full_pipeline_consistency(self, real_data):
        """Test que el pipeline completo produce resultados consistentes con datos reales."""
        returns, ff5_factors, meta = real_data
        
        result = build_gio_portfolios(
            returns_m=returns, ff5_factors=ff5_factors, meta=meta,
            target_n_universe=3, max_per_sector=2, w_max=0.50,
        )
        
        assert not result["universe_gio"].empty
        assert not result["weights_sharpe"].empty
        assert np.isclose(result["weights_sharpe"].sum(), 1.0, rtol=1e-5)



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

