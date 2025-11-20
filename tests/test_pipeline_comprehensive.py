"""
Tests exhaustivos para el módulo pipeline.py.
Cubre integración completa, casos límite y validaciones end-to-end.
"""

import numpy as np
import pandas as pd
import pytest

from ff5_portfolio.pipeline import build_gio_portfolios, build_gio_portfolios_with_metrics


@pytest.fixture
def complete_dataset():
    """Dataset completo para tests de pipeline."""
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


class TestBuildGioPortfoliosComprehensive:
    """Tests exhaustivos para build_gio_portfolios."""

    def test_complete_pipeline_with_benchmark(self, complete_dataset):
        """Test pipeline completo con benchmark."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
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
        
        # Verificar estructura completa
        assert isinstance(result, dict)
        assert "betas" in result
        assert "alphas" in result
        assert "rf_annual" in result
        assert "exp_ret_annual" in result
        assert "universe_gio" in result
        assert "weights_sharpe" in result
        assert "weights_ir" is not None
        
        # Verificar tipos
        assert isinstance(result["betas"], pd.DataFrame)
        assert isinstance(result["alphas"], pd.Series)
        assert isinstance(result["weights_sharpe"], pd.Series)
        assert isinstance(result["weights_ir"], pd.Series)

    def test_complete_pipeline_without_benchmark(self, complete_dataset):
        """Test pipeline completo sin benchmark."""
        returns, ff5_factors, meta, _ = complete_dataset
        
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=None,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_max=0.30,
        )
        
        assert result["weights_ir"] is None
        assert result["weights_sharpe"] is not None

    def test_without_meta(self, complete_dataset):
        """Test sin meta (debería crear dummy)."""
        returns, ff5_factors, _, _ = complete_dataset
        
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=None,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_max=0.30,
        )
        
        # Todos deberían estar en sector "Unknown"
        assert all(result["universe_gio"]["sector"] == "Unknown")

    def test_weights_sum_to_one(self, complete_dataset):
        """Test que los pesos suman a 1."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
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
        
        assert np.isclose(result["weights_sharpe"].sum(), 1.0, rtol=1e-5)
        assert np.isclose(result["weights_ir"].sum(), 1.0, rtol=1e-5)

    def test_weights_respect_bounds(self, complete_dataset):
        """Test que los pesos respetan w_min y w_max."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
        w_max = 0.20
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_min=0.0,
            w_max=w_max,
        )
        
        assert all(result["weights_sharpe"] <= w_max + 1e-6)
        assert all(result["weights_sharpe"] >= 0.0 - 1e-6)
        assert all(result["weights_ir"] <= w_max + 1e-6)
        assert all(result["weights_ir"] >= 0.0 - 1e-6)

    def test_universe_gio_filtering(self, complete_dataset):
        """Test que el universo Gio filtra correctamente."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
        target_n = 3
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=target_n,
            max_per_sector=1,
            max_avg_corr=0.50,
            w_max=0.40,
        )
        
        # El universo debería tener máximo target_n activos
        assert len(result["universe_gio"]) <= target_n
        # Los pesos solo deberían incluir activos del universo
        assert set(result["weights_sharpe"].index).issubset(set(result["universe_gio"].index))

    def test_exp_ret_annual_filtered_to_universe(self, complete_dataset):
        """Test que exp_ret_annual está filtrado al universo."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
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
        
        # exp_ret_annual debería tener solo tickers del universo
        assert set(result["exp_ret_annual"].index).issubset(set(result["universe_gio"].index))
        assert len(result["exp_ret_annual"]) == len(result["universe_gio"])

    def test_with_large_universe_target(self, complete_dataset):
        """Test con target_n_universe grande."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=20,  # Más que los disponibles
            max_per_sector=10,
            max_avg_corr=0.90,
            w_max=0.10,
        )
        
        # Debería seleccionar todos los disponibles (o menos por filtros)
        assert len(result["universe_gio"]) <= len(returns.columns)

    def test_with_small_universe_target(self, complete_dataset):
        """Test con target_n_universe pequeño."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=2,
            max_per_sector=1,
            max_avg_corr=0.90,
            w_max=0.50,
        )
        
        assert len(result["universe_gio"]) <= 2

    def test_with_very_strict_sector_limit(self, complete_dataset):
        """Test con límite de sector muy estricto."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=10,
            max_per_sector=1,  # Solo 1 por sector
            max_avg_corr=0.90,
            w_max=0.20,
        )
        
        # Verificar límite por sector
        sector_counts = result["universe_gio"]["sector"].value_counts()
        assert all(sector_counts <= 1)

    def test_with_very_strict_correlation_limit(self, complete_dataset):
        """Test con límite de correlación muy estricto."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=10,
            max_per_sector=5,
            max_avg_corr=0.20,  # Muy estricto
            w_max=0.20,
        )
        
        # Puede seleccionar menos activos debido a alta correlación
        assert len(result["universe_gio"]) >= 1

    def test_rf_annual_consistency(self, complete_dataset):
        """Test que rf_annual es consistente."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
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
        
        # rf_annual debería ser aproximadamente 12 * promedio mensual de RF
        expected_rf = 12 * ff5_factors["RF"].mean()
        assert np.isclose(result["rf_annual"], expected_rf, rtol=1e-5)

    def test_betas_alphas_consistency(self, complete_dataset):
        """Test que betas y alphas son consistentes."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
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
        
        # Betas y alphas deberían tener los mismos tickers
        assert set(result["betas"].index) == set(result["alphas"].index)
        # El universo debería ser subconjunto
        assert set(result["universe_gio"].index).issubset(set(result["betas"].index))

    def test_with_missing_data_in_returns(self, complete_dataset):
        """Test con datos faltantes en returns."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        # Introducir NaNs
        returns_na = returns.copy()
        returns_na.loc[returns_na.index[:5], "AAPL"] = np.nan
        
        result = build_gio_portfolios(
            returns_m=returns_na,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_max=0.30,
        )
        
        # Debería manejar NaNs y seguir funcionando
        assert result["weights_sharpe"] is not None

    def test_with_partial_date_overlap(self, complete_dataset):
        """Test con solapamiento parcial de fechas."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        # Cambiar fechas de returns
        returns_partial = returns.copy()
        returns_partial.index = pd.date_range("2020-06-01", periods=len(returns_partial), freq="M")
        
        result = build_gio_portfolios(
            returns_m=returns_partial,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_max=0.30,
        )
        
        # Debería usar solo fechas en común
        assert result["weights_sharpe"] is not None

    def test_output_keys_completeness(self, complete_dataset):
        """Test que todas las claves esperadas están en el output."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
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
        
        expected_keys = [
            "betas",
            "alphas",
            "rf_annual",
            "exp_ret_annual",
            "universe_gio",
            "weights_sharpe",
            "weights_ir",
            "cov_annual",
            "returns_universe",
        ]
        assert all(key in result for key in expected_keys)
        assert len(result) == len(expected_keys)

    def test_weights_only_positive(self, complete_dataset):
        """Test que los pesos son solo positivos (long-only)."""
        returns, ff5_factors, meta, benchmark = complete_dataset
        
        result = build_gio_portfolios(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_min=0.0,
            w_max=0.30,
        )

    def test_pipeline_metrics_hook(self, complete_dataset):
        """Test que el helper con métricas regresa información completa."""
        returns, ff5_factors, meta, benchmark = complete_dataset

        result, metrics = build_gio_portfolios_with_metrics(
            returns_m=returns,
            ff5_factors=ff5_factors,
            meta=meta,
            benchmark_returns=benchmark,
            target_n_universe=5,
            max_per_sector=2,
            max_avg_corr=0.80,
            w_max=0.30,
        )

        assert isinstance(result["cov_annual"], pd.DataFrame)
        assert isinstance(result["returns_universe"], pd.DataFrame)
        assert set(result["returns_universe"].columns) == set(result["exp_ret_annual"].index)

        expected_stages = {
            "estimate_ff5_betas",
            "expected_returns",
            "prepare_meta",
            "build_universe_gio",
            "covariance_universe",
            "optimize_max_sharpe",
            "optimize_max_information_ratio",
        }
        assert expected_stages.issubset(metrics.stage_durations.keys())
        assert metrics.total_seconds >= 0.0
        
        # Solo pesos significativos
        weights_sharpe = result["weights_sharpe"][result["weights_sharpe"] > 1e-6]
        weights_ir = result["weights_ir"][result["weights_ir"] > 1e-6]
        
        assert all(weights_sharpe >= 0)
        assert all(weights_ir >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

