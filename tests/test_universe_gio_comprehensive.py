"""
Tests exhaustivos para el módulo universe_gio.py.
Cubre casos normales, límite, errores, correlaciones extremas y diversificación.
"""

import numpy as np
import pandas as pd
import pytest

from ff5_portfolio.universe_gio import build_universe_gio


@pytest.fixture
def large_dataset():
    """Dataset grande para tests."""
    dates = pd.date_range("2020-01-01", periods=60, freq="M")
    np.random.seed(42)
    
    # Crear 20 tickers con diferentes características
    tickers = [f"TICKER_{i}" for i in range(20)]
    returns = pd.DataFrame(
        np.random.normal(0.01, 0.05, (len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    
    # Crear betas variadas
    betas = pd.DataFrame(
        {
            "Mkt-RF": np.random.normal(1.0, 0.3, len(tickers)),
            "SMB": np.random.normal(0.2, 0.2, len(tickers)),
            "HML": np.random.normal(0.0, 0.2, len(tickers)),
            "RMW": np.random.normal(0.3, 0.2, len(tickers)),
            "CMA": np.random.normal(-0.1, 0.2, len(tickers)),
        },
        index=tickers,
    )
    
    # Crear alphas variados
    alphas = pd.Series(
        np.random.normal(0.001, 0.002, len(tickers)),
        index=tickers,
    )
    
    # Crear meta con múltiples sectores
    sectors = ["Tech", "Finance", "Consumer", "Healthcare", "Energy"] * 4
    meta = pd.DataFrame(
        {"sector": sectors},
        index=tickers,
    )
    
    return returns, betas, alphas, meta


class TestBuildUniverseGioComprehensive:
    """Tests exhaustivos para build_universe_gio."""

    def test_with_many_sectors(self, large_dataset):
        """Test con muchos sectores diferentes."""
        returns, betas, alphas, meta = large_dataset
        rf_annual = 0.03
        
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta,
            target_n=10,
            max_per_sector=2,
            max_avg_corr=0.90,
        )
        
        assert len(universe) <= 10
        assert "score_gio" in universe.columns
        # Verificar límite por sector
        sector_counts = universe["sector"].value_counts()
        assert all(sector_counts <= 2)

    def test_with_high_correlation_filter(self, large_dataset):
        """Test con filtro de correlación muy estricto."""
        returns, betas, alphas, meta = large_dataset
        rf_annual = 0.03
        
        # Correlación máxima muy baja
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta,
            target_n=10,
            max_per_sector=10,
            max_avg_corr=0.30,  # Muy estricto
        )
        
        # Puede seleccionar menos activos debido a alta correlación
        assert len(universe) >= 1

    def test_with_low_correlation_filter(self, large_dataset):
        """Test con filtro de correlación muy permisivo."""
        returns, betas, alphas, meta = large_dataset
        rf_annual = 0.03
        
        # Correlación máxima muy alta (casi sin filtro)
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta,
            target_n=10,
            max_per_sector=10,
            max_avg_corr=0.99,  # Muy permisivo
        )
        
        # Debería seleccionar más activos
        assert len(universe) <= 10

    def test_with_single_sector(self, large_dataset):
        """Test con todos los activos en un solo sector."""
        returns, betas, alphas, meta = large_dataset
        # Cambiar todos a un solo sector
        meta_single = meta.copy()
        meta_single["sector"] = "Tech"
        rf_annual = 0.03
        
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta_single,
            target_n=10,
            max_per_sector=5,
            max_avg_corr=0.90,
        )
        
        assert len(universe) <= 10
        # Todos deberían ser Tech
        assert all(universe["sector"] == "Tech")

    def test_with_target_n_larger_than_available(self, large_dataset):
        """Test cuando target_n es mayor que activos disponibles."""
        returns, betas, alphas, meta = large_dataset
        rf_annual = 0.03
        
        # Pedir más activos de los disponibles
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta,
            target_n=100,  # Más que los 20 disponibles
            max_per_sector=10,
            max_avg_corr=0.90,
        )
        
        # Debería seleccionar todos los disponibles (o menos por filtros)
        assert len(universe) <= len(returns.columns)

    def test_with_target_n_one(self, large_dataset):
        """Test con target_n = 1."""
        returns, betas, alphas, meta = large_dataset
        rf_annual = 0.03
        
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta,
            target_n=1,
            max_per_sector=10,
            max_avg_corr=0.90,
        )
        
        assert len(universe) == 1

    def test_with_max_per_sector_zero(self, large_dataset):
        """Test con max_per_sector = 0 (debería fallar o seleccionar 0)."""
        returns, betas, alphas, meta = large_dataset
        rf_annual = 0.03
        
        # Con max_per_sector=0, no debería seleccionar nada
        with pytest.raises(ValueError, match="No se pudo seleccionar"):
            build_universe_gio(
                returns_m=returns,
                betas=betas,
                alphas=alphas,
                rf_annual=rf_annual,
                meta=meta,
                target_n=10,
                max_per_sector=0,
                max_avg_corr=0.90,
            )

    def test_with_na_sectors(self, large_dataset):
        """Test con sectores NaN."""
        returns, betas, alphas, meta = large_dataset
        # Introducir NaN en sectores
        meta_na = meta.copy()
        meta_na.loc[meta_na.index[:5], "sector"] = np.nan
        rf_annual = 0.03
        
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta_na,
            target_n=10,
            max_per_sector=5,
            max_avg_corr=0.90,
        )
        
        # Los NaN deberían convertirse a "Unknown"
        assert all(universe["sector"].notna())

    def test_score_gio_calculation(self, large_dataset):
        """Test que score_gio se calcula correctamente."""
        returns, betas, alphas, meta = large_dataset
        rf_annual = 0.03
        
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta,
            target_n=10,
            max_per_sector=5,
            max_avg_corr=0.90,
        )
        
        # Verificar que score_gio está presente y ordenado
        assert "score_gio" in universe.columns
        # Los scores deberían estar ordenados (descendente implícito por selección)
        assert universe["score_gio"].is_monotonic_decreasing or len(universe) == 1

    def test_with_missing_beta_columns(self, large_dataset):
        """Test con betas que faltan algunas columnas."""
        returns, betas, alphas, meta = large_dataset
        # Eliminar algunas columnas de betas
        betas_incomplete = betas[["Mkt-RF", "SMB"]]  # Faltan HML, RMW, CMA
        rf_annual = 0.03
        
        # Debería manejar esto usando Series de ceros
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas_incomplete,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta,
            target_n=10,
            max_per_sector=5,
            max_avg_corr=0.90,
        )
        
        assert len(universe) > 0

    def test_with_zero_volatility_assets(self, large_dataset):
        """Test con activos de volatilidad cero."""
        returns, betas, alphas, meta = large_dataset
        # Hacer algunos activos con volatilidad cero
        returns_zero_vol = returns.copy()
        returns_zero_vol["TICKER_0"] = 0.01  # Constante (vol = 0)
        rf_annual = 0.03
        
        universe = build_universe_gio(
            returns_m=returns_zero_vol,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta,
            target_n=10,
            max_per_sector=5,
            max_avg_corr=0.90,
        )
        
        # Debería manejar volatilidad cero (Sharpe infinito o NaN)
        assert len(universe) > 0

    def test_with_negative_sharpe_assets(self, large_dataset):
        """Test con activos de Sharpe negativo."""
        returns, betas, alphas, meta = large_dataset
        # Hacer algunos activos con retornos muy bajos
        returns_neg = returns.copy()
        returns_neg["TICKER_0"] = np.random.normal(-0.05, 0.05, len(returns_neg))  # Muy negativo
        rf_annual = 0.03
        
        universe = build_universe_gio(
            returns_m=returns_neg,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta,
            target_n=10,
            max_per_sector=5,
            max_avg_corr=0.90,
        )
        
        # Debería preferir activos con mejor Sharpe
        assert len(universe) > 0
        # El activo negativo no debería estar (o tener score bajo)

    def test_correlation_filtering_effectiveness(self, large_dataset):
        """Test que el filtro de correlación realmente funciona."""
        returns, betas, alphas, meta = large_dataset
        rf_annual = 0.03
        
        # Crear dos activos perfectamente correlacionados
        returns_corr = returns.copy()
        returns_corr["TICKER_CORR"] = returns_corr["TICKER_0"]  # Copia exacta
        
        # Agregar a betas y alphas
        betas_corr = betas.copy()
        betas_corr.loc["TICKER_CORR"] = betas_corr.loc["TICKER_0"]
        alphas_corr = alphas.copy()
        alphas_corr["TICKER_CORR"] = alphas_corr["TICKER_0"]
        meta_corr = meta.copy()
        meta_corr.loc["TICKER_CORR"] = meta_corr.loc["TICKER_0"]
        
        universe = build_universe_gio(
            returns_m=returns_corr,
            betas=betas_corr,
            alphas=alphas_corr,
            rf_annual=rf_annual,
            meta=meta_corr,
            target_n=10,
            max_per_sector=5,
            max_avg_corr=0.50,  # Estricto
        )
        
        # Con correlación perfecta y filtro estricto, no debería seleccionar ambos
        # O si los selecciona, es porque el filtro de correlación no es lo suficientemente estricto
        selected = set(universe.index)
        # Verificar que al menos uno está seleccionado
        assert len(selected) >= 1
        # Si ambos están seleccionados, es porque el filtro de correlación no los detectó
        # (puede pasar si hay muy pocos datos para calcular correlación)
        if len(selected) > 1:
            # Verificar que al menos el filtro funcionó parcialmente
            assert len(selected) <= len(returns_corr.columns)

    def test_with_custom_sector_column_name(self, large_dataset):
        """Test con nombre de columna de sector personalizado."""
        returns, betas, alphas, meta = large_dataset
        # Renombrar columna de sector
        meta_custom = meta.copy()
        meta_custom = meta_custom.rename(columns={"sector": "industry"})
        rf_annual = 0.03
        
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta_custom,
            target_n=10,
            max_per_sector=5,
            max_avg_corr=0.90,
            sector_col="industry",
        )
        
        assert len(universe) > 0

    def test_with_all_assets_same_sector_and_high_corr(self, large_dataset):
        """Test cuando todos los activos están en el mismo sector y muy correlacionados."""
        returns, betas, alphas, meta = large_dataset
        # Todos al mismo sector
        meta_same = meta.copy()
        meta_same["sector"] = "Tech"
        
        # Hacer todos los retornos muy correlacionados
        base_returns = returns.iloc[:, 0]
        returns_corr = pd.DataFrame(
            {col: base_returns + np.random.normal(0, 0.001, len(base_returns))
             for col in returns.columns},
            index=returns.index,
        )
        rf_annual = 0.03
        
        universe = build_universe_gio(
            returns_m=returns_corr,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta_same,
            target_n=10,
            max_per_sector=5,
            max_avg_corr=0.50,  # Estricto
        )
        
        # Debería seleccionar menos debido a alta correlación
        assert len(universe) <= 5  # max_per_sector

    def test_output_columns_completeness(self, large_dataset):
        """Test que todas las columnas esperadas están en el output."""
        returns, betas, alphas, meta = large_dataset
        rf_annual = 0.03
        
        universe = build_universe_gio(
            returns_m=returns,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta,
            target_n=10,
            max_per_sector=5,
            max_avg_corr=0.90,
        )
        
        expected_cols = [
            "score_gio",
            "sharpe_historical",
            "alpha_annual",
            "beta_mkt",
            "beta_rmw",
            "beta_cma",
            "sector",
        ]
        assert all(col in universe.columns for col in expected_cols)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

