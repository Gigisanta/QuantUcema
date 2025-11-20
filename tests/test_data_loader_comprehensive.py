"""
Tests exhaustivos adicionales para el módulo data_loader.py.
Cubre casos límite, errores de red, formatos de datos, etc.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from io import BytesIO

from ff5_portfolio.data_loader import (
    download_stock_data,
    download_ff5_factors,
    get_stock_metadata,
    _normalize_tickers,
    _coerce_to_prices,
)


class TestNormalizeTickersComprehensive:
    """Tests exhaustivos para _normalize_tickers."""

    def test_empty_string(self):
        """Test con string vacío."""
        result = _normalize_tickers("")
        assert result == []

    def test_whitespace_only(self):
        """Test con solo espacios."""
        result = _normalize_tickers("   ,  ,  ")
        assert result == []

    def test_mixed_case(self):
        """Test con mayúsculas y minúsculas mezcladas."""
        result = _normalize_tickers("aapl, MSFT, GoOgL")
        assert all(t.isupper() for t in result)

    def test_special_characters(self):
        """Test con caracteres especiales."""
        result = _normalize_tickers("BRK.B, BF-B, TEST-1")
        assert "BRK-B" in result
        assert "BF-B" in result
        assert "TEST-1" in result

    def test_numbers_in_ticker(self):
        """Test con números en tickers."""
        result = _normalize_tickers("A1, B2, C3")
        assert "A1" in result
        assert "B2" in result
        assert "C3" in result

    def test_very_long_ticker_list(self):
        """Test con lista muy larga de tickers."""
        tickers = [f"TICKER_{i}" for i in range(100)]
        result = _normalize_tickers(tickers)
        assert len(result) == 100

    def test_none_values_filtered(self):
        """Test que None se filtra."""
        result = _normalize_tickers(["AAPL", None, "MSFT", ""])
        assert "AAPL" in result
        assert "MSFT" in result
        assert None not in result


class TestCoerceToPricesComprehensive:
    """Tests exhaustivos para _coerce_to_prices."""

    def test_empty_dataframe(self):
        """Test con DataFrame vacío."""
        result = _coerce_to_prices(pd.DataFrame())
        assert result.empty

    def test_none_input(self):
        """Test con None."""
        result = _coerce_to_prices(None)
        assert result.empty

    def test_single_column_dataframe(self):
        """Test con DataFrame de una columna."""
        df = pd.DataFrame({"Close": [100, 105, 110]})
        result = _coerce_to_prices(df)
        assert not result.empty
        assert "Close" in result.columns or len(result.columns) > 0

    def test_multiindex_with_adj_close(self):
        """Test con MultiIndex y Adj Close."""
        df = pd.DataFrame(
            {
                ("Adj Close", "AAPL"): [100, 105],
                ("Close", "AAPL"): [99, 104],
            }
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        result = _coerce_to_prices(df)
        assert not result.empty

    def test_multiindex_without_adj_close(self):
        """Test con MultiIndex sin Adj Close."""
        df = pd.DataFrame(
            {
                ("Close", "AAPL"): [100, 105],
                ("Open", "AAPL"): [99, 104],
            }
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        result = _coerce_to_prices(df)
        assert not result.empty

    def test_series_input(self):
        """Test con Series."""
        series = pd.Series([100, 105, 110], name="AAPL")
        # _coerce_to_prices maneja Series, pero necesita verificar antes de acceder a .columns
        # Este test verifica que la función maneja Series correctamente
        result = _coerce_to_prices(series, ["AAPL"])
        # Debería convertir Series a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestDownloadStockDataComprehensive:
    """Tests exhaustivos adicionales para download_stock_data."""

    @patch("ff5_portfolio.data_loader.yf.download")
    @patch("ff5_portfolio.data_loader.sleep")
    def test_with_different_batch_sizes(self, mock_sleep, mock_download):
        """Test con diferentes tamaños de batch."""
        dates = pd.date_range("2020-01-01", periods=30, freq="ME")
        np.random.seed(42)
        prices = 100 * (1 + np.random.normal(0.01, 0.05, len(dates))).cumprod()
        
        mock_data = pd.DataFrame(
            {("Adj Close", "AAPL"): prices},
            index=dates,
        )
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        mock_download.return_value = mock_data
        
        for batch_size in [1, 5, 10, 50]:
            returns, _ = download_stock_data(
                ["AAPL"], "2020-01-01", "2022-01-01",
                batch_size=batch_size,
                min_months=12
            )
            assert not returns.empty

    @patch("ff5_portfolio.data_loader.yf.download")
    @patch("ff5_portfolio.data_loader.sleep")
    def test_with_high_nan_ratio(self, mock_sleep, mock_download):
        """Test con series con muchos NaN."""
        dates = pd.date_range("2020-01-01", periods=30, freq="ME")
        # Crear precios con algunos NaN pero suficientes datos válidos
        prices = 100 * (1 + np.random.normal(0.01, 0.05, len(dates))).cumprod()
        prices_series = pd.Series(prices, index=dates)
        prices_series.loc[dates[:10]] = np.nan  # 10/30 = 33% NaN (bajo threshold)
        
        mock_data = pd.DataFrame(
            {("Adj Close", "AAPL"): prices_series},
            index=dates,
        )
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        mock_download.return_value = mock_data
        
        # Debería pasar el filtro de NaN (max_nan_frac = 0.90) y tener suficientes meses
        returns, _ = download_stock_data(
            ["AAPL"], "2020-01-01", "2022-01-01",
            min_months=12
        )
        # Debería tener datos válidos
        assert isinstance(returns, pd.DataFrame)
        assert len(returns.columns) > 0

    def test_invalid_date_range(self):
        """Test con rango de fechas inválido."""
        with pytest.raises((ValueError, Exception)):
            download_stock_data(
                ["AAPL"], "2025-01-01", "2020-01-01",  # Fecha fin antes de inicio
                min_months=12
            )

    @patch("ff5_portfolio.data_loader.yf.download")
    @patch("ff5_portfolio.data_loader.sleep")
    def test_with_all_columns_empty(self, mock_sleep, mock_download):
        """Test cuando todas las columnas están vacías."""
        mock_data = pd.DataFrame()
        mock_download.return_value = mock_data
        
        with pytest.raises(ValueError, match="No se obtuvieron datos"):
            download_stock_data(["INVALID"], "2020-01-01", "2022-01-01")


class TestDownloadFF5FactorsComprehensive:
    """Tests exhaustivos adicionales para download_ff5_factors."""

    @patch("ff5_portfolio.data_loader.requests.get")
    @patch("ff5_portfolio.data_loader.ZipFile")
    def test_with_different_csv_formats(self, mock_zipfile, mock_get):
        """Test con diferentes formatos de CSV."""
        mock_response = Mock()
        mock_response.content = b"fake zip"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        mock_zip = MagicMock()
        mock_zip.__enter__ = Mock(return_value=mock_zip)
        mock_zip.__exit__ = Mock(return_value=None)
        mock_zip.namelist.return_value = ["F-F_Research_Data_5_Factors_2x3.csv"]
        
        # CSV con diferentes formatos de fecha
        csv_content = """Skip
Skip
Skip
202001,0.5,0.1,0.2,0.3,0.4,0.01
202002,0.6,0.2,0.3,0.4,0.5,0.02
202003,0.7,0.3,0.4,0.5,0.6,0.03
"""
        csv_bytes = BytesIO(csv_content.encode())
        mock_zip.open.return_value = csv_bytes
        mock_zipfile.return_value = mock_zip
        
        factors = download_ff5_factors()
        
        assert len(factors) >= 2  # Al menos 2 filas de datos
        assert all(factors.columns == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])

    @patch("ff5_portfolio.data_loader.requests.get")
    def test_timeout_error(self, mock_get):
        """Test con error de timeout."""
        import requests
        mock_get.side_effect = requests.Timeout("Timeout error")
        
        with pytest.raises(RuntimeError, match="Error al descargar"):
            download_ff5_factors()

    @patch("ff5_portfolio.data_loader.requests.get")
    @patch("ff5_portfolio.data_loader.ZipFile")
    def test_invalid_zip_content(self, mock_zipfile, mock_get):
        """Test con contenido ZIP inválido."""
        mock_response = Mock()
        mock_response.content = b"not a zip"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        mock_zipfile.side_effect = Exception("Invalid ZIP")
        
        with pytest.raises(RuntimeError, match="Error al parsear"):
            download_ff5_factors()

    @patch("ff5_portfolio.data_loader.requests.get")
    @patch("ff5_portfolio.data_loader.ZipFile")
    def test_csv_with_no_data_rows(self, mock_zipfile, mock_get):
        """Test con CSV sin filas de datos."""
        mock_response = Mock()
        mock_response.content = b"fake zip"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        mock_zip = MagicMock()
        mock_zip.__enter__ = Mock(return_value=mock_zip)
        mock_zip.__exit__ = Mock(return_value=None)
        mock_zip.namelist.return_value = ["F-F_Research_Data_5_Factors_2x3.csv"]
        
        # CSV solo con headers, sin datos
        csv_content = """Skip
Skip
Skip
"""
        csv_bytes = BytesIO(csv_content.encode())
        mock_zip.open.return_value = csv_bytes
        mock_zipfile.return_value = mock_zip
        
        # CSV sin datos debería causar error al parsear
        with pytest.raises((RuntimeError, ValueError, pd.errors.EmptyDataError)):
            download_ff5_factors()


class TestGetStockMetadataComprehensive:
    """Tests exhaustivos adicionales para get_stock_metadata."""

    @patch("ff5_portfolio.data_loader.yf.Ticker")
    def test_with_none_sector(self, mock_ticker_class):
        """Test cuando sector es None."""
        mock_ticker = Mock()
        mock_ticker.info = {"sector": None}
        mock_ticker_class.return_value = mock_ticker
        
        meta = get_stock_metadata(["AAPL"])
        
        assert meta.loc["AAPL", "sector"] == "Unknown"

    @patch("ff5_portfolio.data_loader.yf.Ticker")
    def test_with_empty_info_dict(self, mock_ticker_class):
        """Test con info vacío."""
        mock_ticker = Mock()
        mock_ticker.info = {}
        mock_ticker_class.return_value = mock_ticker
        
        meta = get_stock_metadata(["AAPL"])
        
        assert meta.loc["AAPL", "sector"] == "Unknown"

    @patch("ff5_portfolio.data_loader.yf.Ticker")
    def test_with_mixed_success_failure(self, mock_ticker_class):
        """Test cuando algunos tickers fallan y otros no."""
        def side_effect(ticker):
            mock = Mock()
            if ticker == "AAPL":
                mock.info = {"sector": "Technology"}
            else:
                raise Exception("Ticker error")
            return mock
        
        mock_ticker_class.side_effect = side_effect
        
        meta = get_stock_metadata(["AAPL", "INVALID"])
        
        assert meta.loc["AAPL", "sector"] == "Technology"
        assert meta.loc["INVALID", "sector"] == "Unknown"

    @patch("ff5_portfolio.data_loader.yf.Ticker")
    def test_with_very_long_ticker_list(self, mock_ticker_class):
        """Test con lista muy larga de tickers."""
        mock_ticker = Mock()
        mock_ticker.info = {"sector": "Technology"}
        mock_ticker_class.return_value = mock_ticker
        
        tickers = [f"TICKER_{i}" for i in range(50)]
        meta = get_stock_metadata(tickers)
        
        assert len(meta) == 50
        assert all(meta["sector"] == "Technology")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

