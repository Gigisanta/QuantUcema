"""
Tests para el módulo data_loader.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock

import ff5_portfolio.data_loader as data_loader_module
from ff5_portfolio.data_loader import (
    download_stock_data,
    download_ff5_factors,
    get_stock_metadata,
    _normalize_tickers,
)


class TestNormalizeTickers:
    """Tests para _normalize_tickers."""

    def test_single_ticker(self):
        """Test normalización de un solo ticker."""
        result = _normalize_tickers("AAPL")
        assert result == ["AAPL"]

    def test_multiple_tickers(self):
        """Test normalización de múltiples tickers."""
        result = _normalize_tickers(["AAPL", "MSFT", "GOOGL"])
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_ticker_with_dot(self):
        """Test normalización de ticker con punto (BRK.B -> BRK-B)."""
        result = _normalize_tickers("BRK.B")
        assert result == ["BRK-B"]

    def test_string_comma_separated(self):
        """Test normalización de string separado por comas."""
        result = _normalize_tickers("AAPL, MSFT, GOOGL")
        assert len(result) == 3
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result

    def test_duplicates_removed(self):
        """Test que se eliminan duplicados."""
        result = _normalize_tickers(["AAPL", "AAPL", "MSFT"])
        assert result == ["AAPL", "MSFT"]


class TestDownloadStockData:
    """Tests para download_stock_data."""

    @patch("ff5_portfolio.data_loader._validate_tickers", return_value=(["AAPL", "MSFT"], []))
    @patch("ff5_portfolio.data_loader.yf.download")
    def test_basic_download(self, mock_download, mock_validate):
        """Test descarga básica de datos."""
        dates = pd.date_range("2020-01-01", periods=30, freq="ME")
        mock_prices = pd.DataFrame(
            {
                ("Adj Close", "AAPL"): np.random.rand(len(dates)),
                ("Adj Close", "MSFT"): np.random.rand(len(dates)),
            },
            index=dates,
        )
        mock_prices.columns = pd.MultiIndex.from_tuples(mock_prices.columns)
        mock_download.return_value = mock_prices

        returns, skipped = download_stock_data(
            ["AAPL", "MSFT"], "2020-01-01", "2022-01-01", min_months=12
        )

        assert isinstance(returns, pd.DataFrame)
        assert "AAPL" in returns.columns
        assert "MSFT" in returns.columns
        assert len(skipped) == 0

    @patch("ff5_portfolio.data_loader._validate_tickers", return_value=(["AAPL"], []))
    @patch("ff5_portfolio.data_loader.yf.download")
    def test_local_cache_reuses_returns(self, mock_download, mock_validate, monkeypatch, tmp_path):
        """Test que el cache local evita descargas repetidas."""
        # Configurar mock de descarga
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        mock_prices = pd.DataFrame(
            {("Adj Close", "AAPL"): np.random.rand(len(dates))}, index=dates
        )
        mock_prices.columns = pd.MultiIndex.from_tuples(mock_prices.columns)
        mock_download.return_value = mock_prices

        # Configurar directorio de cache temporal
        monkeypatch.setattr(data_loader_module, "RETURNS_CACHE_DIR", tmp_path)

        # Primera llamada (debería descargar)
        download_stock_data(["AAPL"], "2020-01-01", "2022-12-31", use_local_cache=True)
        assert mock_download.call_count == 1

        # Segunda llamada (debería usar cache)
        download_stock_data(["AAPL"], "2020-01-01", "2022-12-31", use_local_cache=True)
        assert mock_download.call_count == 1

    def test_empty_tickers(self):
        """Test que falla con tickers vacíos."""
        with pytest.raises(ValueError, match="No se proporcionaron tickers"):
            download_stock_data([], "2020-01-01", "2022-01-01")

    @patch("ff5_portfolio.data_loader.yf.download")
    @patch("ff5_portfolio.data_loader.sleep")
    def test_all_failed_download(self, mock_sleep, mock_download):
        """Test cuando todas las descargas fallan."""
        mock_download.return_value = pd.DataFrame() # Simula una descarga vacía

        # Ahora la función debería lanzar un error porque los tickers son inválidos antes de descargar
        with pytest.raises(ValueError, match="Todos los tickers proporcionados son inválidos o delisted."):
            download_stock_data(["INVALID"], "2020-01-01", "2022-01-01")



class TestDownloadFF5Factors:
    """Tests para download_ff5_factors."""

    @patch("ff5_portfolio.data_loader.requests.get")
    @patch("ff5_portfolio.data_loader.ZipFile")
    def test_basic_download(self, mock_zipfile, mock_get):
        """Test descarga básica de factores FF5."""
        # Mock de respuesta HTTP
        mock_response = Mock()
        mock_response.content = b"fake zip content"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Mock de ZIP file
        mock_zip = MagicMock()
        mock_zip.__enter__ = Mock(return_value=mock_zip)
        mock_zip.__exit__ = Mock(return_value=None)
        mock_zip.namelist.return_value = ["F-F_Research_Data_5_Factors_2x3.csv"]

        # Mock de contenido CSV como BytesIO
        from io import BytesIO
        csv_content = """This line is skipped
This line is skipped
This line is skipped
202001,0.5,0.1,0.2,0.3,0.4,0.01
202002,0.6,0.2,0.3,0.4,0.5,0.02
"""
        csv_bytes = BytesIO(csv_content.encode())
        mock_zip.open.return_value = csv_bytes

        mock_zipfile.return_value = mock_zip

        factors = download_ff5_factors()

        assert isinstance(factors, pd.DataFrame)
        assert "Mkt-RF" in factors.columns
        assert "SMB" in factors.columns
        assert "HML" in factors.columns
        assert "RMW" in factors.columns
        assert "CMA" in factors.columns
        assert "RF" in factors.columns

    @patch("ff5_portfolio.data_loader.requests.get")
    def test_network_error(self, mock_get):
        """Test manejo de error de red."""
        import requests
        mock_get.side_effect = requests.RequestException("Network error")

        with pytest.raises(RuntimeError, match="Error al descargar"):
            download_ff5_factors()


class TestGetStockMetadata:
    """Tests para get_stock_metadata."""

    @patch("ff5_portfolio.data_loader.yf.Tickers")
    def test_basic_metadata(self, mock_tickers_class):
        """Test obtención básica de metadatos."""
        # Mock para yf.Tickers(...).tickers
        mock_ticker_info = Mock()
        mock_ticker_info.info = {"sector": "Technology"}
        mock_tickers_instance = Mock()
        mock_tickers_instance.tickers = [mock_ticker_info]
        mock_tickers_class.return_value = mock_tickers_instance

        meta = get_stock_metadata(["AAPL"])

        assert isinstance(meta, pd.DataFrame)
        assert "sector" in meta.columns
        assert meta.loc["AAPL", "sector"] == "Technology"

    @patch("ff5_portfolio.data_loader.yf.Ticker")
    def test_multiple_tickers(self, mock_ticker_class):
        """Test con múltiples tickers."""
        mock_ticker = Mock()
        mock_ticker.info = {"sector": "Technology"}
        mock_ticker_class.return_value = mock_ticker

        meta = get_stock_metadata(["AAPL", "MSFT"])

        assert len(meta) == 2
        assert "AAPL" in meta.index
        assert "MSFT" in meta.index

    @patch("ff5_portfolio.data_loader.yf.Ticker")
    def test_missing_sector(self, mock_ticker_class):
        """Test cuando el sector no está disponible."""
        mock_ticker = Mock()
        mock_ticker.info = {}  # Sin sector
        mock_ticker_class.return_value = mock_ticker

        meta = get_stock_metadata(["UNKNOWN"])

        assert meta.loc["UNKNOWN", "sector"] == "Unknown"

    @patch("ff5_portfolio.data_loader.yf.Ticker")
    def test_ticker_error(self, mock_ticker_class):
        """Test cuando yfinance falla."""
        mock_ticker_class.side_effect = Exception("Ticker error")

        meta = get_stock_metadata(["ERROR"])

        assert meta.loc["ERROR", "sector"] == "Unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

