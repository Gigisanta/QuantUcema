"""
Carga de datos para la aplicación Streamlit.
"""

import streamlit as st

from ff5_portfolio.data_loader import (
    download_ff5_factors,
    download_stock_data,
    get_sp500_tickers,
    get_stock_metadata,
)
from ff5_portfolio.logging_config import get_logger

logger = get_logger(__name__)


@st.cache_data(ttl=86400)  # Cache por 24 horas
def cached_download_ff5_factors():
    """Cachea factores FF5 que cambian raramente."""
    try:
        return download_ff5_factors()
    except Exception as e:
        logger.error("Error al descargar factores FF5 del cache: %s", e)
        raise


@st.cache_data(ttl=43200)  # Cache por 12 horas
def cached_get_stock_metadata(tickers_str):
    """Cachea metadatos de tickers."""
    try:
        tickers_list = tickers_str.split(",") if isinstance(tickers_str, str) else tickers_str
        return get_stock_metadata(tickers_list)
    except Exception as e:
        logger.error("Error al descargar metadatos del cache: %s", e)
        raise


@st.cache_data(ttl=3600, max_entries=10)  # Cache por 1 hora, máximo 10 entradas
def cached_download_stock_data(tickers_str, start_date, end_date, min_months, validate_tickers):
    """Cachea descarga de datos de acciones basado en parámetros."""
    try:
        tickers_list = tickers_str.split(",") if isinstance(tickers_str, str) else tickers_str
        return download_stock_data(
            tickers_list,
            start_date,
            end_date,
            min_months=min_months,
            validate_tickers=validate_tickers
        )
    except Exception as e:
        logger.error("Error al descargar datos de acciones del cache: %s", e)
        raise


@st.cache_data(ttl=86400)
def cached_sp500_universe():
    """Cachea la lista completa del S&P 500."""
    try:
        return get_sp500_tickers(use_cache=False, cache=None)
    except Exception as e:
        logger.error("Error al descargar universo S&P 500 del cache: %s", e)
        raise


@st.cache_data(ttl=43200)  # Cache por 12 horas
def cached_sp500_universe_validated():
    """Cachea la lista del S&P 500 validada (sin tickers delisted)."""
    try:
        return get_sp500_tickers(use_cache=False, cache=None, validate_tickers=True)
    except Exception as e:
        logger.error("Error al descargar universo S&P 500 validado del cache: %s", e)
        raise


def load_all_data(
    start_date,
    end_date,
    validate_tickers: bool = True,
    include_ir: bool = True,
    progress_callback=None,
) -> dict:
    """
    Orquesta toda la carga de datos necesaria para el pipeline.
    
    Parameters
    ----------
    start_date : date
        Fecha de inicio.
    end_date : date
        Fecha de fin.
    validate_tickers : bool, default True
        Si validar tickers antes de descargar.
    include_ir : bool, default True
        Si incluir descarga de SPY para optimización IR.
    progress_callback : callable, optional
        Función callback para actualizar progreso (recibe mensaje como string).
    
    Returns
    -------
    dict
        Diccionario con keys: sp500_tickers, returns_m, ff5_factors, meta, benchmark_returns.
        benchmark_returns puede ser None si include_ir=False o si falla la descarga.
    """
    skipped_tickers = []
    
    # 0. Obtener tickers del S&P 500
    if progress_callback:
        progress_callback("📋 Obteniendo lista de tickers del S&P 500...")
    logger.info("Obteniendo tickers del S&P 500 (validación: %s)", validate_tickers)
    
    try:
        if validate_tickers:
            sp500_tickers = cached_sp500_universe_validated()
        else:
            sp500_tickers = cached_sp500_universe()
        logger.info("✓ %d tickers del S&P 500 obtenidos", len(sp500_tickers))
    except Exception as cache_error:
        logger.warning("Cache de tickers no disponible (%s). Intentando descarga directa...", cache_error)
        try:
            sp500_tickers = get_sp500_tickers(
                use_cache=True,
                cache=st.session_state,
                validate_tickers=validate_tickers,
            )
            logger.info("✓ %d tickers del S&P 500 descargados directamente", len(sp500_tickers))
        except RuntimeError as e2:
            logger.error("Error crítico al obtener tickers: %s", e2)
            raise RuntimeError(f"No se pudo obtener lista de tickers del S&P 500: {e2}")
        except Exception as e2:
            logger.error("Error inesperado al obtener tickers: %s", e2)
            raise
    
    # 1. Descargar datos de acciones
    if progress_callback:
        progress_callback(f"📥 Descargando datos de {len(sp500_tickers)} acciones del S&P 500...")
    logger.info("Iniciando descarga de datos de acciones: %d tickers", len(sp500_tickers))
    
    tickers_str = ",".join(sp500_tickers)
    returns_m, skipped = cached_download_stock_data(
        tickers_str,
        str(start_date),
        str(end_date),
        min_months=12,
        validate_tickers=validate_tickers,
    )
    skipped_tickers.extend(skipped)
    
    if len(skipped) > 0:
        logger.warning("%d tickers no se pudieron descargar", len(skipped))
    
    if returns_m.shape[1] == 0:
        logger.error("No se obtuvieron datos válidos después de descarga")
        raise ValueError(
            "No se obtuvieron datos válidos. Verificar conexión, tickers y fechas."
        )
    
    logger.info(
        "✓ %d acciones descargadas exitosamente (%d meses de datos)",
        returns_m.shape[1],
        returns_m.shape[0],
    )
    
    # 2. Descargar factores FF5
    if progress_callback:
        progress_callback("📥 Descargando factores FF5...")
    logger.info("Descargando factores FF5")
    ff5_factors = cached_download_ff5_factors()
    logger.info("✓ Factores FF5 descargados (%d meses)", len(ff5_factors))
    
    # 3. Obtener metadatos
    if progress_callback:
        progress_callback("📥 Obteniendo metadatos...")
    logger.info("Obteniendo metadatos para %d tickers", returns_m.shape[1])
    tickers_str = ",".join(returns_m.columns.tolist())
    meta = cached_get_stock_metadata(tickers_str)
    logger.info("✓ Metadatos obtenidos para %d tickers", len(meta))
    
    # 4. Descargar SPY si se necesita IR
    benchmark_returns = None
    if include_ir:
        if progress_callback:
            progress_callback("📥 Descargando datos de SPY (benchmark)...")
        logger.info("Descargando datos de SPY para optimización IR")
        spy_returns, _ = cached_download_stock_data(
            "SPY", str(start_date), str(end_date), min_months=12, validate_tickers=validate_tickers
        )
        if not spy_returns.empty:
            benchmark_returns = spy_returns.iloc[:, 0]
            logger.info("✓ Datos de SPY descargados")
        else:
            logger.warning("No se pudo descargar SPY, se omitirá optimización IR")
            include_ir = False
    
    # 5. Alinear fechas
    logger.info("Alineando fechas entre datasets")
    common_dates = returns_m.index.intersection(ff5_factors.index)
    if benchmark_returns is not None:
        common_dates = common_dates.intersection(benchmark_returns.index)
    
    if len(common_dates) < 12:
        logger.error("Período común insuficiente: %d meses (mínimo 12)", len(common_dates))
        raise ValueError(
            f"Período común insuficiente: {len(common_dates)} meses. Se requieren al menos 12 meses."
        )
    
    returns_m = returns_m.loc[common_dates]
    ff5_factors = ff5_factors.loc[common_dates]
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.loc[common_dates]
    
    logger.info(
        "✓ Datos alineados: %d meses comunes, %d activos",
        len(common_dates),
        returns_m.shape[1],
    )
    
    return {
        "sp500_tickers": sp500_tickers,
        "returns_m": returns_m,
        "ff5_factors": ff5_factors,
        "meta": meta,
        "benchmark_returns": benchmark_returns,
        "skipped_tickers": skipped_tickers,
    }

