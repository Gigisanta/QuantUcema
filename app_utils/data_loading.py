"""
Carga de datos para la aplicación Streamlit.
"""

import streamlit as st

from ff5_portfolio.data_loader import (
    _clean_and_validate_data,
    download_ff5_factors,
    download_stock_data,
    get_sp500_tickers,
    get_stock_metadata,
)
from ff5_portfolio.logging_config import get_logger

logger = get_logger(__name__)


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
    
    sp500_tickers = get_sp500_tickers(use_cache=True, cache=st.session_state, validate_tickers=validate_tickers)
    logger.info("✓ %d tickers del S&P 500 obtenidos", len(sp500_tickers))
    
    # 1. Descargar datos de acciones
    if progress_callback:
        progress_callback(f"📥 Descargando datos de {len(sp500_tickers)} acciones del S&P 500...")
    logger.info("Iniciando descarga de datos de acciones: %d tickers", len(sp500_tickers))
    
    returns_m, skipped = download_stock_data(
        sp500_tickers,
        str(start_date),
        str(end_date),
        min_months=12,
        validate_tickers=False,  # Ya se validaron en el paso anterior
    )
    skipped_tickers.extend(skipped)
    
    if len(skipped) > 0:
        logger.warning("%d tickers no se pudieron descargar", len(skipped))

    # Limpiar y validar datos
    if not returns_m.empty:
        returns_m = _clean_and_validate_data(returns_m)
    
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
    ff5_factors = download_ff5_factors(use_cache=True, cache=st.session_state)
    logger.info("✓ Factores FF5 descargados (%d meses)", len(ff5_factors))
    
    # 3. Obtener metadatos
    if progress_callback:
        progress_callback("📥 Obteniendo metadatos...")
    logger.info("Obteniendo metadatos para %d tickers", returns_m.shape[1])
    meta = get_stock_metadata(returns_m.columns.tolist())
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

