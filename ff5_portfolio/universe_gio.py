"""
Módulo para construcción del universo de acciones "Gio" usando criterios cuantitativos.
"""

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


def build_universe_gio(
    returns_m: pd.DataFrame,
    betas: pd.DataFrame,
    alphas: pd.Series,
    rf_annual: float,
    meta: pd.DataFrame,
    target_n: int,
    max_per_sector: int,
    max_avg_corr: float,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """
    Construye un universo de acciones "Gio" con criterios cuantitativos.

    Selecciona activos basándose en:
    - Sharpe histórico alto
    - Alpha FF5 anual alto
    - RMW (profitability) alto
    - CMA bajo (penalizar asset growth)
    - Volatilidad razonable (penalizar vol muy alta)
    - Beta de mercado cercana a 1

    Aplica filtros de diversificación:
    - Límite de activos por sector
    - Filtro de correlación promedio con picks ya seleccionados

    Parameters
    ----------
    returns_m : pd.DataFrame
        Retornos mensuales de activos. Index = fechas, Columns = tickers.
    betas : pd.DataFrame
        Betas FF5 por ticker. Index = tickers.
    alphas : pd.Series
        Alpha mensual por ticker. Index = tickers.
    rf_annual : float
        Tasa libre de riesgo anual.
    meta : pd.DataFrame
        Metadatos de tickers. Index = tickers, debe tener columna `sector_col`.
    target_n : int
        Número objetivo de activos a seleccionar.
    max_per_sector : int
        Máximo número de activos por sector.
    max_avg_corr : float
        Máximo de correlación promedio permitida con picks ya seleccionados.
    sector_col : str, default "sector"
        Nombre de la columna de sector en meta.

    Returns
    -------
    universe_gio : pd.DataFrame
        Universo seleccionado. Index = tickers, Columns = [score_gio, sharpe_historical,
        alpha_annual, beta_mkt, beta_rmw, beta_cma, sector].
    """
    # Validar que meta tiene la columna de sector
    if sector_col not in meta.columns:
        raise ValueError(
            f"Meta no contiene la columna '{sector_col}'. "
            f"Columnas disponibles: {list(meta.columns)}"
        )

    # Alinear tickers entre returns, betas, alphas y meta
    common_tickers = (
        returns_m.columns.intersection(betas.index)
        .intersection(alphas.index)
        .intersection(meta.index)
    )

    if len(common_tickers) == 0:
        raise ValueError(
            "No hay tickers en común entre returns_m, betas, alphas y meta."
        )
    
    logger.debug(
        "Construyendo universo Gio: %d tickers comunes, objetivo: %d activos",
        len(common_tickers),
        target_n,
    )

    returns_filtered = returns_m[common_tickers]
    betas_filtered = betas.loc[common_tickers]
    alphas_filtered = alphas.loc[common_tickers]
    meta_filtered = meta.loc[common_tickers]
    corr_matrix = returns_filtered.corr()

    # Calcular métricas anualizadas por ticker
    vol_annual = returns_filtered.std() * np.sqrt(12)
    ret_annual = returns_filtered.mean() * 12
    sharpe_historical = (ret_annual - rf_annual) / vol_annual
    alpha_annual = alphas_filtered * 12

    # Extraer betas relevantes
    beta_mkt = betas_filtered["Mkt-RF"] if "Mkt-RF" in betas_filtered.columns else pd.Series(0, index=common_tickers)
    beta_rmw = betas_filtered["RMW"] if "RMW" in betas_filtered.columns else pd.Series(0, index=common_tickers)
    beta_cma = betas_filtered["CMA"] if "CMA" in betas_filtered.columns else pd.Series(0, index=common_tickers)

    # Calcular z-scores de cada métrica (normalización por universo completo)
    def z_score(series: pd.Series) -> pd.Series:
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return pd.Series(0, index=series.index)
        return (series - mean_val) / std_val

    z_sharpe = z_score(sharpe_historical)
    z_alpha = z_score(alpha_annual)
    z_rmw = z_score(beta_rmw)
    z_cma = z_score(beta_cma)
    z_vol = z_score(vol_annual)

    # z-score para beta cercana a 1
    # z_beta_cerca_de_1 = -abs(z_beta_mkt - z_1) donde z_1 corresponde a beta=1
    z_beta_mkt = z_score(beta_mkt)
    beta_1_series = pd.Series(1.0, index=common_tickers)
    z_beta_1 = z_score(beta_1_series)
    z_beta_cerca_de_1 = -(z_beta_mkt - z_beta_1).abs()

    # Calcular score_gio como combinación lineal de z-scores
    # score = z_sharpe + z_alpha + z_rmw - z_cma - z_vol + z_beta_cerca_de_1
    score_gio = (
        z_sharpe
        + z_alpha
        + z_rmw
        - z_cma
        - z_vol
        + z_beta_cerca_de_1
    )

    # Ordenar por score descendente
    tickers_sorted = score_gio.sort_values(ascending=False).index

    # Aplicar filtros de diversificación
    picks = []
    sector_counts = {}
    skipped_sector = 0
    skipped_corr = 0

    for ticker in tickers_sorted:
        if len(picks) >= target_n:
            break

        sector = meta_filtered.loc[ticker, sector_col]
        if pd.isna(sector):
            sector = "Unknown"

        # Filtro por sector: verificar límite
        if sector_counts.get(sector, 0) >= max_per_sector:
            skipped_sector += 1
            logger.debug("%s: Omitido por límite de sector (%s ya tiene %d)", ticker, sector, sector_counts.get(sector, 0))
            continue

        # Filtro por correlación: calcular correlación promedio con picks ya seleccionados
        if len(picks) > 0:
            corr_values = corr_matrix.loc[ticker, picks].dropna()
            if not corr_values.empty:
                avg_corr = corr_values.mean()
                if avg_corr > max_avg_corr:
                    skipped_corr += 1
                    logger.debug("%s: Omitido por alta correlación (%.3f > %.3f)", ticker, avg_corr, max_avg_corr)
                    continue

        # Agregar a picks
        picks.append(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        logger.debug("%s: Agregado al universo (sector: %s, picks: %d/%d)", ticker, sector, len(picks), target_n)
    
    if skipped_sector > 0 or skipped_corr > 0:
        logger.debug("Filtros aplicados: %d omitidos por sector, %d por correlación", skipped_sector, skipped_corr)

    if len(picks) == 0:
        raise ValueError(
            "No se pudo seleccionar ningún activo con los criterios dados."
        )

    # Construir DataFrame de salida
    universe_gio = pd.DataFrame(
        {
            "score_gio": score_gio.loc[picks],
            "sharpe_historical": sharpe_historical.loc[picks],
            "alpha_annual": alpha_annual.loc[picks],
            "beta_mkt": beta_mkt.loc[picks],
            "beta_rmw": beta_rmw.loc[picks],
            "beta_cma": beta_cma.loc[picks],
            "sector": meta_filtered.loc[picks, sector_col].fillna("Unknown"),
        }
    )
    
    logger.info(
        "Universo Gio construido: %d activos seleccionados de %d candidatos (distribución por sector: %s)",
        len(picks),
        len(common_tickers),
        dict(sector_counts),
    )

    return universe_gio

