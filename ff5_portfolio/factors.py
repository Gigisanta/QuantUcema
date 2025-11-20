"""
Módulo para estimación de betas y retornos esperados usando el modelo Fama-French 5 factores.
"""

import numpy as np
import pandas as pd
from statsmodels.api import add_constant

from .logging_config import get_logger

# Columnas requeridas para factores FF5
REQUIRED_FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
MIN_OBSERVATIONS = 12

logger = get_logger(__name__)


def estimate_ff5_betas(
    returns_m: pd.DataFrame, ff5_factors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, float]:
    """
    Estima betas FF5 y alphas por activo usando regresión OLS.

    Para cada activo, estima:
    R_i - RF = alpha + β_m*(Mkt-RF) + β_s*SMB + β_h*HML + β_r*RMW + β_c*CMA + ε

    Parameters
    ----------
    returns_m : pd.DataFrame
        Retornos mensuales de activos. Index = fechas, Columns = tickers. En decimales.
    ff5_factors : pd.DataFrame
        Factores FF5 mensuales. Debe tener columnas: ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"].
        En decimales.

    Returns
    -------
    betas : pd.DataFrame
        Betas FF5 por ticker. Index = tickers, Columns = factores ["Mkt-RF", "SMB", "HML", "RMW", "CMA"].
    alphas : pd.Series
        Alpha mensual por ticker. Index = tickers.
    rf_annual : float
        Tasa libre de riesgo anual (12 * promedio mensual de RF).

    Raises
    ------
    ValueError
        Si faltan columnas requeridas en ff5_factors.
        Si no hay fechas en común entre returns_m y ff5_factors.
    """
    # Validar columnas requeridas
    missing_cols = set(REQUIRED_FF5_COLUMNS) - set(ff5_factors.columns)
    if missing_cols:
        raise ValueError(
            f"Faltan columnas requeridas en ff5_factors: {missing_cols}. "
            f"Columnas requeridas: {REQUIRED_FF5_COLUMNS}"
        )

    # Alinear fechas
    common_dates = returns_m.index.intersection(ff5_factors.index)
    if len(common_dates) == 0:
        raise ValueError(
            "No hay fechas en común entre returns_m y ff5_factors después de alineación."
        )
    
    logger.info(
        "Estimando betas FF5: %d activos, %d meses de datos comunes",
        returns_m.shape[1],
        len(common_dates),
    )

    returns_aligned = returns_m.loc[common_dates]
    factors_aligned = ff5_factors.loc[common_dates, REQUIRED_FF5_COLUMNS]

    # Extraer RF y factores de mercado
    rf = factors_aligned["RF"].values
    factor_vars = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    X = factors_aligned[factor_vars].values

    # Calcular rf_annual
    rf_annual = float(12 * rf.mean())
    logger.debug("Tasa libre de riesgo anual: %.4f%%", rf_annual * 100)

    # Preparar matrices para regresión vectorizada
    X_design = add_constant(X)
    factor_valid_mask = ~np.isnan(X_design).any(axis=1)
    X_design = X_design[factor_valid_mask]

    returns_matrix = returns_aligned.values[factor_valid_mask]
    rf_vector = rf[factor_valid_mask]
    excess_matrix = returns_matrix - rf_vector[:, None]

    # Estimar betas y alphas por ticker
    betas_dict = {}
    alphas_dict = {}
    skipped_count = 0

    for idx, ticker in enumerate(returns_aligned.columns):
        y = excess_matrix[:, idx]
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < MIN_OBSERVATIONS:
            skipped_count += 1
            logger.debug("%s: Insuficientes observaciones (%d < %d)", ticker, valid_mask.sum(), MIN_OBSERVATIONS)
            continue

        X_clean = X_design[valid_mask]
        y_clean = y[valid_mask]

        try:
            params, *_ = np.linalg.lstsq(X_clean, y_clean, rcond=None)
        except np.linalg.LinAlgError:
            skipped_count += 1
            logger.debug("%s: Error en regresión (matriz singular)", ticker)
            continue

        alphas_dict[ticker] = params[0]
        betas_dict[ticker] = dict(zip(factor_vars, params[1:]))
    
    if skipped_count > 0:
        logger.debug("%d activos omitidos por datos insuficientes o errores de regresión", skipped_count)

    if len(betas_dict) == 0:
        raise ValueError(
            f"No hay activos con al menos {MIN_OBSERVATIONS} observaciones válidas."
        )

    # Construir DataFrames de salida
    betas = pd.DataFrame(betas_dict).T
    alphas = pd.Series(alphas_dict)
    
    logger.info(
        "Betas FF5 estimados exitosamente: %d activos con betas válidos",
        len(betas),
    )

    return betas, alphas, rf_annual


def build_expected_returns_ff5(
    betas: pd.DataFrame, ff5_factors: pd.DataFrame, rf_annual: float
) -> pd.Series:
    """
    Construye retornos esperados anuales por activo usando el modelo FF5.

    E[R_i]_annual = rf_annual + β_i · E[F]_annual

    Parameters
    ----------
    betas : pd.DataFrame
        Betas FF5 por ticker. Index = tickers, Columns = factores FF5.
    ff5_factors : pd.DataFrame
        Factores FF5 mensuales históricos. Usado para calcular expectativas de factores.
    rf_annual : float
        Tasa libre de riesgo anual.

    Returns
    -------
    exp_returns : pd.Series
        Retornos esperados anuales por activo. Index = tickers.
    """
    factor_vars = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    # Validar que ff5_factors tiene las columnas necesarias
    missing_factors = set(factor_vars) - set(ff5_factors.columns)
    if missing_factors:
        raise ValueError(
            f"ff5_factors no contiene todas las columnas de factores requeridas: {missing_factors}"
        )

    # Validar que betas tiene las columnas necesarias
    missing_betas = set(factor_vars) - set(betas.columns)
    if missing_betas:
        raise ValueError(
            f"Betas no contiene todas las columnas de factores requeridas: {missing_betas}"
        )

    # Calcular expectativas anuales de factores
    exp_factors_annual = 12 * ff5_factors[factor_vars].mean()
    logger.debug("Expectativas anuales de factores: %s", exp_factors_annual.to_dict())

    # Calcular retornos esperados anuales por activo
    # E[R_i] = rf + β_i · E[F]
    exp_returns = rf_annual + (betas[factor_vars] @ exp_factors_annual)
    
    logger.info(
        "Retornos esperados calculados para %d activos (rango: %.2f%% - %.2f%%)",
        len(exp_returns),
        exp_returns.min() * 100,
        exp_returns.max() * 100,
    )

    return exp_returns

