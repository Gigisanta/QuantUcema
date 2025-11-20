"""
Módulo pipeline: función orquestadora principal para construcción de carteras Gio.
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, TypeVar

import pandas as pd

from .factors import estimate_ff5_betas, build_expected_returns_ff5
from .logging_config import get_logger
from .optimization import optimize_max_sharpe, optimize_max_information_ratio
from .universe_gio import build_universe_gio

logger = get_logger(__name__)

_T = TypeVar("_T")


@dataclass
class PipelineMetrics:
    """Almacena tiempos de cada etapa del pipeline cuantitativo."""

    stage_durations: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0

    def record(self, stage: str, seconds: float) -> None:
        """Registra la duración de una etapa específica."""
        self.stage_durations[stage] = seconds


def _time_stage(metrics: PipelineMetrics, stage: str, fn: Callable[[], _T]) -> _T:
    """Ejecuta una función y registra su duración en metrics."""
    logger.info("Iniciando etapa: %s", stage)
    start = perf_counter()
    result = fn()
    elapsed = perf_counter() - start
    metrics.record(stage, elapsed)
    logger.info("Etapa '%s' completada en %.2f segundos", stage, elapsed)
    return result


def build_gio_portfolios(
    returns_m: pd.DataFrame,
    ff5_factors: pd.DataFrame,
    meta: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    target_n_universe: int = 30,
    max_per_sector: int = 4,
    max_avg_corr: float = 0.80,
    sector_col: str = "sector",
    w_min: float = 0.0,
    w_max: float = 0.10,
) -> dict:
    """
    Orquesta la construcción completa de carteras Gio.

    Pipeline:
    1. Estima betas y alphas FF5
    2. Construye retornos esperados anuales
    3. Arma universo Gio (stock picking cuantitativo)
    4. Optimiza cartera de máximo Sharpe
    5. Opcionalmente optimiza cartera de máximo IR si se provee benchmark

    Parameters
    ----------
    returns_m : pd.DataFrame
        Retornos mensuales de activos. Index = fechas, Columns = tickers.
    ff5_factors : pd.DataFrame
        Factores FF5 mensuales. Columnas: ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"].
    meta : pd.DataFrame | None, default None
        Metadatos de tickers. Index = tickers, columna `sector_col`. Si None, se crea dummy.
    benchmark_returns : pd.Series | None, default None
        Retornos mensuales del benchmark (ej. SPY). Si None, no se optimiza IR.
    target_n_universe : int, default 30
        Número objetivo de activos en universo Gio.
    max_per_sector : int, default 4
        Máximo número de activos por sector.
    max_avg_corr : float, default 0.80
        Máximo de correlación promedio permitida con picks ya seleccionados.
    sector_col : str, default "sector"
        Nombre de la columna de sector en meta.
    w_min : float, default 0.0
        Peso mínimo por activo en optimización.
    w_max : float, default 0.10
        Peso máximo por activo en optimización.

    Returns
    -------
    result : dict
        Diccionario con:
        - 'betas': DataFrame de betas FF5 por ticker
        - 'alphas': Series de alphas mensuales por ticker
        - 'rf_annual': float con tasa libre de riesgo anual
        - 'exp_ret_annual': Series de retornos esperados anuales (filtrado al universo Gio)
        - 'universe_gio': DataFrame del universo seleccionado
        - 'weights_sharpe': Series de pesos de cartera máximo Sharpe (index = tickers universo)
        - 'weights_ir': Series de pesos de cartera máximo IR (index = tickers universo) o None
        - 'cov_annual': DataFrame con la covarianza anualizada del universo Gio
        - 'returns_universe': DataFrame de retornos mensuales (ya filtrado al universo Gio)
    """
    result, _ = build_gio_portfolios_with_metrics(
        returns_m=returns_m,
        ff5_factors=ff5_factors,
        meta=meta,
        benchmark_returns=benchmark_returns,
        target_n_universe=target_n_universe,
        max_per_sector=max_per_sector,
        max_avg_corr=max_avg_corr,
        sector_col=sector_col,
        w_min=w_min,
        w_max=w_max,
    )
    return result


def build_gio_portfolios_with_metrics(
    returns_m: pd.DataFrame,
    ff5_factors: pd.DataFrame,
    meta: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    target_n_universe: int = 30,
    max_per_sector: int = 4,
    max_avg_corr: float = 0.80,
    sector_col: str = "sector",
    w_min: float = 0.0,
    w_max: float = 0.10,
) -> tuple[dict, PipelineMetrics]:
    """
    Variante de build_gio_portfolios que retorna métricas de performance.

    Returns
    -------
    result : dict
        Mismo diccionario que build_gio_portfolios.
    metrics : PipelineMetrics
        Duración de cada etapa y tiempo total del pipeline.
    """
    metrics = PipelineMetrics()
    total_start = perf_counter()
    
    logger.info(
        "Iniciando pipeline Gio: %d activos, %d meses de datos, universo objetivo: %d",
        returns_m.shape[1],
        returns_m.shape[0],
        target_n_universe,
    )

    # 1. Estimar betas y alphas FF5
    logger.debug("Estimando betas FF5 para %d activos", returns_m.shape[1])
    betas, alphas, rf_annual = _time_stage(
        metrics, "estimate_ff5_betas", lambda: estimate_ff5_betas(returns_m, ff5_factors)
    )
    logger.debug("Betas estimados: %d activos con %d factores", betas.shape[0], betas.shape[1])

    # 2. Construir retornos esperados anuales
    logger.debug("Construyendo retornos esperados anuales")
    exp_ret_annual = _time_stage(
        metrics,
        "expected_returns",
        lambda: build_expected_returns_ff5(betas, ff5_factors, rf_annual),
    )
    logger.debug("Retornos esperados calculados para %d activos", len(exp_ret_annual))

    # 3. Manejar meta si es None
    def _prepare_meta() -> pd.DataFrame:
        if meta is not None:
            return meta
        return pd.DataFrame({"sector": "Unknown"}, index=exp_ret_annual.index)

    meta_resolved = _time_stage(metrics, "prepare_meta", _prepare_meta)

    # 4. Armar universo Gio
    logger.info(
        "Construyendo universo Gio: objetivo %d activos, max %d por sector, max correlación %.2f",
        target_n_universe,
        max_per_sector,
        max_avg_corr,
    )
    universe_gio = _time_stage(
        metrics,
        "build_universe_gio",
        lambda: build_universe_gio(
            returns_m=returns_m,
            betas=betas,
            alphas=alphas,
            rf_annual=rf_annual,
            meta=meta_resolved,
            target_n=target_n_universe,
            max_per_sector=max_per_sector,
            max_avg_corr=max_avg_corr,
            sector_col=sector_col,
        ),
    )
    logger.info("Universo Gio construido: %d activos seleccionados", len(universe_gio))

    # 5. Filtrar returns y exp_ret_annual al universo Gio
    universe_tickers = universe_gio.index
    returns_universe = returns_m[universe_tickers]
    exp_ret_universe = exp_ret_annual.loc[universe_tickers]
    
    # Liberar memoria de returns_m completo y exp_ret_annual completo
    # (ya no se necesitan, solo el universo filtrado)
    del returns_m
    del exp_ret_annual

    # 6. Construir matriz de covarianza anual del universo (ya vectorizado)
    cov_annual = _time_stage(
        metrics, "covariance_universe", lambda: returns_universe.cov() * 12
    )

    # 7. Optimizar máximo Sharpe
    logger.info(
        "Optimizando cartera de máximo Sharpe: %d activos, pesos entre %.2f%% y %.2f%%",
        len(universe_tickers),
        w_min * 100,
        w_max * 100,
    )
    weights_sharpe_arr = _time_stage(
        metrics,
        "optimize_max_sharpe",
        lambda: optimize_max_sharpe(
            mu=exp_ret_universe,
            cov=cov_annual,
            rf=rf_annual,
            w_min=w_min,
            w_max=w_max,
        ),
    )
    weights_sharpe = pd.Series(weights_sharpe_arr, index=universe_tickers)
    active_positions = (weights_sharpe > 1e-4).sum()
    logger.info(
        "Optimización Sharpe completada: %d posiciones activas, peso total: %.4f",
        active_positions,
        weights_sharpe.sum(),
    )

    # 8. Optimizar máximo IR si se provee benchmark
    if benchmark_returns is not None:
        logger.info(
            "Optimizando cartera de máximo Information Ratio vs benchmark: %d activos",
            len(universe_tickers),
        )
        weights_ir_arr = _time_stage(
            metrics,
            "optimize_max_information_ratio",
            lambda: optimize_max_information_ratio(
                returns_m=returns_universe,
                benchmark_returns=benchmark_returns,
                mu_expected=exp_ret_universe,
                rf=rf_annual,
                w_min=w_min,
                w_max=w_max,
            ),
        )
        weights_ir = pd.Series(weights_ir_arr, index=universe_tickers)
        active_positions_ir = (weights_ir > 1e-4).sum()
        logger.info(
            "Optimización IR completada: %d posiciones activas, peso total: %.4f",
            active_positions_ir,
            weights_ir.sum(),
        )
    else:
        logger.debug("Benchmark no proporcionado, omitiendo optimización IR")
        weights_ir = None

    metrics.total_seconds = perf_counter() - total_start
    logger.info(
        "Pipeline completado exitosamente en %.2f segundos (total)",
        metrics.total_seconds,
    )

    # Retornar diccionario con todos los resultados
    return {
        "betas": betas,
        "alphas": alphas,
        "rf_annual": rf_annual,
        "exp_ret_annual": exp_ret_universe,
        "universe_gio": universe_gio,
        "weights_sharpe": weights_sharpe,
        "weights_ir": weights_ir,
        "cov_annual": cov_annual,
        "returns_universe": returns_universe,
    }, metrics


