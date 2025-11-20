"""
Módulo para optimización de carteras: máximo Sharpe e Information Ratio.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .logging_config import get_logger

logger = get_logger(__name__)


def optimize_max_sharpe(
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    rf: float,
    w_min: float = 0.0,
    w_max: float = 0.10,
    initial_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Optimiza cartera de máximo Sharpe ratio.

    Maximiza: Sharpe = (w'μ - rf) / sqrt(w'Σw)

    Sujeto a:
    - sum(w) = 1
    - w_min <= w_i <= w_max

    Parameters
    ----------
    mu : pd.Series | np.ndarray
        Retornos esperados anuales. Si Series, index = tickers.
    cov : pd.DataFrame | np.ndarray
        Matriz de covarianza anual. Si DataFrame, index/columns = tickers.
    rf : float
        Tasa libre de riesgo anual.
    w_min : float, default 0.0
        Peso mínimo por activo.
    w_max : float, default 0.10
        Peso máximo por activo.
    initial_weights : np.ndarray | None, default None
        Vector opcional para warm-start (se normaliza automáticamente).

    Returns
    -------
    w_opt : np.ndarray
        Pesos óptimos. Mismo orden que mu.

    Raises
    ------
    ValueError
        Si mu y cov tienen tamaños inconsistentes.
    """
    # Convertir a arrays numpy
    if isinstance(mu, pd.Series):
        mu_arr = mu.values
        tickers = mu.index
    else:
        mu_arr = np.asarray(mu)
        tickers = None

    if isinstance(cov, pd.DataFrame):
        if tickers is not None:
            # Alinear cov con mu si ambos son DataFrames/Series
            try:
                cov_arr = cov.loc[tickers, tickers].values
            except KeyError:
                raise ValueError(
                    f"No hay índices comunes entre mu y cov. "
                    f"Mu tiene índices: {list(tickers)}, cov tiene índices: {list(cov.index)}"
                )
        else:
            cov_arr = cov.values
    else:
        cov_arr = np.asarray(cov)

    n = len(mu_arr)

    # Validar tamaños
    if cov_arr.shape != (n, n):
        raise ValueError(
            f"Tamaño inconsistente: mu tiene {n} elementos, pero cov es {cov_arr.shape}. "
            f"Debe ser ({n}, {n})."
        )

    if w_min * n > 1.0:
        raise ValueError(
            f"Infeasible bounds: w_min={w_min} implica suma mínima {w_min * n:.2f} > 1.0"
        )

    # Función objetivo: minimizar -Sharpe (equivale a maximizar Sharpe)
    def objective(w: np.ndarray) -> float:
        portfolio_return = w @ mu_arr - rf
        portfolio_vol = np.sqrt(w @ cov_arr @ w)
        if portfolio_vol < 1e-8:
            return 1e6
        return -(portfolio_return / portfolio_vol)

    def objective_grad(w: np.ndarray) -> np.ndarray:
        portfolio_return = w @ mu_arr - rf
        cov_w = cov_arr @ w
        portfolio_var = w @ cov_w
        if portfolio_var < 1e-12:
            return np.zeros_like(w)
        portfolio_vol = np.sqrt(portfolio_var)
        numerator = mu_arr / portfolio_vol
        denominator = (portfolio_return / (portfolio_vol**3)) * cov_w
        return -(numerator - denominator)

    # Validar que los bounds permiten solución factible
    if w_max * n < 1.0:
        # Si el límite máximo no permite sumar a 1, ajustar w_max temporalmente
        w_max_effective = max(w_max, 1.0 / n)
    else:
        w_max_effective = w_max

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds
    bounds = [(w_min, w_max_effective) for _ in range(n)]

    # Inicialización: pesos iguales o warm start
    if initial_weights is not None:
        if len(initial_weights) != n:
            raise ValueError("initial_weights debe tener el mismo largo que mu.")
        w0 = np.clip(initial_weights.astype(float), w_min, w_max_effective)
        if w0.sum() == 0:
            w0 = np.ones(n) / n
    else:
        w0 = np.ones(n) / n
    w0 = w0 / w0.sum()

    # Optimizar
    logger.debug(
        "Iniciando optimización IR: %d activos, bounds [%.2f, %.2f], benchmark retorno=%.2f%%",
        n,
        w_min,
        w_max_effective,
        benchmark_annual * 100,
    )
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        logger.error("Optimización IR falló: %s", result.message)
        raise RuntimeError(
            f"Optimización falló: {result.message}. "
            f"Intentar ajustar parámetros o verificar datos de entrada."
        )

    w_opt = result.x

    # Asegurar que suma 1 (por posibles errores numéricos)
    w_opt = w_opt / w_opt.sum()
    
    # Calcular métricas del resultado
    portfolio_return = w_opt @ mu_arr
    alpha = portfolio_return - benchmark_annual
    te = np.sqrt(w_opt @ cov_diff @ w_opt)
    ir = alpha / te if te > 1e-8 else 0.0
    active_positions = (w_opt > 1e-4).sum()
    
    logger.info(
        "Optimización IR completada: IR=%.3f, alpha=%.2f%%, TE=%.2f%%, posiciones activas=%d",
        ir,
        alpha * 100,
        te * 100,
        active_positions,
    )

    return w_opt


def optimize_max_information_ratio(
    returns_m: pd.DataFrame,
    benchmark_returns: pd.Series,
    mu_expected: pd.Series | np.ndarray,
    rf: float,
    w_min: float = 0.0,
    w_max: float = 0.10,
    initial_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Optimiza cartera de máximo Information Ratio vs benchmark.

    Maximiza: IR = alpha / TE
    donde:
    - alpha = E[R_port] - E[R_bench]
    - TE = std(R_port - R_bench) = sqrt(w'Σ_diff w)

    Parameters
    ----------
    returns_m : pd.DataFrame
        Retornos mensuales de activos. Index = fechas, Columns = tickers.
    benchmark_returns : pd.Series
        Retornos mensuales del benchmark. Index = fechas.
    mu_expected : pd.Series | np.ndarray
        Retornos esperados anuales por activo. Si Series, index = tickers.
    rf : float
        Tasa libre de riesgo anual (no usado directamente en la función objetivo).
    w_min : float, default 0.0
        Peso mínimo por activo.
    w_max : float, default 0.10
        Peso máximo por activo.
    initial_weights : np.ndarray | None, default None
        Vector opcional para warm-start (se normaliza automáticamente).

    Returns
    -------
    w_opt : np.ndarray
        Pesos óptimos. Mismo orden que mu_expected.

    Raises
    ------
    ValueError
        Si returns_m y benchmark_returns no tienen fechas alineadas.
        Si mu_expected y returns_m tienen tickers inconsistentes.
    """
    # Validar alineación de fechas
    common_dates = returns_m.index.intersection(benchmark_returns.index)
    if len(common_dates) == 0:
        raise ValueError(
            "No hay fechas en común entre returns_m y benchmark_returns."
        )

    returns_aligned = returns_m.loc[common_dates]
    benchmark_aligned = benchmark_returns.loc[common_dates]

    # Calcular retornos diferenciales históricos
    diff_returns = returns_aligned.subtract(benchmark_aligned, axis=0)

    # Construir matriz de covarianza anual de diferencias
    cov_diff_df = diff_returns.cov() * 12

    # Convertir mu_expected a array y alinear con tickers de returns
    if isinstance(mu_expected, pd.Series):
        common_tickers = returns_aligned.columns.intersection(mu_expected.index)
        if len(common_tickers) == 0:
            raise ValueError(
                "No hay tickers en común entre mu_expected y returns_m.columns."
            )
        mu_arr = mu_expected.loc[common_tickers].values
        returns_aligned = returns_aligned[common_tickers]
        cov_diff = cov_diff_df.loc[common_tickers, common_tickers].values
        tickers = common_tickers
    else:
        mu_arr = np.asarray(mu_expected)
        tickers = returns_aligned.columns
        n_expected = len(mu_arr)
        n_actual = len(returns_aligned.columns)
        if n_expected != n_actual:
            raise ValueError(
                f"Tamaño inconsistente: mu_expected tiene {n_expected} elementos, "
                f"pero returns_m tiene {n_actual} columnas."
            )
        cov_diff = cov_diff_df.values

    n = len(mu_arr)

    # Calcular retorno esperado anual del benchmark
    benchmark_annual = benchmark_aligned.mean() * 12

    if w_min * n > 1.0:
        raise ValueError(
            f"Infeasible bounds: w_min={w_min} implica suma mínima {w_min * n:.2f} > 1.0"
        )

    # Función objetivo: minimizar -IR (equivale a maximizar IR)
    def objective(w: np.ndarray) -> float:
        portfolio_alpha = w @ mu_arr - benchmark_annual
        te = np.sqrt(w @ cov_diff @ w)
        if te < 1e-8:
            return 1e6
        return -(portfolio_alpha / te)

    def objective_grad(w: np.ndarray) -> np.ndarray:
        portfolio_alpha = w @ mu_arr - benchmark_annual
        cov_w = cov_diff @ w
        te_sq = w @ cov_w
        if te_sq < 1e-12:
            return np.zeros_like(w)
        te = np.sqrt(te_sq)
        numerator = mu_arr / te
        denominator = (portfolio_alpha / (te**3)) * cov_w
        return -(numerator - denominator)

    # Validar que los bounds permiten solución factible
    if w_max * n < 1.0:
        # Si el límite máximo no permite sumar a 1, ajustar w_max temporalmente
        w_max_effective = max(w_max, 1.0 / n)
    else:
        w_max_effective = w_max

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds
    bounds = [(w_min, w_max_effective) for _ in range(n)]

    if initial_weights is not None:
        if len(initial_weights) != n:
            raise ValueError("initial_weights debe tener el mismo largo que mu_expected.")
        w0 = np.clip(initial_weights.astype(float), w_min, w_max_effective)
        if w0.sum() == 0:
            w0 = np.ones(n) / n
    else:
        w0 = np.ones(n) / n
    w0 = w0 / w0.sum()

    # Optimizar
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(
            f"Optimización falló: {result.message}. "
            f"Intentar ajustar parámetros o verificar datos de entrada."
        )

    w_opt = result.x

    # Asegurar que suma 1 (por posibles errores numéricos)
    w_opt = w_opt / w_opt.sum()

    return w_opt

