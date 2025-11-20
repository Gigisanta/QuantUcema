"""
Cálculos de métricas y resumen de carteras.
"""

import numpy as np
import pandas as pd


def prepare_portfolio_summary(
    result: dict, benchmark_returns: pd.Series | None
) -> dict:
    """
    Calcula métricas de cartera reusables entre tabs.
    
    Parameters
    ----------
    result : dict
        Resultado del pipeline con keys: weights_sharpe, weights_ir, returns_universe,
        cov_annual, exp_ret_annual, rf_annual.
    benchmark_returns : pd.Series | None
        Retornos mensuales del benchmark (SPY).
    
    Returns
    -------
    dict
        Diccionario con métricas de carteras Sharpe e IR.
    """
    returns_universe = result["returns_universe"]
    cov_annual = result["cov_annual"]

    weights_sharpe = result["weights_sharpe"]
    sharpe_cov = cov_annual.loc[weights_sharpe.index, weights_sharpe.index]
    exp_ret_sharpe = result["exp_ret_annual"].loc[weights_sharpe.index]

    port_ret_sharpe = float(weights_sharpe @ exp_ret_sharpe)
    port_vol_sharpe = float(np.sqrt(weights_sharpe @ sharpe_cov.values @ weights_sharpe))
    sharpe_ratio = (port_ret_sharpe - result["rf_annual"]) / port_vol_sharpe
    port_returns_sharpe = (returns_universe[weights_sharpe.index] @ weights_sharpe).dropna()

    summary = {
        "sharpe": {
            "expected_return": port_ret_sharpe,
            "volatility": port_vol_sharpe,
            "sharpe_ratio": sharpe_ratio,
            "weights_count": int((weights_sharpe > 1e-4).sum()),
        },
        "ir": None,
        "cumulative": {
            "sharpe": (1 + port_returns_sharpe).cumprod() * 100,
            "ir": None,
            "benchmark": None,
        },
    }

    if benchmark_returns is not None:
        summary["cumulative"]["benchmark"] = (1 + benchmark_returns).cumprod() * 100

    weights_ir = result["weights_ir"]
    if weights_ir is not None and benchmark_returns is not None:
        exp_ret_ir = result["exp_ret_annual"].loc[weights_ir.index]
        cov_ir = cov_annual.loc[weights_ir.index, weights_ir.index]
        port_ret_ir = float(weights_ir @ exp_ret_ir)
        port_vol_ir = float(np.sqrt(weights_ir @ cov_ir.values @ weights_ir))
        spy_ret_annual = benchmark_returns.mean() * 12
        alpha_ir = port_ret_ir - spy_ret_annual

        port_returns_ir = (returns_universe[weights_ir.index] @ weights_ir).dropna()
        aligned_bench = benchmark_returns.loc[port_returns_ir.index]
        diff_returns = port_returns_ir - aligned_bench
        te_ir = diff_returns.std() * np.sqrt(12)
        ir_ratio = alpha_ir / te_ir if te_ir > 0 else 0.0

        summary["ir"] = {
            "expected_return": port_ret_ir,
            "volatility": port_vol_ir,
            "alpha": alpha_ir,
            "information_ratio": ir_ratio,
            "weights_count": int((weights_ir > 1e-4).sum()),
        }
        summary["cumulative"]["ir"] = (1 + port_returns_ir).cumprod() * 100

    return summary

