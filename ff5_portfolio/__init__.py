"""
Paquete ff5_portfolio: construcción de carteras usando modelo Fama-French 5 factores.
"""

from .factors import estimate_ff5_betas, build_expected_returns_ff5
from .universe_gio import build_universe_gio
from .optimization import optimize_max_sharpe, optimize_max_information_ratio
from .pipeline import build_gio_portfolios, build_gio_portfolios_with_metrics, PipelineMetrics
from .data_loader import (
    download_stock_data,
    download_ff5_factors,
    get_stock_metadata,
)

__all__ = [
    "estimate_ff5_betas",
    "build_expected_returns_ff5",
    "build_universe_gio",
    "optimize_max_sharpe",
    "optimize_max_information_ratio",
    "build_gio_portfolios",
    "build_gio_portfolios_with_metrics",
    "PipelineMetrics",
    "download_stock_data",
    "download_ff5_factors",
    "get_stock_metadata",
]

