"""
Herramienta de profiling para el pipeline completo de ff5_portfolio.

Descarga datos del S&P 500 (subconjunto configurable), ejecuta el pipeline y
reporta métricas de tiempo por etapa para facilitar comparaciones antes/después
de optimizaciones de performance.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd

from ff5_portfolio.data_loader import (
    download_ff5_factors,
    download_stock_data,
    get_sp500_tickers,
    get_stock_metadata,
)
from ff5_portfolio.pipeline import build_gio_portfolios_with_metrics


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profilea el pipeline Gio con descargas reales de datos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", default="2020-01-01", help="Fecha inicial (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"),
                        help="Fecha final (YYYY-MM-DD).")
    parser.add_argument("--max-tickers", type=int, default=50, help="Número máximo de tickers del S&P 500.")
    parser.add_argument("--min-months", type=int, default=12, help="Mínimo de meses requeridos por ticker.")
    parser.add_argument("--target-n", type=int, default=30, help="Tamaño objetivo del universo Gio.")
    parser.add_argument("--max-per-sector", type=int, default=4, help="Máx. activos por sector.")
    parser.add_argument("--max-avg-corr", type=float, default=0.80, help="Máx. correlación promedio permitida.")
    parser.add_argument("--w-max", type=float, default=0.10, help="Peso máximo por activo en optimización.")
    parser.add_argument("--include-ir", action="store_true", help="Incluye optimización IR vs SPY.")
    parser.add_argument("--log-level", default="INFO", help="Nivel de logging (DEBUG, INFO, etc).")
    parser.add_argument("--metrics-csv", type=Path, help="Ruta opcional para guardar métricas en CSV.")
    parser.add_argument("--no-cache", action="store_true", help="Deshabilita caches de data_loader.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)

    logging.info("Iniciando profiling del pipeline Gio...")
    overall_start = perf_counter()

    tickers = get_sp500_tickers(use_cache=not args.no_cache, cache=None)
    tickers = tickers[: args.max_tickers]
    logging.info("Tickers S&P 500 seleccionados: %d", len(tickers))

    stage_timings: dict[str, float] = {}

    # Descarga de retornos
    stage_start = perf_counter()
    returns_m, skipped = download_stock_data(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        min_months=args.min_months,
    )
    stage_timings["download_stock_data"] = perf_counter() - stage_start
    logging.info("Retornos descargados para %d tickers (%d fallidos).", returns_m.shape[1], len(skipped))

    # Factores FF5
    stage_start = perf_counter()
    ff5_factors = download_ff5_factors()
    stage_timings["download_ff5_factors"] = perf_counter() - stage_start

    # Metadatos
    stage_start = perf_counter()
    meta = get_stock_metadata(returns_m.columns.tolist())
    stage_timings["get_stock_metadata"] = perf_counter() - stage_start

    # Benchmark SPY opcional
    benchmark_returns = None
    if args.include_ir:
        stage_start = perf_counter()
        spy_returns, _ = download_stock_data(
            tickers=["SPY"],
            start_date=args.start_date,
            end_date=args.end_date,
            min_months=args.min_months,
        )
        stage_timings["download_spy"] = perf_counter() - stage_start
        benchmark_returns = spy_returns.iloc[:, 0]

    # Pipeline principal
    result, metrics = build_gio_portfolios_with_metrics(
        returns_m=returns_m,
        ff5_factors=ff5_factors,
        meta=meta,
        benchmark_returns=benchmark_returns,
        target_n_universe=args.target_n,
        max_per_sector=args.max_per_sector,
        max_avg_corr=args.max_avg_corr,
        w_max=args.w_max,
    )

    total_time = perf_counter() - overall_start
    logging.info("Pipeline completado. Universo Gio: %d activos.", len(result["universe_gio"]))

    print("\n==== Métricas de Pipeline ====")
    for stage, seconds in metrics.stage_durations.items():
        print(f"{stage:35s}: {seconds:8.2f} s")
    print(f"{'pipeline_total':35s}: {metrics.total_seconds:8.2f} s")

    print("\n==== Métricas de Descarga ====")
    for stage, seconds in stage_timings.items():
        print(f"{stage:35s}: {seconds:8.2f} s")

    print(f"\nTiempo total end-to-end: {total_time:.2f} s")

    if args.metrics_csv:
        df_metrics = pd.DataFrame(
            [
                {"stage": stage, "seconds": seconds, "type": "pipeline"}
                for stage, seconds in metrics.stage_durations.items()
            ]
            + [
                {"stage": stage, "seconds": seconds, "type": "data_loader"}
                for stage, seconds in stage_timings.items()
            ]
        )
        df_metrics.to_csv(args.metrics_csv, index=False)
        logging.info("Métricas guardadas en %s", args.metrics_csv)


if __name__ == "__main__":
    main()


