"""
Ejemplo de uso del módulo ff5_portfolio.
"""

import numpy as np
import pandas as pd
from ff5_portfolio import build_gio_portfolios

# Configurar semilla para reproducibilidad
np.random.seed(42)

# Crear datos sintéticos de ejemplo
print("=" * 60)
print("EJEMPLO DE USO: ff5_portfolio")
print("=" * 60)

# 1. Generar retornos mensuales de activos (24 meses, 10 tickers)
print("\n1. Generando retornos mensuales de activos...")
dates = pd.date_range("2020-01-01", periods=24, freq="M")
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]

returns_m = pd.DataFrame(
    np.random.normal(0.01, 0.05, (24, 10)),
    index=dates,
    columns=tickers,
)
print(f"   ✓ Retornos generados: {returns_m.shape[0]} meses × {returns_m.shape[1]} activos")

# 2. Generar factores FF5 mensuales
print("\n2. Generando factores FF5 mensuales...")
ff5_factors = pd.DataFrame(
    {
        "Mkt-RF": np.random.normal(0.01, 0.05, 24),
        "SMB": np.random.normal(0.002, 0.03, 24),
        "HML": np.random.normal(0.001, 0.04, 24),
        "RMW": np.random.normal(0.003, 0.02, 24),
        "CMA": np.random.normal(-0.001, 0.025, 24),
        "RF": np.random.normal(0.002, 0.001, 24),
    },
    index=dates,
)
print(f"   ✓ Factores FF5 generados: {ff5_factors.shape[0]} meses")

# 3. Generar metadatos (sectores)
print("\n3. Generando metadatos de sectores...")
sectors = ["Tech", "Tech", "Tech", "Consumer", "Tech", "Tech", "Tech", "Finance", "Finance", "Consumer"]
meta = pd.DataFrame(
    {"sector": sectors},
    index=tickers,
)
print(f"   ✓ Metadatos generados para {len(meta)} tickers")

# 4. Generar retornos de benchmark (SPY simulado)
print("\n4. Generando retornos de benchmark (SPY)...")
benchmark_returns = pd.Series(
    np.random.normal(0.01, 0.04, 24),
    index=dates,
)
print(f"   ✓ Benchmark generado: {len(benchmark_returns)} meses")

# 5. Ejecutar pipeline completo
print("\n5. Ejecutando pipeline completo de construcción de carteras...")
print("   - Estimando betas FF5")
print("   - Construyendo retornos esperados")
print("   - Armando universo Gio")
print("   - Optimizando máximo Sharpe")
print("   - Optimizando máximo Information Ratio")

result = build_gio_portfolios(
    returns_m=returns_m,
    ff5_factors=ff5_factors,
    meta=meta,
    benchmark_returns=benchmark_returns,
    target_n_universe=5,
    max_per_sector=2,
    max_avg_corr=0.80,
    sector_col="sector",
    w_min=0.0,
    w_max=0.25,
)

# 6. Mostrar resultados
print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

print(f"\n✓ RF Anual: {result['rf_annual']:.4f} ({result['rf_annual']*100:.2f}%)")

print(f"\n✓ Universo Gio seleccionado ({len(result['universe_gio'])} activos):")
print(result['universe_gio'][['score_gio', 'sharpe_historical', 'alpha_annual', 'sector']].round(4))

print(f"\n✓ Cartera de Máximo Sharpe:")
weights_sharpe = result['weights_sharpe']
for ticker, weight in weights_sharpe.items():
    print(f"   {ticker}: {weight:.4f} ({weight*100:.2f}%)")
print(f"   Suma de pesos: {weights_sharpe.sum():.4f}")

print(f"\n✓ Cartera de Máximo Information Ratio:")
weights_ir = result['weights_ir']
for ticker, weight in weights_ir.items():
    print(f"   {ticker}: {weight:.4f} ({weight*100:.2f}%)")
print(f"   Suma de pesos: {weights_ir.sum():.4f}")

print(f"\n✓ Retornos esperados anuales (universo Gio):")
for ticker, ret in result['exp_ret_annual'].items():
    print(f"   {ticker}: {ret:.4f} ({ret*100:.2f}%)")

print("\n" + "=" * 60)
print("✓ Ejecución completada exitosamente")
print("=" * 60)

