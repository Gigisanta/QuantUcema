Quiero que actúes como un arquitecto de proyecto Python muy prolijo, pensando en:
- reusabilidad,
- claridad de API,
- y facilidad para integrarlo en notebooks (Colab, Jupyter) y otros repos.

Voy a empezar desde una carpeta VACÍA. Tu tarea es diseñar y crear TODO el proyecto desde cero.

--------------------------------------------------
ROL Y OBJETIVO GENERAL
--------------------------------------------------

Tu rol:
- Diseñar y generar un mini-repo en Python cuyo núcleo sea un módulo de construcción de carteras basado en el modelo Fama-French de 5 factores (FF5).
- El módulo debe encapsular el “stock picking cuantitativo” tipo “Universo Gio” y las optimizaciones de carteras:
  - máximo Sharpe,
  - máximo Information Ratio vs benchmark.

Mi objetivo:
- Usar este proyecto para un TP de Quant en UCEMA (modelo FF5, S&P 500, etc.).
- Poder reusar el módulo como pieza plug-and-play en otros proyectos (por ejemplo Cactus / AutoAdvisor).
- Mantener la parte de datos (descargas desde Yahoo/Ken French) FUERA del módulo central. El módulo trabaja con DataFrames ya preparados.

--------------------------------------------------
ALCANCE FUNCIONAL DEL MÓDULO
--------------------------------------------------

Quiero un paquete Python llamado: `ff5_portfolio`.

Este paquete debe proveer, como API pública, las siguientes funciones de alto nivel:

1) `estimate_ff5_betas(returns_m, ff5_factors)`
   - Input:
     - `returns_m`: DataFrame de retornos mensuales de activos (index = fechas, columns = tickers, en decimales).
     - `ff5_factors`: DataFrame de factores FF5 mensuales, en decimales, con columnas:
       ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"].
   - Output:
     - `betas`: DataFrame con betas FF5 por ticker.
     - `alphas`: Series con alpha mensual por ticker.
     - `rf_annual`: float con tasa libre de riesgo anual (aprox 12 * promedio mensual de RF).
   - Detalles:
     - Alinear fechas entre returns y factores.
     - Usar regresión OLS (statsmodels) por cada activo:
       R_i - RF = alpha + beta_m (Mkt-RF) + beta_s SMB + beta_h HML + beta_r RMW + beta_c CMA + error.
     - Filtrar activos con muy pocos datos (ej. menos de 12 obs válidas).

2) `build_expected_returns_ff5(betas, ff5_factors, rf_annual)`
   - Usa las betas y los factores FF5 para estimar:
     - E[R_i]_annual = rf_annual + beta_i · (E[F]_annual).
   - Output:
     - Series con retornos esperados anuales por activo.

3) `build_universe_gio(returns_m, betas, alphas, rf_annual, meta, target_n, max_per_sector, max_avg_corr, sector_col="sector")`
   - Construye un universo de acciones “Gio” con criterios cuantitativos:
     - Métricas por activo (anualizadas):
       - Volatilidad,
       - retorno promedio,
       - Sharpe histórico = (ret_anual - rf_annual) / vol,
       - alpha anual FF5,
       - betas a factores (al menos Mkt-RF, RMW, CMA).
     - Factores que quiero que uses en el “score Gio”:
       - Sharpe histórico (alto = mejor),
       - alpha anual (alto = mejor),
       - RMW (profitability) alto,
       - CMA bajo (penalizar asset growth alto),
       - volatilidad razonable (penalizar vol muy alta),
       - beta de mercado cercana a 1 (ni 0.2 ni 2).
     - Armar un score total como combinación lineal de z-scores:
       - z_sharpe, z_alpha, z_rmw, -z_cma, -z_vol, z_beta_cerca_de_1.
     - Ordenar activos por `score_gio` de mayor a menor.
     - Aplicar filtros de diversificación:
       - límite de activos por sector (ej. max_per_sector),
       - filtro de correlación:
         - ir agregando activos en orden de score,
         - para cada candidato, calcular su correlación promedio con picks ya elegidos,
         - si `avg_corr > max_avg_corr`, se salta,
         - parar cuando se llega a `target_n`.
   - Input:
     - `returns_m`: DataFrame mensual de retornos.
     - `betas`, `alphas`, `rf_annual`: salidas de `estimate_ff5_betas`.
     - `meta`: DataFrame indexado por ticker, con al menos una columna de sector (por defecto `sector`).
   - Output:
     - DataFrame `universe_gio` indexado por ticker, con:
       - `score_gio`,
       - Sharpe histórico,
       - alpha anual,
       - betas relevantes,
       - sector.

4) `optimize_max_sharpe(mu, cov, rf, w_min=0.0, w_max=0.10)`
   - Resolver:
     - max Sharpe = (wᵀ μ – rf) / sqrt(wᵀ Σ w),
       sujeto a:
       - sum(w) = 1,
       - w_min ≤ w_i ≤ w_max.
   - Usar `scipy.optimize.minimize` con SLSQP.
   - Inputs:
     - `mu`: retornos esperados anuales (Series o array).
     - `cov`: matriz de covarianza anual (DataFrame o array).
     - `rf`: tasa libre de riesgo anual.
   - Output:
     - `w_opt`: array de pesos (mismo orden que mu).

5) `optimize_max_information_ratio(returns_m, benchmark_returns, mu_expected, rf, w_min=0.0, w_max=0.10)`
   - Optimizar cartera de máximo Information Ratio vs benchmark:
     - IR = alpha / TE,
       donde:
       - alpha = E[R_port – R_bench],
       - TE = std(R_port – R_bench).
   - Usar:
     - `mu_expected` (anual) como retorno esperado de cada activo,
     - estadísticas históricas mensuales (covarianzas) para calcular tracking error.
   - Constraints:
     - sum(w) = 1,
     - w_min ≤ w_i ≤ w_max.
   - Input:
     - `returns_m`: DataFrame de retornos mensuales de activos.
     - `benchmark_returns`: Series de retornos mensuales del benchmark (ej. SPY).
     - `mu_expected`: retornos esperados anuales.
     - `rf`: solo para consistencia, no entra directo en la función objetivo.
   - Output:
     - `w_opt`: array de pesos.

6) `build_gio_portfolios(returns_m, ff5_factors, meta=None, benchmark_returns=None, target_n_universe=30, max_per_sector=4, max_avg_corr=0.80, sector_col="sector", w_min=0.0, w_max=0.10)`
   - Función “orquestador” que:
     1. Llama a `estimate_ff5_betas` para obtener betas, alphas, rf_annual.
     2. Llama a `build_expected_returns_ff5` para E[R]_annual.
     3. Llama a `build_universe_gio` para armar universo Gio.
     4. Construye matriz de covarianza anual sobre ese universo.
     5. Optimiza:
        - cartera de máximo Sharpe → `weights_sharpe`.
        - si `benchmark_returns` no es None, cartera de máximo IR → `weights_ir`.
   - Output: diccionario con:
     - `betas`, `alphas`, `rf_annual`,
     - `exp_ret_annual`,
     - `universe_gio`,
     - `weights_sharpe`,
     - `weights_ir`.

--------------------------------------------------
ESTRUCTURA DE CARPETAS QUE QUIERO
--------------------------------------------------

Desde esta carpeta vacía, quiero que construyas algo así:

- `ff5_portfolio/`
  - `__init__.py`
  - `factors.py`
  - `universe_gio.py`
  - `optimization.py`
  - `pipeline.py`
- `requirements.txt`
- `README.md`
- (opcional, más tarde) `examples/` con ejemplos de uso.

Requerimientos específicos:

- `__init__.py` debe exponer la API principal:

  from .factors import estimate_ff5_betas, build_expected_returns_ff5
  from .universe_gio import build_universe_gio
  from .optimization import optimize_max_sharpe, optimize_max_information_ratio
  from .pipeline import build_gio_portfolios

- `requirements.txt` con lo mínimo necesario:
  - pandas
  - numpy
  - statsmodels
  - scipy

- `README.md` en español, corto y claro:
  - qué hace el módulo,
  - cómo instalar deps,
  - ejemplo simple de uso desde un notebook.

--------------------------------------------------
ESTILO DE CÓDIGO Y BUENAS PRÁCTICAS
--------------------------------------------------

Quiero que respetes:

- Python 3.10+.
- Typing moderado (type hints en firmas de funciones públicas).
- Docstrings en formato sencillo (no hace falta NumpyDoc ultra estricto, pero sí parámetros y returns).
- Nada de I/O de red en este módulo:
  - NO llamar a yfinance, NO descargar cosas.
  - El módulo asume que le paso DataFrames ya armados.
- No hardcodear rutas, ni nombres de archivos.
- No mezclar lógica de negocio con plotting:
  - este módulo NO hace gráficos. Solo calcula y devuelve objetos (Series, DataFrames, arrays).
- Errores claros:
  - si faltan columnas en factores, levantar `ValueError` con mensaje claro,
  - si no hay tickers en común, idem.

--------------------------------------------------
INTEGRACIÓN PENSADA PARA COLAB / NOTEBOOK
--------------------------------------------------

Quiero que el diseño contemple este flujo típico en un notebook (fuera del módulo):

- Cargar precios históricos de acciones y armar `returns_m`.
- Cargar factores FF5 desde la web de Ken French y armar `ff5_factors`.
- Cargar metadatos de tickers (sector, etc.) en `meta`.
- Cargar retornos de SPY en `spy_ret`.

Luego, el usuario hace algo así:

```python
from ff5_portfolio import build_gio_portfolios

result = build_gio_portfolios(
    returns_m=returns_m,
    ff5_factors=ff5_factors,
    meta=meta,                 # DataFrame con columna 'sector'
    benchmark_returns=spy_ret, # opcional
    target_n_universe=30,
    max_per_sector=4,
    max_avg_corr=0.80,
    sector_col="sector",
    w_min=0.0,
    w_max=0.10,
)

universe_gio   = result["universe_gio"]
weights_sharpe = result["weights_sharpe"]
weights_ir     = result["weights_ir"]
