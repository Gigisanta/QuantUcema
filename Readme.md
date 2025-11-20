# ff5_portfolio

`ff5_portfolio` es un mini–módulo en Python para construir carteras cuantitativas usando:

- el modelo de **Fama–French 5 factores (FF5)**  
- un universo de acciones curado mediante **stock picking cuantitativo** ("Universo Gio")  
- optimización de carteras de:
  - **máximo Sharpe**
  - **máximo Information Ratio vs un benchmark** (por ejemplo, SPY)

Está pensado para:

- Trabajos prácticos de finanzas cuantitativas (ej. UCEMA, modelo FF5 sobre S&P 500).
- Integrarse fácilmente en otros proyectos (dashboards, backtests, research, etc.).
- Usarse desde **notebooks (Colab, Jupyter)**, scripts Python, o **interfaz web interactiva (Streamlit)**.

---

## Instalación de dependencias

Dentro del proyecto:

```bash
pip install -r requirements.txt
```

El módulo usa las siguientes librerías:

* `pandas` - Manipulación de datos
* `numpy` - Cálculos numéricos
* `statsmodels` - Regresiones OLS para estimación de betas
* `scipy` - Optimización de carteras
* `yfinance` - Descarga de datos de acciones
* `streamlit` - Interfaz web interactiva (opcional)
* `requests` - Descarga de factores FF5 desde Ken French
* `matplotlib` y `seaborn` - Visualizaciones

---

## Modo de uso

### Opción 1: Interfaz Web Interactiva (Recomendado)

La forma más fácil de usar el módulo es a través de la interfaz web Streamlit:

```bash
streamlit run app.py
```

Esto abrirá una interfaz web en tu navegador donde podrás:

1. **Ingresar tickers** manualmente (separados por comas) o cargar desde un archivo CSV
2. **Seleccionar rango de fechas** para el análisis
3. **Configurar parámetros** del modelo (tamaño del universo, límites por sector, etc.)
4. **Ver resultados** en múltiples tabs:
   - Resumen con métricas principales
   - Universo Gio con activos seleccionados
   - Composición de cartera de máximo Sharpe
   - Composición de cartera de máximo IR (si está habilitado)
   - Gráficos interactivos (frontera eficiente, evolución temporal)

La interfaz descarga automáticamente:
- Precios de acciones desde yfinance
- Factores FF5 desde Ken French
- Metadatos (sectores) de las acciones
- Datos de SPY como benchmark (si se requiere)

### Opción 2: Uso Programático

Si preferís usar el módulo desde código Python, podés:

**Opción A: Con descarga automática de datos**

```python
from ff5_portfolio import build_gio_portfolios
from ff5_portfolio.data_loader import (
    download_stock_data,
    download_ff5_factors,
    get_stock_metadata,
)

# Descargar datos automáticamente
returns_m, skipped = download_stock_data(
    ["AAPL", "MSFT", "GOOGL"], 
    start_date="2020-01-01", 
    end_date="2024-12-31"
)
ff5_factors = download_ff5_factors()
meta = get_stock_metadata(["AAPL", "MSFT", "GOOGL"])

# Construir carteras
result = build_gio_portfolios(
    returns_m=returns_m,
    ff5_factors=ff5_factors,
    meta=meta,
    target_n_universe=30,
    max_per_sector=4,
    max_avg_corr=0.80,
)
```

**Opción B: Con datos propios**

Si ya tenés los datos preparados:

```python
from ff5_portfolio import build_gio_portfolios

# Supongamos que ya preparaste:
# returns_m       -> DataFrame (fechas x tickers) de retornos mensuales
# ff5_factors     -> DataFrame de factores FF5 mensuales
# meta            -> DataFrame indexado por ticker, con columna 'sector'
# spy_returns_m   -> Series de retornos mensuales del SPY (benchmark)

result = build_gio_portfolios(
    returns_m=returns_m,
    ff5_factors=ff5_factors,
    meta=meta,
    benchmark_returns=spy_returns_m,   # opcional
    target_n_universe=30,
    max_per_sector=4,
    max_avg_corr=0.80,
)
```

## Funcionalidades del módulo

El módulo se encarga de:

* **Descargar datos** automáticamente desde yfinance y Ken French (opcional)
* Estimar **betas y alphas FF5**
* Construir **retornos esperados** vía FF5
* Armar el **universo Gio** (stock picking cuantitativo)
* Optimizar carteras de **máx Sharpe** y **máx Information Ratio**

---

## API principal

Desde un notebook o script podés importar:

```python
from ff5_portfolio import (
    estimate_ff5_betas,
    build_expected_returns_ff5,
    build_universe_gio,
    optimize_max_sharpe,
    optimize_max_information_ratio,
    build_gio_portfolios,
)
```

Las funciones clave:

* `estimate_ff5_betas(returns_m, ff5_factors)`
  Estima betas y alphas FF5 por activo y devuelve además `rf_annual`.

* `build_expected_returns_ff5(betas, ff5_factors, rf_annual)`
  Construye retornos esperados anuales por activo usando el modelo FF5.

* `build_universe_gio(returns_m, betas, alphas, rf_annual, meta, ...)`
  Arma un universo de acciones "Gio" usando:

  * Sharpe histórico
  * alpha FF5
  * exposición a factores de calidad (RMW, CMA)
  * control de volatilidad
  * beta de mercado cercana a 1
  * límites por sector y filtros de correlación entre activos

* `optimize_max_sharpe(mu, cov, rf, w_min, w_max)`
  Devuelve los pesos de la cartera de máximo Sharpe bajo restricciones (long-only, cap por activo).

* `optimize_max_information_ratio(returns_m, benchmark_returns, mu_expected, rf, ...)`
  Devuelve los pesos de la cartera que maximiza el Information Ratio vs un benchmark.

* `build_gio_portfolios(...)`
  Orquesta todo junto:

  * estima FF5,
  * construye retornos esperados,
  * arma universo Gio,
  * optimiza máx Sharpe,
  * y opcionalmente máx IR.

---

## Ejemplo rápido de uso en un notebook

```python
import pandas as pd
from ff5_portfolio import build_gio_portfolios

# Supongamos que ya preparaste:
# returns_m       -> DataFrame (fechas x tickers) de retornos mensuales
# ff5_factors     -> DataFrame de factores FF5 mensuales
# meta            -> DataFrame indexado por ticker, con columna 'sector'
# spy_returns_m   -> Series de retornos mensuales del SPY (benchmark)

result = build_gio_portfolios(
    returns_m=returns_m,
    ff5_factors=ff5_factors,
    meta=meta,
    benchmark_returns=spy_returns_m,   # podés pasar None si no querés máx IR
    target_n_universe=30,
    max_per_sector=4,
    max_avg_corr=0.80,
    sector_col="sector",
    w_min=0.0,
    w_max=0.10,
)

universe_gio   = result["universe_gio"]    # universo de acciones seleccionadas
weights_sharpe = result["weights_sharpe"]  # cartera de máximo Sharpe
weights_ir     = result["weights_ir"]      # cartera de máximo IR (puede ser None)
```

A partir de estos objetos, podés:

* graficar la composición de cartera,
* hacer backtests vs SPY,
* exportar los pesos a un dashboard,
* integrarlo en otros sistemas de inversión.

---

## Ejemplo de uso completo

### Desde la UI

1. Ejecutar `streamlit run app.py`
2. En el sidebar, ingresar tickers: `AAPL, MSFT, GOOGL, AMZN, TSLA`
3. Seleccionar fechas: inicio `2020-01-01`, fin `2024-12-31`
4. Ajustar parámetros según preferencia
5. Hacer clic en "Calcular Carteras"
6. Explorar resultados en los diferentes tabs

### Desde código Python

Ver `example_usage.py` para un ejemplo completo con datos sintéticos.

## Estado del proyecto

Esta versión incluye:

* **API clara y reutilizable** para:
  * estimar FF5,
  * armar universos cuantitativos,
  * optimizar carteras.
* **Interfaz web interactiva** con Streamlit
* **Descarga automática de datos** desde yfinance y Ken French
* **Visualizaciones** de resultados y análisis
* **Tests completos** para todos los módulos

Es fácil de extender:

* nuevos factores,
* otras reglas de stock picking,
* constraints adicionales en la optimización,
* nuevas visualizaciones en la UI.
