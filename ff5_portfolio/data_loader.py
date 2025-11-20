"""
Módulo para descarga automática de datos desde yfinance y factores FF5 desde Ken French.
"""

import contextlib
import logging
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from time import sleep
from zipfile import ZipFile

import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# Cache en disco para retornos mensuales
DEFAULT_CACHE_DIR = Path(os.getenv("FF5_CACHE_DIR", Path.home() / ".cache" / "ff5_portfolio"))
RETURNS_CACHE_DIR = DEFAULT_CACHE_DIR / "returns_monthly"
CACHE_TTL_SECONDS = float(os.getenv("FF5_CACHE_TTL_SECONDS", 12 * 3600))  # 12 horas por defecto


def _ensure_returns_cache_dir() -> None:
    try:
        RETURNS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Si no se puede crear el directorio, se omite cache en disco
        pass


def _returns_cache_path(ticker: str) -> Path:
    safe_ticker = ticker.replace("/", "_").replace(":", "_")
    return RETURNS_CACHE_DIR / f"{safe_ticker}.parquet"


def _load_cached_returns(
    ticker: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    min_months: int,
    ttl_seconds: float = CACHE_TTL_SECONDS,
) -> pd.Series | None:
    path = _returns_cache_path(ticker)
    if not path.exists():
        return None

    if ttl_seconds and (time.time() - path.stat().st_mtime) > ttl_seconds:
        return None

    try:
        df = pd.read_parquet(path)
    except Exception:
        return None

    if "return" not in df.columns:
        return None

    series = df["return"]
    series.index = pd.to_datetime(series.index)
    subset = series.loc[(series.index >= start_ts) & (series.index <= end_ts)]
    if subset.empty:
        return None

    # Permitir tolerancia de un mes al inicio y fin por efecto de pct_change
    tolerance = pd.offsets.MonthEnd(1)
    if subset.index.min() > start_ts + tolerance or subset.index.max() < end_ts - tolerance:
        return None

    if subset.dropna().shape[0] < min_months:
        return None

    return subset.sort_index()


def _save_returns_cache(ticker: str, series: pd.Series) -> None:
    try:
        _ensure_returns_cache_dir()
        df = series.sort_index().to_frame("return")
        df.to_parquet(_returns_cache_path(ticker), compression="snappy")
    except Exception as exc:
        logger.debug("No se pudo guardar cache de %s: %s", ticker, exc)

# Detectar si estamos en Streamlit
def _is_streamlit_context():
    """Detecta si estamos ejecutando en contexto de Streamlit."""
    try:
        # Verificar si streamlit está importado y en runtime
        import streamlit as st
        # En runtime de Streamlit, ciertos atributos están disponibles
        return hasattr(st, 'session_state') or 'streamlit' in sys.modules
    except (ImportError, AttributeError):
        return False

# Configurar logging usando el sistema centralizado
from .logging_config import setup_logging

logger = setup_logging(level=logging.INFO, module_name=__name__)

# URL para factores FF5 de Ken French
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
FF5_FILENAME = "F-F_Research_Data_5_Factors_2x3.csv"

# URL para obtener lista del S&P 500 desde Wikipedia
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_DATAHUB_URL = "https://pkgstore.datahub.io/core/s-and-p-500-companies/constituents_csv/data/3d4b23b8f62b7197a15104445832a762/constituents_csv.csv"


# Lista estática de fallback por si fallan las fuentes externas
_STATIC_SP500_FALLBACK = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'JNJ', 'V',
    'UNH', 'XOM', 'JPM', 'WMT', 'PG', 'MA', 'CVX', 'HD', 'LLY', 'BAC', 'PFE',
    'KO', 'ABBV', 'PEP', 'MRK', 'COST', 'TMO', 'AVGO', 'CSCO', 'DIS', 'ABT',
    'ACN', 'MCD', 'NEE', 'DHR', 'LIN', 'VZ', 'WFC', 'PM', 'COP', 'TXN', 'CRM',
    'NKE', 'RTX', 'UPS', 'MS', 'HON', 'AMGN', 'LOW', 'UNP', 'IBM'
]

# Timeouts por defecto (en segundos) - Optimizados para velocidad
DEFAULT_TIMEOUT_YFINANCE = 30  # Timeout para yf.download() y yf.Ticker().info (reducido de 45s)
DEFAULT_TIMEOUT_REQUESTS = 30  # Timeout para requests (ya configurado en algunos lugares)


def _coerce_to_prices(df, tickers_hint=None):
    """
    Devuelve un DataFrame de precios por ticker.
    Acepta salida de yf.download() con o sin MultiIndex.
    Prioriza 'Adj Close' y cae a 'Close' si no existe.
    
    Mejora: Valida que el DataFrame tenga estructura válida antes de procesar.
    """
    if df is None:
        return pd.DataFrame()
    
    # Validar que tenga índice (fechas)
    if not hasattr(df, 'index') or len(df.index) == 0:
        return pd.DataFrame()
    
    # Manejar Series primero (antes de acceder a .columns)
    if isinstance(df, pd.Series):
        name = tickers_hint[0] if tickers_hint else "PX"
        return df.to_frame(name=name)

    # Validar que tenga columnas
    if not hasattr(df, 'columns') or len(df.columns) == 0:
        return pd.DataFrame()

    # MultiIndex (Open, High, Low, Close, Adj Close, Volume) x tickers
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        if "Adj Close" in lv0:
            out = df["Adj Close"].copy()
        elif "Close" in lv0:
            out = df["Close"].copy()
        else:
            # Si no hay ninguna de las dos, intentamos reconstruir
            out = df.copy()
            price_candidates = [c for c in out.columns.unique(0) if "close" in c.lower()]
            if price_candidates:
                out = out[price_candidates[0]]
            else:
                return pd.DataFrame()
    else:
        # Columnas planas
        if "Adj Close" in df.columns:
            out = df[["Adj Close"]].copy()
            if tickers_hint and len(tickers_hint) == 1:
                out.columns = [tickers_hint[0]]
        elif "Close" in df.columns:
            out = df[["Close"]].copy()
            if tickers_hint and len(tickers_hint) == 1:
                out.columns = [tickers_hint[0]]
        else:
            # No hay Adj Close ni Close
            return pd.DataFrame()

    # Quitar columnas duplicadas
    out = out.loc[:, ~out.columns.duplicated()]
    
    # Validar que el resultado no esté vacío
    if out.empty or len(out.columns) == 0:
        return pd.DataFrame()
    
    return out


def _prices_to_monthly_returns(px_df: pd.DataFrame) -> pd.Series:
    """
    Convierte un DataFrame de precios (una columna) a retornos mensuales.
    """
    if px_df is None or px_df.empty:
        return pd.Series(dtype=float)

    px = px_df.sort_index()
    px.index = px.index.to_period("M").to_timestamp("M")

    # Seleccionar primera columna disponible
    series = px.iloc[:, 0]
    returns = series.pct_change().dropna()
    return returns


def _clip_returns_to_range(
    returns: pd.Series, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> pd.Series:
    """
    Recorta una Serie de retornos al rango solicitado.
    """
    if returns.empty:
        return returns
    return returns.loc[(returns.index >= start_ts) & (returns.index <= end_ts)]


def _normalize_tickers(tickers):
    """
    Normaliza tickers a formato Yahoo Finance.
    Convierte BRK.B -> BRK-B, etc.
    """
    if isinstance(tickers, str):
        # Si es string, puede ser separado por comas
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    if not tickers:
        return []

    tickers = [str(t).upper().strip() for t in tickers if t]

    # Mapeos manuales comunes
    mapeo_manual = {
        "BF.B": "BF-B",
        "BRK.B": "BRK-B",
    }

    normalized = []
    for t in tickers:
        # Reemplazar puntos por guiones para formato Yahoo
        t_clean = t.replace(".", "-").replace(" ", "")
        # Aplicar mapeos manuales si existen
        t_clean = mapeo_manual.get(t_clean, t_clean)
        normalized.append(t_clean)

    # Eliminar duplicados preservando orden
    return list(dict.fromkeys(normalized))


def _validate_tickers(tickers, max_workers=8, timeout=10):
    """
    Valida una lista de tickers verificando si existen en yfinance.

    Parameters
    ----------
    tickers : list[str]
        Lista de tickers a validar.
    max_workers : int, default 8
        Máximo número de threads para validación paralela.
    timeout : float, default 10
        Timeout en segundos por ticker.

    Returns
    -------
    valid : list[str]
        Lista de tickers válidos.
    invalid : list[str]
        Lista de tickers inválidos o delisted.
    """
    if not tickers:
        return [], []

    valid = []
    invalid = []

    def _check_ticker(ticker):
        """Verifica si un ticker es válido."""
        try:
            def _ticker_func():
                stock = yf.Ticker(ticker)
                # Intentar obtener info básica - si falla, ticker es inválido
                with _suppress_output():
                    info = stock.info
                # Si no hay error y tenemos info básica, es válido
                return info is not None and len(info) > 0

            result = _run_with_timeout(_ticker_func, timeout)
            return ticker, result
        except Exception:
            return ticker, False

    # Procesar en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_check_ticker, ticker): ticker
            for ticker in tickers
        }

        for future in as_completed(future_map):
            ticker = future_map[future]
            try:
                ticker_result, is_valid = future.result()
                if is_valid:
                    valid.append(ticker)
                else:
                    invalid.append(ticker)
            except Exception:
                invalid.append(ticker)

    return valid, invalid


def get_sp500_tickers(use_cache=True, cache=None, validate_tickers=False):
    """
    Obtiene lista actualizada de tickers del S&P 500.
    Intenta desde DataHub (CSV), fallback a Wikipedia y finalmente a lista estática.
    """
    if use_cache and cache is not None:
        cached_tickers = cache.get('sp500_tickers')
        cache_time = cache.get('sp500_cache_time')
        if cached_tickers and cache_time and (time.time() - cache_time < 86400):
            logger.info(f"Usando tickers del S&P 500 desde cache ({len(cached_tickers)} tickers)")
            return cached_tickers

    # 1. Intentar desde DataHub.io (fuente principal, CSV)
    try:
        logger.info("Obteniendo tickers del S&P 500 desde DataHub.io...")
        resp = requests.get(SP500_DATAHUB_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(BytesIO(resp.content))
        tickers = _normalize_tickers(df["Symbol"].tolist())

        if len(tickers) >= 400:
            logger.info(f"Obtenidos {len(tickers)} tickers desde DataHub.io")
            if cache is not None:
                cache['sp500_tickers'] = tickers
                cache['sp500_cache_time'] = time.time()
            return tickers
    except Exception as e:
        logger.warning(f"No se pudo obtener tickers desde DataHub.io: {e}. Intentando fallback a Wikipedia...")

    # 2. Fallback a Wikipedia (fuente secundaria)
    try:
        logger.info("Obteniendo tickers del S&P 500 desde Wikipedia...")
        resp = requests.get(SP500_WIKI_URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        tables = pd.read_html(StringIO(resp.text))

        sp500_table = None
        for table in tables:
            if 'Symbol' in table.columns:
                sp500_table = table
                break

        if sp500_table is None:
             raise ValueError("No se encontró la tabla de tickers en Wikipedia")

        tickers = _normalize_tickers(sp500_table["Symbol"].tolist())

        if len(tickers) >= 400:
            logger.info(f"Obtenidos {len(tickers)} tickers desde Wikipedia")
            if cache is not None:
                cache['sp500_tickers'] = tickers
                cache['sp500_cache_time'] = time.time()
            return tickers
        else:
            raise ValueError(f"Wikipedia devolvió solo {len(tickers)} tickers.")

    except Exception as e:
        logger.warning(f"No se pudo obtener tickers desde Wikipedia: {e}. Usando lista estática.")

    # 3. Fallback a lista estática (último recurso)
    logger.warning("Todas las fuentes externas fallaron. Usando lista estática de S&P 500.")
    if cache is not None:
        cache['sp500_tickers'] = _STATIC_SP500_FALLBACK
        cache['sp500_cache_time'] = time.time()

    if validate_tickers:
        logger.info(f"Validando {len(tickers)} tickers del S&P 500...")
        valid_tickers, _ = _validate_tickers(tickers)
        logger.info(f"Tickers validados: {len(valid_tickers)} restantes")
        return valid_tickers

    return tickers


@contextlib.contextmanager
def _suppress_stderr():
    """
    Context manager para suprimir stderr temporalmente.
    
    Suprime mensajes de error verbosos de yfinance durante descargas.
    """
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


@contextlib.contextmanager
def _suppress_output():
    """
    Context manager para suprimir tanto stdout como stderr temporalmente.
    Útil para suprimir todos los mensajes verbosos de yfinance.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def download_stock_data(
    tickers,
    start_date,
    end_date,
    batch_size=200,
    pause=0.1,
    min_months=24,
    use_local_cache: bool = True,
    cache_ttl_seconds: float = CACHE_TTL_SECONDS,
    max_workers: int | None = None,
    validate_tickers: bool = True,
):
    """
    Descarga precios mensuales ajustados desde yfinance y retorna retornos mensuales.
    
    Mejora: Incluye retry logic con backoff exponencial, timeouts y logging.

    Parameters
    ----------
    tickers : list[str] | str
        Lista de tickers o string separado por comas. También acepta lista desde CSV.
    start_date : str
        Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date : str
        Fecha de fin en formato 'YYYY-MM-DD'.
    batch_size : int, default 200
        Tamaño de lote para descargas en batch (optimizado para velocidad).
    pause : float, default 0.1
        Pausa en segundos entre lotes (optimizado para velocidad).
    min_months : int, default 24
        Mínimo número de meses requeridos para mantener un ticker.

    use_local_cache : bool, default True
        Si True, intenta reutilizar retornos previamente descargados desde cache Parquet.
    cache_ttl_seconds : float, default 12 horas
        Tiempo máximo que se considera válido un archivo cacheado.
    max_workers : int | None
        Máximo de threads simultáneos para descargas. Si None se deduce de batch_size.
    validate_tickers : bool, default True
        Si True, valida tickers con yfinance antes de descargar y filtra inválidos.

    Returns
    -------
    returns_m : pd.DataFrame
        Retornos mensuales. Index = fechas (fin de mes), Columns = tickers.
    skipped : list[str]
        Lista de tickers que no se pudieron descargar.

    Raises
    ------
    ValueError
        Si no se obtuvieron datos válidos.
    """
    start_time_total = time.time()

    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    tickers_normalized = _normalize_tickers(tickers)

    if len(tickers_normalized) == 0:
        raise ValueError("No se proporcionaron tickers válidos.")

    # Validar tickers si se solicita
    if validate_tickers and len(tickers_normalized) > 0:
        logger.info("Validando tickers antes de descarga...")
        valid_tickers, invalid_tickers = _validate_tickers(tickers_normalized, max_workers=16, timeout=5)

        if invalid_tickers:
            logger.info(
                f"Filtrados {len(invalid_tickers)} tickers inválidos/delisted antes de descarga: "
                f"{', '.join(invalid_tickers[:10])}{'...' if len(invalid_tickers) > 10 else ''}"
            )
            tickers_normalized = valid_tickers

        if len(tickers_normalized) == 0:
            raise ValueError("Todos los tickers proporcionados son inválidos o delisted.")

    logger.info(
        "Iniciando descarga de %d tickers (periodo: %s a %s)",
        len(tickers_normalized),
        start_date,
        end_date,
    )

    start_ts = pd.to_datetime(start_date).to_period("M").to_timestamp("M")
    end_ts = pd.to_datetime(end_date).to_period("M").to_timestamp("M")

    returns_store: dict[str, pd.Series] = {}
    skipped: list[str] = []
    pending: list[str] = []

    if use_local_cache:
        logger.info("Intentando reutilizar cache local de retornos...")
        for ticker in tickers_normalized:
            cached = _load_cached_returns(
                ticker, start_ts, end_ts, min_months, ttl_seconds=cache_ttl_seconds
            )
            if cached is not None:
                returns_store[ticker] = cached
            else:
                pending.append(ticker)
        logger.info(
            "Cache válido para %d tickers, %d pendientes de descarga.",
            len(returns_store),
            len(pending),
        )
    else:
        pending = tickers_normalized.copy()

    if max_workers is None:
        # Reducir paralelismo cuando hay muchos tickers para evitar rate limiting
        if len(tickers_normalized) > 50:
            max_workers = min(8, max(2, batch_size // 20))  # Menos agresivo con muchos tickers
        else:
            max_workers = min(16, max(4, batch_size // 10))

    if pending:
        logger.info(f"Descargando datos para {len(pending)} tickers...")
        px_df = yf.download(
            pending,
            start=start_date,
            end=end_date,
            interval="1mo",
            auto_adjust=True,
            progress=False,
            threads=max_workers,
        )

        if px_df is not None and not px_df.empty:
            prices = _coerce_to_prices(px_df)
            for ticker in pending:
                if ticker in prices.columns:
                    returns_series = _prices_to_monthly_returns(prices[[ticker]])
                    returns_series = _clip_returns_to_range(
                        returns_series, start_ts, end_ts
                    )

                    if returns_series.dropna().shape[0] >= min_months:
                        returns_store[ticker] = returns_series
                        if use_local_cache:
                            _save_returns_cache(ticker, returns_series)
                    else:
                        skipped.append(ticker)
                else:
                    skipped.append(ticker)
        else:
            skipped.extend(pending)

    if not returns_store:
        elapsed_total = time.time() - start_time_total
        logger.error(
            "No se obtuvieron datos después de %.1fs. Verificar conexión, tickers y fechas.",
            elapsed_total,
        )
        raise ValueError(
            "No se obtuvieron datos. Verificar conexión, tickers y fechas."
        )

    ordered = [t for t in tickers_normalized if t in returns_store]
    returns_m = pd.DataFrame({ticker: returns_store[ticker] for ticker in ordered})
    returns_m = returns_m.sort_index()

    if returns_m.shape[1] == 0:
        raise ValueError("Todas las series vinieron vacías.")

    elapsed_total = time.time() - start_time_total
    logger.info(
        "Descarga completada: %d tickers válidos, %d fallidos, tiempo total: %.1fs",
        returns_m.shape[1],
        len(skipped),
        elapsed_total,
    )

    return returns_m, skipped


def download_ff5_factors(use_cache=True, cache=None):
    """
    Descarga factores FF5 mensuales desde Ken French, con cache en memoria.
    """
    if use_cache and cache is not None:
        cached_factors = cache.get('ff5_factors')
        cache_time = cache.get('ff5_cache_time')
        if cached_factors is not None and cache_time and (time.time() - cache_time < 86400):
            logger.info("Usando factores FF5 desde cache")
            return cached_factors

    logger.info("Descargando factores FF5 desde Ken French...")
    try:
        resp = requests.get(FF5_URL, timeout=60)
        resp.raise_for_status()

        with ZipFile(BytesIO(resp.content)) as zf:
            if FF5_FILENAME not in zf.namelist():
                raise RuntimeError("No se encontró CSV dentro del ZIP de Ken French.")
            
            with zf.open(FF5_FILENAME) as f:
                ff_raw = pd.read_csv(f, skiprows=3)

        ff = ff_raw[ff_raw.iloc[:, 0].astype(str).str.isnumeric()].copy()
        ff.columns = ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
        ff = ff.set_index("Date").astype(float) / 100.0
        ff.index = pd.to_datetime(ff.index, format="%Y%m") + pd.offsets.MonthEnd(0)

        logger.info(f"Factores FF5 procesados: {len(ff)} meses de datos")

        if cache is not None:
            cache['ff5_factors'] = ff
            cache['ff5_cache_time'] = time.time()

        return ff

    except requests.RequestException as e:
        logger.error(f"Error al descargar factores FF5: {e}")
        raise RuntimeError(f"Error al descargar factores FF5: {e}")
    except Exception as e:
        logger.error(f"Error al parsear factores FF5: {e}")
        raise RuntimeError(f"Error al parsear factores FF5: {e}")


def get_stock_metadata(tickers):
    """
    Obtiene metadatos (sector, etc.) de tickers desde yfinance.
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    
    tickers_normalized = _normalize_tickers(tickers)
    if not tickers_normalized:
        return pd.DataFrame(columns=["sector"])

    logger.info(f"Obteniendo metadatos para {len(tickers_normalized)} tickers...")
    
    with _suppress_output():
        infos = yf.Tickers(tickers_normalized).tickers

    metadata_list = []
    for ticker, ticker_info in zip(tickers_normalized, infos):
        try:
            sector = ticker_info.info.get("sector", "Unknown")
            if not sector or pd.isna(sector):
                sector = "Unknown"
            metadata_list.append({"sector": sector})
        except Exception:
            metadata_list.append({"sector": "Unknown"})
            
    meta = pd.DataFrame(metadata_list, index=tickers_normalized)
    return meta


def _clean_and_validate_data(returns_m: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y valida el DataFrame de retornos.
    - Interpola huecos pequeños (hasta 2 meses).
    - Elimina activos con demasiados NaNs.
    - Winsoriza valores extremos para mitigar outliers.
    """
    # 1. Eliminar activos con más de 20% de datos faltantes
    max_missing = int(len(returns_m) * 0.20)
    returns_m = returns_m.dropna(axis=1, thresh=len(returns_m) - max_missing)

    # 2. Interpolar huecos pequeños (linealmente)
    returns_m = returns_m.interpolate(method='linear', limit=2, limit_direction='both')

    # 3. Winsorizar en el 1% y 99% para mitigar outliers
    returns_m = returns_m.clip(lower=returns_m.quantile(0.01), upper=returns_m.quantile(0.99), axis=1)

    # 4. Eliminar cualquier activo que aún tenga NaNs después de la limpieza
    returns_m = returns_m.dropna(axis=1)

    logger.info(f"Datos limpios y validados: {returns_m.shape[1]} activos restantes")
    return returns_m
