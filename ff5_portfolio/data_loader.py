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
    Obtiene lista actualizada de tickers del S&P 500 desde Wikipedia.

    Esta función SOLO usa datos reales descargados desde Wikipedia.
    No hay fallbacks con datos estáticos o hardcodeados.

    Parameters
    ----------
    use_cache : bool, default True
        Si True, usa cache si está disponible.
    cache : dict, optional
        Diccionario de cache (típicamente session state de Streamlit).
        Debe tener clave 'sp500_tickers' y 'sp500_cache_time'.
    validate_tickers : bool, default False
        Si True, valida cada ticker con yfinance y filtra los inválidos.

    Returns
    -------
    tickers : list[str]
        Lista de tickers del S&P 500 normalizados.

    Raises
    ------
    RuntimeError
        Si no se puede obtener la lista del S&P 500 desde Wikipedia.
        Incluye mensaje claro sobre cómo resolver problemas de conexión.
    """
    start_time = time.time()
    
    # Verificar cache si está disponible
    # IMPORTANTE: Solo usar cache si tiene al menos 400 tickers (validación de datos reales)
    if use_cache and cache is not None:
        cached_tickers = cache.get('sp500_tickers')
        cache_time = cache.get('sp500_cache_time')
        if cached_tickers is not None and cache_time is not None:
            # Validar que el cache contiene datos reales (al menos 400 tickers)
            if len(cached_tickers) >= 400:
                # Cache válido por 24 horas
                if time.time() - cache_time < 86400:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Usando tickers del S&P 500 desde cache "
                        f"({len(cached_tickers)} tickers, {elapsed:.2f}s)"
                    )
                    return cached_tickers
            else:
                # Cache inválido (probablemente de lista estática antigua), limpiarlo
                logger.warning(
                    f"Cache contiene solo {len(cached_tickers)} tickers, "
                    "probablemente de lista estática antigua. Limpiando cache y descargando datos reales."
                )
                cache.pop('sp500_tickers', None)
                cache.pop('sp500_cache_time', None)
    
    logger.info("Obteniendo lista de tickers del S&P 500 desde Wikipedia...")
    
    # Intentar descargar desde Wikipedia con múltiples reintentos
    max_retries = 3
    base_delay = 2.0
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    last_error = None
    for attempt in range(max_retries):
        try:
            logger.debug(f"Intentando descargar desde Wikipedia (intento {attempt + 1}/{max_retries})...")
            
            # Aumentar timeout en reintentos
            timeout = 10 + (attempt * 5)
            resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=timeout)
            resp.raise_for_status()
            
            # Parsear HTML con pandas
            from io import StringIO
            tables = pd.read_html(StringIO(resp.text))
            
            if not tables:
                raise ValueError("No se encontraron tablas en la página de Wikipedia")
            
            # Buscar tabla que contenga los tickers del S&P 500
            sp500_table = None
            for table in tables:
                # Buscar tabla que tenga columna de símbolos
                if 'Symbol' in table.columns or 'Ticker symbol' in table.columns:
                    sp500_table = table
                    break
            
            # Si no encontramos por nombre de columna, buscar en todas las tablas
            if sp500_table is None:
                for table in tables:
                    # Verificar si la tabla tiene un tamaño razonable (S&P 500 tiene ~500 filas)
                    if len(table) >= 400:
                        # Verificar si alguna columna parece contener símbolos de tickers
                        for col in table.columns:
                            sample_values = table[col].dropna().head(10).astype(str)
                            # Los tickers suelen ser strings cortos en mayúsculas
                            if all(len(v) <= 6 and v.isupper() for v in sample_values if v):
                                sp500_table = table
                                symbol_col = col
                                break
                        if sp500_table is not None:
                            break
            
            if sp500_table is None and len(tables) > 0:
                # Último recurso: usar la tabla más grande
                sp500_table = max(tables, key=len)
                if len(sp500_table) < 400:
                    raise ValueError(
                        f"La tabla más grande encontrada tiene solo {len(sp500_table)} filas, "
                        "esperado al menos 400 para S&P 500"
                    )
            
            if sp500_table is None:
                raise ValueError("No se pudo identificar la tabla del S&P 500 en Wikipedia")
            
            # Identificar columna de símbolos
            symbol_col = None
            for col in sp500_table.columns:
                col_lower = str(col).lower()
                if 'symbol' in col_lower or 'ticker' in col_lower:
                    symbol_col = col
                    break
            
            if symbol_col is None:
                # Intentar primera columna si no encontramos por nombre
                symbol_col = sp500_table.columns[0]
            
            # Extraer tickers
            tickers_raw = sp500_table[symbol_col].dropna().astype(str).tolist()
            
            # Normalizar tickers
            tickers = _normalize_tickers(tickers_raw)
            
            # Validar que se obtuvieron tickers razonables
            if len(tickers) < 400:
                raise ValueError(
                    f"Se obtuvieron solo {len(tickers)} tickers desde Wikipedia, "
                    "esperado al menos 400 para S&P 500. "
                    "Esto puede indicar un problema con el parsing de la página."
                )
            
            # Validar que los tickers tienen formato razonable (no vacíos, no demasiado largos)
            invalid_tickers = [t for t in tickers if not t or len(t) > 10]
            if invalid_tickers:
                logger.warning(
                    f"Se encontraron {len(invalid_tickers)} tickers con formato inválido, "
                    "serán filtrados"
                )
                tickers = [t for t in tickers if t and len(t) <= 10]
            
            if len(tickers) < 400:
                raise ValueError(
                    f"Después de filtrar tickers inválidos, quedan solo {len(tickers)} tickers, "
                    "esperado al menos 400 para S&P 500"
                )
            
            # Validar tickers con yfinance si se solicita
            if validate_tickers:
                logger.info("Validando tickers del S&P 500 con yfinance...")
                valid_tickers, invalid_tickers = _validate_tickers(tickers, max_workers=16, timeout=5)
                
                if invalid_tickers:
                    logger.info(
                        f"Filtrados {len(invalid_tickers)} tickers inválidos/delisted: "
                        f"{', '.join(invalid_tickers[:10])}{'...' if len(invalid_tickers) > 10 else ''}"
                    )
                    tickers = valid_tickers
                
                if len(tickers) < 300:  # Permitir menos si se validó
                    logger.warning(
                        f"Después de validación, quedan {len(tickers)} tickers válidos. "
                        "Esto puede indicar problemas de conexión con yfinance."
                    )
            
            elapsed = time.time() - start_time
            logger.info(
                f"Tickers del S&P 500 obtenidos desde Wikipedia: {len(tickers)} tickers "
                f"({elapsed:.2f}s)"
            )
            
            # Guardar en cache si está disponible
            if cache is not None:
                cache['sp500_tickers'] = tickers
                cache['sp500_cache_time'] = time.time()
            
            return tickers
            
        except requests.RequestException as e:
            last_error = e
            elapsed = time.time() - start_time
            logger.warning(
                f"Error de red obteniendo tickers desde Wikipedia (intento {attempt + 1}/{max_retries}, "
                f"elapsed: {elapsed:.1f}s): {str(e)}"
            )
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Reintentando en {delay:.1f} segundos...")
                sleep(delay)
            else:
                break
        except Exception as e:
            last_error = e
            elapsed = time.time() - start_time
            logger.warning(
                f"Error obteniendo tickers desde Wikipedia (intento {attempt + 1}/{max_retries}, "
                f"elapsed: {elapsed:.1f}s): {str(e)}"
            )
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Reintentando en {delay:.1f} segundos...")
                sleep(delay)
            else:
                break
    
    # Si llegamos aquí, todos los intentos fallaron
    elapsed_total = time.time() - start_time
    error_msg = (
        f"No se pudo obtener la lista de tickers del S&P 500 desde Wikipedia "
        f"después de {max_retries} intentos ({elapsed_total:.1f}s).\n\n"
        f"Error: {str(last_error) if last_error else 'Desconocido'}\n\n"
        "Posibles causas:\n"
        "- Problemas de conexión a internet\n"
        "- Wikipedia temporalmente no disponible\n"
        "- Firewall o proxy bloqueando el acceso\n\n"
        "Sugerencias:\n"
        "- Verifica tu conexión a internet\n"
        "- Intenta nuevamente en unos minutos\n"
        "- Verifica que puedas acceder a https://en.wikipedia.org/wiki/List_of_S%26P_500_companies\n"
        "- Si el problema persiste, verifica configuración de firewall/proxy"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


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
    Nota: No suprime nuestros logs (logging module usa handlers separados).
    """
    # Guardar referencias originales
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Solo suprimir si estamos seguros de que podemos restaurar
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        # Restaurar siempre, incluso si hubo excepciones
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception:
            pass  # Si la restauración falla, continuar


class YFinanceTimeoutError(Exception):
    """Excepción para timeouts en operaciones de yfinance."""
    pass


def _run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """
    Ejecuta una función con timeout usando threading.

    Parameters
    ----------
    func : callable
        Función a ejecutar.
    timeout_seconds : float
        Timeout en segundos.
    *args, **kwargs
        Argumentos para la función.

    Returns
    -------
    result
        Resultado de la función.

    Raises
    ------
    YFinanceTimeoutError
        Si la función no completa en el tiempo especificado.
    """
    result = [None]
    exception = [None]

    def target():
        try:
            # Ejecutar función sin manipular logging
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread aún corriendo = timeout
        # Usar print en lugar de logger para evitar problemas de streams
        print(f"WARNING: Timeout después de {timeout_seconds}s en {func.__name__}", file=sys.stderr)
        raise YFinanceTimeoutError(f"Operación excedió timeout de {timeout_seconds}s")

    if exception[0] is not None:
        raise exception[0]

    return result[0]


def _download_ticker_with_retry(
    ticker, start_date, end_date, max_retries=3, base_delay=1.0, timeout=DEFAULT_TIMEOUT_YFINANCE
):
    """
    Descarga un ticker individual con retry logic, backoff exponencial y timeout.
    
    Suprime mensajes de error verbosos de yfinance pero loggea nuestros mensajes.
    
    Parameters
    ----------
    ticker : str
        Ticker a descargar.
    start_date : str
        Fecha de inicio.
    end_date : str
        Fecha de fin.
    max_retries : int, default 3
        Número máximo de reintentos.
    base_delay : float, default 1.0
        Delay base en segundos para backoff exponencial.
    timeout : float, default DEFAULT_TIMEOUT_YFINANCE
        Timeout en segundos para cada intento.
    
    Returns
    -------
    px_df : pd.DataFrame | None
        DataFrame de precios o None si falla.
    error_msg : str | None
        Mensaje de error si falla, None si tiene éxito.
    """
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            def _download_func():
                # Configurar yfinance para ser silencioso
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Deshabilitar logging temporalmente para yfinance
                    yf_logging = logging.getLogger("yfinance")
                    original_level = yf_logging.level
                    yf_logging.setLevel(logging.ERROR)

                    try:
                        return yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            interval="1mo",
                            auto_adjust=True,
                            progress=False,
                            threads=False,
                        )
                    finally:
                        yf_logging.setLevel(original_level)
            
            # Ejecutar con timeout
            try:
                tmp = _run_with_timeout(_download_func, timeout)
            except YFinanceTimeoutError:
                elapsed = time.time() - start_time
                logger.warning(
                    f"Timeout descargando {ticker} (intento {attempt + 1}/{max_retries}, "
                    f"elapsed: {elapsed:.1f}s)"
                )
                if attempt < max_retries - 1:
                    sleep(base_delay * (2 ** attempt))
                    continue
                return None, f"Timeout después de {timeout}s"
            
            elapsed = time.time() - start_time

            # Validar que la respuesta no esté vacía
            if tmp is None or (hasattr(tmp, 'empty') and tmp.empty):
                try:
                    logger.debug(f"{ticker}: Respuesta vacía (intento {attempt + 1}, elapsed: {elapsed:.1f}s)")
                except Exception:
                    pass  # Logging falló, continuar silenciosamente
                if attempt < max_retries - 1:
                    sleep(base_delay * (2 ** attempt))
                    continue
                return None, "Respuesta vacía de yfinance"

            px1 = _coerce_to_prices(tmp, [ticker])

            if px1 is not None and not px1.empty and len(px1.columns) > 0:
                # Validar que tenga datos válidos (no solo NaN)
                if px1.iloc[:, 0].notna().sum() > 0:
                    try:
                        logger.debug(f"{ticker}: Descargado exitosamente ({elapsed:.1f}s)")
                    except Exception:
                        pass  # Logging falló, continuar silenciosamente
                    return px1, None

            try:
                logger.debug(
                    f"{ticker}: Sin datos válidos (intento {attempt + 1}, elapsed: {elapsed:.1f}s)"
                )
            except Exception:
                pass  # Logging falló, continuar silenciosamente
            if attempt < max_retries - 1:
                sleep(base_delay * (2 ** attempt))
                continue

            return None, f"Sin datos válidos después de {max_retries} intentos"

        except YFinanceTimeoutError:
            # Ya manejado arriba
            if attempt < max_retries - 1:
                sleep(base_delay * (2 ** attempt))
                continue
            return None, f"Timeout después de {timeout}s"
        except Exception as e:
            error_msg = str(e)
            elapsed = time.time() - start_time if 'start_time' in locals() else 0

            # Intentar logging pero no fallar si hay problemas con streams
            try:
                # Detectar tipo de error para logging apropiado
                if "YFTzMissingError" in str(type(e)) or "possibly delisted" in error_msg:
                    logger.debug(f"{ticker}: Posiblemente delisted (intento {attempt + 1}, elapsed: {elapsed:.1f}s)")
                elif "Expecting value" in error_msg or "timezone" in error_msg.lower():
                    logger.debug(f"{ticker}: Error de API (intento {attempt + 1}, elapsed: {elapsed:.1f}s)")
                else:
                    logger.warning(
                        f"{ticker}: Error en intento {attempt + 1}/{max_retries} "
                        f"(elapsed: {elapsed:.1f}s): {error_msg}"
                    )
            except Exception:
                pass  # Logging falló, continuar silenciosamente

            # Detectar errores específicos de yfinance
            if ("YFTzMissingError" in str(type(e)) or
                "possibly delisted" in error_msg or
                "Expecting value" in error_msg or
                "timezone" in error_msg.lower()):
                # Errores de API - no reintentar
                return None, f"Error de API: {error_msg}"
            else:
                # Error no relacionado con API, no reintentar
                return None, f"Error: {error_msg}"
    
    return None, f"Falló después de {max_retries} intentos"


def _download_returns_parallel(
    tickers: list[str],
    start_date: str,
    end_date: str,
    min_months: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    max_workers: int,
) -> tuple[dict[str, pd.Series], list[str]]:
    """
    Descarga retornos mensuales en paralelo usando ThreadPoolExecutor.
    """
    results: dict[str, pd.Series] = {}
    skipped: list[str] = []

    if not tickers:
        return results, skipped

    worker_count = max(1, min(max_workers, 32))
    total_tickers = len(tickers)
    
    logger.debug(
        "Iniciando descarga paralela de %d tickers con %d workers",
        total_tickers,
        worker_count,
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_download_ticker_with_retry, ticker, start_date, end_date): ticker
            for ticker in tickers
        }

        completed = 0
        for future in as_completed(future_map):
            ticker = future_map[future]
            completed += 1
            
            # Log progreso cada 10% o cada 10 tickers, lo que sea menor
            if completed % max(1, min(10, total_tickers // 10)) == 0:
                logger.info(
                    "Progreso descarga: %d/%d tickers procesados (%.1f%%)",
                    completed,
                    total_tickers,
                    100.0 * completed / total_tickers,
                )
            
            try:
                px_df, error_msg = future.result()
            except Exception as exc:
                try:
                    logger.warning("%s: excepción durante descarga paralela: %s", ticker, exc)
                except Exception:
                    pass  # Logging falló, continuar silenciosamente
                skipped.append(ticker)
                continue

            if px_df is None:
                skipped.append(ticker)
                try:
                    logger.debug("%s: descarga fallida (%s)", ticker, error_msg or "sin datos")
                except Exception:
                    pass  # Logging falló, continuar silenciosamente
                continue

            returns_series = _prices_to_monthly_returns(px_df)
            returns_series = _clip_returns_to_range(returns_series, start_ts, end_ts)

            if returns_series.dropna().shape[0] < min_months:
                skipped.append(ticker)
                try:
                    logger.debug("%s: menos de %d meses válidos.", ticker, min_months)
                except Exception:
                    pass  # Logging falló, continuar silenciosamente
                continue

            results[ticker] = returns_series
            logger.debug("%s: descargado exitosamente", ticker)

    logger.info(
        "Descarga paralela completada: %d exitosos, %d fallidos",
        len(results),
        len(skipped),
    )
    return results, skipped


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
        total_batches = (len(pending) + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, len(pending), batch_size), 1):
            lote = pending[start : start + batch_size]
            batch_start_time = time.time()
            logger.info(
                "Procesando lote %d/%d (%d tickers, desde %s hasta %s)",
                batch_idx,
                total_batches,
                len(lote),
                lote[0],
                lote[-1],
            )

            downloaded, skipped_batch = _download_returns_parallel(
                tickers=lote,
                start_date=start_date,
                end_date=end_date,
                min_months=min_months,
                start_ts=start_ts,
                end_ts=end_ts,
                max_workers=min(max_workers, len(lote)),
            )

            returns_store.update(downloaded)
            skipped.extend(skipped_batch)

            if use_local_cache:
                for ticker, series in downloaded.items():
                    _save_returns_cache(ticker, series)

            elapsed_batch = time.time() - batch_start_time
            logger.info(
                "Lote %d completado: %d descargados, %d fallidos (%.1fs)",
                batch_idx,
                len(downloaded),
                len(skipped_batch),
                elapsed_batch,
            )
            sleep(pause)

    if not returns_store:
        elapsed_total = time.time() - start_time_total
        logger.error(
            "No se obtuvieron datos después de %.1fs. Verificar conexión, tickers y fechas.",
            elapsed_total,
        )
        raise ValueError("No se obtuvieron datos. Verificar conexión, tickers y fechas.")

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


def download_ff5_factors():
    """
    Descarga factores FF5 mensuales desde Ken French.

    Returns
    -------
    ff5_factors : pd.DataFrame
        Factores FF5 mensuales. Index = fechas (fin de mes),
        Columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"].
        Valores en decimales.

    Raises
    ------
    RuntimeError
        Si no se puede descargar o parsear el archivo.
    """
    start_time = time.time()
    logger.info("Descargando factores FF5 desde Ken French...")
    
    try:
        resp = requests.get(FF5_URL, timeout=60)
        resp.raise_for_status()
        elapsed = time.time() - start_time
        logger.info(f"Factores FF5 descargados ({elapsed:.2f}s)")
    except requests.RequestException as e:
        elapsed = time.time() - start_time
        logger.error(f"Error al descargar factores FF5 después de {elapsed:.2f}s: {e}")
        raise RuntimeError(f"Error al descargar factores FF5: {e}")
    
    try:
        logger.debug("Parseando archivo ZIP de factores FF5...")
        with ZipFile(BytesIO(resp.content)) as zf:
            # Buscar el archivo CSV dentro del ZIP
            if FF5_FILENAME in zf.namelist():
                nombre = FF5_FILENAME
            else:
                # Fallback: tomar el primer .csv que encuentre
                candidatos = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not candidatos:
                    raise RuntimeError("No se encontró CSV dentro del ZIP de Ken French.")
                nombre = candidatos[0]
            
            with zf.open(nombre) as f:
                ff_raw = pd.read_csv(f, skiprows=3)
        logger.debug("CSV parseado exitosamente")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error al parsear factores FF5 después de {elapsed:.2f}s: {e}")
        raise RuntimeError(f"Error al parsear factores FF5: {e}")
    
    # Limpiar tabla: quedar solo con filas de fechas YYYYMM
    ff = ff_raw[ff_raw.iloc[:, 0].astype(str).str.isnumeric()].copy()
    ff.columns = ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    ff = ff.set_index("Date").astype(float) / 100.0  # Convertir a decimales
    
    # Convertir índice a fechas
    ff.index = pd.to_datetime(ff.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    
    elapsed_total = time.time() - start_time
    logger.info(
        f"Factores FF5 procesados: {len(ff)} meses de datos "
        f"(periodo: {ff.index[0].date()} a {ff.index[-1].date()}, "
        f"tiempo total: {elapsed_total:.2f}s)"
    )
    
    return ff


def get_stock_metadata(tickers, batch_size=100, pause=0.05):
    """
    Obtiene metadatos (sector, etc.) de tickers desde yfinance.
    
    Procesa en batches con timeout y logging para evitar bloqueos.

    Parameters
    ----------
    tickers : list[str] | str
        Lista de tickers o string separado por comas.
    batch_size : int, default 100
        Tamaño de lote para procesar metadatos (optimizado para velocidad).
    pause : float, default 0.05
        Pausa en segundos entre batches para evitar rate limiting (optimizado).

    Returns
    -------
    meta : pd.DataFrame
        Metadatos indexados por ticker. Columnas: ['sector'].
        Tickers no encontrados tendrán sector 'Unknown'.
    """
    start_time = time.time()
    
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    
    tickers_normalized = _normalize_tickers(tickers)
    
    if len(tickers_normalized) == 0:
        logger.warning("No se proporcionaron tickers para metadatos.")
        return pd.DataFrame(columns=["sector"])
    
    logger.info(f"Obteniendo metadatos para {len(tickers_normalized)} tickers...")
    
    metadata_list = []
    total_batches = (len(tickers_normalized) + batch_size - 1) // batch_size
    
    for batch_idx, i in enumerate(range(0, len(tickers_normalized), batch_size), 1):
        batch = tickers_normalized[i:i + batch_size]
        batch_start_time = time.time()
        
        logger.info(
            f"Procesando metadatos lote {batch_idx}/{total_batches} "
            f"({len(batch)} tickers)"
        )
        
        for ticker_idx, ticker in enumerate(batch, 1):
            try:
                def _get_info_func():
                    stock = yf.Ticker(ticker)
                    # Suprimir output verboso de yfinance
                    with _suppress_output():
                        return stock.info
                
                # Ejecutar con timeout
                try:
                    info = _run_with_timeout(_get_info_func, DEFAULT_TIMEOUT_YFINANCE)
                except YFinanceTimeoutError:
                    logger.warning(f"{ticker}: Timeout obteniendo metadatos")
                    metadata_list.append({"sector": "Unknown"})
                    continue
                
                sector = info.get("sector", "Unknown")
                if not sector or pd.isna(sector):
                    sector = "Unknown"
                
                metadata_list.append({"sector": sector})
                
            except Exception as e:
                logger.debug(f"{ticker}: Error obteniendo metadatos: {str(e)}")
                # Si falla, usar sector Unknown
                metadata_list.append({"sector": "Unknown"})
            
            # Log progreso cada 10 tickers o al final del batch
            if ticker_idx % 10 == 0 or ticker_idx == len(batch):
                elapsed = time.time() - batch_start_time
                logger.debug(
                    f"Lote {batch_idx}: {ticker_idx}/{len(batch)} tickers procesados "
                    f"({elapsed:.1f}s)"
                )
        
        # Pausa entre batches
        if batch_idx < total_batches:
            sleep(pause)
        
        elapsed = time.time() - batch_start_time
        logger.info(f"Lote {batch_idx} completado ({elapsed:.1f}s)")
    
    elapsed_total = time.time() - start_time
    logger.info(
        f"Metadatos obtenidos para {len(metadata_list)} tickers "
        f"(tiempo total: {elapsed_total:.1f}s)"
    )
    
    meta = pd.DataFrame(metadata_list, index=tickers_normalized)
    return meta

