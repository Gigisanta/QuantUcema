"""
Aplicación Streamlit para ff5_portfolio.
UI interactiva para construir carteras usando modelo Fama-French 5 factores.
"""

import logging
import warnings

import streamlit as st

from app_utils import config, data_loading, portfolio_summary, visualizations
from ff5_portfolio import build_gio_portfolios_with_metrics
from ff5_portfolio.logging_config import setup_logging
from ff5_portfolio.streamlit_logger import StreamlitLogHandler

# Configurar logging
logger = setup_logging(level=logging.INFO, module_name=__name__)

warnings.filterwarnings("ignore")

# Configuración de página
st.set_page_config(
    page_title="FF5 Portfolio Builder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título principal
st.title("📊 FF5 Portfolio Builder")
st.markdown(
    "Construcción de carteras optimizadas usando el modelo Fama-French 5 factores"
)

# Obtener configuración del sidebar
sidebar_config = config.get_sidebar_config()

# Sección de logs en sidebar
with st.sidebar.expander("📋 Logs del Sistema", expanded=False):
    log_level_filter = st.selectbox(
        "Filtrar por nivel",
        ["Todos", "DEBUG", "INFO", "WARNING", "ERROR"],
        key="log_level_filter",
    )
    
    # Obtener handler de Streamlit si existe
    streamlit_handler = None
    for handler in logger.handlers:
        if isinstance(handler, StreamlitLogHandler):
            streamlit_handler = handler
            break
    
    if streamlit_handler:
        # Mostrar logs filtrados
        level_map = {
            "Todos": None,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        selected_level = level_map.get(log_level_filter, None)
        logs = streamlit_handler.get_logs(level=selected_level, max_entries=100)
        
        if logs:
            # Mostrar logs en orden inverso (más recientes primero)
            log_container = st.container()
            with log_container:
                for log_entry in reversed(logs[-50:]):  # Mostrar últimos 50
                    level = log_entry["level"]
                    emoji = streamlit_handler.LEVEL_EMOJIS.get(level, "•")
                    color = streamlit_handler.LEVEL_COLORS.get(level, "#000000")
                    
                    log_text = f"{emoji} `{log_entry['timestamp']}` **{log_entry['module']}** {log_entry['raw_message']}"
                    
                    if level >= logging.ERROR:
                        st.error(log_text)
                    elif level >= logging.WARNING:
                        st.warning(log_text)
                    else:
                        st.markdown(
                            f"<span style='color: {color}'>{log_text}</span>",
                            unsafe_allow_html=True
                        )
        else:
            st.caption("No hay logs aún.")
        
        if st.button("🗑️ Limpiar Logs", use_container_width=True):
            streamlit_handler.clear()
            st.rerun()
    else:
        st.caption("Logs no disponibles (handler no configurado)")

# Contenedor principal
if sidebar_config["run_calculation"]:
    # Progress bar general
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Etapa 1: Cargar datos (0-60%)
        status_text.info("📥 Cargando datos...")
        progress_bar.progress(10)
        
        data = data_loading.load_all_data(
            start_date=sidebar_config["start_date"],
            end_date=sidebar_config["end_date"],
            validate_tickers=sidebar_config["validate_tickers"],
            include_ir=sidebar_config["include_ir"],
            progress_callback=lambda msg: status_text.info(msg),
        )
        
        progress_bar.progress(60)
        
        # Mostrar resumen de datos cargados
        if len(data["skipped_tickers"]) > 0:
            skipped_count = len(data["skipped_tickers"])
            skipped_sample = ', '.join(data["skipped_tickers"][:10])
            if skipped_count > 10:
                skipped_sample += f" ... y {skipped_count - 10} más"
            st.warning(
                f"⚠️ {skipped_count} tickers no se pudieron descargar: {skipped_sample}. "
                "Esto puede deberse a símbolos delisted o problemas temporales de conexión."
            )
        
        st.success(
            f"✓ {data['returns_m'].shape[1]} acciones descargadas exitosamente "
            f"({data['returns_m'].shape[0]} meses de datos)"
        )
        
        # Etapa 2: Ejecutar pipeline (60-90%)
        status_text.info("⚙️ Calculando carteras optimizadas...")
        progress_bar.progress(70)
        
        result, pipeline_metrics = build_gio_portfolios_with_metrics(
            returns_m=data["returns_m"],
            ff5_factors=data["ff5_factors"],
            meta=data["meta"],
            benchmark_returns=data["benchmark_returns"] if sidebar_config["include_ir"] else None,
            target_n_universe=sidebar_config["target_n_universe"],
            max_per_sector=sidebar_config["max_per_sector"],
            max_avg_corr=sidebar_config["max_avg_corr"],
            w_min=0.0,
            w_max=sidebar_config["w_max"],
        )
        
        progress_bar.progress(90)
        
        # Validar resultados
        required_keys = ["weights_sharpe", "universe_gio", "exp_ret_annual", "cov_annual", "returns_universe"]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            logger.error("Faltan campos requeridos en el resultado: %s", missing_keys)
            st.error(f"❌ Error interno: Faltan campos requeridos en el resultado: {missing_keys}")
            st.stop()

        if result["weights_sharpe"] is None or result["weights_sharpe"].empty:
            logger.error("No se pudo calcular la cartera de máximo Sharpe")
            st.error("❌ Error: No se pudo calcular la cartera de máximo Sharpe")
            st.stop()

        if result["universe_gio"].empty:
            logger.error("El universo Gio está vacío")
            st.error("❌ Error: El universo Gio está vacío")
            st.stop()

        # Etapa 3: Preparar resumen (90-100%)
        status_text.info("📊 Preparando resultados...")
        progress_bar.progress(95)
        
        portfolio_summary_data = portfolio_summary.prepare_portfolio_summary(
            result, data["benchmark_returns"] if sidebar_config["include_ir"] else None
        )
        
        progress_bar.progress(100)
        status_text.success("✅ Cálculo completado exitosamente!")
        
        # Ocultar progress bar después de un momento
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        # Mostrar resultados en tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Resumen", "🎯 Universo Gio", "📈 Máx Sharpe", "🎯 Máx IR", "📉 Gráficos"]
        )
        
        with tab1:
            visualizations.render_summary_tab(
                result, portfolio_summary_data, pipeline_metrics
            )
        
        with tab2:
            visualizations.render_universe_tab(result)
        
        with tab3:
            visualizations.render_sharpe_tab(result)
        
        with tab4:
            visualizations.render_ir_tab(result)
        
        with tab5:
            visualizations.render_charts_tab(
                portfolio_summary_data,
                data["benchmark_returns"] if sidebar_config["include_ir"] else None,
            )
    
    except ValueError as e:
        error_msg = str(e)
        logger.error("Error de validación: %s", error_msg, exc_info=True)
        st.error(f"❌ Error de validación: {error_msg}")
        if "No se obtuvieron datos" in error_msg:
            st.info(
                "💡 **Sugerencias:**\n"
                "- Verifica tu conexión a internet\n"
                "- El rango de fechas puede ser muy amplio o muy reciente\n"
                "- yfinance puede estar experimentando problemas temporales\n"
                "- Intenta reducir el rango de fechas o esperar unos minutos"
            )
        st.exception(e)
    except RuntimeError as e:
        error_msg = str(e)
        logger.error("Error de runtime: %s", error_msg, exc_info=True)
        st.error(f"❌ Error crítico: {error_msg}")
        st.info(
            "💡 **Sugerencias para resolver el problema:**\n"
            "- Verifica tu conexión a internet\n"
            "- Intenta nuevamente en unos minutos\n"
            "- Verifica que puedas acceder a Wikipedia en tu navegador\n"
            "- Si usas proxy/firewall, verifica que permita acceso a Wikipedia"
        )
        st.exception(e)
    except Exception as e:
        error_msg = str(e)
        logger.error("Error durante el cálculo: %s", error_msg, exc_info=True)
        st.error(f"❌ Error durante el cálculo: {error_msg}")
        st.info(
            "💡 Si el error persiste, verifica:\n"
            "- Conexión a internet estable\n"
            "- Que el rango de fechas sea válido\n"
            "- Que yfinance esté funcionando correctamente"
        )
        st.exception(e)

else:
    # Mensaje inicial
    st.info(
        """
    👋 **Bienvenido al FF5 Portfolio Builder**
    
    Esta aplicación te permite construir carteras optimizadas usando el modelo Fama-French 5 factores.
    
    **Instrucciones:**
    1. Selecciona el rango de fechas para el análisis
    2. Ajusta los parámetros del modelo según tus preferencias
    3. Haz clic en "Calcular Carteras" para ver los resultados
    
    **Características:**
    - **Análisis automático del S&P 500**: Utiliza todos los tickers del S&P 500 automáticamente
    - Descarga automática de datos desde yfinance
    - Construcción de universo Gio con stock picking cuantitativo
    - Optimización de máximo Sharpe ratio
    - Optimización de máximo Information Ratio vs SPY
    - Visualizaciones interactivas de resultados
    - **Logging en tiempo real**: Ve qué está pasando en cada momento
    
    **Datos reales y actualizados:**
    - La aplicación SOLO usa datos reales descargados desde fuentes externas
    - Lista del S&P 500: descargada desde Wikipedia (actualizada automáticamente)
    - Precios históricos: descargados desde yfinance (datos reales de mercado)
    - Factores FF5: descargados desde Ken French (datos académicos oficiales)
    - Metadatos: obtenidos desde yfinance (sectores reales)
    
    **Nota:** Este proceso requiere conexión a internet y puede tomar varios minutos 
    dependiendo de la cantidad de tickers y velocidad de conexión.
    """
    )
