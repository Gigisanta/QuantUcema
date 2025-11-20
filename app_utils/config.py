"""
Configuración del sidebar de la aplicación Streamlit.
"""

from datetime import date

import pandas as pd
import streamlit as st


def get_sidebar_config() -> dict:
    """
    Obtiene toda la configuración del sidebar.
    
    Returns
    -------
    dict
        Diccionario con parámetros de configuración:
        - start_date: fecha de inicio
        - end_date: fecha de fin
        - target_n_universe: tamaño objetivo del universo Gio
        - max_per_sector: máximo activos por sector
        - max_avg_corr: máxima correlación promedio
        - w_max: peso máximo por activo
        - include_ir: si incluir optimización IR
        - validate_tickers: si validar tickers antes de descargar
    """
    # Inicializar cache de S&P 500 en session state
    if 'sp500_tickers' not in st.session_state:
        st.session_state.sp500_tickers = None
        st.session_state.sp500_cache_time = None
    
    # Sidebar con configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Información sobre universo S&P 500
    with st.sidebar.expander("📊 Universo de Análisis", expanded=False):
        st.info(
            "La aplicación utiliza automáticamente todos los tickers del **S&P 500** "
            "para construir el universo de análisis. Puedes elegir validar tickers "
            "para filtrar automáticamente aquellos delisted o inválidos, "
            "asegurando datos de mejor calidad."
        )
    
    # Selector de fechas
    with st.sidebar.expander("📅 Rango de Fechas", expanded=True):
        col1, col2 = st.columns(2)
        start_date = col1.date_input(
            "Fecha inicio", 
            value=pd.to_datetime("2020-01-01").date(),
            key="start_date"
        )
        end_date_default = pd.to_datetime(date.today()).date()
        end_date = col2.date_input(
            "Fecha fin", 
            value=end_date_default,
            key="end_date"
        )
    
    # Parámetros del modelo
    with st.sidebar.expander("🎯 Parámetros del Modelo", expanded=True):
        target_n_universe = st.slider(
            "Tamaño universo Gio",
            min_value=5,
            max_value=100,
            value=30,
            help="Número objetivo de activos en el universo Gio",
            key="target_n_universe"
        )
        max_per_sector = st.slider(
            "Máx activos por sector",
            min_value=1,
            max_value=10,
            value=4,
            help="Máximo número de activos por sector",
            key="max_per_sector"
        )
        max_avg_corr = st.slider(
            "Máx correlación promedio",
            min_value=0.0,
            max_value=1.0,
            value=0.80,
            step=0.05,
            help="Máxima correlación promedio permitida entre activos",
            key="max_avg_corr"
        )
        w_max = st.slider(
            "Peso máximo por activo (%)",
            min_value=1,
            max_value=50,
            value=10,
            help="Peso máximo permitido por activo en la cartera",
            key="w_max"
        ) / 100.0
    
    # Opciones avanzadas
    with st.sidebar.expander("⚙️ Opciones Avanzadas", expanded=False):
        include_ir = st.checkbox(
            "Incluir optimización IR vs SPY",
            value=True,
            help="Optimizar cartera de máximo Information Ratio vs SPY",
            key="include_ir"
        )
        
        validate_tickers = st.checkbox(
            "Validar tickers antes de descargar",
            value=True,
            help="Verificar que los tickers existen en yfinance antes de descargar datos (más lento pero más confiable)",
            key="validate_tickers"
        )
    
    # Botón para ejecutar
    run_calculation = st.sidebar.button("🚀 Calcular Carteras", type="primary", use_container_width=True)
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "target_n_universe": target_n_universe,
        "max_per_sector": max_per_sector,
        "max_avg_corr": max_avg_corr,
        "w_max": w_max,
        "include_ir": include_ir,
        "validate_tickers": validate_tickers,
        "run_calculation": run_calculation,
    }

