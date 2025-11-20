"""
Configuración centralizada de logging para ff5_portfolio.

Detecta automáticamente si está corriendo en Streamlit o en un script normal
y configura los handlers apropiados.
"""

import logging
import sys
from typing import Optional

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except (ImportError, AttributeError):
    _STREAMLIT_AVAILABLE = False


def _is_streamlit_context() -> bool:
    """
    Detecta si estamos en un contexto de Streamlit.
    
    Returns
    -------
    bool
        True si estamos en Streamlit, False en caso contrario.
    """
    if not _STREAMLIT_AVAILABLE:
        return False
    
    try:
        # Verificar si streamlit está disponible y si estamos en una sesión activa
        # Esto es más confiable que verificar el stack
        return hasattr(st, 'session_state')
    except Exception:
        return False


def setup_logging(
    level: int = logging.INFO,
    module_name: Optional[str] = None,
    use_streamlit_handler: bool = True
) -> logging.Logger:
    """
    Configura logging para un módulo específico.
    
    Si estamos en Streamlit y use_streamlit_handler=True, agrega el handler
    de Streamlit. También agrega un StreamHandler estándar para scripts normales.
    
    Parameters
    ----------
    level : int, default logging.INFO
        Nivel de logging (logging.DEBUG, logging.INFO, etc.).
    module_name : str, optional
        Nombre del módulo. Si None, usa el nombre del módulo que llama.
    use_streamlit_handler : bool, default True
        Si True y estamos en Streamlit, agrega el handler de Streamlit.
    
    Returns
    -------
    logging.Logger
        Logger configurado.
    """
    if module_name is None:
        # Obtener nombre del módulo que llama
        import inspect
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            module_name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            module_name = 'unknown'
    
    logger = logging.getLogger(module_name)
    
    # Evitar configurar múltiples veces
    if logger.handlers:
        logger.setLevel(level)
        return logger
    
    logger.setLevel(level)
    
    # Formato para logs
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para scripts normales (consola)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler de Streamlit si estamos en Streamlit
    if use_streamlit_handler and _is_streamlit_context():
        try:
            from .streamlit_logger import StreamlitLogHandler
            
            streamlit_handler = StreamlitLogHandler()
            streamlit_handler.setLevel(level)
            # Formato más simple para Streamlit (sin timestamp duplicado)
            streamlit_formatter = logging.Formatter(
                fmt='%(message)s',
                datefmt='%H:%M:%S'
            )
            streamlit_handler.setFormatter(streamlit_formatter)
            logger.addHandler(streamlit_handler)
        except Exception:
            # Si falla agregar handler de Streamlit, continuar sin él
            pass
    
    # Evitar propagación al root logger para evitar duplicados
    logger.propagate = False
    
    return logger


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Obtiene un logger configurado para el módulo especificado.
    
    Si el logger no existe, lo crea y configura automáticamente.
    
    Parameters
    ----------
    module_name : str, optional
        Nombre del módulo. Si None, usa el nombre del módulo que llama.
    
    Returns
    -------
    logging.Logger
        Logger configurado.
    """
    if module_name is None:
        import inspect
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            module_name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            module_name = 'unknown'
    
    logger = logging.getLogger(module_name)
    
    # Si el logger no tiene handlers, configurarlo
    if not logger.handlers:
        setup_logging(module_name=module_name)
    
    return logger

