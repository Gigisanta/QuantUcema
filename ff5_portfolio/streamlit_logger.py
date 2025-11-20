"""
Handler de logging personalizado para Streamlit.

Muestra logs en la UI de Streamlit en tiempo real con colores y formato estructurado.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Deque, Optional

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class StreamlitLogHandler(logging.Handler):
    """
    Handler de logging que muestra logs en Streamlit.
    
    Mantiene un buffer circular de logs y los muestra en un contenedor de Streamlit.
    """
    
    # Colores para cada nivel de log
    LEVEL_COLORS = {
        logging.DEBUG: "#888888",      # Gris
        logging.INFO: "#1f77b4",       # Azul
        logging.WARNING: "#ff7f0e",    # Amarillo/Naranja
        logging.ERROR: "#d62728",      # Rojo
        logging.CRITICAL: "#8b0000",   # Rojo oscuro
    }
    
    # Emojis para cada nivel
    LEVEL_EMOJIS = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨",
    }
    
    def __init__(self, max_logs: int = 500, container_key: str = "log_container"):
        """
        Inicializa el handler.
        
        Parameters
        ----------
        max_logs : int, default 500
            Número máximo de logs a mantener en el buffer.
        container_key : str, default "log_container"
            Clave para el contenedor de Streamlit en session_state.
        """
        super().__init__()
        self.max_logs = max_logs
        self.container_key = container_key
        self._log_buffer: Deque[dict] = deque(maxlen=max_logs)
        self._setup_session_state()
    
    def _setup_session_state(self) -> None:
        """Inicializa el buffer de logs en session_state si estamos en Streamlit."""
        if STREAMLIT_AVAILABLE:
            if self.container_key not in st.session_state:
                st.session_state[self.container_key] = deque(maxlen=self.max_logs)
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Procesa un log record y lo agrega al buffer.
        
        Parameters
        ----------
        record : logging.LogRecord
            Record de log a procesar.
        """
        try:
            # Formatear el mensaje
            msg = self.format(record)
            
            # Crear entrada de log estructurada
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                "level": record.levelno,
                "levelname": record.levelname,
                "module": record.module,
                "message": msg,
                "raw_message": record.getMessage(),
            }
            
            # Agregar al buffer
            self._log_buffer.append(log_entry)
            
            # Si estamos en Streamlit, también agregar a session_state
            if STREAMLIT_AVAILABLE:
                if self.container_key in st.session_state:
                    st.session_state[self.container_key].append(log_entry)
                    
        except Exception:
            # Si hay error al procesar el log, no fallar silenciosamente
            # pero tampoco romper la aplicación
            self.handleError(record)
    
    def get_logs(
        self, 
        level: Optional[int] = None, 
        max_entries: Optional[int] = None
    ) -> list[dict]:
        """
        Obtiene logs del buffer, opcionalmente filtrados por nivel.
        
        Parameters
        ----------
        level : int, optional
            Nivel mínimo de log a incluir (logging.DEBUG, logging.INFO, etc.).
        max_entries : int, optional
            Número máximo de entradas a retornar.
        
        Returns
        -------
        list[dict]
            Lista de entradas de log.
        """
        logs = list(self._log_buffer)
        
        # Filtrar por nivel si se especifica
        if level is not None:
            logs = [log for log in logs if log["level"] >= level]
        
        # Limitar número de entradas
        if max_entries is not None:
            logs = logs[-max_entries:]
        
        return logs
    
    def clear(self) -> None:
        """Limpia el buffer de logs."""
        self._log_buffer.clear()
        if STREAMLIT_AVAILABLE:
            if self.container_key in st.session_state:
                st.session_state[self.container_key].clear()
    
    def render_in_streamlit(
        self, 
        show_timestamp: bool = True,
        show_module: bool = True,
        max_display: int = 100
    ) -> None:
        """
        Renderiza los logs en Streamlit.
        
        Parameters
        ----------
        show_timestamp : bool, default True
            Mostrar timestamp en cada log.
        show_module : bool, default True
            Mostrar nombre del módulo en cada log.
        max_display : int, default 100
            Número máximo de logs a mostrar.
        """
        if not STREAMLIT_AVAILABLE:
            return
        
        logs = self.get_logs(max_entries=max_display)
        
        if not logs:
            st.caption("No hay logs aún.")
            return
        
        # Mostrar logs en orden inverso (más recientes primero)
        for log_entry in reversed(logs):
            level = log_entry["level"]
            color = self.LEVEL_COLORS.get(level, "#000000")
            emoji = self.LEVEL_EMOJIS.get(level, "•")
            
            # Construir mensaje formateado
            parts = [emoji]
            
            if show_timestamp:
                parts.append(f"`{log_entry['timestamp']}`")
            
            if show_module:
                parts.append(f"**{log_entry['module']}**")
            
            parts.append(log_entry["raw_message"])
            
            # Mostrar con color según nivel
            if level >= logging.ERROR:
                st.error(" ".join(parts))
            elif level >= logging.WARNING:
                st.warning(" ".join(parts))
            else:
                # Para INFO y DEBUG, usar markdown con color
                st.markdown(
                    f"<span style='color: {color}'>" + " ".join(parts) + "</span>",
                    unsafe_allow_html=True
                )

