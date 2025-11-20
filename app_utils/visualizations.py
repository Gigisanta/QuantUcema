"""
Visualizaciones para la aplicación Streamlit.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ff5_portfolio.pipeline import PipelineMetrics


def render_summary_tab(
    result: dict,
    portfolio_summary: dict,
    pipeline_metrics: PipelineMetrics,
) -> None:
    """Renderiza el tab de resumen."""
    st.header("Resumen de Resultados")
    
    col1, col2, col3, col4 = st.columns(4)
    
    sharpe_summary = portfolio_summary["sharpe"]
    
    with col1:
        st.metric("RF Anual", f"{result['rf_annual']:.2%}")
    with col2:
        st.metric("Retorno Esperado (Sharpe)", f"{sharpe_summary['expected_return']:.2%}")
    with col3:
        st.metric("Volatilidad (Sharpe)", f"{sharpe_summary['volatility']:.2%}")
    with col4:
        st.metric("Sharpe Ratio", f"{sharpe_summary['sharpe_ratio']:.2f}")
    
    st.markdown("---")
    
    # Información del universo
    st.subheader("Universo Gio")
    st.write(f"**Activos seleccionados:** {len(result['universe_gio'])}")
    st.write(f"**Activos en cartera Sharpe:** {sharpe_summary['weights_count']}")
    
    if portfolio_summary["ir"] is not None:
        ir_summary = portfolio_summary["ir"]
        st.write(f"**Activos en cartera IR:** {ir_summary['weights_count']}")
        
        st.markdown("---")
        st.subheader("Cartera Máx IR")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Retorno Esperado", f"{ir_summary['expected_return']:.2%}")
        with col2:
            st.metric("Volatilidad", f"{ir_summary['volatility']:.2%}")
        with col3:
            st.metric("Alpha vs SPY", f"{ir_summary['alpha']:.2%}")
        with col4:
            st.metric("Information Ratio", f"{ir_summary['information_ratio']:.2f}")

    if pipeline_metrics.stage_durations:
        metrics_df = pd.DataFrame(
            list(pipeline_metrics.stage_durations.items()),
            columns=["Etapa", "Segundos"],
        )
        metrics_df = metrics_df.sort_values("Segundos", ascending=False)
        metrics_df.loc[len(metrics_df)] = ["TOTAL", pipeline_metrics.total_seconds]
        st.markdown("---")
        st.subheader("Tiempos del Pipeline")
        st.dataframe(metrics_df, width='stretch', hide_index=True)


def render_universe_tab(result: dict) -> None:
    """Renderiza el tab del universo Gio."""
    st.header("Universo Gio - Activos Seleccionados")
    
    universe_df = result["universe_gio"].copy()
    universe_df["score_gio"] = universe_df["score_gio"].round(4)
    universe_df["sharpe_historical"] = universe_df["sharpe_historical"].round(4)
    universe_df["alpha_annual"] = universe_df["alpha_annual"].round(4)
    
    st.dataframe(universe_df, width='stretch')
    
    # Gráfico de scores
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(20, len(universe_df))
    top_universe = universe_df.nlargest(top_n, "score_gio")
    
    ax.barh(range(len(top_universe)), top_universe["score_gio"].values)
    ax.set_yticks(range(len(top_universe)))
    ax.set_yticklabels(top_universe.index)
    ax.set_xlabel("Score Gio")
    ax.set_title(f"Top {top_n} Activos por Score Gio")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)


def render_sharpe_tab(result: dict) -> None:
    """Renderiza el tab de cartera máximo Sharpe."""
    st.header("Cartera de Máximo Sharpe")

    # Filtrar pesos significativos
    weights_sharpe = result["weights_sharpe"]
    weights_display = weights_sharpe[weights_sharpe > 1e-4].sort_values(
        ascending=False
    )
    
    st.subheader("Pesos de la Cartera")
    weights_df = pd.DataFrame(
        {
            "Ticker": weights_display.index,
            "Peso": weights_display.values,
            "Peso (%)": (weights_display.values * 100).round(2),
        }
    )
    st.dataframe(weights_df, width='stretch', hide_index=True)
    
    # Gráfico de composición
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de barras
    ax1.barh(range(len(weights_display)), weights_display.values * 100)
    ax1.set_yticks(range(len(weights_display)))
    ax1.set_yticklabels(weights_display.index)
    ax1.set_xlabel("Peso (%)")
    ax1.set_title("Composición de Cartera (Máx Sharpe)")
    ax1.invert_yaxis()
    
    # Gráfico de pie (top 10)
    top_10 = weights_display.head(10)
    if len(weights_display) > 10:
        other_weight = weights_display.iloc[10:].sum()
        top_10 = pd.concat([top_10, pd.Series({"Otros": other_weight})])
    
    ax2.pie(
        top_10.values * 100,
        labels=top_10.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax2.set_title("Distribución Top 10")
    
    plt.tight_layout()
    st.pyplot(fig)


def render_ir_tab(result: dict) -> None:
    """Renderiza el tab de cartera máximo IR."""
    if result["weights_ir"] is not None:
        st.header("Cartera de Máximo Information Ratio")
        
        weights_ir = result["weights_ir"]
        weights_ir_display = weights_ir[weights_ir > 1e-4].sort_values(
            ascending=False
        )
        
        st.subheader("Pesos de la Cartera")
        weights_ir_df = pd.DataFrame(
            {
                "Ticker": weights_ir_display.index,
                "Peso": weights_ir_display.values,
                "Peso (%)": (weights_ir_display.values * 100).round(2),
            }
        )
        st.dataframe(weights_ir_df, width='stretch', hide_index=True)
        
        # Gráfico de composición
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gráfico de barras
        ax1.barh(range(len(weights_ir_display)), weights_ir_display.values * 100)
        ax1.set_yticks(range(len(weights_ir_display)))
        ax1.set_yticklabels(weights_ir_display.index)
        ax1.set_xlabel("Peso (%)")
        ax1.set_title("Composición de Cartera (Máx IR)")
        ax1.invert_yaxis()
        
        # Gráfico de pie
        top_10_ir = weights_ir_display.head(10)
        if len(weights_ir_display) > 10:
            other_weight_ir = weights_ir_display.iloc[10:].sum()
            top_10_ir = pd.concat(
                [top_10_ir, pd.Series({"Otros": other_weight_ir})]
            )
        
        ax2.pie(
            top_10_ir.values * 100,
            labels=top_10_ir.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title("Distribución Top 10")
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ La optimización IR no se ejecutó. Actívala en la configuración.")


def render_charts_tab(
    portfolio_summary: dict,
    benchmark_returns: pd.Series | None,
) -> None:
    """Renderiza el tab de gráficos."""
    st.header("Gráficos y Análisis")
    
    # Frontera eficiente (simplificada)
    st.subheader("Comparación de Carteras")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sharpe_summary = portfolio_summary["sharpe"]
    ax.scatter(
        sharpe_summary["volatility"] * 100,
        sharpe_summary["expected_return"] * 100,
        c="red",
        marker="*",
        s=300,
        label=f"Máx Sharpe ({sharpe_summary['sharpe_ratio']:.2f})",
        zorder=5,
    )
    
    # Punto de cartera IR si existe
    if portfolio_summary["ir"] is not None:
        ir_summary = portfolio_summary["ir"]
        ax.scatter(
            ir_summary["volatility"] * 100,
            ir_summary["expected_return"] * 100,
            c="orange",
            marker="*",
            s=300,
            label="Máx IR",
            zorder=5,
        )
    
    # Punto de SPY si existe
    if portfolio_summary["cumulative"]["benchmark"] is not None and benchmark_returns is not None:
        spy_ret_annual = benchmark_returns.mean() * 12
        spy_vol_annual = benchmark_returns.std() * np.sqrt(12)
        ax.scatter(
            spy_vol_annual * 100,
            spy_ret_annual * 100,
            c="gray",
            marker="o",
            s=200,
            label="SPY",
            zorder=5,
        )
    
    ax.set_xlabel("Volatilidad Anual (%)")
    ax.set_ylabel("Retorno Esperado Anual (%)")
    ax.set_title("Riesgo vs Retorno: Comparación de Carteras")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Evolución temporal (backtest simplificado)
    st.subheader("Evolución Temporal (Backtest)")
    
    cum_sharpe = portfolio_summary["cumulative"]["sharpe"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cum_sharpe.index, cum_sharpe.values, label="Cartera Máx Sharpe", linewidth=2)
    
    if portfolio_summary["cumulative"]["benchmark"] is not None:
        cum_spy = portfolio_summary["cumulative"]["benchmark"]
        ax.plot(cum_spy.index, cum_spy.values, label="SPY", linestyle="--", linewidth=2)
    
    if portfolio_summary["cumulative"]["ir"] is not None:
        cum_ir = portfolio_summary["cumulative"]["ir"]
        ax.plot(cum_ir.index, cum_ir.values, label="Cartera Máx IR", linewidth=2)
    
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor (base 100)")
    ax.set_title("Evolución Acumulada de Carteras")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

