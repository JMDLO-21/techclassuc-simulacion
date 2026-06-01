"""
visualizacion.py
================
Genera todas las gráficas del proyecto TechClassUC con Matplotlib.
Cada función guarda su figura en outputs/ y la cierra para liberar memoria.
"""

from __future__ import annotations

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sin GUI para entornos headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def _guardar(fig: plt.Figure, nombre: str) -> str:
    """Guarda la figura en outputs/ y la cierra."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ruta = os.path.join(OUTPUTS_DIR, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gráfica guardada: {ruta}")
    print(f"  ✔ Guardada: {nombre}")
    return ruta


def grafica_evolucion_temporal(
    evolucion: List[Tuple[float, int]],
    t_warm: float = 60.0,
    nombre: str = "01_evolucion_temporal.png",
) -> str:
    """
    Evolución temporal del número de clientes en el sistema (réplica representativa).

    Parameters
    ----------
    evolucion : List[(tiempo, n_clientes)]
    t_warm : float
        Período de calentamiento (se marca con línea vertical).
    nombre : str
        Nombre del archivo de salida.
    """
    if not evolucion:
        logger.warning("Evolución temporal vacía; gráfica omitida.")
        return ""

    tiempos = [e[0] for e in evolucion]
    n_clientes = [e[1] for e in evolucion]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(tiempos, n_clientes, where="post", color="#2196F3", linewidth=0.9, label="Clientes en sistema")
    ax.axvline(t_warm, color="#F44336", linestyle="--", linewidth=1.5, label=f"Fin warm-up ({t_warm:.0f} min)")
    ax.fill_betweenx([0, max(n_clientes, default=1) + 1], 0, t_warm, alpha=0.08, color="#F44336", label="Período transitorio")

    ax.set_xlabel("Tiempo de simulación (minutos)", fontsize=12)
    ax.set_ylabel("Clientes en el sistema", fontsize=12)
    ax.set_title("Evolución temporal del número de clientes en el sistema\nTechClassUC — Réplica representativa", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, nombre)


def grafica_histograma_wq(
    wq_lista: List[float],
    wq_analitico: Optional[float] = None,
    nombre: str = "02_histograma_wq.png",
) -> str:
    """
    Histograma de tiempos de espera Wq de los clientes atendidos.

    Parameters
    ----------
    wq_lista : List[float]
        Tiempos de espera individuales en minutos.
    wq_analitico : Optional[float]
        Valor teórico M/M/c de Wq (se dibuja como línea vertical).
    nombre : str
    """
    if not wq_lista:
        logger.warning("Lista Wq vacía; histograma omitido.")
        return ""

    arr = np.array(wq_lista)
    fig, ax = plt.subplots(figsize=(10, 5))

    n_bins = min(50, max(10, len(arr) // 10))
    ax.hist(arr, bins=n_bins, color="#4CAF50", edgecolor="white", alpha=0.85, density=True, label="Distribución Wq simulada")

    # Curva exponencial ajustada
    if arr.mean() > 0:
        x_fit = np.linspace(0, arr.max(), 300)
        rate = 1.0 / arr.mean()
        y_fit = rate * np.exp(-rate * x_fit)
        ax.plot(x_fit, y_fit, "k--", linewidth=1.5, label=f"Exp ajustada (media={arr.mean():.2f} min)")

    ax.axvline(arr.mean(), color="#FF5722", linewidth=2, label=f"Media simulada = {arr.mean():.2f} min")
    if wq_analitico is not None:
        ax.axvline(wq_analitico, color="#9C27B0", linewidth=2, linestyle=":", label=f"Wq analítico = {wq_analitico:.2f} min")

    ax.set_xlabel("Tiempo de espera en cola Wq (minutos)", fontsize=12)
    ax.set_ylabel("Densidad", fontsize=12)
    ax.set_title("Distribución de tiempos de espera Wq\nTechClassUC — Réplica representativa", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, nombre)


def grafica_wq_vs_servidores(
    cs: List[int],
    wq_simulado: List[float],
    wq_analitico: List[float],
    umbral: float = 10.0,
    nombre: str = "03_wq_vs_servidores.png",
) -> str:
    """
    Curva de Wq promedio vs. número de servidores c (curva de capacidad).

    Parameters
    ----------
    cs : List[int]
    wq_simulado : List[float]
    wq_analitico : List[float]
    umbral : float
        Línea horizontal de objetivo (default = 10 min).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cs, wq_simulado,  "o-", color="#2196F3", linewidth=2, markersize=7, label="Simulado (Montecarlo)")
    ax.plot(cs, wq_analitico, "s--", color="#FF9800", linewidth=1.8, markersize=6, label="Analítico M/M/c")
    ax.axhline(umbral, color="#F44336", linestyle=":", linewidth=1.5, label=f"Objetivo Wq < {umbral} min")

    ax.set_xlabel("Número de técnicos (c)", fontsize=12)
    ax.set_ylabel("Tiempo promedio en cola Wq (minutos)", fontsize=12)
    ax.set_title("Curva de capacidad: Wq vs. número de técnicos\nTechClassUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(cs)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _guardar(fig, nombre)


def grafica_rho_vs_lambda(
    lambdas: List[float],
    cs: List[int],
    mu: float,
    nombre: str = "04_rho_vs_lambda.png",
) -> str:
    """
    Gráfica de ρ vs. λ para distintos valores de c.

    Parameters
    ----------
    lambdas : List[float]
    cs : List[int]
    mu : float
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colores = plt.cm.tab10(np.linspace(0, 0.7, len(cs)))

    for idx, c in enumerate(cs):
        rhos = [lam / (c * mu) for lam in lambdas]
        rhos_validos = [r if r < 1.0 else np.nan for r in rhos]
        ax.plot(lambdas, rhos_validos, "o-", color=colores[idx], linewidth=2, markersize=6, label=f"c = {c}")

    ax.axhline(1.0, color="#F44336", linestyle="--", linewidth=1.5, label="ρ = 1 (límite estabilidad)")
    ax.fill_between(lambdas, 1.0, 1.2, alpha=0.1, color="#F44336", label="Zona inestable")

    ax.set_xlabel("Tasa de llegada λ (clientes/hora)", fontsize=12)
    ax.set_ylabel("Factor de utilización ρ", fontsize=12)
    ax.set_title("Factor de utilización ρ vs. λ para distintos c\nTechClassUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, 1.25)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, nombre)


def grafica_distribucion_medias(
    wq_todas: List[float],
    wq_medio: float,
    ic_95: Tuple[float, float],
    nombre: str = "05_distribucion_medias_replicas.png",
) -> str:
    """
    Distribución de las medias de Wq entre réplicas (verificación TCL).

    Parameters
    ----------
    wq_todas : List[float]
        Media de Wq por réplica.
    wq_medio : float
        Media global de todas las réplicas.
    ic_95 : Tuple[float, float]
        Intervalo de confianza 95%.
    """
    if len(wq_todas) < 3:
        return ""

    arr = np.array(wq_todas)
    fig, ax = plt.subplots(figsize=(9, 5))

    n_bins = max(7, len(arr) // 5)
    ax.hist(arr, bins=n_bins, color="#9C27B0", edgecolor="white", alpha=0.8, density=True, label="Distribución de medias Wq")

    # Curva normal ajustada
    x_fit = np.linspace(arr.min() - arr.std(), arr.max() + arr.std(), 300)
    y_norm = scipy_stats.norm.pdf(x_fit, arr.mean(), arr.std())
    ax.plot(x_fit, y_norm, "k-", linewidth=2, label="Normal ajustada (TCL)")

    ax.axvline(wq_medio, color="#F44336", linewidth=2, label=f"Media = {wq_medio:.2f} min")
    ax.axvline(ic_95[0], color="#FF9800", linestyle="--", linewidth=1.5, label=f"IC 95%: [{ic_95[0]:.2f}, {ic_95[1]:.2f}]")
    ax.axvline(ic_95[1], color="#FF9800", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Media de Wq por réplica (minutos)", fontsize=12)
    ax.set_ylabel("Densidad", fontsize=12)
    ax.set_title("Distribución de medias Wq entre réplicas\nTechClassUC — Verificación Teorema Central del Límite", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, nombre)


def heatmap_wq(
    tabla_wq: np.ndarray,
    lambdas: List[float],
    cs: List[int],
    umbral: float = 10.0,
    nombre: str = "06_heatmap_wq.png",
) -> str:
    """
    Heatmap de Wq en función de λ y c.

    Parameters
    ----------
    tabla_wq : np.ndarray (n_cs x n_lambdas)
    lambdas : List[float]
    cs : List[int]
    umbral : float
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    # Enmascarar valores NaN (configuraciones inestables)
    tabla_masked = np.ma.masked_invalid(tabla_wq)
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color="#444444")

    im = ax.imshow(tabla_masked, aspect="auto", cmap=cmap, origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wq promedio (minutos)", fontsize=11)

    # Anotaciones
    for i in range(len(cs)):
        for j in range(len(lambdas)):
            val = tabla_wq[i, j]
            if not np.isnan(val):
                color_txt = "white" if val > umbral * 1.5 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=color_txt, fontweight="bold")

    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:.0f}" for l in lambdas], fontsize=9)
    ax.set_yticks(range(len(cs)))
    ax.set_yticklabels([str(c) for c in cs], fontsize=9)
    ax.set_xlabel("Tasa de llegada λ (clientes/hora)", fontsize=12)
    ax.set_ylabel("Número de técnicos c", fontsize=12)
    ax.set_title(f"Heatmap Wq promedio (min) — Objetivo: Wq < {umbral} min\nTechClassUC", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _guardar(fig, nombre)


def heatmap_rho(
    tabla_rho: np.ndarray,
    lambdas: List[float],
    cs: List[int],
    nombre: str = "07_heatmap_rho.png",
) -> str:
    """Heatmap de ρ en función de λ y c."""
    fig, ax = plt.subplots(figsize=(11, 6))
    tabla_masked = np.ma.masked_invalid(tabla_rho)
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color="#444444")
    im = ax.imshow(tabla_masked, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Utilización ρ", fontsize=11)

    for i in range(len(cs)):
        for j in range(len(lambdas)):
            val = tabla_rho[i, j]
            if not np.isnan(val):
                color_txt = "white" if val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color_txt)

    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:.0f}" for l in lambdas], fontsize=9)
    ax.set_yticks(range(len(cs)))
    ax.set_yticklabels([str(c) for c in cs], fontsize=9)
    ax.set_xlabel("Tasa de llegada λ (clientes/hora)", fontsize=12)
    ax.set_ylabel("Número de técnicos c", fontsize=12)
    ax.set_title("Heatmap de utilización ρ — Análisis de sensibilidad\nTechClassUC", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _guardar(fig, nombre)


def grafica_optimizacion(
    historial: List[Dict],
    umbral: float = 10.0,
    c_optimo: Optional[int] = None,
    nombre: str = "08_optimizacion_servidores.png",
) -> str:
    """
    Gráfica de Wq vs c para la optimización automática.

    Parameters
    ----------
    historial : List[dict]
        Lista de {'c': int, 'wq': float, 'estable': bool}.
    umbral : float
    c_optimo : Optional[int]
    """
    estables = [h for h in historial if h.get("estable", False)]
    if not estables:
        return ""

    cs_plot = [h["c"] for h in estables]
    wqs_plot = [h["wq"] for h in estables]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cs_plot, wqs_plot, "D-", color="#2196F3", linewidth=2, markersize=8, label="Wq simulado")
    ax.axhline(umbral, color="#F44336", linestyle="--", linewidth=1.8, label=f"Objetivo: Wq < {umbral} min")

    if c_optimo is not None:
        idx_opt = next((i for i, h in enumerate(estables) if h["c"] == c_optimo), None)
        if idx_opt is not None:
            ax.scatter([cs_plot[idx_opt]], [wqs_plot[idx_opt]], s=200, zorder=5,
                       color="#4CAF50", marker="*", label=f"c óptimo = {c_optimo}")

    ax.set_xlabel("Número de técnicos (c)", fontsize=12)
    ax.set_ylabel("Tiempo promedio en cola Wq (minutos)", fontsize=12)
    ax.set_title("Optimización automática: mínimo c para Wq < objetivo\nTechClassUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(cs_plot)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, nombre)


def grafica_no_estacionario(
    evolucion: List[Tuple[float, int]],
    nombre: str = "09_llegadas_no_estacionarias.png",
) -> str:
    """
    Evolución temporal con llegadas no estacionarias, mostrando los tramos horarios.
    """
    if not evolucion:
        return ""

    from simulacion_des import TASA_POR_HORA

    tiempos = [e[0] for e in evolucion]
    n_clientes = [e[1] for e in evolucion]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(tiempos, n_clientes, where="post", color="#009688", linewidth=0.9, label="Clientes en sistema")

    colores_tramo = ["#FFF9C4", "#FFF176", "#FFEE58", "#FDD835", "#F9A825"]
    for idx, (t_ini, tasa) in enumerate(TASA_POR_HORA):
        t_fin = TASA_POR_HORA[idx + 1][0] if idx + 1 < len(TASA_POR_HORA) else max(tiempos, default=480)
        ax.axvspan(t_ini, t_fin, alpha=0.25, color=colores_tramo[idx % len(colores_tramo)],
                   label=f"λ={tasa:.0f}/h ({t_ini:.0f}–{t_fin:.0f} min)")

    ax.set_xlabel("Tiempo de simulación (minutos)", fontsize=12)
    ax.set_ylabel("Clientes en el sistema", fontsize=12)
    ax.set_title("Llegadas no estacionarias: variación de λ por tramo horario\nTechClassUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, nombre)


def grafica_reneging(
    historial_clientes: List[Any],
    nombre: str = "10_reneging_abandonos.png",
) -> str:
    """
    Gráfica de proporción de abandonos de cola (reneging) por tiempo de llegada.
    """
    from cliente import Cliente

    if not historial_clientes:
        return ""

    tiempos_llegada_aten = [c.t_llegada for c in historial_clientes if not c.abandono]
    tiempos_llegada_aban = [c.t_llegada for c in historial_clientes if c.abandono]

    if not tiempos_llegada_aban:
        logger.info("Sin abandonos registrados; gráfica de reneging omitida.")
        return ""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(tiempos_llegada_aten, bins=30, alpha=0.7, color="#4CAF50", label=f"Atendidos ({len(tiempos_llegada_aten)})")
    ax.hist(tiempos_llegada_aban, bins=30, alpha=0.7, color="#F44336", label=f"Abandonaron ({len(tiempos_llegada_aban)})")
    ax.set_xlabel("Tiempo de llegada (minutos)", fontsize=12)
    ax.set_ylabel("Número de clientes", fontsize=12)
    ax.set_title("Distribución de clientes atendidos vs abandonos (Reneging)\nTechClassUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, nombre)


def grafica_warmup_welch(
    serie_prom: List[float],
    idx_warmup: int,
    nombre: str = "11_welch_warmup.png",
) -> str:
    """
    Visualiza la detección del período de calentamiento por el método de Welch.
    """
    if not serie_prom:
        return ""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(serie_prom, color="#607D8B", linewidth=1.2, alpha=0.8, label="Wq promedio (Welch)")
    ax.axvline(idx_warmup, color="#F44336", linewidth=2, linestyle="--",
               label=f"Inicio estado estacionario (idx={idx_warmup})")
    ax.fill_betweenx([0, max(serie_prom, default=1) * 1.1], 0, idx_warmup,
                     alpha=0.1, color="#F44336", label="Período transitorio")
    ax.set_xlabel("Número de cliente (orden de llegada)", fontsize=12)
    ax.set_ylabel("Wq promedio acumulado (minutos)", fontsize=12)
    ax.set_title("Detección automática del período de calentamiento — Método de Welch\nTechClassUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, nombre)
