"""
sensibilidad.py
===============
Análisis de sensibilidad del sistema TechClassUC variando λ y c.
Incluye:
- Barrido de parámetros para construir tablas de calor (heatmaps).
- Detección automática del período de calentamiento (método de Welch).
- Optimización automática: encontrar el mínimo c tal que Wq < umbral.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from montecarlo import correr_replicas
from analitico import calcular_metricas_mmc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Método de Welch para detección automática del período de calentamiento
# ---------------------------------------------------------------------------

def detectar_warmup_welch(
    serie: List[float],
    ventana: int = 5,
) -> int:
    """
    Detecta el índice de inicio del estado estacionario mediante el método
    de Welch (promedio móvil sobre múltiples réplicas).

    El método suaviza la serie temporal con una ventana deslizante y busca
    el primer índice donde la serie suavizada deja de decrecer de forma
    sostenida (derivada ≈ 0).

    Parameters
    ----------
    serie : List[float]
        Serie temporal de una métrica (p. ej., Wq acumulado por evento).
    ventana : int
        Tamaño de la ventana de suavizado.

    Returns
    -------
    int
        Índice estimado donde termina el transitorio.
    """
    if len(serie) < 2 * ventana + 1:
        return 0

    arr = np.array(serie, dtype=float)
    # Suavizado con ventana deslizante
    suavizado = np.convolve(arr, np.ones(ventana) / ventana, mode="valid")

    # Buscar donde la derivada primera cambia de signo (mínimo local global)
    derivada = np.diff(suavizado)
    for i in range(1, len(derivada)):
        if derivada[i] >= 0 and derivada[i - 1] < 0:
            return int(i + ventana)

    # Si no hay cruce, usar el 10% de la serie como fallback
    return max(1, int(0.10 * len(serie)))


def calcular_warmup_welch_desde_replicas(
    lambda_base: float,
    mu: float,
    c: int,
    n_replicas: int = 10,
    t_sim: float = 480.0,
    semilla_base: int = 42,
) -> float:
    """
    Calcula el período de calentamiento óptimo usando el método de Welch
    sobre múltiples réplicas cortas.

    Parameters
    ----------
    lambda_base : float
    mu : float
    c : int
    n_replicas : int
        Número de réplicas para promediar.
    t_sim : float
        Duración de cada réplica en minutos.
    semilla_base : int

    Returns
    -------
    float
        Tiempo de calentamiento recomendado en minutos.
    """
    from simulacion_des import correr_una_replica

    series_wq: List[List[float]] = []
    for i in range(n_replicas):
        try:
            res = correr_una_replica(
                lambda_base=lambda_base,
                mu=mu,
                c=c,
                t_sim=t_sim,
                t_warm=0.0,          # sin warm-up para observar el transitorio
                semilla=semilla_base + i,
                usar_prioridad=False,
                no_estacionario=False,
            )
            if res["wq_lista"]:
                series_wq.append(res["wq_lista"])
        except ValueError:
            pass

    if not series_wq:
        return 60.0  # fallback por defecto

    # Truncar al mínimo tamaño de serie
    min_len = min(len(s) for s in series_wq)
    if min_len < 10:
        return 60.0

    # Promediar series entre réplicas
    matriz = np.array([s[:min_len] for s in series_wq])
    serie_prom = np.mean(matriz, axis=0).tolist()

    idx = detectar_warmup_welch(serie_prom, ventana=max(3, min_len // 20))

    # Convertir índice de cliente a tiempo aproximado
    lambda_min = lambda_base / 60.0
    t_por_cliente = 1.0 / lambda_min if lambda_min > 0 else 1.0
    t_warmup_est = idx * t_por_cliente

    # Acotar entre 10 y 120 minutos
    t_warmup_est = max(10.0, min(t_warmup_est, 120.0))
    logger.info(f"Welch warm-up estimado: {t_warmup_est:.1f} minutos (índice={idx})")
    return float(t_warmup_est)


# ---------------------------------------------------------------------------
# Barrido de sensibilidad
# ---------------------------------------------------------------------------

def barrido_sensibilidad(
    lambdas: List[float],
    cs: List[int],
    mu: float,
    N: int = 20,
    t_sim: float = 480.0,
    t_warm: float = 60.0,
    semilla_base: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta el barrido completo de sensibilidad variando λ y c.

    Para cada combinación (c, λ) ejecuta N réplicas y registra
    Wq_prom, Lq_prom y ρ.

    Parameters
    ----------
    lambdas : List[float]
        Valores de tasa de llegada a evaluar (clientes/hora).
    cs : List[int]
        Valores de número de servidores a evaluar.
    mu : float
        Tasa de servicio fija.
    N : int
        Réplicas por configuración.
    t_sim : float
    t_warm : float
    semilla_base : int
    verbose : bool

    Returns
    -------
    dict con:
        - tabla_wq  : np.ndarray (len(cs) x len(lambdas))
        - tabla_lq  : np.ndarray
        - tabla_rho : np.ndarray
        - lambdas   : List[float]
        - cs        : List[int]
        - analitico_wq : np.ndarray (valores teóricos de Wq)
    """
    n_c = len(cs)
    n_l = len(lambdas)

    tabla_wq  = np.full((n_c, n_l), np.nan)
    tabla_lq  = np.full((n_c, n_l), np.nan)
    tabla_rho = np.full((n_c, n_l), np.nan)
    tabla_anal_wq = np.full((n_c, n_l), np.nan)

    total = n_c * n_l
    contador = 0

    for i, c in enumerate(cs):
        for j, lam in enumerate(lambdas):
            contador += 1
            rho = lam / (c * mu)
            if rho >= 1.0:
                if verbose:
                    print(f"  [{contador}/{total}] c={c}, λ={lam:.1f} → INESTABLE (ρ={rho:.2f}), omitido")
                continue

            if verbose:
                print(f"  [{contador}/{total}] c={c}, λ={lam:.1f} | ρ={rho:.3f} ...", end=" ", flush=True)
            try:
                res = correr_replicas(
                    N=N,
                    lambda_base=lam,
                    mu=mu,
                    c=c,
                    t_sim=t_sim,
                    t_warm=t_warm,
                    semilla_base=semilla_base,
                    usar_prioridad=False,
                    verbose=False,
                )
                tabla_wq[i, j]  = res["medias"]["wq_promedio"]
                tabla_lq[i, j]  = res["medias"]["lq_promedio"]
                tabla_rho[i, j] = res["medias"]["rho"]

                # Analítico
                anal = calcular_metricas_mmc(lam, mu, c)
                tabla_anal_wq[i, j] = anal["Wq"]

                if verbose:
                    print(f"Wq={tabla_wq[i,j]:.2f} min")
            except Exception as e:
                logger.warning(f"Error en c={c}, λ={lam}: {e}")
                if verbose:
                    print(f"ERROR: {e}")

    return {
        "tabla_wq":      tabla_wq,
        "tabla_lq":      tabla_lq,
        "tabla_rho":     tabla_rho,
        "analitico_wq":  tabla_anal_wq,
        "lambdas":       lambdas,
        "cs":            cs,
    }


# ---------------------------------------------------------------------------
# Optimización automática: mínimo c para cumplir objetivo de Wq
# ---------------------------------------------------------------------------

def optimizar_servidores(
    lambda_base: float,
    mu: float,
    umbral_wq_min: float = 10.0,
    c_min: int = 1,
    c_max: int = 15,
    N: int = 20,
    t_sim: float = 480.0,
    t_warm: float = 60.0,
    semilla_base: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Encuentra el número mínimo de técnicos c tal que Wq_promedio < umbral_wq_min.

    Usa búsqueda lineal ascendente sobre c (más robusta que binaria para
    funciones no monótonas en simulación).

    Parameters
    ----------
    lambda_base : float
    mu : float
    umbral_wq_min : float
        Objetivo de tiempo de espera en minutos (default = 10 min).
    c_min : int
        Mínimo c a evaluar.
    c_max : int
        Máximo c a evaluar antes de declarar no convergencia.
    N : int
        Réplicas por configuración.
    t_sim : float
    t_warm : float
    semilla_base : int
    verbose : bool

    Returns
    -------
    dict con:
        - c_optimo : int
        - wq_optimo : float
        - historial : List[dict] — (c, wq) evaluados
        - cumple : bool
    """
    historial: List[Dict] = []
    c_optimo: Optional[int] = None
    wq_optimo: Optional[float] = None

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Optimización: Wq < {umbral_wq_min} min | λ={lambda_base} | μ={mu}")
        print(f"{'='*55}")

    for c in range(c_min, c_max + 1):
        rho = lambda_base / (c * mu)
        if rho >= 1.0:
            if verbose:
                print(f"  c={c:2d} → INESTABLE (ρ={rho:.3f}), continuando...")
            historial.append({"c": c, "wq": float("inf"), "rho": rho, "estable": False})
            continue

        try:
            res = correr_replicas(
                N=N,
                lambda_base=lambda_base,
                mu=mu,
                c=c,
                t_sim=t_sim,
                t_warm=t_warm,
                semilla_base=semilla_base,
                usar_prioridad=False,
                verbose=False,
            )
            wq = res["medias"]["wq_promedio"]
            historial.append({"c": c, "wq": wq, "rho": rho, "estable": True})

            if verbose:
                cumple_label = "✓ CUMPLE" if wq < umbral_wq_min else "✗"
                print(f"  c={c:2d} | ρ={rho:.3f} | Wq={wq:6.2f} min  {cumple_label}")

            if wq < umbral_wq_min and c_optimo is None:
                c_optimo = c
                wq_optimo = wq
                break  # encontramos el mínimo c que cumple

        except Exception as e:
            logger.warning(f"Error optimizando c={c}: {e}")

    cumple = c_optimo is not None
    if verbose:
        if cumple:
            print(f"\n  → Solución óptima: c = {c_optimo} técnicos (Wq = {wq_optimo:.2f} min)")
        else:
            print(f"\n  → No se encontró solución con c ≤ {c_max}")
        print(f"{'='*55}\n")

    return {
        "c_optimo":  c_optimo if c_optimo else c_max,
        "wq_optimo": wq_optimo if wq_optimo else float("inf"),
        "historial": historial,
        "cumple":    cumple,
        "umbral":    umbral_wq_min,
    }
