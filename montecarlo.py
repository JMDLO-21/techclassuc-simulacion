"""
montecarlo.py
=============
Ejecuta N réplicas independientes de la simulación DES y calcula
estadísticas de Montecarlo: medias, desviaciones estándar e intervalos
de confianza al 95% para todas las métricas de desempeño.
"""

from __future__ import annotations

import math
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats as scipy_stats

from simulacion_des import correr_una_replica

logger = logging.getLogger(__name__)


def correr_replicas(
    N: int,
    lambda_base: float,
    mu: float,
    c: int,
    t_sim: float = 480.0,
    t_warm: float = 60.0,
    semilla_base: int = 42,
    t_max_espera: Optional[float] = None,
    usar_prioridad: bool = True,
    prob_urgente: float = 0.15,
    no_estacionario: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta N réplicas independientes de la simulación y agrega resultados.

    Cada réplica usa semilla = semilla_base + i para garantizar
    independencia y reproducibilidad.

    Parameters
    ----------
    N : int
        Número de réplicas a ejecutar.
    lambda_base : float
        Tasa de llegada base (clientes/hora).
    mu : float
        Tasa de servicio (clientes/hora por técnico).
    c : int
        Número de técnicos.
    t_sim : float
        Duración de cada réplica (minutos).
    t_warm : float
        Período de calentamiento a descartar (minutos).
    semilla_base : int
        Semilla inicial; réplica i usa semilla_base + i.
    t_max_espera : Optional[float]
        Tiempo máximo de espera en cola (reneging). None = desactivado.
    usar_prioridad : bool
        Si True, activa colas con prioridad para clientes urgentes.
    prob_urgente : float
        Probabilidad de que un cliente sea urgente.
    no_estacionario : bool
        Si True, la tasa de llegada varía por tramo horario.
    verbose : bool
        Si True, imprime progreso en consola.

    Returns
    -------
    dict con claves:
        - resultados_replicas : List[dict] — resultado crudo de cada réplica
        - medias : dict[str, float] — media de cada métrica
        - desv_std : dict[str, float] — desviación estándar
        - ic_95 : dict[str, Tuple[float, float]] — intervalo de confianza 95%
        - n_replicas : int
        - n_minimo_replicas : int — N mínimo para error relativo ≤ 5%
        - wq_todas : List[float] — lista de Wq promedio por réplica
        - evolucion_rep0 : List — evolución temporal de la réplica 0
        - wq_lista_rep0 : List[float] — Wq individuales de la réplica 0
    """
    metricas = ["wq_promedio", "ws_promedio", "lq_promedio", "rho"]
    resultados: List[Dict[str, Any]] = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Montecarlo: {N} réplicas | λ={lambda_base} | μ={mu} | c={c}")
        print(f"{'='*60}")

    for i in range(N):
        semilla = semilla_base + i
        try:
            res = correr_una_replica(
                lambda_base=lambda_base,
                mu=mu,
                c=c,
                t_sim=t_sim,
                t_warm=t_warm,
                semilla=semilla,
                t_max_espera=t_max_espera,
                usar_prioridad=usar_prioridad,
                prob_urgente=prob_urgente,
                no_estacionario=no_estacionario,
            )
            resultados.append(res)
            if verbose and (i + 1) % 5 == 0:
                print(f"  Réplica {i+1}/{N} completada — "
                      f"Wq={res['wq_promedio']:.2f} min, ρ={res['rho']:.3f}")
        except ValueError as e:
            logger.error(f"Réplica {i} fallida: {e}")
            raise

    # Agregar métricas
    medias: Dict[str, float] = {}
    desv_std: Dict[str, float] = {}
    ic_95: Dict[str, tuple] = {}

    for metrica in metricas:
        valores = np.array([r[metrica] for r in resultados])
        media = float(np.mean(valores))
        std = float(np.std(valores, ddof=1))
        t_crit = scipy_stats.t.ppf(0.975, df=N - 1)
        margen = t_crit * std / math.sqrt(N)

        medias[metrica] = media
        desv_std[metrica] = std
        ic_95[metrica] = (media - margen, media + margen)

    # Número mínimo de réplicas para error relativo ≤ 5% en Wq
    n_minimo = _calcular_n_minimo(
        [r["wq_promedio"] for r in resultados],
        error_relativo=0.05,
    )

    wq_todas = [r["wq_promedio"] for r in resultados]

    if verbose:
        _imprimir_resumen(medias, desv_std, ic_95, n_minimo, N)

    return {
        "resultados_replicas": resultados,
        "medias": medias,
        "desv_std": desv_std,
        "ic_95": ic_95,
        "n_replicas": N,
        "n_minimo_replicas": n_minimo,
        "wq_todas": wq_todas,
        "evolucion_rep0": resultados[0]["evolucion_temporal"] if resultados else [],
        "wq_lista_rep0": resultados[0]["wq_lista"] if resultados else [],
    }


def _calcular_n_minimo(valores: List[float], error_relativo: float = 0.05) -> int:
    """
    Estima el número mínimo de réplicas necesario para que el semiancho
    del IC 95% sea ≤ error_relativo * media.

    Parameters
    ----------
    valores : List[float]
        Muestra piloto de valores de la métrica.
    error_relativo : float
        Error relativo máximo deseado (0.05 = 5%).

    Returns
    -------
    int
        N mínimo estimado.
    """
    if len(valores) < 2:
        return 30
    arr = np.array(valores)
    media = float(np.mean(arr))
    if media == 0:
        return 30
    std = float(np.std(arr, ddof=1))
    t_crit = scipy_stats.t.ppf(0.975, df=len(valores) - 1)
    # n = (t * s / (e * media))^2
    n = math.ceil((t_crit * std / (error_relativo * media)) ** 2)
    return max(n, 1)


def _imprimir_resumen(
    medias: Dict[str, float],
    desv_std: Dict[str, float],
    ic_95: Dict[str, tuple],
    n_minimo: int,
    n_actual: int,
) -> None:
    """Imprime tabla de resultados Montecarlo en consola."""
    print(f"\n{'─'*65}")
    print(f"  {'Métrica':<22} {'Media':>10} {'Std':>10} {'IC 95% inf':>12} {'IC 95% sup':>12}")
    print(f"{'─'*65}")
    etiquetas = {
        "wq_promedio": "Wq prom (min)",
        "ws_promedio": "Ws prom (min)",
        "lq_promedio": "Lq prom (clientes)",
        "rho":         "ρ utilización",
    }
    for k, label in etiquetas.items():
        lo, hi = ic_95[k]
        print(f"  {label:<22} {medias[k]:>10.4f} {desv_std[k]:>10.4f} {lo:>12.4f} {hi:>12.4f}")
    print(f"{'─'*65}")
    print(f"  N actual: {n_actual} | N mínimo sugerido (error ≤ 5%): {n_minimo}")
    print(f"{'─'*65}\n")
