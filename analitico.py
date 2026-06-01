"""
analitico.py
============
Implementa las fórmulas cerradas del modelo M/M/c para calcular
las métricas teóricas de desempeño y comparar con resultados simulados.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


def calcular_p0(lambda_: float, mu: float, c: int) -> float:
    """
    Calcula P₀: probabilidad de que el sistema esté vacío en M/M/c.

    P₀ = [ Σ_{n=0}^{c-1} (λ/μ)^n/n!  +  (λ/μ)^c / (c! · (1-ρ)) ]^{-1}

    Parameters
    ----------
    lambda_ : float
        Tasa de llegada (clientes/hora).
    mu : float
        Tasa de servicio por servidor (clientes/hora).
    c : int
        Número de servidores.

    Returns
    -------
    float
        Probabilidad P₀ ∈ (0, 1].

    Raises
    ------
    ValueError
        Si el sistema es inestable (ρ ≥ 1).
    """
    rho = lambda_ / (c * mu)
    if rho >= 1.0:
        raise ValueError(
            f"Sistema M/M/c inestable: ρ = {rho:.4f} ≥ 1. "
            "No existe solución en estado estacionario."
        )
    a = lambda_ / mu  # intensidad de tráfico (Erlang)

    suma = sum((a ** n) / math.factorial(n) for n in range(c))
    ultimo = (a ** c) / (math.factorial(c) * (1.0 - rho))
    return 1.0 / (suma + ultimo)


def calcular_metricas_mmc(
    lambda_: float,
    mu: float,
    c: int,
) -> Dict[str, float]:
    """
    Calcula todas las métricas analíticas del modelo M/M/c.

    Parameters
    ----------
    lambda_ : float
        Tasa de llegada (clientes/hora).
    mu : float
        Tasa de servicio por servidor (clientes/hora).
    c : int
        Número de servidores.

    Returns
    -------
    dict con claves:
        rho   : factor de utilización
        P0    : probabilidad de sistema vacío
        Lq    : clientes promedio en cola
        Wq    : tiempo promedio en cola (minutos)
        L     : clientes promedio en sistema
        W     : tiempo promedio en sistema (minutos)
    """
    rho = lambda_ / (c * mu)
    p0 = calcular_p0(lambda_, mu, c)
    a = lambda_ / mu

    # Lq (número esperado en cola)
    Lq = (p0 * (a ** c) * rho) / (math.factorial(c) * (1.0 - rho) ** 2)

    # Wq (tiempo en cola) → convertido a minutos
    Wq_horas = Lq / lambda_
    Wq_min = Wq_horas * 60.0

    # L y W (sistema completo)
    L = Lq + (lambda_ / mu)
    W_horas = L / lambda_
    W_min = W_horas * 60.0

    return {
        "rho": rho,
        "P0": p0,
        "Lq": Lq,
        "Wq": Wq_min,   # en minutos
        "L": L,
        "W": W_min,      # en minutos
    }


def comparar_con_simulacion(
    analitico: Dict[str, float],
    simulado: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Compara resultados analíticos con los simulados y calcula el error relativo.

    Parameters
    ----------
    analitico : dict
        Métricas analíticas (salida de calcular_metricas_mmc).
    simulado : dict
        Métricas simuladas (medias de Montecarlo).

    Returns
    -------
    dict
        Para cada métrica comparable: {analitico, simulado, error_relativo_%}.
    """
    # Mapa de equivalencia entre nombres
    mapa = {
        "Wq": "wq_promedio",
        "Lq": "lq_promedio",
        "rho": "rho",
    }

    comparacion: Dict[str, Dict[str, float]] = {}
    for nombre_anal, nombre_sim in mapa.items():
        val_anal = analitico.get(nombre_anal, None)
        val_sim = simulado.get(nombre_sim, None)
        if val_anal is None or val_sim is None:
            continue
        if val_anal != 0:
            error = abs(val_sim - val_anal) / abs(val_anal) * 100.0
        else:
            error = float("nan")
        comparacion[nombre_anal] = {
            "analitico": val_anal,
            "simulado": val_sim,
            "error_relativo_%": error,
        }
    return comparacion


def imprimir_comparacion(comparacion: Dict[str, Dict[str, float]]) -> None:
    """Imprime tabla comparativa analítico vs simulado en consola."""
    print(f"\n{'='*62}")
    print("  Comparación Analítico M/M/c vs Simulación Montecarlo")
    print(f"{'='*62}")
    print(f"  {'Métrica':<12} {'Analítico':>12} {'Simulado':>12} {'Error %':>10}")
    print(f"{'─'*62}")
    etiquetas = {"Wq": "Wq (min)", "Lq": "Lq (clientes)", "rho": "ρ"}
    for k, datos in comparacion.items():
        label = etiquetas.get(k, k)
        print(
            f"  {label:<12} {datos['analitico']:>12.4f} "
            f"{datos['simulado']:>12.4f} {datos['error_relativo_%']:>9.2f}%"
        )
    print(f"{'='*62}\n")
