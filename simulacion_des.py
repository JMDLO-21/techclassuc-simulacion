"""
simulacion_des.py
=================
Implementa la simulación de eventos discretos (DES) del sistema de colas
TechClassUC usando SimPy.

Incluye:
- Proceso de llegada de clientes (estacionario y no estacionario).
- Proceso de atención con posible abandono de cola (reneging).
- Soporte de prioridades (urgente/normal) mediante PriorityResource.
- Recolección de métricas durante estado estacionario (post warm-up).
"""

from __future__ import annotations

import random
import simpy
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from cliente import Cliente
from servidor import crear_recurso, EstadisticasServidor


# ---------------------------------------------------------------------------
# Tasas de llegada no estacionarias (clientes/hora por tramo horario)
# Representa una jornada de 8 horas con mayor carga al mediodía
# ---------------------------------------------------------------------------
TASA_POR_HORA: List[Tuple[float, float]] = [
    (0.0,   8.0),   # 0–60 min  → mañana temprano (baja demanda)
    (60.0,  12.0),  # 60–180 min → mañana plena
    (180.0, 16.0),  # 180–300 min → mediodía (pico)
    (300.0, 10.0),  # 300–420 min → tarde
    (420.0, 6.0),   # 420–480 min → cierre
]


def _tasa_llegada_en(t: float, lambda_base: float, no_estacionario: bool) -> float:
    """
    Devuelve la tasa de llegada instantánea según el tiempo de simulación.

    Parameters
    ----------
    t : float
        Tiempo actual en la simulación (minutos).
    lambda_base : float
        Tasa base en clientes/hora (parámetro de configuración).
    no_estacionario : bool
        Si True, la tasa varía según el tramo horario del día.

    Returns
    -------
    float
        Tasa de llegada en clientes/minuto.
    """
    if not no_estacionario:
        return lambda_base / 60.0

    for inicio, tasa in reversed(TASA_POR_HORA):
        if t >= inicio:
            return tasa / 60.0
    return lambda_base / 60.0


def proceso_cliente(
    env: simpy.Environment,
    cliente: Cliente,
    recurso: simpy.Resource | simpy.PriorityResource,
    mu: float,
    stats: EstadisticasServidor,
    historial: List[Cliente],
    t_warm: float,
    t_max_espera: Optional[float],
    usar_prioridad: bool,
) -> Any:
    """
    Proceso SimPy que modela el ciclo de vida de un cliente en el sistema.

    Flujo: llega → solicita servidor → espera (con posible abandono) →
           es atendido → libera servidor → registra métricas.

    Parameters
    ----------
    env : simpy.Environment
    cliente : Cliente
    recurso : simpy.Resource o simpy.PriorityResource
    mu : float
        Tasa de servicio en clientes/hora.
    stats : EstadisticasServidor
    historial : List[Cliente]
        Lista donde se acumulan los clientes procesados.
    t_warm : float
        Período de calentamiento en minutos; métricas sólo después de este tiempo.
    t_max_espera : Optional[float]
        Tiempo máximo de espera en cola (reneging). None = sin abandono.
    usar_prioridad : bool
        Si True, la solicitud al recurso incluye la prioridad del cliente.
    """
    mu_min = mu / 60.0  # convertir a clientes/minuto

    # Solicitar servidor (con o sin prioridad)
    if usar_prioridad:
        req = recurso.request(priority=cliente.prioridad)
    else:
        req = recurso.request()

    # Reneging: el cliente abandona si no es atendido en t_max_espera minutos
    if t_max_espera is not None:
        resultado = yield req | env.timeout(t_max_espera)
        if req not in resultado:
            # El cliente abandonó
            cliente.abandono = True
            recurso.release(req)
            if env.now >= t_warm:
                stats.clientes_abandonaron += 1
                historial.append(cliente)
            return
    else:
        yield req

    # Inicio de atención
    cliente.t_inicio_atencion = env.now

    # Tiempo de servicio ~ Exponencial(μ)
    t_servicio = random.expovariate(mu_min)
    yield env.timeout(t_servicio)

    # Fin de atención
    cliente.t_fin_atencion = env.now
    recurso.release(req)

    # Registrar métricas sólo después del warm-up
    if env.now >= t_warm:
        stats.clientes_atendidos += 1
        stats.tiempo_ocupado_acum += t_servicio
        historial.append(cliente)


def proceso_llegadas(
    env: simpy.Environment,
    lambda_base: float,
    mu: float,
    recurso: simpy.Resource | simpy.PriorityResource,
    stats: EstadisticasServidor,
    historial: List[Cliente],
    t_sim: float,
    t_warm: float,
    t_max_espera: Optional[float],
    usar_prioridad: bool,
    prob_urgente: float,
    no_estacionario: bool,
    evolucion_temporal: List[Tuple[float, int]],
) -> Any:
    """
    Proceso SimPy que genera llegadas de clientes al sistema.

    Utiliza tiempos entre llegadas exponenciales (proceso de Poisson).
    Si no_estacionario=True, la tasa λ varía según el tramo horario.

    Parameters
    ----------
    env : simpy.Environment
    lambda_base : float
        Tasa base de llegada en clientes/hora.
    mu : float
        Tasa de servicio en clientes/hora.
    recurso : simpy.Resource o simpy.PriorityResource
    stats : EstadisticasServidor
    historial : List[Cliente]
    t_sim : float
        Duración total de la simulación en minutos.
    t_warm : float
        Período de calentamiento en minutos.
    t_max_espera : Optional[float]
        Tiempo máximo de espera para reneging.
    usar_prioridad : bool
    prob_urgente : float
        Probabilidad de que un cliente sea urgente (prioridad 0).
    no_estacionario : bool
        Si True, λ varía con el tiempo.
    evolucion_temporal : List[Tuple[float, int]]
        Lista donde se registran (tiempo, clientes_en_sistema) para graficación.
    """
    tipos = ["soporte", "mantenimiento", "reclamo"]
    id_counter = 0

    while env.now < t_sim:
        tasa_actual = _tasa_llegada_en(env.now, lambda_base, no_estacionario)
        t_entre_llegadas = random.expovariate(tasa_actual)

        yield env.timeout(t_entre_llegadas)

        if env.now >= t_sim:
            break

        id_counter += 1
        prioridad = 0 if random.random() < prob_urgente else 1
        cliente = Cliente(
            id_cliente=id_counter,
            tipo=random.choice(tipos),
            prioridad=prioridad,
            t_llegada=env.now,
        )

        # Registrar estado del sistema en este instante
        n_en_sistema = len(recurso.queue) + recurso.count
        evolucion_temporal.append((env.now, n_en_sistema))

        env.process(
            proceso_cliente(
                env, cliente, recurso, mu, stats, historial,
                t_warm, t_max_espera, usar_prioridad,
            )
        )


def correr_una_replica(
    lambda_base: float,
    mu: float,
    c: int,
    t_sim: float = 480.0,
    t_warm: float = 60.0,
    semilla: int = 42,
    t_max_espera: Optional[float] = None,
    usar_prioridad: bool = True,
    prob_urgente: float = 0.15,
    no_estacionario: bool = False,
) -> Dict[str, Any]:
    """
    Ejecuta una réplica completa de la simulación DES.

    Parameters
    ----------
    lambda_base : float
        Tasa de llegada base en clientes/hora.
    mu : float
        Tasa de servicio en clientes/hora por técnico.
    c : int
        Número de técnicos (servidores en paralelo).
    t_sim : float
        Duración de la simulación en minutos.
    t_warm : float
        Período de calentamiento en minutos (se descarta para métricas).
    semilla : int
        Semilla aleatoria para reproducibilidad.
    t_max_espera : Optional[float]
        Tiempo máximo de espera antes de abandono. None = sin reneging.
    usar_prioridad : bool
        Si True, usa PriorityResource para gestionar urgentes.
    prob_urgente : float
        Probabilidad de que un cliente sea urgente.
    no_estacionario : bool
        Si True, la tasa de llegada varía por tramo horario.

    Returns
    -------
    dict
        Diccionario con métricas:
        - wq_promedio : float (tiempo promedio en cola, minutos)
        - ws_promedio : float (tiempo promedio en sistema, minutos)
        - lq_promedio : float (clientes promedio en cola)
        - rho : float (utilización observada)
        - n_atendidos : int
        - n_abandonaron : int
        - historial : List[Cliente]
        - evolucion_temporal : List[Tuple[float, int]]
        - wq_lista : List[float]

    Raises
    ------
    ValueError
        Si la configuración es inestable (ρ ≥ 1) con llegadas estacionarias.
    """
    rho_teorico = lambda_base / (c * mu)
    if rho_teorico >= 1.0 and not no_estacionario:
        raise ValueError(
            f"Sistema inestable: ρ = {rho_teorico:.3f} ≥ 1. "
            f"Aumente c o reduzca λ antes de simular."
        )

    random.seed(semilla)
    env = simpy.Environment()
    stats = EstadisticasServidor(capacidad=c)
    historial: List[Cliente] = []
    evolucion_temporal: List[Tuple[float, int]] = []

    recurso = crear_recurso(env, c, prioridad=usar_prioridad)

    env.process(
        proceso_llegadas(
            env, lambda_base, mu, recurso, stats, historial,
            t_sim, t_warm, t_max_espera, usar_prioridad,
            prob_urgente, no_estacionario, evolucion_temporal,
        )
    )
    env.run(until=t_sim)

    # Calcular métricas del período estacionario
    t_efectivo = t_sim - t_warm
    clientes_post_warm = [c_ for c_ in historial if not c_.abandono]

    wq_lista = [c_.calcular_wq() for c_ in clientes_post_warm if c_.calcular_wq() is not None]
    ws_lista = [c_.calcular_ws() for c_ in clientes_post_warm if c_.calcular_ws() is not None]

    wq_prom = float(np.mean(wq_lista)) if wq_lista else 0.0
    ws_prom = float(np.mean(ws_lista)) if ws_lista else 0.0

    # Lq via Ley de Little: Lq = λ_efectiva * Wq
    lambda_min = lambda_base / 60.0
    lq_prom = lambda_min * wq_prom

    rho_obs = stats.utilization(t_efectivo)

    return {
        "wq_promedio": wq_prom,
        "ws_promedio": ws_prom,
        "lq_promedio": lq_prom,
        "rho": rho_obs,
        "n_atendidos": stats.clientes_atendidos,
        "n_abandonaron": stats.clientes_abandonaron,
        "historial": historial,
        "evolucion_temporal": evolucion_temporal,
        "wq_lista": wq_lista,
    }
