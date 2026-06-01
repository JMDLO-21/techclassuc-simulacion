"""
servidor.py
===========
Configuración del recurso SimPy que modela los técnicos de TechClassUC
y estadísticas acumuladas de utilización por servidor.
"""

from __future__ import annotations

import simpy
from dataclasses import dataclass, field
from typing import List


@dataclass
class EstadisticasServidor:
    """
    Estadísticas acumuladas del grupo de servidores durante una réplica.

    Attributes
    ----------
    capacidad : int
        Número de técnicos (servidores en paralelo).
    clientes_atendidos : int
        Total de clientes que completaron su atención.
    clientes_abandonaron : int
        Total de clientes que abandonaron la cola (reneging).
    tiempo_ocupado_acum : float
        Suma de tiempos de servicio de todos los clientes atendidos (minutos).
    """

    capacidad: int
    clientes_atendidos: int = 0
    clientes_abandonaron: int = 0
    tiempo_ocupado_acum: float = 0.0

    def utilization(self, t_sim: float) -> float:
        """
        Calcula la utilización promedio del grupo de servidores.

        Parameters
        ----------
        t_sim : float
            Duración efectiva de la simulación (sin warm-up) en minutos.

        Returns
        -------
        float
            Fracción promedio de tiempo en que cada servidor estuvo ocupado.
        """
        if t_sim <= 0 or self.capacidad <= 0:
            return 0.0
        return self.tiempo_ocupado_acum / (self.capacidad * t_sim)


def crear_recurso(
    env: simpy.Environment,
    capacidad: int,
    prioridad: bool = False,
) -> simpy.Resource | simpy.PriorityResource:
    """
    Crea el recurso SimPy para modelar los técnicos del sistema.

    Parameters
    ----------
    env : simpy.Environment
        Entorno de simulación activo.
    capacidad : int
        Número de técnicos disponibles (c en M/M/c).
    prioridad : bool
        Si True, usa PriorityResource para clientes urgentes.

    Returns
    -------
    simpy.Resource o simpy.PriorityResource
        Recurso SimPy configurado.

    Raises
    ------
    ValueError
        Si la capacidad es menor o igual a cero.
    """
    if capacidad <= 0:
        raise ValueError(f"La capacidad del servidor debe ser > 0. Recibido: {capacidad}")
    if prioridad:
        return simpy.PriorityResource(env, capacity=capacidad)
    return simpy.Resource(env, capacity=capacidad)
