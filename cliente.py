"""
cliente.py
==========
Define la entidad Cliente del sistema de simulación TechClassUC.
Cada cliente representa una solicitud de servicio que entra al sistema de colas.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Cliente:
    """
    Representa una solicitud de servicio que llega al centro de atención TechClassUC.

    Attributes
    ----------
    id_cliente : int
        Identificador único del cliente.
    tipo : str
        Tipo de solicitud: 'soporte', 'mantenimiento' o 'reclamo'.
    prioridad : int
        Prioridad en cola: 0 = urgente, 1 = normal (menor valor = mayor prioridad en SimPy).
    t_llegada : float
        Tiempo de llegada al sistema (minutos desde inicio de simulación).
    t_inicio_atencion : Optional[float]
        Tiempo en que inicia su atención. None si aún no ha sido atendido.
    t_fin_atencion : Optional[float]
        Tiempo en que termina su atención. None si aún no ha finalizado.
    abandono : bool
        True si el cliente abandonó la cola antes de ser atendido (reneging).
    """

    id_cliente: int
    tipo: str = "soporte"
    prioridad: int = 1          # 1 = normal, 0 = urgente
    t_llegada: float = 0.0
    t_inicio_atencion: Optional[float] = None
    t_fin_atencion: Optional[float] = None
    abandono: bool = False

    def calcular_wq(self) -> Optional[float]:
        """
        Calcula el tiempo de espera en cola (Wq).

        Returns
        -------
        float or None
            Wq = t_inicio_atencion - t_llegada, o None si el cliente no fue atendido.
        """
        if self.t_inicio_atencion is not None:
            return max(0.0, self.t_inicio_atencion - self.t_llegada)
        return None

    def calcular_ws(self) -> Optional[float]:
        """
        Calcula el tiempo total en el sistema (Ws).

        Returns
        -------
        float or None
            Ws = t_fin_atencion - t_llegada, o None si el cliente no terminó su atención.
        """
        if self.t_fin_atencion is not None:
            return max(0.0, self.t_fin_atencion - self.t_llegada)
        return None

    def calcular_ts(self) -> Optional[float]:
        """
        Calcula el tiempo de servicio efectivo (tiempo siendo atendido).

        Returns
        -------
        float or None
            ts = t_fin_atencion - t_inicio_atencion, o None si no completó atención.
        """
        if self.t_inicio_atencion is not None and self.t_fin_atencion is not None:
            return max(0.0, self.t_fin_atencion - self.t_inicio_atencion)
        return None

    def __repr__(self) -> str:
        prioridad_label = "URGENTE" if self.prioridad == 0 else "normal"
        wq = self.calcular_wq()
        return (
            f"Cliente(id={self.id_cliente}, tipo={self.tipo}, "
            f"prio={prioridad_label}, llegada={self.t_llegada:.2f}, "
            f"Wq={wq:.2f if wq is not None else 'N/A'})"
        )
