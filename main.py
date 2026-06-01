"""
main.py
=======
Punto de entrada del proyecto TechClassUC — Modelos de Simulación.

Orquesta todos los módulos: DES, Montecarlo, Analítico, Sensibilidad y
Visualización. Los parámetros pueden ajustarse desde config/params.json
o usando los valores base del documento del proyecto.

Uso:
    python main.py
    python main.py --lambda 10 --mu 4 --c 3 --replicas 30
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from typing import Dict, Any

# ─── configuración de rutas ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for d in ("outputs", "reports", "logs", "config"):
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

# ─── logging ──────────────────────────────────────────────────────────────
log_path = os.path.join(BASE_DIR, "logs", "simulacion.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")


# ─── imports internos ─────────────────────────────────────────────────────
from simulacion_des import correr_una_replica
from montecarlo import correr_replicas
from analitico import calcular_metricas_mmc, comparar_con_simulacion, imprimir_comparacion
from sensibilidad import (
    barrido_sensibilidad,
    optimizar_servidores,
    calcular_warmup_welch_desde_replicas,
    detectar_warmup_welch,
)
import visualizacion as viz


# ──────────────────────────────────────────────────────────────────────────
# Parámetros y configuración
# ──────────────────────────────────────────────────────────────────────────

PARAMS_DEFAULT: Dict[str, Any] = {
    "lambda_base": 10.0,    # clientes/hora
    "mu":          4.0,     # clientes/hora por técnico
    "c":           3,       # técnicos (configuración base)
    "t_sim":       480.0,   # minutos (jornada 8 horas)
    "t_warm":      60.0,    # minutos de calentamiento
    "N_replicas":  30,      # réplicas Montecarlo
    "semilla_base": 42,
    "umbral_wq":   10.0,    # minutos (objetivo de espera)
    "t_max_espera": 20.0,   # minutos máx antes de abandonar (reneging)
    "prob_urgente": 0.15,   # fracción de clientes urgentes
}


def cargar_params(ruta: str = None) -> Dict[str, Any]:
    """Carga parámetros desde config/params.json si existe; si no, usa defaults."""
    if ruta is None:
        ruta = os.path.join(BASE_DIR, "config", "params.json")
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        params = {**PARAMS_DEFAULT, **overrides}
        logger.info(f"Parámetros cargados desde {ruta}")
    else:
        params = PARAMS_DEFAULT.copy()
        # guardar defaults para referencia
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        logger.info("Usando parámetros base (config/params.json creado).")
    return params


def parsear_args() -> argparse.Namespace:
    """Parsea argumentos opcionales de línea de comandos."""
    parser = argparse.ArgumentParser(description="TechClassUC — Simulación M/M/c")
    parser.add_argument("--lambda", dest="lambda_base", type=float, default=None)
    parser.add_argument("--mu",     dest="mu",          type=float, default=None)
    parser.add_argument("--c",      dest="c",           type=int,   default=None)
    parser.add_argument("--replicas", dest="N_replicas", type=int,  default=None)
    parser.add_argument("--t_sim",  dest="t_sim",       type=float, default=None)
    parser.add_argument("--t_warm", dest="t_warm",      type=float, default=None)
    parser.add_argument("--quick",  action="store_true", help="Modo rápido: menos réplicas y barrido reducido")
    return parser.parse_args()


def validar_estabilidad(lambda_: float, mu: float, c: int) -> float:
    """Verifica ρ < 1 y lanza advertencia o error."""
    rho = lambda_ / (c * mu)
    if rho >= 1.0:
        msg = (
            f"\n{'!'*60}\n"
            f"  SISTEMA INESTABLE: ρ = {rho:.4f} ≥ 1\n"
            f"  λ = {lambda_}, μ = {mu}, c = {c}\n"
            f"  Aumente c o reduzca λ para garantizar estabilidad.\n"
            f"{'!'*60}"
        )
        logger.error(msg)
        print(msg)
        sys.exit(1)
    logger.info(f"Estabilidad verificada: ρ = {rho:.4f} < 1  ✓")
    return rho


def imprimir_banner(params: Dict[str, Any], rho: float) -> None:
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         TechClassUC — Simulación de Colas M/M/c              ║
║         Modelos de Simulación  |  Julián Morales de la Ossa  ║
╚══════════════════════════════════════════════════════════════╝

  Parámetros base:
    λ (llegadas)       = {params['lambda_base']:.1f} clientes/hora
    μ (servicio/técnico)= {params['mu']:.1f} clientes/hora
    c (técnicos)       = {params['c']}
    ρ (utilización)    = {rho:.4f}
    T_sim              = {params['t_sim']:.0f} minutos
    T_warm             = {params['t_warm']:.0f} minutos
    N_réplicas         = {params['N_replicas']}
    Umbral Wq objetivo = {params['umbral_wq']:.1f} minutos
""")


def guardar_reporte_json(datos: Dict[str, Any], ruta: str) -> None:
    """Guarda el reporte de resultados en formato JSON."""
    def serializable(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return str(obj)

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, default=serializable, ensure_ascii=False)
    logger.info(f"Reporte JSON guardado: {ruta}")


# ──────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_inicio_total = time.time()

    args = parsear_args()
    params = cargar_params()

    # Overrides desde CLI
    for key in ("lambda_base", "mu", "c", "N_replicas", "t_sim", "t_warm"):
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val

    modo_rapido = args.quick
    if modo_rapido:
        params["N_replicas"] = 10
        logger.info("Modo rápido activado: N=10 réplicas.")

    lambda_base = params["lambda_base"]
    mu          = params["mu"]
    c           = params["c"]
    t_sim       = params["t_sim"]
    N           = params["N_replicas"]
    semilla     = params["semilla_base"]
    umbral_wq   = params["umbral_wq"]
    t_max_esp   = params["t_max_espera"]
    prob_urg    = params["prob_urgente"]

    # ── 0. Validación de estabilidad ────────────────────────────────────
    rho = validar_estabilidad(lambda_base, mu, c)
    imprimir_banner(params, rho)

    # ── 1. Warm-up automático (Welch) ───────────────────────────────────
    print("\n[1/7] Detectando período de calentamiento (método de Welch)...")
    t_warm_welch = calcular_warmup_welch_desde_replicas(
        lambda_base=lambda_base, mu=mu, c=c,
        n_replicas=min(10, N), t_sim=t_sim, semilla_base=semilla,
    )
    print(f"       → Warm-up estimado por Welch: {t_warm_welch:.1f} min "
          f"(configurado: {params['t_warm']:.1f} min)")

    # Usamos el máximo entre el estimado y el configurado para mayor seguridad
    t_warm = max(params["t_warm"], t_warm_welch)
    print(f"       → Warm-up final aplicado: {t_warm:.1f} minutos")

    # Gráfica Welch (usamos la réplica piloto)
    try:
        res_piloto = correr_una_replica(
            lambda_base, mu, c, t_sim, t_warm=0.0,
            semilla=semilla, usar_prioridad=False,
        )
        if res_piloto["wq_lista"]:
            idx_w = detectar_warmup_welch(res_piloto["wq_lista"])
            viz.grafica_warmup_welch(res_piloto["wq_lista"], idx_w)
    except Exception as e:
        logger.warning(f"No se pudo generar gráfica Welch: {e}")

    # ── 2. Simulación DES base (una réplica representativa) ──────────────
    print("\n[2/7] Ejecutando réplica representativa (DES con todas las extensiones)...")
    res_base = correr_una_replica(
        lambda_base=lambda_base,
        mu=mu,
        c=c,
        t_sim=t_sim,
        t_warm=t_warm,
        semilla=semilla,
        t_max_espera=t_max_esp,
        usar_prioridad=True,
        prob_urgente=prob_urg,
        no_estacionario=False,
    )
    print(f"       → Atendidos: {res_base['n_atendidos']} | "
          f"Abandonaron: {res_base['n_abandonaron']} | "
          f"Wq={res_base['wq_promedio']:.2f} min | ρ={res_base['rho']:.3f}")

    viz.grafica_evolucion_temporal(res_base["evolucion_temporal"], t_warm)
    viz.grafica_reneging(res_base["historial"])

    # Réplica con llegadas no estacionarias
    print("       → Ejecutando réplica con llegadas no estacionarias...")
    res_noest = correr_una_replica(
        lambda_base, mu, c, t_sim, t_warm,
        semilla=semilla + 100,
        usar_prioridad=True,
        prob_urgente=prob_urg,
        no_estacionario=True,
    )
    viz.grafica_no_estacionario(res_noest["evolucion_temporal"])

    # ── 3. Montecarlo base ───────────────────────────────────────────────
    # Se ejecutan DOS corridas:
    #   a) M/M/c puro (sin reneging, sin prioridad) → para comparación analítica
    #   b) Completo (con todas las extensiones) → para análisis operativo
    print(f"\n[3/7] Montecarlo base: {N} réplicas (λ={lambda_base}, c={c})...")
    print("       → Corrida A: M/M/c puro (para validación analítica)...")
    mc_puro = correr_replicas(
        N=N,
        lambda_base=lambda_base,
        mu=mu,
        c=c,
        t_sim=t_sim,
        t_warm=t_warm,
        semilla_base=semilla,
        t_max_espera=None,       # sin reneging
        usar_prioridad=False,    # sin prioridad → M/M/c estándar
        prob_urgente=0.0,
        no_estacionario=False,
        verbose=True,
    )
    print("       → Corrida B: M/M/c con extensiones (operativo)...")
    mc_base = correr_replicas(
        N=N,
        lambda_base=lambda_base,
        mu=mu,
        c=c,
        t_sim=t_sim,
        t_warm=t_warm,
        semilla_base=semilla,
        t_max_espera=t_max_esp,
        usar_prioridad=True,
        prob_urgente=prob_urg,
        no_estacionario=False,
        verbose=False,
    )

    # ── 4. Validación analítica M/M/c ────────────────────────────────────
    print("\n[4/7] Comparación analítica M/M/c (usando corrida pura)...")
    analitico_base = calcular_metricas_mmc(lambda_base, mu, c)
    print(f"       Analítico → Wq={analitico_base['Wq']:.4f} min, "
          f"Lq={analitico_base['Lq']:.4f}, ρ={analitico_base['rho']:.4f}")
    # Comparar contra la corrida pura (mismas condiciones que el modelo analítico)
    comparacion = comparar_con_simulacion(analitico_base, mc_puro["medias"])
    imprimir_comparacion(comparacion)

    viz.grafica_histograma_wq(mc_puro["wq_lista_rep0"], wq_analitico=analitico_base["Wq"])
    viz.grafica_distribucion_medias(
        mc_puro["wq_todas"],
        mc_puro["medias"]["wq_promedio"],
        mc_puro["ic_95"]["wq_promedio"],
    )

    # ── 5. Análisis de sensibilidad ──────────────────────────────────────
    print("\n[5/7] Análisis de sensibilidad (barrido c × λ)...")
    lambdas_sens = [6.0, 8.0, 10.0, 12.0, 14.0] if not modo_rapido else [8.0, 10.0, 12.0]
    cs_sens      = [2, 3, 4, 5, 6]               if not modo_rapido else [2, 3, 4, 5]
    N_sens       = 15                             if not modo_rapido else 8

    sens = barrido_sensibilidad(
        lambdas=lambdas_sens,
        cs=cs_sens,
        mu=mu,
        N=N_sens,
        t_sim=t_sim,
        t_warm=t_warm,
        semilla_base=semilla,
        verbose=True,
    )

    viz.heatmap_wq(sens["tabla_wq"], lambdas_sens, cs_sens, umbral=umbral_wq)
    viz.heatmap_rho(sens["tabla_rho"], lambdas_sens, cs_sens)

    # Curvas Wq vs c y ρ vs λ para la configuración base
    wq_sim_vs_c  = []
    wq_anal_vs_c = []
    for ci in cs_sens:
        idx_c = cs_sens.index(ci)
        # Buscar el índice de lambda_base en lambdas_sens (más cercano)
        idx_l = min(range(len(lambdas_sens)), key=lambda k: abs(lambdas_sens[k] - lambda_base))
        val_sim  = sens["tabla_wq"][idx_c, idx_l]
        val_anal = sens["analitico_wq"][idx_c, idx_l]
        wq_sim_vs_c.append(float(val_sim)  if not np.isnan(val_sim)  else np.nan)
        wq_anal_vs_c.append(float(val_anal) if not np.isnan(val_anal) else np.nan)

    viz.grafica_wq_vs_servidores(cs_sens, wq_sim_vs_c, wq_anal_vs_c, umbral=umbral_wq)
    viz.grafica_rho_vs_lambda(lambdas_sens, cs_sens, mu)

    # ── 6. Optimización automática ───────────────────────────────────────
    print(f"\n[6/7] Optimización: mínimo c para Wq < {umbral_wq} min...")
    opt = optimizar_servidores(
        lambda_base=lambda_base,
        mu=mu,
        umbral_wq_min=umbral_wq,
        c_min=1,
        c_max=10,
        N=N_sens,
        t_sim=t_sim,
        t_warm=t_warm,
        semilla_base=semilla,
        verbose=True,
    )
    viz.grafica_optimizacion(opt["historial"], umbral=umbral_wq, c_optimo=opt["c_optimo"])

    # ── 7. Reporte final ─────────────────────────────────────────────────
    print("\n[7/7] Generando reporte final...")
    elapsed = time.time() - t_inicio_total

    reporte = {
        "parametros": params,
        "t_warm_aplicado": t_warm,
        "rho_teorico": rho,
        "analitico": analitico_base,
        "montecarlo_puro_mmc": {
            "medias":   mc_puro["medias"],
            "desv_std": mc_puro["desv_std"],
            "ic_95":    {k: list(v) for k, v in mc_puro["ic_95"].items()},
            "n_replicas": mc_puro["n_replicas"],
            "n_minimo_replicas": mc_puro["n_minimo_replicas"],
        },
        "montecarlo_operativo": {
            "medias":   mc_base["medias"],
            "desv_std": mc_base["desv_std"],
            "ic_95":    {k: list(v) for k, v in mc_base["ic_95"].items()},
            "n_replicas": mc_base["n_replicas"],
        },
        "comparacion_analitico_vs_simulado": comparacion,
        "optimizacion": {
            "c_optimo":  opt["c_optimo"],
            "wq_optimo": opt["wq_optimo"],
            "cumple":    opt["cumple"],
            "umbral_min": umbral_wq,
        },
        "tiempo_ejecucion_seg": round(elapsed, 2),
    }

    ruta_json = os.path.join(BASE_DIR, "reports", "reporte_final.json")
    guardar_reporte_json(reporte, ruta_json)

    # Imprimir resumen en consola
    _imprimir_reporte_final(reporte, mc_puro, mc_base, analitico_base, opt, elapsed)


def _imprimir_reporte_final(reporte, mc_puro, mc_base, analitico, opt, elapsed):
    sep = "═" * 65
    print(f"\n{sep}")
    print("  REPORTE FINAL — TechClassUC")
    print(sep)
    p = reporte["parametros"]
    print(f"  λ={p['lambda_base']} c/h | μ={p['mu']} c/h | c={p['c']} técnicos | ρ={reporte['rho_teorico']:.4f}")
    print(f"  Warm-up aplicado: {reporte['t_warm_aplicado']:.1f} min | Réplicas: {mc_puro['n_replicas']}")
    print(f"  N mínimo sugerido: {mc_puro['n_minimo_replicas']} réplicas (error ≤ 5%)")
    print(f"\n  ── Validación M/M/c puro (sin reneging) ──")
    print(f"  {'Métrica':<20} {'Analítico':>12} {'Simulado':>12} {'IC 95%':>22}")
    print(f"  {'─'*68}")
    map_keys = [("Wq (min)", "Wq", "wq_promedio"), ("Lq (clientes)", "Lq", "lq_promedio"), ("ρ", "rho", "rho")]
    for label, ka, ks in map_keys:
        anal_val = analitico.get(ka, float("nan"))
        sim_val  = mc_puro["medias"].get(ks, float("nan"))
        lo, hi   = mc_puro["ic_95"].get(ks, (float("nan"), float("nan")))
        print(f"  {label:<20} {anal_val:>12.4f} {sim_val:>12.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"\n  ── Modelo operativo (con reneging + prioridades) ──")
    for label, ka, ks in map_keys:
        sim_val  = mc_base["medias"].get(ks, float("nan"))
        lo, hi   = mc_base["ic_95"].get(ks, (float("nan"), float("nan")))
        print(f"  {label:<20} {'—':>12} {sim_val:>12.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"\n  Optimización: c_óptimo = {opt['c_optimo']} técnicos "
          f"(Wq = {opt['wq_optimo']:.2f} min < {opt['umbral']} min objetivo)")
    print(f"  Cumple objetivo: {'✓ SÍ' if opt['cumple'] else '✗ NO'}")
    print(f"\n  Gráficas guardadas en: outputs/")
    print(f"  Reporte JSON:          reports/reporte_final.json")
    print(f"  Log:                   logs/simulacion.log")
    print(f"\n  Tiempo total de ejecución: {elapsed:.1f} segundos")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
