"""
app_render.py
=============
Servidor web Flask para despliegue en Render.
Expone endpoints REST que permiten ejecutar la simulación y
descargar resultados sin necesidad de una interfaz gráfica local.
"""

from __future__ import annotations

import os
import json
import threading
import time
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

# Estado global de la simulación
_estado: dict = {"ejecutando": False, "completado": False, "error": None, "inicio": None}
_lock = threading.Lock()


def _ejecutar_simulacion(params: dict) -> None:
    """Ejecuta main() en un hilo separado."""
    import sys
    sys.argv = ["main.py"]  # limpiar argumentos CLI

    # Aplicar parámetros al config
    config_path = os.path.join(os.path.dirname(__file__), "config", "params.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(params, f, indent=2)

    try:
        from main import main
        main()
        with _lock:
            _estado["completado"] = True
            _estado["ejecutando"] = False
    except Exception as e:
        with _lock:
            _estado["error"] = str(e)
            _estado["ejecutando"] = False


@app.route("/health", methods=["GET"])
def health():
    """Health check para Render."""
    return jsonify({"status": "ok", "service": "TechClassUC-Simulacion"}), 200


@app.route("/", methods=["GET"])
def index():
    """Página de bienvenida con instrucciones de uso."""
    return jsonify({
        "servicio": "TechClassUC — Simulación M/M/c",
        "version": "1.0.0",
        "endpoints": {
            "GET  /health":      "Health check",
            "POST /simular":     "Ejecutar simulación con parámetros JSON",
            "GET  /estado":      "Consultar estado de la simulación",
            "GET  /resultados":  "Obtener resultados JSON del último run",
            "GET  /grafica/<n>": "Descargar gráfica PNG por número (01–11)",
        },
        "ejemplo_post": {
            "url": "/simular",
            "body": {
                "lambda_base": 10,
                "mu": 4,
                "c": 3,
                "N_replicas": 30,
                "t_sim": 480,
                "t_warm": 60,
                "umbral_wq": 10
            }
        }
    })


@app.route("/simular", methods=["POST"])
def simular():
    """Lanza la simulación con los parámetros recibidos por POST."""
    with _lock:
        if _estado["ejecutando"]:
            return jsonify({"error": "Simulación ya en curso. Consulte /estado"}), 409

    datos = request.get_json(silent=True) or {}
    params_default = {
        "lambda_base": 10.0, "mu": 4.0, "c": 3,
        "t_sim": 480.0, "t_warm": 60.0,
        "N_replicas": 30, "semilla_base": 42,
        "umbral_wq": 10.0, "t_max_espera": 20.0,
        "prob_urgente": 0.15,
    }
    params = {**params_default, **datos}

    # Validar estabilidad antes de lanzar
    rho = params["lambda_base"] / (params["c"] * params["mu"])
    if rho >= 1.0:
        return jsonify({"error": f"Sistema inestable: ρ={rho:.3f} ≥ 1"}), 400

    with _lock:
        _estado.update({"ejecutando": True, "completado": False, "error": None, "inicio": time.time()})

    hilo = threading.Thread(target=_ejecutar_simulacion, args=(params,), daemon=True)
    hilo.start()
    return jsonify({"mensaje": "Simulación iniciada", "parametros": params, "rho": rho}), 202


@app.route("/estado", methods=["GET"])
def estado():
    """Devuelve el estado actual de la simulación."""
    with _lock:
        s = dict(_estado)
    if s["inicio"]:
        s["segundos_transcurridos"] = round(time.time() - s["inicio"], 1)
    return jsonify(s)


@app.route("/resultados", methods=["GET"])
def resultados():
    """Devuelve el reporte JSON generado por la última simulación."""
    ruta = os.path.join(os.path.dirname(__file__), "reports", "reporte_final.json")
    if not os.path.exists(ruta):
        return jsonify({"error": "No hay resultados aún. Ejecute POST /simular primero."}), 404
    with open(ruta, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/grafica/<nombre>", methods=["GET"])
def descargar_grafica(nombre: str):
    """Descarga una gráfica PNG del directorio outputs/."""
    # Seguridad: sólo archivos PNG
    if not nombre.endswith(".png"):
        nombre = nombre + ".png"
    ruta = os.path.join(os.path.dirname(__file__), "outputs", nombre)
    if not os.path.exists(ruta):
        graficas_disp = [f for f in os.listdir(os.path.join(os.path.dirname(__file__), "outputs"))
                         if f.endswith(".png")] if os.path.isdir(
            os.path.join(os.path.dirname(__file__), "outputs")) else []
        return jsonify({"error": f"Gráfica no encontrada: {nombre}",
                        "disponibles": graficas_disp}), 404
    return send_file(ruta, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
