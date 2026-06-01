from __future__ import annotations
import os, sys, json, threading, time
from flask import Flask, jsonify, request, send_file, render_template, abort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

_estado: dict = {"ejecutando": False, "completado": False, "error": None, "inicio": None}
_lock = threading.Lock()

def _run_sim(params: dict) -> None:
    os.makedirs(os.path.join(BASE_DIR, 'config'), exist_ok=True)
    with open(os.path.join(BASE_DIR, 'config', 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    sys.argv = ['main.py']
    try:
        import importlib, main as m
        importlib.reload(m)
        m.main()
        with _lock:
            _estado.update({"completado": True, "ejecutando": False})
    except Exception as e:
        with _lock:
            _estado.update({"error": str(e), "ejecutando": False})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/api/health')
def api_health():
    return jsonify({"status": "ok"}), 200

@app.route('/api/simular', methods=['POST'])
def simular():
    with _lock:
        if _estado['ejecutando']:
            return jsonify({"error": "Simulación en curso"}), 409
    p = request.get_json(silent=True) or {}
    defaults = {"lambda_base": 10.0, "mu": 4.0, "c": 3, "t_sim": 480.0,
                "t_warm": 60.0, "N_replicas": 30, "semilla_base": 42,
                "umbral_wq": 10.0, "t_max_espera": 20.0, "prob_urgente": 0.15}
    params = {**defaults, **p}
    rho = params['lambda_base'] / (params['c'] * params['mu'])
    if rho >= 1.0:
        return jsonify({"error": f"Sistema inestable ρ={rho:.3f}"}), 400
    with _lock:
        _estado.update({"ejecutando": True, "completado": False, "error": None, "inicio": time.time()})
    threading.Thread(target=_run_sim, args=(params,), daemon=True).start()
    return jsonify({"ok": True, "rho": rho}), 202

@app.route('/api/estado')
def estado():
    with _lock:
        s = dict(_estado)
    if s['inicio']:
        s['segundos'] = round(time.time() - s['inicio'], 1)
    return jsonify(s)

@app.route('/api/resultados')
def resultados():
    ruta = os.path.join(BASE_DIR, 'reports', 'reporte_final.json')
    if not os.path.exists(ruta):
        return jsonify({"error": "Sin resultados aún"}), 404
    with open(ruta, encoding='utf-8') as f:
        return jsonify(json.load(f))

@app.route('/api/grafica/<nombre>')
def grafica(nombre: str):
    if not nombre.endswith('.png'):
        nombre += '.png'
    ruta = os.path.join(BASE_DIR, 'outputs', nombre)
    if not os.path.exists(ruta):
        abort(404)
    return send_file(ruta, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  TechClassUC UI → http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)