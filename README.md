# TechClassUC — Simulación de Colas M/M/c

**Proyecto Integrador | Modelos de Simulación**  
Julián Morales de la Ossa · Corporación Universitaria Remington

---

## Descripción

Simulación computacional del sistema de atención al cliente de **TechClassUC**, una empresa de soporte técnico. El proyecto combina:

- **Simulación de Eventos Discretos (DES)** con SimPy (modelo M/M/c)
- **Simulación de Montecarlo** con múltiples réplicas independientes
- **Validación analítica** contra fórmulas cerradas M/M/c
- **Análisis de sensibilidad** variando λ y c
- **Extensiones**: prioridades, reneging, llegadas no estacionarias, detección Welch, optimización automática

---

## Estructura del Proyecto

```
proyecto/
├── cliente.py           # Entidad Cliente con atributos y cálculo de Wq/Ws
├── servidor.py          # Recurso SimPy y estadísticas de servidores
├── simulacion_des.py    # Modelo DES: llegadas, colas, atención (SimPy)
├── montecarlo.py        # N réplicas + IC 95% + N mínimo de réplicas
├── analitico.py         # Fórmulas cerradas M/M/c: P₀, Lq, Wq, L, W
├── sensibilidad.py      # Barrido λ×c, método de Welch, optimización
├── visualizacion.py     # 11 gráficas Matplotlib (PNG)
├── main.py              # Punto de entrada principal
├── app_render.py        # Servidor Flask para Render
├── requirements.txt
├── Dockerfile.render
├── render.yaml
├── outputs/             # Gráficas generadas (.png)
├── reports/             # reporte_final.json
├── logs/                # simulacion.log
└── config/              # params.json (parámetros editables)
```

---

## Parámetros Base

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| λ | 10 clientes/hora | Tasa de llegada |
| μ | 4 clientes/hora | Tasa de servicio por técnico |
| c | 3 técnicos | Configuración base |
| T_sim | 480 min | Jornada de 8 horas |
| T_warm | 60 min | Período de calentamiento |
| N | 30 réplicas | Réplicas Montecarlo |
| ρ | 0.833 | Factor de utilización |

---

## Ejecución Local

### Requisitos

- Python 3.12+
- pip

### Instalación

```bash
cd proyecto
pip install -r requirements.txt
```

### Ejecutar simulación completa

```bash
python main.py
```

### Opciones de línea de comandos

```bash
# Parámetros personalizados
python main.py --lambda 12 --mu 4 --c 4 --replicas 50

# Modo rápido (menos réplicas, barrido reducido)
python main.py --quick
```

### Editar parámetros sin CLI

Edita `config/params.json` antes de ejecutar:

```json
{
  "lambda_base": 10.0,
  "mu": 4.0,
  "c": 3,
  "t_sim": 480.0,
  "t_warm": 60.0,
  "N_replicas": 30,
  "semilla_base": 42,
  "umbral_wq": 10.0,
  "t_max_espera": 20.0,
  "prob_urgente": 0.15
}
```

---

## Ejecución con Docker

### Construir imagen

```bash
docker build -f Dockerfile.render -t techclassuc-sim .
```

### Ejecutar localmente con Docker

```bash
docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs techclassuc-sim
```

### Ejecutar simulación via API

```bash
# Lanzar simulación con parámetros base
curl -X POST http://localhost:8000/simular \
  -H "Content-Type: application/json" \
  -d '{"lambda_base": 10, "mu": 4, "c": 3, "N_replicas": 30}'

# Consultar estado
curl http://localhost:8000/estado

# Obtener resultados
curl http://localhost:8000/resultados

# Descargar gráfica
curl -O http://localhost:8000/grafica/01_evolucion_temporal.png
```

---

## Despliegue en Render

### Opción A — Deploy automático (recomendado)

1. Haz fork del repositorio en GitHub
2. Ve a [render.com](https://render.com) → **New → Blueprint**
3. Conecta el repositorio → Render detecta `render.yaml` automáticamente
4. Click en **Apply** → el servicio se despliega con Auto Deploy activado

### Opción B — Deploy manual

1. Ve a [render.com](https://render.com) → **New → Web Service**
2. Conecta el repositorio
3. Configuración:
   - **Runtime**: Docker
   - **Dockerfile Path**: `./Dockerfile.render`
   - **Health Check Path**: `/health`
4. Agrega variables de entorno:
   - `MPLBACKEND=Agg`
   - `PYTHONUNBUFFERED=1`

### Endpoints disponibles en Render

Una vez desplegado en `https://tu-app.onrender.com`:

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Info y documentación de endpoints |
| `/health` | GET | Health check |
| `/simular` | POST | Lanzar simulación (body JSON con parámetros) |
| `/estado` | GET | Estado de la simulación en curso |
| `/resultados` | GET | Resultados JSON del último run |
| `/grafica/<nombre>` | GET | Descargar gráfica PNG |

---

## Gráficas Generadas

| Archivo | Descripción |
|---------|-------------|
| `01_evolucion_temporal.png` | Clientes en sistema a lo largo del tiempo |
| `02_histograma_wq.png` | Distribución de tiempos de espera Wq |
| `03_wq_vs_servidores.png` | Curva Wq vs. número de técnicos |
| `04_rho_vs_lambda.png` | Factor de utilización vs. λ para distintos c |
| `05_distribucion_medias_replicas.png` | Distribución de medias (verificación TCL) |
| `06_heatmap_wq.png` | Heatmap Wq en función de λ y c |
| `07_heatmap_rho.png` | Heatmap ρ en función de λ y c |
| `08_optimizacion_servidores.png` | Curva de optimización: mínimo c |
| `09_llegadas_no_estacionarias.png` | Evolución con λ variable por tramo horario |
| `10_reneging_abandonos.png` | Clientes atendidos vs. abandonos |
| `11_welch_warmup.png` | Detección de warm-up por método de Welch |

---

## Extensiones Implementadas

| Extensión | Estado | Módulo |
|-----------|--------|--------|
| Prioridad urgente (PriorityResource) | ✅ | `simulacion_des.py` |
| Abandono de cola (Reneging) | ✅ | `simulacion_des.py` |
| Llegadas no estacionarias | ✅ | `simulacion_des.py` |
| Detección Welch (warm-up automático) | ✅ | `sensibilidad.py` |
| Optimización automática de c | ✅ | `sensibilidad.py` |

---

## Resultados Esperados (parámetros base)

Con λ=10, μ=4, c=3 (ρ=0.833):

| Métrica | Analítico | Simulado (aprox.) |
|---------|-----------|-------------------|
| Wq (min) | ~14.5 | ~13–16 |
| Lq (clientes) | ~2.42 | ~2.1–2.7 |
| ρ | 0.833 | ~0.80–0.86 |

> La optimización automática determinará que se necesitan al menos **4 técnicos** para cumplir el objetivo de Wq < 10 min.

---

## Licencia

Proyecto académico — Corporación Universitaria Remington · 2025
