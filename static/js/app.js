/* ── app.js ─────────────────────────────────────────────────
   TechClassUC — Frontend logic
   ─────────────────────────────────────────────────────────── */

// ── Reactive ρ preview ────────────────────────────────────
function actualizarRho() {
  const lambda = parseFloat(document.getElementById('lambda_base').value) || 0;
  const mu     = parseFloat(document.getElementById('mu').value) || 1;
  const c      = parseInt(document.getElementById('c').value) || 1;
  const rho    = lambda / (c * mu);

  const valEl  = document.getElementById('rho-val');
  const barEl  = document.getElementById('rho-bar');
  const warnEl = document.getElementById('rho-warn');

  valEl.textContent = rho.toFixed(4);
  const pct = Math.min(rho * 100, 100);
  barEl.style.width = pct + '%';

  if (rho >= 1) {
    barEl.style.background = 'var(--danger)';
    valEl.style.color = 'var(--danger)';
    warnEl.style.display = 'block';
  } else if (rho >= 0.8) {
    barEl.style.background = 'var(--warn)';
    valEl.style.color = 'var(--warn)';
    warnEl.style.display = 'none';
  } else {
    barEl.style.background = 'var(--accent2)';
    valEl.style.color = 'var(--accent)';
    warnEl.style.display = 'none';
  }
}

['lambda_base','mu','c'].forEach(id => {
  document.getElementById(id).addEventListener('input', actualizarRho);
});
actualizarRho();

// Toggle reneging field visibility
document.getElementById('usar_reneging').addEventListener('change', function() {
  document.getElementById('reneging-field').style.display = this.checked ? 'block' : 'none';
});

// ── Tabs ──────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-' + btn.dataset.tab).classList.add('active');
  });
});

// ── Estado de la simulación ───────────────────────────────
let pollingInterval = null;

function setBadge(text, cls) {
  const b = document.getElementById('status-badge');
  b.textContent = text;
  b.className = 'badge ' + (cls || '');
}

function setProgress(pct, label) {
  document.getElementById('progress-wrap').style.display = 'block';
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-label').textContent = label;
}

// ── Lanzar simulación ─────────────────────────────────────
async function lanzarSimulacion() {
  const rho = parseFloat(document.getElementById('rho-val').textContent);
  if (rho >= 1) {
    alert('Sistema inestable (ρ ≥ 1). Ajusta λ, μ o c antes de simular.');
    return;
  }

  const btn = document.getElementById('btn-run');
  btn.disabled = true;
  setBadge('⬤ Ejecutando...', 'running');

  const usarReneging = document.getElementById('usar_reneging').checked;

  const params = {
    lambda_base:   parseFloat(document.getElementById('lambda_base').value),
    mu:            parseFloat(document.getElementById('mu').value),
    c:             parseInt(document.getElementById('c').value),
    N_replicas:    parseInt(document.getElementById('N_replicas').value),
    t_sim:         parseFloat(document.getElementById('t_sim').value),
    t_warm:        parseFloat(document.getElementById('t_warm').value),
    umbral_wq:     parseFloat(document.getElementById('umbral_wq').value),
    t_max_espera:  usarReneging ? parseFloat(document.getElementById('t_max_espera').value) : null,
    usar_prioridad: document.getElementById('usar_prioridad').checked,
    no_estacionario: document.getElementById('no_estacionario').checked,
    usar_welch:    document.getElementById('usar_welch').checked,
    usar_optimizacion: document.getElementById('usar_optimizacion').checked,
    prob_urgente:  parseFloat(document.getElementById('prob_urgente').value),
    semilla_base:  42,
  };

  try {
    const resp = await fetch('/api/simular', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = await resp.json();
    if (!resp.ok) { throw new Error(data.error || 'Error al iniciar'); }
    iniciarPolling();
  } catch (err) {
    setBadge('⬤ Error', 'error');
    btn.disabled = false;
    alert('Error: ' + err.message);
  }
}

function iniciarPolling() {
  const etapas = [
    [10,  'Verificando estabilidad...'],
    [18,  'Detectando warm-up (Welch)...'],
    [28,  'Réplica representativa (DES)...'],
    [45,  'Ejecutando réplicas Montecarlo...'],
    [62,  'Comparación analítica M/M/c...'],
    [78,  'Análisis de sensibilidad...'],
    [90,  'Optimización automática de c...'],
    [97,  'Generando reporte final...'],
  ];
  let etapaIdx = 0;

  if (pollingInterval) clearInterval(pollingInterval);
  pollingInterval = setInterval(async () => {
    try {
      // Avanzar la barra de progreso suavemente
      if (etapaIdx < etapas.length) {
        setProgress(etapas[etapaIdx][0], etapas[etapaIdx][1]);
        etapaIdx++;
      }

      const resp = await fetch('/api/estado');
      const estado = await resp.json();

      if (estado.completado) {
        clearInterval(pollingInterval);
        setProgress(100, 'Simulación completada ✓');
        setTimeout(() => cargarResultados(), 400);
        document.getElementById('btn-run').disabled = false;
        setBadge('⬤ Completado', '');
      } else if (estado.error) {
        clearInterval(pollingInterval);
        setBadge('⬤ Error', 'error');
        setProgress(0, 'Error: ' + estado.error);
        document.getElementById('btn-run').disabled = false;
        alert('Error en la simulación: ' + estado.error);
      }
    } catch (e) { /* continuar */ }
  }, 900);
}

// ── Cargar resultados ─────────────────────────────────────
async function cargarResultados() {
  try {
    const resp = await fetch('/api/resultados');
    if (!resp.ok) return;
    const data = await resp.json();
    mostrarResultados(data);
  } catch (e) {
    console.error('Error cargando resultados:', e);
  }
}

async function cargarResultadosGuardados() {
  setBadge('⬤ Cargando...', 'running');
  await cargarResultados();
  setBadge('⬤ Listo', '');
}

// ── Mostrar resultados ────────────────────────────────────
function mostrarResultados(data) {
  // Mostrar KPIs
  const mc = data.montecarlo_puro_mmc || data.montecarlo || {};
  const medias = mc.medias || {};
  document.getElementById('kpi-wq').textContent   = (medias.wq_promedio || 0).toFixed(2) + ' min';
  document.getElementById('kpi-lq').textContent   = (medias.lq_promedio || 0).toFixed(2);
  document.getElementById('kpi-rho').textContent  = (medias.rho || 0).toFixed(3);

  const opt = data.optimizacion || {};
  document.getElementById('kpi-optimo').textContent = opt.c_optimo
    ? `${opt.c_optimo} (Wq=${(opt.wq_optimo||0).toFixed(1)} min)`
    : '—';

  document.getElementById('kpis').style.display = 'grid';
  document.getElementById('tabs').style.display = 'flex';
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('analitico-content').style.display = 'block';

  // Poblar tablas
  poblarTablaAnalitica(data.analitico || {});
  poblarTablaSimulada(medias, mc.ic_95 || {});
  poblarTablaComparacion(data.comparacion_analitico_vs_simulado || {});
  poblarIC(medias, mc.ic_95 || {});

  // Cargar gráficas
  cargarGraficas();

  // Reporte JSON
  document.getElementById('json-view').textContent = JSON.stringify(data, null, 2);
}

function poblarTablaAnalitica(anal) {
  const tabla = document.getElementById('tabla-analitica');
  const filas = [
    ['ρ (utilización)',    anal.rho,  'rho'],
    ['P₀ (sistema vacío)', anal.P0,   'p0'],
    ['Lq (clientes cola)', anal.Lq,   'lq'],
    ['Wq (tiempo cola)',   anal.Wq ? anal.Wq.toFixed(4) + ' min' : '—', 'wq'],
    ['L (clientes sist.)', anal.L,    'l'],
    ['W (tiempo sistema)', anal.W ? anal.W.toFixed(4) + ' min' : '—', 'w'],
  ];
  tabla.innerHTML = `<thead><tr><th>Métrica</th><th>Valor analítico</th></tr></thead><tbody>
    ${filas.map(([label, val]) => `<tr><td>${label}</td><td class="val-mono">${typeof val === 'number' ? val.toFixed(4) : (val||'—')}</td></tr>`).join('')}
  </tbody>`;
}

function poblarTablaSimulada(medias, ic) {
  const tabla = document.getElementById('tabla-simulada');
  const fmt = (v, ic_key) => {
    if (v === undefined) return '—';
    const lo = ic[ic_key] ? ic[ic_key][0].toFixed(3) : '—';
    const hi = ic[ic_key] ? ic[ic_key][1].toFixed(3) : '—';
    return `<span class="val-mono">${v.toFixed(4)}</span><br><small style="color:var(--text3)">[${lo}, ${hi}]</small>`;
  };
  tabla.innerHTML = `<thead><tr><th>Métrica</th><th>Media simulada (IC 95%)</th></tr></thead><tbody>
    <tr><td>Wq promedio (min)</td><td>${fmt(medias.wq_promedio, 'wq_promedio')}</td></tr>
    <tr><td>Ws promedio (min)</td><td>${fmt(medias.ws_promedio, 'ws_promedio')}</td></tr>
    <tr><td>Lq (clientes)</td><td>${fmt(medias.lq_promedio, 'lq_promedio')}</td></tr>
    <tr><td>ρ utilización</td><td>${fmt(medias.rho, 'rho')}</td></tr>
  </tbody>`;
}

function poblarTablaComparacion(comp) {
  const tabla = document.getElementById('tabla-comparacion');
  const etiquetas = { Wq: 'Wq (min)', Lq: 'Lq (clientes)', rho: 'ρ utilización' };
  const filas = Object.entries(comp).map(([k, v]) => {
    const err = v['error_relativo_%'];
    let cls = err < 10 ? 'error-good' : err < 25 ? 'error-warn' : 'error-bad';
    return `<tr>
      <td>${etiquetas[k] || k}</td>
      <td class="val-mono">${v.analitico.toFixed(4)}</td>
      <td class="val-mono">${v.simulado.toFixed(4)}</td>
      <td><span class="error-badge ${cls}">${err.toFixed(2)}%</span></td>
    </tr>`;
  }).join('');
  tabla.innerHTML = `<thead><tr><th>Métrica</th><th>Analítico</th><th>Simulado</th><th>Error relativo</th></tr></thead><tbody>${filas}</tbody>`;
}

function poblarIC(medias, ic) {
  const container = document.getElementById('ic-chart');
  const metricas = [
    { key: 'wq_promedio', label: 'Wq (min)' },
    { key: 'ws_promedio', label: 'Ws (min)' },
    { key: 'lq_promedio', label: 'Lq (clientes)' },
    { key: 'rho',         label: 'ρ utilización' },
  ];

  container.innerHTML = metricas.map(({ key, label }) => {
    if (!ic[key]) return '';
    const lo  = ic[key][0];
    const hi  = ic[key][1];
    const med = medias[key] || 0;
    const rng = hi - lo || 1;

    // Normalizar a 0–100 sobre el rango extendido
    const ext  = rng * 0.5;
    const total = rng + 2 * ext;
    const leftPct  = ((lo - (lo - ext)) / total * 100).toFixed(1);
    const widthPct = (rng / total * 100).toFixed(1);
    const medPct   = (((med - (lo - ext)) / total) * 100).toFixed(1);

    return `<div class="ic-row">
      <span class="ic-label">${label}</span>
      <div class="ic-bar-wrap">
        <div class="ic-track"></div>
        <div class="ic-fill" style="left:${leftPct}%;width:${widthPct}%"></div>
        <div class="ic-dot" style="left:${medPct}%"></div>
      </div>
      <span class="ic-nums">[${lo.toFixed(3)}, ${hi.toFixed(3)}]</span>
    </div>`;
  }).join('');
}

// ── Cargar gráficas ───────────────────────────────────────
const GRAFICAS = [
  { key: '01_evolucion_temporal',          label: 'Evolución temporal',           tab: 'graficas' },
  { key: '02_histograma_wq',               label: 'Histograma Wq',                tab: 'graficas' },
  { key: '03_wq_vs_servidores',            label: 'Wq vs Nº de técnicos',         tab: 'graficas' },
  { key: '04_rho_vs_lambda',               label: 'ρ vs λ',                       tab: 'graficas' },
  { key: '05_distribucion_medias_replicas',label: 'Distribución medias (TCL)',     tab: 'graficas' },
  { key: '06_heatmap_wq',                  label: 'Heatmap Wq (sensibilidad)',     tab: 'sensibilidad' },
  { key: '07_heatmap_rho',                 label: 'Heatmap ρ (sensibilidad)',      tab: 'sensibilidad' },
  { key: '08_optimizacion_servidores',     label: 'Optimización c mínimo',         tab: 'extensiones' },
  { key: '09_llegadas_no_estacionarias',   label: 'Llegadas no estacionarias',     tab: 'extensiones' },
  { key: '10_reneging_abandonos',          label: 'Reneging / Abandonos',          tab: 'extensiones' },
  { key: '11_welch_warmup',                label: 'Warm-up (método de Welch)',      tab: 'extensiones' },
];

async function cargarGraficas() {
  const grids = {
    graficas:     document.getElementById('graficas-grid'),
    sensibilidad: document.getElementById('sensibilidad-grid'),
    extensiones:  document.getElementById('extensiones-grid'),
  };
  Object.values(grids).forEach(g => g.innerHTML = '');

  for (const g of GRAFICAS) {
    const url = `/api/grafica/${g.key}.png?t=${Date.now()}`;
    try {
      const resp = await fetch(url, { method: 'HEAD' });
      if (!resp.ok) continue;
    } catch { continue; }

    const card = document.createElement('div');
    card.className = 'grafica-card';
    card.innerHTML = `
      <div class="grafica-header">
        <span class="grafica-nombre">${g.label}</span>
        <span class="grafica-expand">↗ ampliar</span>
      </div>
      <div class="grafica-img-wrap">
        <img class="grafica-img" src="${url}" alt="${g.label}" loading="lazy" />
      </div>`;
    card.addEventListener('click', () => abrirLightbox(url, g.label));
    grids[g.tab].appendChild(card);
  }
}

// ── Lightbox ──────────────────────────────────────────────
function abrirLightbox(src, caption) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox-caption').textContent = caption;
  document.getElementById('lightbox').classList.add('open');
  document.body.style.overflow = 'hidden';
}
function cerrarLightbox() {
  document.getElementById('lightbox').classList.remove('open');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') cerrarLightbox(); });

// ── Inicialización: intentar cargar resultados existentes ──
window.addEventListener('load', async () => {
  try {
    const r = await fetch('/api/resultados');
    if (r.ok) {
      const data = await r.json();
      if (data && data.analitico) mostrarResultados(data);
    }
  } catch {}
});
