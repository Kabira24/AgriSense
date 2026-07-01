/* ── AgriSense AI — app.js ──────────────────────────────────────────────────
   Handles:
     • Particle canvas animation
     • Slider ↔ number input sync with real-time Farm Health Score updates
     • Preset quick-fill buttons
     • Weather auto-fill widget (geocode → summary → prefill form fields)
     • 7-day weather forecast strip (results panel)
     • API call to /predict        (crop recommendation, port 8000)
     • API call to /profit         (profit estimation service, port 8001)
     • API call to /weather        (weather service, port 8002)
     • API call to /risk           (risk assessment engine, port 8003)
     • API call to /advise         (AI advisor / Gemini, port 8004)
     • API call to /planner        (crop operations planner, port 8005)
     • API call to /latest-sensor  (IoT sensor feed, port 8006)
     • Input mode toggle: Manual Mode vs Live Sensor Mode
     • Live sensor polling: fetches /latest-sensor every 10 s in Sensor Mode
     • Tab switcher navigation controller
     • Real-time Farm Health Score calculation (pH, Temperature, Moisture)
     • Live Diagnostics health polling for all 7 microservices
     • Operations timeline rendering with planting window status
     • Interactive crop navigation shortcut buttons
   ─────────────────────────────────────────────────────────────────────────── */

'use strict';

// ── Backend URL Configuration ─────────────────────────────────────────────────
// Single configurable URL replaces the previous 7 separate port constants.
// To configure for production, set window.AGRISENSE_BACKEND_URL in index.html
// (see the <script> block near the bottom of index.html).
const BACKEND_URL = (() => {
  // 1. Explicit override from index.html (production Render URL)
  if (window.AGRISENSE_BACKEND_URL) return window.AGRISENSE_BACKEND_URL;
  // 2. Local development fallback
  if (window.location.hostname === 'localhost' ||
      window.location.hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  // 3. Same-origin fallback (e.g. if backend and frontend are co-hosted)
  return window.location.origin;
})();

// All service bases now point to the same unified backend.
// The variable names are preserved so no other line in this file needs to change.
const API_BASE     = BACKEND_URL;
const PROFIT_BASE  = BACKEND_URL;
const WEATHER_BASE = BACKEND_URL;
const RISK_BASE    = BACKEND_URL;
const ADVISOR_BASE = BACKEND_URL;
const PLANNER_BASE = BACKEND_URL;
const SENSOR_BASE  = BACKEND_URL;

// ── Sensor Mode State ────────────────────────────────────────────────────────
// Tracks whether the form is in Manual or Live Sensor polling mode.
let _sensorMode    = 'manual';   // 'manual' | 'sensor'
let _sensorPollId  = null;       // setInterval handle for the 10 s polling loop


// Stores the last successful responses for cross-service wiring.
let _lastPredictResult  = null;
let _lastInputValues    = null;
let _lastProfitResult   = null;
let _lastWeatherGeo     = null;    // { lat, lon, city, state, country } from geocode
let _lastWeatherSummary = null;    // full summary from /weather/summary (has rainfall_forecast_mm)

// Globally accessible tab switcher function
let switchTab = null;

// ── Crop emoji map ────────────────────────────────────────────────────────────
const CROP_EMOJI = {
  apple: '🍎', banana: '🍌', blackgram: '🫘', chickpea: '🌱',
  coconut: '🥥', coffee: '☕', cotton: '🌿', grapes: '🍇',
  jute: '🌾', kidneybeans: '🫘', lentil: '🌿', maize: '🌽',
  mango: '🥭', mothbeans: '🫘', mungbean: '🫘', muskmelon: '🍈',
  orange: '🍊', papaya: '🍈', pigeonpeas: '🌱', pomegranate: '🍎',
  rice: '🌾', watermelon: '🍉',
};

// ── Services check mappings ──────────────────────────────────────────────────
const SERVICES = [
  { name: 'predict', url: `${API_BASE}/health` },
  { name: 'profit',  url: `${PROFIT_BASE}/health` },
  { name: 'weather', url: `${WEATHER_BASE}/health` },
  { name: 'risk',    url: `${RISK_BASE}/health` },
  { name: 'advisor', url: `${ADVISOR_BASE}/health` },
  { name: 'planner', url: `${PLANNER_BASE}/health` },
  { name: 'sensor',  url: `${SENSOR_BASE}/health` },  // IoT sensor feed (port 8006)
];

// ── Weather condition icon map (Open-Meteo precipitation → icon) ─────────────
function precipIcon(mm) {
  if (mm === null || mm === undefined) return '☁️';
  if (mm === 0)   return '☀️';
  if (mm < 1)     return '🌤️';
  if (mm < 5)     return '🌦️';
  if (mm < 15)    return '🌧️';
  return '⛈️';
}

// ── Presets ───────────────────────────────────────────────────────────────────
const PRESETS = {
  tropical:   { ph: 6.0,  temperature: 30.0, humidity: 90.0, rainfall: 280.0 },
  'semi-arid':{ ph: 7.5,  temperature: 35.0, humidity: 35.0, rainfall: 50.0  },
  temperate:  { ph: 6.5,  temperature: 18.0, humidity: 65.0, rainfall: 100.0 },
  monsoon:    { ph: 6.8,  temperature: 28.0, humidity: 85.0, rainfall: 200.0 },
};

// ── Particle canvas ───────────────────────────────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', () => { resize(); initParticleSet(); });

  function initParticleSet() {
    const count = Math.floor((W * H) / 14000);
    particles = Array.from({ length: count }, () => ({
      x:  Math.random() * W,
      y:  Math.random() * H,
      r:  Math.random() * 1.5 + 0.4,
      dx: (Math.random() - 0.5) * 0.3,
      dy: (Math.random() - 0.5) * 0.3,
      a:  Math.random() * 0.5 + 0.1,
    }));
  }
  initParticleSet();

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(16,185,129,${p.a})`;
      ctx.fill();
      p.x += p.dx; p.y += p.dy;
      if (p.x < 0 || p.x > W) p.dx *= -1;
      if (p.y < 0 || p.y > H) p.dy *= -1;
    });
    requestAnimationFrame(draw);
  }
  draw();
}

// ── Slider ↔ input sync ───────────────────────────────────────────────────────
const FIELDS = ['ph', 'temperature', 'humidity', 'rainfall'];

function syncSliders() {
  FIELDS.forEach(name => {
    const input  = document.getElementById(name);
    const slider = document.getElementById(`${name}-slider`);
    if (!input || !slider) return;
    input.value = slider.value;
    slider.addEventListener('input', () => {
      input.value = slider.value;
      clearError(name);
      updateFarmHealthScore();
    });
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      if (!isNaN(v)) slider.value = v;
      clearError(name);
      updateFarmHealthScore();
    });
  });
}

// Helper: set a field + slider value
function setFieldValue(name, value) {
  const input  = document.getElementById(name);
  const slider = document.getElementById(`${name}-slider`);
  if (input)  input.value  = value;
  if (slider) slider.value = value;
  clearError(name);
  updateFarmHealthScore();
}

// ── Live Diagnostics & Health Checks ──────────────────────────────────────────
async function checkHealth() {
  const dot   = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  
  // 1. Check core predictor liveness (original sidebar dot)
  let predictOk = false;
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) predictOk = true;
  } catch (_) { /* ignore */ }

  if (predictOk) {
    dot.classList.add('online');
    label.textContent = 'Model Online';
  } else {
    dot.classList.remove('online');
    label.textContent = 'API Offline';
  }

  // 2. Query all 7 microservices concurrently for the System Health tab
  SERVICES.forEach(async svc => {
    const sDot = document.getElementById(`health-${svc.name}-dot`);
    const sLabel = document.getElementById(`health-${svc.name}-label`);
    if (!sDot || !sLabel) return;

    let ok = false;
    try {
      const res = await fetch(svc.url, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const payload = await res.json().catch(() => null);
        ok = res.ok;
      }
    } catch (_) { /* ignore */ }

    const sBadge = sDot.closest('.diagnostic-status');
    if (ok) {
      sDot.classList.add('online');
      sLabel.textContent = 'online';
      sLabel.style.color = '#15803d';
      if (sBadge) sBadge.classList.add('online-status');
    } else {
      sDot.classList.remove('online');
      sLabel.textContent = 'offline';
      sLabel.style.color = '#dc2626';
      if (sBadge) sBadge.classList.remove('online-status');
    }
  });
}

// ── Dynamic Farm Health Score Calibrator ──────────────────────────────────────
function updateFarmHealthScore() {
  const phInput = document.getElementById('ph');
  const tempInput = document.getElementById('temperature');
  const humInput = document.getElementById('humidity');
  const rainInput = document.getElementById('rainfall');

  if (!phInput || !tempInput || !humInput || !rainInput) return;

  const ph = parseFloat(phInput.value) || 6.5;
  const temperature = parseFloat(tempInput.value) || 28.0;
  const humidity = parseFloat(humInput.value) || 82.0;
  const rainfall = parseFloat(rainInput.value) || 120.0;

  // pH score: sweet spot 6.5
  const phScore = Math.max(0, 100 - Math.abs(ph - 6.5) * 25);
  // Temp score: sweet spot 24C
  const tempScore = Math.max(0, 100 - Math.abs(temperature - 24) * 3);
  // Moisture score: humidity sweet spot 70%, rainfall sweet spot 1000mm
  const humScore = Math.max(0, 100 - Math.abs(humidity - 70) * 1.5);
  const rainScore = rainfall >= 1000
    ? Math.max(0, 100 - (rainfall - 1000) * 0.03)
    : Math.max(0, (rainfall / 1000) * 100);
  const moistureScore = (humScore * 0.6) + (rainScore * 0.4);

  // Weighted total score
  const totalScore = Math.round((phScore * 0.4) + (tempScore * 0.3) + (moistureScore * 0.3));

  // Update DOM values
  document.getElementById('health-score-number').textContent = `${totalScore}`;
  
  const statusEl = document.getElementById('health-score-status');
  if (totalScore >= 85) {
    statusEl.textContent = 'Optimal';
    statusEl.style.color = '#16a34a';
  } else if (totalScore >= 70) {
    statusEl.textContent = 'Good';
    statusEl.style.color = '#3b82f6';
  } else if (totalScore >= 50) {
    statusEl.textContent = 'Moderate';
    statusEl.style.color = '#f59e0b';
  } else {
    statusEl.textContent = 'Poor';
    statusEl.style.color = '#ef4444';
  }

  // Update SVG radial stroke offset
  const gaugeFill = document.getElementById('health-gauge-fill');
  if (gaugeFill) {
    // 440 circumference
    const offset = 440 - (440 * totalScore / 100);
    gaugeFill.style.strokeDashoffset = offset;
  }

  // Update breakdown progress bars & labels
  document.getElementById('health-ph-val').textContent = ph.toFixed(1);
  document.getElementById('health-ph-bar').style.width = `${phScore}%`;

  document.getElementById('health-temp-val').textContent = `${temperature.toFixed(1)}°C`;
  document.getElementById('health-temp-bar').style.width = `${tempScore}%`;

  document.getElementById('health-moisture-val').textContent = `${humidity.toFixed(1)}%`;
  document.getElementById('health-moisture-bar').style.width = `${moistureScore}%`;
}

// ── Validation ────────────────────────────────────────────────────────────────
function setError(name, msg) {
  const el = document.getElementById(`err-${name}`);
  const grp = document.getElementById(`group-${name}`);
  if (el)  el.textContent = msg;
  if (grp) grp.classList.toggle('has-error', !!msg);
}
function clearError(name) { setError(name, ''); }

function validateForm(data) {
  let ok = true;
  const rules = [
    { name: 'ph',          min: 0,   max: 14,   label: 'pH'          },
    { name: 'temperature', min: -10, max: 60,   label: 'Temperature' },
    { name: 'humidity',    min: 0,   max: 100,  label: 'Humidity'    },
    { name: 'rainfall',    min: 0,   max: 5000, label: 'Rainfall'    },
  ];
  FIELDS.forEach(n => clearError(n));
  rules.forEach(({ name, min, max, label }) => {
    const v = data[name];
    if (v === '' || v === null || isNaN(v)) {
      setError(name, `${label} is required.`);
      ok = false;
    } else if (v < min || v > max) {
      setError(name, `${label} must be between ${min} and ${max}.`);
      ok = false;
    }
  });
  return ok;
}

// ── Collect form values ───────────────────────────────────────────────────────
function collectValues() {
  return Object.fromEntries(
    FIELDS.map(n => [n, parseFloat(document.getElementById(n).value)])
  );
}

// ── Fill from preset ──────────────────────────────────────────────────────────
function fillPreset(key) {
  const p = PRESETS[key];
  FIELDS.forEach(n => {
    setFieldValue(n, p[n]);
  });
  // Clear any weather geo lock when a preset is chosen
  clearWeatherFill();
}

// ── Predict API call ──────────────────────────────────────────────────────────
async function callPredict(payload) {
  const res = await fetch(`${API_BASE}/predict`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Profit API call ───────────────────────────────────────────────────────────
async function callProfit(crop, yieldTHa = 2.5, areaHa = 1, state = null) {
  const res = await fetch(`${PROFIT_BASE}/profit`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ crop, yield_t_ha: yieldTHa, area_ha: areaHa, state }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Risk API call ──────────────────────────────────────────────────────
async function callRisk(crop, ph, temperature, humidity, rainfall) {
  const res = await fetch(`${RISK_BASE}/risk`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ crop, ph, temperature, humidity, rainfall }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Weather API calls ─────────────────────────────────────────────────────────

/** Resolve city name → { lat, lon, city, state, country, timezone } */
async function callGeocode(city, country = null) {
  const params = new URLSearchParams({ city });
  if (country) params.append('country', country);
  const res = await fetch(`${WEATHER_BASE}/weather/geocode?${params}`, {
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/** Crop-ready weather summary: temperature, humidity, rainfall (annualised) */
async function callWeatherSummary(lat, lon, days = 14) {
  const params = new URLSearchParams({ lat, lon, days });
  const res = await fetch(`${WEATHER_BASE}/weather/summary?${params}`, {
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/** Full 7-day daily forecast for the weather strip */
async function callWeatherForecast(lat, lon, days = 7) {
  const params = new URLSearchParams({ lat, lon, days });
  const res = await fetch(`${WEATHER_BASE}/weather?${params}`, {
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Advisor API call ─────────────────────────────────────────────────────
async function callAdvise(predictResult, inputValues) {
  const topCrop = predictResult.top_crops[0].crop;
  const p = _lastProfitResult;
  const g = _lastWeatherGeo;

  // ── Step 1: Fetch real risk data from the Risk Engine ─────────────────
  let riskData = null;
  try {
    riskData = await callRisk(
      topCrop,
      inputValues.ph,
      inputValues.temperature,
      inputValues.humidity,
      inputValues.rainfall,
    );
  } catch (_) {
    // Risk service unreachable — send null scores; advisor will still work
  }

  // ── Step 2: Resolve correct rainfall_forecast_mm ───────────────────
  const rainfallForecastMm = _lastWeatherSummary
    ? _lastWeatherSummary.rainfall_forecast_mm
    : null;

  const payload = {
    crop_recommendation: {
      top_crops: predictResult.top_crops.map(c => ({
        rank:       c.rank,
        crop:       c.crop,
        confidence: c.confidence,
      })),
    },
    profit_estimate: {
      crop:                topCrop,
      production_quintals: p ? p.production_quintals : null,
      market_price:        p ? p.market_price_modal  : null,
      revenue:             p ? p.revenue              : null,
      cost:                p ? p.cost                 : null,
      profit:              p ? p.profit               : null,
      profit_margin_pct:   p ? p.profit_margin_pct   : null,
      break_even_price:    p ? p.break_even_price     : null,
      price_available:     p ? p.price_available      : false,
    },
    weather: {
      temperature:          inputValues.temperature,
      humidity:             inputValues.humidity,
      rainfall_forecast_mm: rainfallForecastMm,
      summary: g
        ? `${g.city}${g.state ? ', ' + g.state : ''} — Temp ${inputValues.temperature}°C, Humidity ${inputValues.humidity}%, Annual rainfall ${inputValues.rainfall} mm/yr`
        : `Temperature ${inputValues.temperature}°C, Humidity ${inputValues.humidity}%, Annual rainfall ${inputValues.rainfall} mm/yr`,
    },
    risk_assessment: riskData ? {
      crop:           riskData.crop,
      soil_risk:      riskData.soil.score,
      disease_risk:   riskData.disease.score,
      water_risk:     riskData.water.score,
      weather_risk:   riskData.weather.score,
      composite_risk: riskData.overall.score,
      risk_level:     riskData.overall.level,
    } : {
      crop:           topCrop,
      composite_risk: 0,
      risk_level:     'Unknown',
    },
  };

  const res = await fetch(`${ADVISOR_BASE}/advise`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Typewriter effect ─────────────────────────────────────────────────────────
function typeWriter(el, text, speed = 14) {
  el.textContent = '';
  const lines = text.split('\n');
  let lineIdx = 0, charIdx = 0;
  function tick() {
    if (lineIdx >= lines.length) return;
    const line = lines[lineIdx];
    if (charIdx < line.length) {
      el.textContent += line[charIdx++];
      setTimeout(tick, speed);
    } else {
      el.textContent += '\n';
      lineIdx++; charIdx = 0;
      setTimeout(tick, speed * 2);
    }
  }
  tick();
}

// ── Render crop recommendations ───────────────────────────────────────────────
function renderResults(data) {
  const echo = document.getElementById('results-echo');
  const labels = { ph: '🧪 pH', temperature: '🌡️ Temp', humidity: '💧 Humidity', rainfall: '🌧️ Rainfall' };
  const units  = { ph: '',       temperature: ' °C',     humidity: ' %',          rainfall: ' mm'       };
  echo.innerHTML = FIELDS.map(n => `
    <span class="echo-chip">
      ${labels[n]}: <strong>${data.input_echo[n]}${units[n]}</strong>
    </span>`).join('');

  const list = document.getElementById('crops-list');
  list.innerHTML = data.top_crops.map(crop => {
    const emoji = CROP_EMOJI[crop.crop.toLowerCase()] || '🌿';
    const pct   = crop.confidence.toFixed(1);
    return `
      <div class="crop-card rank-${crop.rank}" id="crop-card-${crop.rank}">
        <div class="crop-glow"></div>
        <div class="crop-row">
          <div class="crop-row-left">
            <div class="crop-rank-badge">#${crop.rank}</div>
            <span class="crop-emoji">${emoji}</span>
            <span class="crop-name">${crop.crop}</span>
          </div>
          <span class="confidence-pct">${pct}%</span>
        </div>
        <div class="confidence-bar-wrap">
          <div class="confidence-bar-track">
            <div class="confidence-bar-fill" data-pct="${crop.confidence / 100}"
                 style="transform:scaleX(0)"></div>
          </div>
        </div>
        <div class="crop-card-footer">
          <button type="button" class="crop-action-link select-crop-plan" data-crop="${crop.crop.toLowerCase()}">
            📅 Plan Operations
          </button>
        </div>
      </div>`;
  }).join('');

  requestAnimationFrame(() => {
    document.querySelectorAll('.confidence-bar-fill').forEach(bar => {
      bar.style.transform = `scaleX(${parseFloat(bar.dataset.pct)})`;
    });
  });

  // Attach navigation shortcut triggers on recommended crop action links
  setTimeout(() => {
    document.querySelectorAll('.select-crop-plan').forEach(btn => {
      btn.addEventListener('click', () => {
        const crop = btn.dataset.crop;
        const plannerSelect = document.getElementById('planner-crop');
        if (plannerSelect) {
          plannerSelect.value = crop;
        }
        const plannerDate = document.getElementById('planner-date');
        if (plannerDate && !plannerDate.value) {
          plannerDate.value = new Date().toISOString().split('T')[0];
        }
        if (switchTab) switchTab('planner');
      });
    });
  }, 100);
}

// ── Weather strip renderer ────────────────────────────────────────────────────
function renderWeatherStrip(forecast, geoInfo) {
  const titleEl = document.getElementById('ws-title');
  if (geoInfo) {
    const loc = [geoInfo.city, geoInfo.state, geoInfo.country].filter(Boolean).join(', ');
    titleEl.textContent = `${forecast.forecast_days}-Day Forecast — ${loc}`;
  }

  const strip = document.getElementById('weather-strip');
  strip.innerHTML = forecast.daily.map(day => {
    const date   = new Date(day.date + 'T00:00:00');
    const dayStr = date.toLocaleDateString('en-IN', { weekday: 'short' });
    const dateStr= date.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
    const icon   = precipIcon(day.precipitation_mm);
    const tMax   = day.temp_max_c !== null ? `${day.temp_max_c}°` : '—';
    const tMin   = day.temp_min_c !== null ? `${day.temp_min_c}°` : '—';
    const rain   = day.precipitation_mm !== null ? `${day.precipitation_mm} mm` : '—';
    const hum    = day.humidity_mean_pct !== null ? `${Math.round(day.humidity_mean_pct)}%` : '—';

    return `
      <div class="ws-day">
        <span class="ws-day-name">${dayStr}</span>
        <span class="ws-day-date">${dateStr}</span>
        <span class="ws-day-icon">${icon}</span>
        <span class="ws-day-temp-max">${tMax}</span>
        <span class="ws-day-temp-min">${tMin}</span>
        <div class="ws-day-footer">
          <span class="ws-day-rain" title="Precipitation">${rain}</span>
          <span class="ws-day-hum"  title="Humidity">${hum}</span>
        </div>
      </div>`;
  }).join('');

  document.getElementById('weather-strip-section').hidden = false;
}

function hideWeatherStrip() {
  document.getElementById('weather-strip-section').hidden = true;
  document.getElementById('weather-strip').innerHTML = '';
}

// ── Weather widget UI helpers ─────────────────────────────────────────────────
function wwSetStatus(msg, isError = false) {
  const el = document.getElementById('ww-status');
  el.textContent = msg;
  el.className   = 'ww-status' + (isError ? ' ww-status--error' : ' ww-status--info');
  el.hidden = !msg;
}

// Custom wrapper to prevent widget collapse or overflow
function wwShowFilled(text) {
  document.getElementById('ww-filled-text').textContent = text;
  document.getElementById('ww-filled').hidden  = false;
  document.getElementById('ww-status').hidden  = true;
}

function clearWeatherFill() {
  _lastWeatherGeo = null;
  document.getElementById('ww-filled').hidden = true;
  document.getElementById('ww-status').hidden = true;
  document.getElementById('ww-city').value    = '';
  document.getElementById('ww-country').value = '';
}

// ── Profit card helpers ───────────────────────────────────────────────────────
function fmt(n, decimals = 0) {
  if (n === null || n === undefined) return '—';
  return '₹' + Number(n).toLocaleString('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtQ(n) {
  if (n === null || n === undefined) return '—';
  return Number(n).toLocaleString('en-IN', { maximumFractionDigits: 2 }) + ' q';
}

function profitClass(profit) {
  if (profit === null || profit === undefined) return '';
  return profit >= 0 ? 'profit-positive' : 'profit-negative';
}

function renderProfitCard(p) {
  const crop      = p.crop;
  const emoji     = CROP_EMOJI[crop] || '🌿';
  const available = p.price_available;

  const revenueRange = (available && p.revenue_min !== null && p.revenue_max !== null)
    ? `<span class="profit-range">${fmt(p.revenue_min)} – ${fmt(p.revenue_max)}</span>`
    : '';

  const profitLine = available && p.profit !== null
    ? `<div class="profit-stat profit-stat--highlight ${profitClass(p.profit)}">
         <span class="ps-label">Net Profit</span>
         <span class="ps-value">${fmt(p.profit)}</span>
         ${ p.profit_margin_pct !== null
            ? `<span class="ps-badge">${p.profit_margin_pct.toFixed(1)}% margin</span>`
            : '' }
       </div>`
    : `<div class="profit-stat profit-stat--unavail">
         <span class="ps-label">Price data unavailable</span>
         <span class="ps-value ps-muted">No Agmarknet data for ${crop}</span>
       </div>`;

  const priceSource = p.price_source
    ? `<span class="profit-source">📍 Price source: ${p.price_source}</span>`
    : '';
  const priceDate = p.price_date
    ? `<span class="profit-source">📅 ${p.price_date}</span>`
    : '';

  document.getElementById('profit-card').innerHTML = `
    <div class="profit-header">
      <div class="profit-header-left">
        <span class="profit-emoji">${emoji}</span>
        <div>
          <h3 class="profit-title">Profit Estimate — <span class="profit-crop">${crop}</span></h3>
          <p class="profit-sub">Per hectare · default yield ${p.yield_t_ha} t/ha · ${p.area_ha} ha</p>
        </div>
      </div>
      <div class="profit-meta">
        ${priceSource}
        ${priceDate}
      </div>
    </div>
    <div class="profit-stats">
      <div class="profit-stat">
        <span class="ps-label">Production</span>
        <span class="ps-value">${fmtQ(p.production_quintals)}</span>
      </div>
      <div class="profit-stat">
        <span class="ps-label">Market Price (modal)</span>
        <span class="ps-value">${available ? fmt(p.market_price_modal) + '/q' : '—'}</span>
      </div>
      <div class="profit-stat">
        <span class="ps-label">Revenue (modal)</span>
        <span class="ps-value">${available ? fmt(p.revenue) : '—'}</span>
        ${revenueRange}
      </div>
      <div class="profit-stat">
        <span class="ps-label">Cultivation Cost</span>
        <span class="ps-value">${fmt(p.cost)}</span>
      </div>
      ${profitLine}
      <div class="profit-stat">
        <span class="ps-label">Break-even Price</span>
        <span class="ps-value">${p.break_even_price !== null ? fmt(p.break_even_price) + '/q' : '—'}</span>
      </div>
    </div>
  `;
}

// ── Profit section state helpers ──────────────────────────────────────────────
function showProfitLoading() {
  document.getElementById('profit-loading').hidden = false;
  document.getElementById('profit-card').hidden    = true;
  document.getElementById('profit-error').hidden   = true;
  document.getElementById('profit-section').hidden = false;
}
function showProfitCard(p) {
  _lastProfitResult = p;
  renderProfitCard(p);
  document.getElementById('profit-loading').hidden = true;
  document.getElementById('profit-card').hidden    = false;
  document.getElementById('profit-error').hidden   = true;
  document.getElementById('profit-section').hidden = false;
}
function showProfitError(msg) {
  document.getElementById('profit-loading').hidden        = true;
  document.getElementById('profit-card').hidden           = true;
  document.getElementById('profit-error').hidden          = false;
  document.getElementById('profit-error-msg').textContent = msg;
  document.getElementById('profit-section').hidden        = false;
}
function hideProfitSection() {
  document.getElementById('profit-section').hidden = true;
  _lastProfitResult = null;
}

// ── Show / hide panels ────────────────────────────────────────────────────────
function showForm() {
  document.getElementById('predict-form').hidden    = false;
  document.getElementById('results-placeholder').hidden = false;
  document.getElementById('results-panel').hidden   = true;
  document.getElementById('error-panel').hidden     = true;
  document.getElementById('loading-overlay').hidden = true;
  _lastPredictResult  = null;
  _lastInputValues    = null;
  _lastWeatherSummary = null;
  hideProfitSection();
  hideWeatherStrip();
  resetAdvisorUI();
}
function showLoading() {
  document.getElementById('loading-overlay').hidden = false;
  document.getElementById('results-placeholder').hidden = true;
  document.getElementById('results-panel').hidden   = true;
  document.getElementById('error-panel').hidden     = true;
}
function hideLoading() {
  document.getElementById('loading-overlay').hidden = true;
}

function showResults(data, inputValues) {
  _lastPredictResult = data;
  _lastInputValues   = inputValues;
  hideProfitSection();
  hideWeatherStrip();
  resetAdvisorUI();
  renderResults(data);
  document.getElementById('predict-form').hidden    = false; // Keep form open on the left
  document.getElementById('results-placeholder').hidden = true;
  document.getElementById('results-panel').hidden   = false;
  document.getElementById('error-panel').hidden     = true;
  hideLoading();
}

function showError(msg) {
  document.getElementById('error-msg').textContent = msg;
  document.getElementById('predict-form').hidden    = false; // Keep form open
  document.getElementById('results-placeholder').hidden = true;
  document.getElementById('results-panel').hidden   = true;
  document.getElementById('error-panel').hidden     = false;
  hideLoading();
}

// ── Advisor UI helpers ────────────────────────────────────────────────────────
function resetAdvisorUI() {
  document.getElementById('advisor-panel').hidden   = true;
  document.getElementById('advisor-loading').hidden = true;
  document.getElementById('advisor-error').hidden   = true;
  document.getElementById('advisor-btn').hidden     = false;
  document.getElementById('advisor-advice').textContent = '';
}
function showAdvisorLoading() {
  document.getElementById('advisor-btn').hidden     = true;
  document.getElementById('advisor-loading').hidden = false;
  document.getElementById('advisor-panel').hidden   = true;
  document.getElementById('advisor-error').hidden   = true;
}
function showAdvisorResult(advice) {
  document.getElementById('advisor-loading').hidden = true;
  document.getElementById('advisor-error').hidden   = true;
  document.getElementById('advisor-panel').hidden   = false;
  document.getElementById('advisor-btn').hidden     = true;
  typeWriter(document.getElementById('advisor-advice'), advice);
}
function showAdvisorError(msg) {
  document.getElementById('advisor-loading').hidden  = true;
  document.getElementById('advisor-panel').hidden    = true;
  document.getElementById('advisor-btn').hidden      = false;
  document.getElementById('advisor-error').hidden    = false;
  document.getElementById('advisor-error-msg').textContent = msg;
}

// ── Sowing Calendar Operations Planner UI Helpers ─────────────────────────────
function showPlannerLoading() {
  document.getElementById('planner-placeholder').hidden = true;
  document.getElementById('planner-loading').hidden     = false;
  document.getElementById('planner-error').hidden       = true;
  document.getElementById('planner-results').hidden     = true;
}
function showPlannerError(msg) {
  document.getElementById('planner-placeholder').hidden = true;
  document.getElementById('planner-loading').hidden     = true;
  document.getElementById('planner-error').hidden       = false;
  document.getElementById('planner-error-msg').textContent = msg;
  document.getElementById('planner-results').hidden     = true;
}
function showPlannerResults(data) {
  document.getElementById('planner-placeholder').hidden = true;
  document.getElementById('planner-loading').hidden     = true;
  document.getElementById('planner-error').hidden       = true;
  document.getElementById('planner-results').hidden     = false;

  document.getElementById('planner-title-crop').textContent = `${data.display_name} Operations`;
  document.getElementById('planner-season').textContent = `Season: ${data.season}`;
  document.getElementById('planner-sowing-date').textContent = `Sowing Window: ${data.sowing_date}`;
  document.getElementById('planner-harvest-date').textContent = `Harvest Est: ${data.estimated_harvest_date}`;

  // Window status banner
  const banner = document.getElementById('planner-window-banner');
  banner.className = 'planting-window-banner';
  let emoji = '✅';
  if (data.planting_window_status === 'optimal') {
    banner.classList.add('optimal');
    emoji = '✅';
  } else if (data.planting_window_status.includes('suboptimal')) {
    banner.classList.add('suboptimal');
    emoji = '⚠️';
  } else {
    banner.classList.add('outside');
    emoji = '❌';
  }
  banner.textContent = `${emoji} ${data.window_message}`;

  // Chronological timeline rendering
  const timeline = document.getElementById('planner-timeline');
  timeline.innerHTML = data.schedule.map(evt => {
    const impClass = evt.importance.toLowerCase();
    const dateObj  = new Date(evt.planned_date);
    const dateStr  = dateObj.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });

    let detailsHtml = '';
    if (evt.details) {
      detailsHtml = Object.entries(evt.details)
        .filter(([_, val]) => val !== null && val !== undefined)
        .map(([key, val]) => {
          let label = key.replace(/_/g, ' ');
          let displayVal = typeof val === 'object' ? JSON.stringify(val) : val;
          return `<span class="timeline-detail-chip"><strong>${label}:</strong> ${displayVal}</span>`;
        }).join('');
    }

    return `
      <div class="timeline-item ${impClass}">
        <div class="timeline-badge"></div>
        <div class="timeline-content">
          <div class="timeline-item-header">
            <span class="timeline-stage">${evt.stage_name}</span>
            <span class="timeline-das">${evt.days_after_sowing === 0 ? 'Sowing Day' : `DAS: ${evt.days_after_sowing}`}</span>
          </div>
          <p class="timeline-desc">${evt.description}</p>
          <div class="timeline-details">
            <span class="importance-tag ${impClass}">${evt.importance}</span>
            <span class="timeline-detail-chip">📅 ${dateStr}</span>
            ${detailsHtml}
          </div>
        </div>
      </div>`;
  }).join('');
}

// ── Live Sensor Feed helpers ───────────────────────────────────────────

/**
 * Fetch /latest-sensor from sensor.py and update the dashboard card values.
 * Also optionally auto-fill the soil form if Sensor Mode is active.
 */
async function fetchAndApplySensorData(autofillForm = false) {
  try {
    const res = await fetch(`${SENSOR_BASE}/latest-sensor`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // ── Update sensor feed card on Dashboard ───────────────────────────
    const tempEl     = document.getElementById('sensor-temp');
    const humEl      = document.getElementById('sensor-humidity');
    const moistEl    = document.getElementById('sensor-moisture');
    const phEl       = document.getElementById('sensor-ph');
    if (tempEl)  tempEl.textContent  = data.temperature.toFixed(1);
    if (humEl)   humEl.textContent   = data.humidity.toFixed(1);
    if (moistEl) moistEl.textContent = data.soil_moisture.toFixed(1);
    if (phEl)    phEl.textContent    = data.ph.toFixed(1);

    // ── Auto-fill the Crop Analysis form in Sensor Mode ──────────────────
    if (autofillForm) {
      setFieldValue('temperature', data.temperature.toFixed(1));
      setFieldValue('humidity',    data.humidity.toFixed(1));
      setFieldValue('ph',          data.ph.toFixed(1));
      // soil_moisture is % vol; we don't map to rainfall directly,
      // but we can fill humidity. Rainfall remains user-editable in sensor mode.
      // To map soil_moisture → rainfall, add logic here in the future.
    }

  } catch (_) {
    // Sensor service offline — silently ignore; dashboard shows '--'
  }
}

/**
 * Start polling the sensor endpoint every 10 seconds.
 * Sensor data is applied to both the dashboard card and the form.
 */
function startSensorPolling() {
  if (_sensorPollId !== null) return;  // already running
  fetchAndApplySensorData(true);       // immediate first fetch
  _sensorPollId = setInterval(() => fetchAndApplySensorData(true), 10_000);
}

/**
 * Stop the sensor polling loop.
 */
function stopSensorPolling() {
  if (_sensorPollId !== null) {
    clearInterval(_sensorPollId);
    _sensorPollId = null;
  }
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  syncSliders();
  updateFarmHealthScore();
  checkHealth();
  setInterval(checkHealth, 30_000);

  // Initial sensor feed fetch (dashboard card only, no form autofill in manual mode)
  fetchAndApplySensorData(false);
  // Refresh sensor card every 30 s even in manual mode (dashboard display only)
  setInterval(() => { if (_sensorMode === 'manual') fetchAndApplySensorData(false); }, 30_000);

  // Set default calendar sowing date to today
  const plannerDate = document.getElementById('planner-date');
  if (plannerDate) {
    plannerDate.value = new Date().toISOString().split('T')[0];
  }

  // ── Tab Controller Logic ──────────────────────────────────────────────────
  const menuItems = document.querySelectorAll('.menu-item');
  const tabPanes  = document.querySelectorAll('.tab-pane');
  const pageTitleEl = document.getElementById('page-title');

  switchTab = function(tabId) {
    menuItems.forEach(mi => {
      if (mi.dataset.tab === tabId) {
        mi.classList.add('active');
        pageTitleEl.textContent = mi.textContent.trim();
      } else {
        mi.classList.remove('active');
      }
    });

    tabPanes.forEach(pane => {
      if (pane.id === `tab-${tabId}`) {
        pane.classList.add('active');
      } else {
        pane.classList.remove('active');
      }
    });
  };

  menuItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      switchTab(item.dataset.tab);
    });
  });

  // Welcome card shortcut buttons
  document.getElementById('shortcut-predict-btn').addEventListener('click', (e) => {
    e.preventDefault();
    switchTab('analysis');
  });
  document.getElementById('shortcut-planner-btn').addEventListener('click', (e) => {
    e.preventDefault();
    switchTab('planner');
  });

  // ── Input Mode Toggle (Manual ↔ Live Sensor) ───────────────────────────
  const modeManualBtn = document.getElementById('mode-manual-btn');
  const modeSensorBtn = document.getElementById('mode-sensor-btn');
  const sensorModeStatus = document.getElementById('sensor-mode-status');
  const sensorModeText   = document.getElementById('sensor-mode-text');

  function setInputMode(mode) {
    _sensorMode = mode;

    if (mode === 'sensor') {
      modeManualBtn.classList.remove('active');
      modeSensorBtn.classList.add('active');
      sensorModeStatus.hidden = false;
      sensorModeText.textContent = 'Live Sensor Mode active — fetching every 10 s…';
      // Disable the four soil inputs so user can’t override sensor values
      ['ph', 'temperature', 'humidity', 'ph-slider', 'temperature-slider', 'humidity-slider'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.setAttribute('readonly', 'true');
      });
      startSensorPolling();
    } else {
      // Manual mode
      modeManualBtn.classList.add('active');
      modeSensorBtn.classList.remove('active');
      sensorModeStatus.hidden = true;
      // Re-enable the inputs
      ['ph', 'temperature', 'humidity', 'ph-slider', 'temperature-slider', 'humidity-slider'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.removeAttribute('readonly');
      });
      stopSensorPolling();
    }
  }

  modeManualBtn.addEventListener('click', () => setInputMode('manual'));
  modeSensorBtn.addEventListener('click', () => setInputMode('sensor'));

  // Preset buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => fillPreset(btn.dataset.preset));
  });

  // Reset buttons
  document.getElementById('reset-btn').addEventListener('click', showForm);
  document.getElementById('error-reset-btn').addEventListener('click', showForm);

  // ── Weather widget toggle ─────────────────────────────────────────────────
  const wwToggle = document.getElementById('ww-toggle');
  const wwBody   = document.getElementById('ww-body');
  wwToggle.addEventListener('click', () => {
    const open = wwBody.hidden;
    wwBody.hidden = !open;
    wwToggle.setAttribute('aria-expanded', open);
    wwToggle.innerHTML = open
      ? 'Hide <span class="ww-chevron ww-chevron--open">▾</span>'
      : 'Show <span class="ww-chevron">▸</span>';
  });

  // ── Weather fetch button ──────────────────────────────────────────────────
  async function triggerWeatherFill() {
    const city    = document.getElementById('ww-city').value.trim();
    const country = document.getElementById('ww-country').value.trim() || null;
    if (!city) {
      wwSetStatus('Please enter a city name.', true);
      return;
    }

    const btn = document.getElementById('ww-fetch-btn');
    btn.disabled   = true;
    btn.textContent = 'Fetching…';
    wwSetStatus('Geocoding city…');
    document.getElementById('ww-filled').hidden = true;

    try {
      // Step 1: Geocode
      const geo = await callGeocode(city, country);
      _lastWeatherGeo = geo;
      wwSetStatus(`Found: ${geo.city}${geo.state ? ', ' + geo.state : ''}, ${geo.country} (${geo.latitude.toFixed(3)}, ${geo.longitude.toFixed(3)}) — Fetching 14-day summary…`);

      // Step 2: Summary (14-day for better rainfall annualisation)
      const summary = await callWeatherSummary(geo.latitude, geo.longitude, 14);
      _lastWeatherSummary = summary;   // store for advisor rainfall_forecast_mm

      // Step 3: Fill weather form fields (temperature, humidity, rainfall)
      if (summary.temperature !== null) setFieldValue('temperature', summary.temperature.toFixed(1));
      if (summary.humidity    !== null) setFieldValue('humidity',    summary.humidity.toFixed(1));
      if (summary.rainfall    !== null) setFieldValue('rainfall',    Math.round(summary.rainfall));

      const loc = [geo.city, geo.state, geo.country].filter(Boolean).join(', ');
      wwShowFilled(
        `Filled from ${loc} — ${summary.temperature}°C · ${summary.humidity}% humidity · ${Math.round(summary.rainfall)} mm/yr rainfall`
      );

    } catch (err) {
      wwSetStatus(`Error: ${err.message}`, true);
      _lastWeatherGeo = null;
    } finally {
      btn.disabled    = false;
      btn.textContent = 'Fetch';
    }
  }

  document.getElementById('ww-fetch-btn').addEventListener('click', triggerWeatherFill);

  // Also trigger on Enter inside the city input
  document.getElementById('ww-city').addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); triggerWeatherFill(); }
  });

  // Clear weather fill
  document.getElementById('ww-clear-btn').addEventListener('click', () => {
    clearWeatherFill();
    wwSetStatus('Weather data cleared. Fields are now editable.', false);
  });

  // ── Form submit ───────────────────────────────────────────────────────────
  document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const values = collectValues();
    if (!validateForm(values)) return;

    const btn = document.getElementById('predict-btn');
    btn.disabled = true;
    showLoading();

    try {
      const result = await callPredict(values);
      showResults(result, values);

      // ── Profit: fetch for the top crop (non-blocking) ─────────────────
      const topCrop = result.top_crops[0].crop;
      showProfitLoading();
      callProfit(topCrop)
        .then(p  => showProfitCard(p))
        .catch(() => showProfitError(
          `Profit service offline.\nStart it with:\n  python backend:/profit.py`
        ));

      // ── Weather strip: show if we have a location, fetch daily forecast ─
      if (_lastWeatherGeo) {
        const geo = _lastWeatherGeo;
        callWeatherForecast(geo.latitude, geo.longitude, 7)
          .then(fc => renderWeatherStrip(fc, geo))
          .catch(() => { /* silent fail — strip stays hidden */ });
      }

    } catch (err) {
      showError(
        `Could not reach the prediction service.\n\n${err.message}\n\n` +
        `Make sure the backend is running:\n  python backend:/predict.py`
      );
    } finally {
      btn.disabled = false;
    }
  });

  // ── Advisor Simulated Fallback for Demos ────────────────────────────────────
  function generateSimulatedAdvice(crop, inputs, profit) {
    const cName = crop.charAt(0).toUpperCase() + crop.slice(1);
    const ph = inputs.ph;
    const temp = inputs.temperature;
    const hum = inputs.humidity;
    const rain = inputs.rainfall;

    let phAdvice = ph < 5.5 
      ? "Your soil is acidic. Consider applying agricultural lime (calcium carbonate) to raise the pH to a more optimal 6.0 - 6.5 range."
      : ph > 7.5
      ? "Your soil is slightly alkaline. Consider adding organic matter or sulfur to lower the pH."
      : "Your soil pH of " + ph + " is in the optimal range. Maintain organic composting to sustain soil health.";

    let weatherAdvice = temp > 30 
      ? "High temperatures detected (" + temp + "°C). Ensure regular mulching to retain soil moisture and prevent thermal stress."
      : temp < 15
      ? "Cooler temperatures detected (" + temp + "°C). Slow growth cycles might be observed. Monitor seedling establishment."
      : "Temperatures are optimal (" + temp + "°C) for biochemical plant development.";

    let waterAdvice = hum > 80
      ? "High humidity (" + hum + "%) increases fungal disease risks. Implement proper spacing and weed control to promote airflow."
      : "Humidity is moderate (" + hum + "%). Normal transpiration rates expected.";

    let profitAdvice = profit && profit.price_available
      ? `Market rates for ${cName} are holding at ₹${profit.market_price_modal}/q. With estimated production of ${profit.production_quintals} q, focus on minimizing transport costs to secure your ₹${profit.profit.toLocaleString('en-IN')}/ha margin.`
      : `Market cost is estimated at ₹${profit ? profit.cost.toLocaleString('en-IN') : '45,000'}/ha. Monitor local Mandi pricing to ensure high profit margins.`;

    return `### Agronomic Strategy for ${cName}

1. **Soil Management**:
   ${phAdvice}

2. **Microclimate Adaptation**:
   ${weatherAdvice}

3. **Disease & Pest Prevention**:
   ${waterAdvice} Monitor crops for blight or rust symptoms during high humidity intervals.

4. **Financial Optimization**:
   ${profitAdvice} Focus on quality grading to command premium rates.`;
  }

  // ── Advisor button ─────────────────────────────────────────────────────────
  async function triggerAdvisor() {
    if (!_lastPredictResult || !_lastInputValues) return;
    showAdvisorLoading();
    try {
      const { advice } = await callAdvise(_lastPredictResult, _lastInputValues);
      showAdvisorResult(advice);
    } catch (err) {
      if (err.message.includes('GEMINI_API_KEY') || err.message.includes('503') || err.message.includes('Failed to fetch') || err.message.includes('Failed') || err.message.includes('HTTP 503')) {
        const advice = generateSimulatedAdvice(_lastPredictResult.top_crops[0].crop, _lastInputValues, _lastProfitResult);
        showAdvisorResult(advice + "\n\n*(Note: This advice was simulated locally because GEMINI_API_KEY is not set on the server)*");
      } else {
        showAdvisorError(
          `Could not reach the AI Advisor.\n${err.message}\n\n` +
          `Make sure the advisor service is running:\n  GEMINI_API_KEY=<key> python backend:/advisor.py`
        );
      }
    }
  }

  document.getElementById('advisor-btn').addEventListener('click', triggerAdvisor);
  document.getElementById('advisor-retry-btn').addEventListener('click', triggerAdvisor);
  document.getElementById('advisor-error-retry').addEventListener('click', triggerAdvisor);

  // ── Sowing Operations Planner form listener ──────────────────────────────────
  const plannerForm = document.getElementById('planner-form');
  if (plannerForm) {
    plannerForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const crop = document.getElementById('planner-crop').value;
      const sowingDate = document.getElementById('planner-date').value;

      if (!crop || !sowingDate) {
        showPlannerError('Please select a recommended crop and enter a target sowing date.');
        return;
      }

      showPlannerLoading();

      try {
        const res = await fetch(`${PLANNER_BASE}/planner/schedule`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ crop, sowing_date: sowingDate }),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        showPlannerResults(data);
      } catch (err) {
        showPlannerError(
          `Could not generate sowing calendar timeline.\n\n${err.message}\n\n` +
          `Make sure the Crop Planner service is running:\n  python backend:/planner.py`
        );
      }
    });
  }
});
