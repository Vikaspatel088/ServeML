const $ = (id) => document.getElementById(id);
const state = { metadata: null, metrics: null, toastTimer: null, health: null };
const pages = ['overview', 'playground', 'model', 'metrics', 'docs'];
const nice = (value) => String(value || '').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const safe = (value) => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function toast(message, error = false) {
  const el = $('toast'); el.textContent = message; el.className = `toast show${error ? ' error' : ''}`;
  clearTimeout(state.toastTimer); state.toastTimer = setTimeout(() => el.className = 'toast', 2600);
}
function navigate(page) {
  if (!pages.includes(page)) page = 'overview';
  document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === `page-${page}`));
  document.querySelectorAll('.nav-item').forEach(el => { const active = el.dataset.page === page; el.classList.toggle('active', active); if (active) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current'); });
  $('crumbPage').textContent = nice(page);
  $('sidebar').classList.remove('open'); $('mobileMenu').setAttribute('aria-expanded', 'false'); $('mobileMenu').setAttribute('aria-label', 'Open navigation');
  history.replaceState(null, '', `#${page}`);
  if (page === 'metrics' || page === 'overview') refreshMetrics();
}
document.querySelectorAll('[data-page]').forEach(el => el.addEventListener('click', () => navigate(el.dataset.page)));
document.querySelectorAll('[data-go]').forEach(el => el.addEventListener('click', () => navigate(el.dataset.go)));
$('mobileMenu').addEventListener('click', () => { const open = $('sidebar').classList.toggle('open'); $('mobileMenu').setAttribute('aria-expanded', String(open)); $('mobileMenu').setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation'); });
$('refresh').addEventListener('click', () => { refreshAll(); toast('Console data refreshed'); });
document.addEventListener('keydown', e => { if ((e.key.toLowerCase() === 'p') && !['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) navigate('playground'); if (e.key === 'Escape') { $('sidebar').classList.remove('open'); $('mobileMenu').setAttribute('aria-expanded', 'false'); $('mobileMenu').setAttribute('aria-label', 'Open navigation'); } });

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body?.error?.message || body?.detail?.message || body?.detail || `Request failed (${response.status})`);
  return body;
}
function formatSize(bytes) { return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`; }
function formatDuration(seconds) { return (seconds / 3600).toFixed(1); }
function formatDate(value) { return value ? new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—'; }

async function refreshMetadata() {
  const m = await api('/api/v1/model/metadata'); state.metadata = m;
  $('overviewModelName').textContent = m.name; $('overviewModelVersion').textContent = `v${m.version}`;
  $('overviewAlgorithm').textContent = m.algorithm.replace('Classifier', '');
  $('frameworkTag').textContent = m.framework.toUpperCase(); $('featureCount').textContent = m.feature_names.length; $('classCount').textContent = m.target_classes.length;
  $('accuracyValue').innerHTML = `${(m.accuracy * 100).toFixed(1)}<small>%</small>`;
  $('accuracyTrack').style.width = `${Math.max(0, Math.min(100, m.accuracy * 100))}%`;
  $('testMethod').textContent = `${100 - m.evaluation.test_size * 100} / ${m.evaluation.test_size * 100}`;
  $('trainingSamples').textContent = m.training_samples.toLocaleString(); $('artifactSize').textContent = formatSize(m.artifact.size_bytes);
  $('detailName').textContent = m.name; $('detailVersion').textContent = `v${m.version}`;
  $('detailStatus').textContent = nice(m.status); $('detailAlgorithm').textContent = m.algorithm;
  $('detailFramework').textContent = m.framework; $('detailDataset').textContent = m.dataset;
  $('detailEstimators').textContent = m.n_estimators.toLocaleString(); $('detailAccuracy').textContent = `${(m.accuracy * 100).toFixed(1)}%`;
  $('detailTraining').textContent = m.training_samples.toLocaleString(); $('detailTrained').textContent = formatDate(m.training_timestamp);
  $('detailArtifact').textContent = `${m.artifact.filename} · ${formatSize(m.artifact.size_bytes)}`;
  $('featureList').innerHTML = m.feature_names.map((name, i) => `<div class="contract-row"><span>${safe(nice(name))}</span><code>float · ${i < 2 ? 'sepal' : 'petal'}</code></div>`).join('');
  $('classList').innerHTML = m.target_classes.map((name, i) => `<span class="class-chip">${String(i).padStart(2, '0')} · ${safe(name)}</span>`).join('');
  $('modelContractCounts').innerHTML = `${m.feature_names.length} FEATURES <span>·</span> ${m.target_classes.length} CLASSES`;
}
async function refreshMetrics() {
  try {
    const m = await api('/api/v1/metrics'); state.metrics = m;
    const apiStatus = $('metricsApiStatus');
    if (apiStatus) { apiStatus.classList.remove('offline'); apiStatus.innerHTML = '<i></i> AUTO-REFRESH 10S'; }
    $('metricRequests').textContent = m.total_predictions.toLocaleString(); $('metricRequestNote').textContent = m.total_predictions ? `${m.successful_predictions.toLocaleString()} successful · ${m.failed_predictions.toLocaleString()} failed` : 'No requests recorded yet'; $('metricLatency').textContent = m.successful_predictions ? m.average_latency_ms.toFixed(2) : '—';
    $('metricSuccess').textContent = m.total_predictions ? m.success_rate.toFixed(1) : '—'; $('metricSuccessNote').textContent = m.total_predictions ? `${m.successful_predictions.toLocaleString()} successful predictions` : 'No requests recorded yet';
    $('metricUptime').textContent = formatDuration(m.uptime_seconds); $('updatedAt').textContent = `Updated ${new Date(m.observed_at).toLocaleTimeString()}`;
    $('metricsTotal').textContent = m.total_predictions.toLocaleString(); $('metricsLatency').textContent = m.successful_predictions ? m.average_latency_ms.toFixed(2) : '—';
    $('metricsSuccess').textContent = m.total_predictions ? m.success_rate.toFixed(1) : '—'; $('metricsFailures').textContent = m.total_predictions ? `${m.failed_predictions} failed ${m.failed_predictions === 1 ? 'request' : 'requests'}` : 'No requests recorded yet';
    renderRecent(m.recent_predictions, $('recentList')); renderRecent(m.recent_predictions, $('metricsRecent')); renderDistribution(m.prediction_distribution);
  } catch (error) {
    $('updatedAt').textContent = 'API unavailable';
    const apiStatus = $('metricsApiStatus');
    if (apiStatus) { apiStatus.classList.add('offline'); apiStatus.innerHTML = '<i></i> API UNAVAILABLE'; }
  }
}
async function refreshHealth() {
  const status = $('serviceStatus');
  const dot = $('sidebarHealthDot');
  try {
    const health = await api('/health');
    state.health = health;
    status.classList.remove('offline'); status.innerHTML = '<i></i> API ONLINE';
    $('sidebarHealthLabel').textContent = 'Runtime connected';
    dot.classList.remove('offline');
    $('modelStatusLine').classList.remove('offline'); $('modelStatusLine').innerHTML = '<span class="pulse"></span> MODEL ONLINE <span class="divider">/</span> LOCAL PROCESS';
    $('modelAvailabilityText').textContent = 'The trained classifier is loaded and ready to predict Iris species.';
    $('modelPageStatus').classList.remove('offline'); $('modelPageStatus').innerHTML = '<i></i> SERVING';
    $('detailStatus').textContent = 'Serving';
    $('loadedStep').classList.add('complete'); $('loadedStep').classList.remove('offline'); $('loadedState').textContent = 'DONE';
    $('servingStep').classList.remove('offline'); $('servingState').textContent = 'LIVE';
  } catch {
    state.health = null;
    status.classList.add('offline'); status.innerHTML = '<i></i> API OFFLINE';
    $('sidebarHealthLabel').textContent = 'Runtime unavailable';
    dot.classList.add('offline');
    $('modelStatusLine').classList.add('offline'); $('modelStatusLine').innerHTML = '<span class="pulse"></span> MODEL OFFLINE <span class="divider">/</span> HEALTH CHECK FAILED';
    $('modelAvailabilityText').textContent = 'The service is unavailable. Inference can resume when the model is ready.';
    $('modelPageStatus').classList.add('offline'); $('modelPageStatus').innerHTML = '<i></i> OFFLINE';
    $('detailStatus').textContent = 'Offline';
    $('loadedStep').classList.remove('complete'); $('loadedStep').classList.add('offline'); $('loadedState').textContent = 'UNAVAILABLE';
    $('servingStep').classList.add('offline'); $('servingState').textContent = 'OFFLINE';
  }
}
function renderRecent(items, target) {
  if (!items?.length) { target.innerHTML = `<div class="empty-state"><span class="empty-icon">⌁</span><strong>No inference activity yet</strong><p>Run your first prediction to see it here.</p><button class="text-button" data-go="playground">Open playground →</button></div>`; target.querySelector('[data-go]')?.addEventListener('click', () => navigate('playground')); return; }
  target.innerHTML = items.slice(0, 8).map(row => `<div class="recent-row"><span class="species-dot">✳</span><div><strong>${safe(nice(row.prediction))}</strong><small>${new Date(row.timestamp).toLocaleTimeString()} · ${row.latency_ms.toFixed(2)} ms</small></div><span class="confidence">${(row.confidence * 100).toFixed(1)}%</span></div>`).join('');
}
function renderDistribution(values) {
  const target = $('distribution'); const entries = Object.entries(values || {});
  if (!entries.length) { target.innerHTML = `<div class="empty-state compact"><span class="empty-icon">⌁</span><strong>No predictions to chart</strong><p>Class counts appear after inference calls.</p></div>`; return; }
  const max = Math.max(...entries.map(([, value]) => value), 1);
  target.innerHTML = entries.map(([name, count]) => `<div class="dist-row"><span>${safe(name)}</span><div><span style="width:${count / max * 100}%"></span></div><strong>${count}</strong></div>`).join('');
}
async function refreshAll() {
  await Promise.all([refreshMetadata().catch(() => { $('updatedAt').textContent = 'Model metadata unavailable'; }), refreshMetrics(), refreshHealth()]);
}

const form = $('predictForm');
form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = $('runButton'); button.disabled = true; button.innerHTML = '<span>◌</span> Validating…';
  $('requestState').textContent = 'VALIDATING · Checking measurement inputs';
  $('responseCode').textContent = 'PENDING'; $('resultBody').classList.remove('hidden'); $('resultOutput').classList.add('hidden');
  try {
    const payload = Object.fromEntries(new FormData(form).entries());
    Object.keys(payload).forEach(key => payload[key] = Number(payload[key]));
    $('requestState').textContent = 'INFERENCING · Sending request to the loaded model'; button.innerHTML = '<span>◌</span> Running model…';
    const result = await api('/api/v1/predict', { method: 'POST', body: JSON.stringify(payload) });
    showResult(result); $('responseCode').textContent = '200 OK'; $('requestState').textContent = 'COMPLETE · Response received from the model API'; toast('Inference completed successfully'); await Promise.all([refreshMetrics(), refreshHealth()]);
  } catch (error) { $('responseCode').textContent = 'ERROR'; $('requestState').textContent = 'FAILED · Inference request did not complete'; $('resultBody').innerHTML = `<div class="empty-state"><span class="empty-icon">!</span><strong>Inference failed</strong><p>${safe(error.message)}</p></div>`; await refreshHealth(); toast(error.message, true); }
  finally { button.disabled = false; button.innerHTML = '<span>▶</span> Run inference'; }
});
function showResult(result) {
  $('resultBody').classList.add('hidden'); const output = $('resultOutput'); output.classList.remove('hidden');
  const confidence = Math.max(...Object.values(result.probabilities));
  const probabilities = Object.entries(result.probabilities).sort((a,b) => b[1] - a[1]);
  output.innerHTML = `<div class="result-class"><div class="model-glyph">✳</div><div><small>PREDICTED SPECIES</small><strong>${safe(result.prediction)}</strong></div></div><div class="result-confidence"><div class="confidence-head"><span>Top class confidence</span><strong>${(confidence * 100).toFixed(1)}%</strong></div><div class="confidence-track"><span style="width:${(confidence * 100).toFixed(1)}%"></span></div></div><div class="probability-list">${probabilities.map(([name, score]) => `<div class="probability-row"><span>${safe(name)}</span><div class="probability-track"><span style="width:${(score * 100).toFixed(1)}%"></span></div><strong>${(score * 100).toFixed(1)}%</strong></div>`).join('')}</div><div class="result-meta"><span>${safe(result.model.name)} · v${safe(result.model.version)}</span><span>${result.latency_ms.toFixed(2)} ms</span></div>`;
}
$('sampleInput').addEventListener('click', () => {
  const samples = [{sepal_length:5.1,sepal_width:3.5,petal_length:1.4,petal_width:.2},{sepal_length:6.4,sepal_width:3.2,petal_length:4.5,petal_width:1.5},{sepal_length:6.3,sepal_width:3.3,petal_length:6,petal_width:2.5}];
  const sample = samples[Math.floor(Math.random() * samples.length)]; Object.entries(sample).forEach(([key, value]) => form.elements[key].value = value.toFixed(1));
});
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  const text = $(button.dataset.copy).innerText;
  try { await navigator.clipboard.writeText(text); button.innerHTML = 'Copied ✓'; setTimeout(() => button.innerHTML = 'Copy <span>▢</span>', 1500); }
  catch { toast('Clipboard access is unavailable', true); }
}));

const initialPage = location.hash.slice(1); navigate(pages.includes(initialPage) ? initialPage : 'overview');
refreshAll(); setInterval(() => { refreshMetrics(); refreshHealth(); }, 10000);
