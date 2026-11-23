// Punch Text — text out of a batch of videos, entirely in the browser. Nothing is uploaded anywhere:
// the audio is decoded here, Whisper runs in worker.js, results live in this browser's IndexedDB.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const SR = 16000;

/* ---------- languages: Whisper's codes, names from the browser in Ukrainian ---------- */
const POPULAR = ['uk', 'en', 'es', 'fr', 'ar', 'de', 'it', 'pt', 'pl', 'tr', 'ru'];
const ALL = 'en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue'.split(' ');
const names = (() => { try { return new Intl.DisplayNames(['uk'], { type: 'language' }); } catch { return null; } })();
const langName = (c) => { if (!c) return ''; const n = names?.of(c === 'jw' ? 'jv' : c); return n ? n[0].toUpperCase() + n.slice(1) : c; };

/* ---------- models ---------- */
const MODELS = {
  turbo: { label: 'Точна', sub: 'Whisper large-v3-turbo', size: '≈ 1,6 ГБ' },
  small: { label: 'Легка', sub: 'Whisper small', size: '≈ 0,6 ГБ' },
};

/* ---------- settings kept in this browser ---------- */
const PREF = 'punch-text:prefs';
const prefs = (() => { try { return { lang: '', model: 'turbo', pick: false, ...JSON.parse(localStorage.getItem(PREF)) }; } catch { return { lang: '', model: 'turbo', pick: false }; } })();
const savePrefs = () => { try { localStorage.setItem(PREF, JSON.stringify(prefs)); } catch {} };

/* ---------- IndexedDB: finished transcripts ---------- */
const db = new Promise((res, rej) => {
  const r = indexedDB.open('punch-text', 1);
  r.onupgradeneeded = () => r.result.createObjectStore('items', { keyPath: 'id' });
  r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
}).catch(() => null);
async function tx(mode, fn) {
  const d = await db; if (!d) return null;
  return new Promise((res, rej) => { const t = d.transaction('items', mode); const q = fn(t.objectStore('items')); t.oncomplete = () => res(q?.result); t.onerror = () => rej(t.error); });
}
const dbAll = () => tx('readonly', (s) => s.getAll()).catch(() => []);
const dbPut = (it) => tx('readwrite', (s) => s.put(strip(it))).catch(() => {});
const dbDel = (id) => tx('readwrite', (s) => s.delete(id)).catch(() => {});
const dbClear = () => tx('readwrite', (s) => s.clear()).catch(() => {});
const strip = ({ file, ...rest }) => rest; // the video itself is never stored

/* ---------- state ---------- */
const items = [];            // newest first
let gpu = { webgpu: false, f16: false };
let worker = null, model = { key: '', state: 'idle', loaded: 0, total: 0 }, busy = false, current = null;

/* ---------- device ---------- */
async function detectGpu() {
  if (!navigator.gpu) return { webgpu: false, f16: false };
  try { const a = await navigator.gpu.requestAdapter(); return a ? { webgpu: true, f16: a.features.has('shader-f16') } : { webgpu: false, f16: false }; }
  catch { return { webgpu: false, f16: false }; }
}
const canTurbo = () => gpu.webgpu && gpu.f16;
const deviceFor = (m) => (gpu.webgpu ? 'webgpu' : 'wasm');

/* ---------- worker ---------- */
let waiters = [];
function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data: m }) => {
    if (m.type === 'load-progress') { model.loaded = m.loaded; model.total = m.total; renderModel(); return; }
    if (m.type === 'warming') { model.state = 'warming'; renderModel(); return; }
    if (m.type === 'ready') { model.state = 'ready'; renderModel(); }
    if (m.type === 'error' && m.stage === 'load') { model.state = 'error'; model.error = m.message; renderModel(); }
    const it = m.id && items.find((x) => x.id === m.id);
    if (it) onItemMessage(it, m);
    waiters = waiters.filter((w) => !w(m));
  };
  worker.onerror = (e) => { console.error(e); model.state = 'error'; model.error = e.message || 'Воркер упав'; renderModel(); waiters.forEach((w) => w({ type: 'error', stage: 'load', message: model.error })); waiters = []; worker = null; };
  return worker;
}
const waitFor = (pred) => new Promise((res) => waiters.push((m) => (pred(m) ? (res(m), true) : false)));

async function ensureModel() {
  const key = `${prefs.model}:${deviceFor(prefs.model)}`;
  if (model.key === key && model.state === 'ready') return true;
  model = { key, state: 'loading', loaded: 0, total: 0 };
  renderModel();
  try { await navigator.storage?.persist?.(); } catch {}
  getWorker().postMessage({ type: 'load', model: prefs.model, device: deviceFor(prefs.model) });
  const m = await waitFor((x) => x.type === 'ready' || (x.type === 'error' && x.stage === 'load'));
  refreshStorage();
  return m.type === 'ready';
}

/* ---------- audio out of a video, in the browser ---------- */
async function decode(file) {
  const buf = await file.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx({ sampleRate: SR });
  try {
    const ab = await ctx.decodeAudioData(buf);
    if (ab.numberOfChannels === 1) return ab.getChannelData(0).slice();
    const out = new Float32Array(ab.length);
    for (let c = 0; c < ab.numberOfChannels; c++) { const d = ab.getChannelData(c); for (let i = 0; i < out.length; i++) out[i] += d[i]; }
    for (let i = 0; i < out.length; i++) out[i] /= ab.numberOfChannels;
    return out;
  } finally { ctx.close?.(); }
}

/* ---------- queue ---------- */
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
function addFiles(list) {
  const files = [...list].filter((f) => /^(video|audio)\//.test(f.type) || /\.(mp4|mov|m4v|webm|mkv|avi|mp3|m4a|wav|aac|ogg|opus|flac)$/i.test(f.name));
  if (!files.length) return toast('Це не відео й не аудіо', true);
  // with "Обрати шматок" on, each file first waits in the editor (status 'pick') instead of going straight to the queue
  for (const f of files) items.unshift({ id: uid(), file: f, name: f.name, size: f.size, langSet: prefs.lang || null, model: prefs.model, status: prefs.pick ? 'pick' : 'queued', segments: [], createdAt: Date.now() });
  renderList();
  if (prefs.pick && !ed.it) openNextPick();
  pump();
}

async function pump() {
  if (busy) return;
  busy = true; renderControls();
  try {
    while (true) {
      const it = [...items].reverse().find((x) => x.status === 'queued'); // oldest first
      if (!it) break;
      if (it.model !== prefs.model) it.model = prefs.model;
      if (!(await ensureModel())) { for (const x of items) if (x.status === 'queued') { x.status = 'error'; x.error = 'Модель не завантажилась: ' + (model.error || ''); renderRow(x); } break; }
      current = it;
      it.status = 'audio'; renderRow(it);
      let audio;
      try { audio = await decode(it.file); }
      catch { it.status = 'error'; it.error = 'Браузер не зміг дістати звук з цього файлу (спробуйте MP4 або MP3)'; renderRow(it); continue; }
      if (!items.includes(it)) continue; // cleared while its audio was being decoded
      it.duration = audio.length / SR;
      let offset = 0;
      if (it.range) { offset = it.range[0]; audio = audio.slice(Math.floor(it.range[0] * SR), Math.ceil(it.range[1] * SR)); }
      it.status = 'run'; it.done = 0; it.total = 0; it.segments = []; renderRow(it);
      getWorker().postMessage({ type: 'run', id: it.id, audio, offset, language: it.langSet }, [audio.buffer]);
      await waitFor((m) => m.id === it.id && ['done', 'cancelled', 'error'].includes(m.type));
      current = null;
    }
  } finally { busy = false; current = null; renderControls(); }
}

function onItemMessage(it, m) {
  if (m.type === 'progress') {
    it.done = m.done; it.total = m.total; it.elapsed = m.elapsed; it.langs = m.langs; it.segments.push(...m.fresh);
    renderRow(it);
  } else if (m.type === 'done' || m.type === 'cancelled') {
    it.segments = m.segments; it.langs = m.langs?.length ? m.langs : it.langs;
    it.status = m.type === 'done' ? 'done' : 'cancelled';
    it.finishedAt = Date.now();
    delete it.file;
    if (it.segments.length || it.status === 'done') dbPut(it);
    renderRow(it); renderControls(); refreshStorage();
    if (m.type === 'done' && !items.some((x) => ['queued', 'audio', 'run'].includes(x.status))) toast('Готово — тексти збережено в цьому браузері');
  } else if (m.type === 'error') {
    it.status = 'error'; it.error = m.message; delete it.file; renderRow(it); renderControls();
  }
}

/* ---------- text formats ---------- */
const pad = (n, w = 2) => String(n).padStart(w, '0');
function tc(sec, sep = ',', short = false) {
  const ms = Math.round(sec * 1000), h = Math.floor(ms / 3600000), m = Math.floor(ms / 60000) % 60, s = Math.floor(ms / 1000) % 60;
  if (short) return (h ? h + ':' + pad(m) : m) + ':' + pad(s);
  return `${pad(h)}:${pad(m)}:${pad(s)}${sep}${pad(ms % 1000, 3)}`;
}
// plain text: a new paragraph after a pause longer than 1.5 s
function toTxt(segs) {
  let out = '';
  segs.forEach((s, i) => { out += i === 0 ? s.text : (s.start - segs[i - 1].end > 1.5 ? '\n\n' : ' ') + s.text; });
  return out.trim() + '\n';
}
const toSrt = (segs) => segs.map((s, i) => `${i + 1}\n${tc(s.start)} --> ${tc(s.end)}\n${s.text}\n`).join('\n');
const toVtt = (segs) => 'WEBVTT\n\n' + segs.map((s) => `${tc(s.start, '.')} --> ${tc(s.end, '.')}\n${s.text}\n`).join('\n');
const FORMATS = { txt: toTxt, srt: toSrt, vtt: toVtt };
const base = (name) => name.replace(/\.[^.]+$/, '');

function download(name, blob) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

/* ---------- a tiny ZIP writer (stored, no compression — text is small) ---------- */
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
const crc32 = (b) => { let c = 0xffffffff; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
function zip(files) {
  const enc = new TextEncoder(), parts = [], central = [];
  let off = 0;
  const d = new Date(), dt = ((d.getFullYear() - 1980) << 25) | ((d.getMonth() + 1) << 21) | (d.getDate() << 16) | (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  for (const f of files) {
    const name = enc.encode(f.name), data = enc.encode(f.text), crc = crc32(data);
    const h = new DataView(new ArrayBuffer(30));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); h.setUint32(10, dt, true);
    h.setUint32(14, crc, true); h.setUint32(18, data.length, true); h.setUint32(22, data.length, true); h.setUint16(26, name.length, true);
    const c = new DataView(new ArrayBuffer(46));
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true); c.setUint32(12, dt, true);
    c.setUint32(16, crc, true); c.setUint32(20, data.length, true); c.setUint32(24, data.length, true); c.setUint16(28, name.length, true); c.setUint32(42, off, true);
    parts.push(h, name, data); central.push(c, name);
    off += 30 + name.length + data.length;
  }
  const size = central.reduce((s, p) => s + p.byteLength, 0);
  const e = new DataView(new ArrayBuffer(22));
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true); e.setUint32(12, size, true); e.setUint32(16, off, true);
  return new Blob([...parts, ...central, e], { type: 'application/zip' });
}
function downloadAll() {
  const done = items.filter((x) => x.segments?.length && ['done', 'cancelled'].includes(x.status));
  if (!done.length) return;
  const used = new Map(), files = [];
  for (const it of [...done].reverse()) {
    let b = base(it.name); const n = used.get(b) || 0; used.set(b, n + 1); if (n) b += ` (${n + 1})`;
    files.push({ name: `${b}.txt`, text: toTxt(it.segments) }, { name: `${b}.srt`, text: toSrt(it.segments) });
  }
  download(`punch-text-${new Date().toISOString().slice(0, 10)}.zip`, zip(files));
}

/* ---------- rendering ---------- */
const fmtSize = (b) => (b >= 1e9 ? (b / 1e9).toFixed(1).replace('.', ',') + ' ГБ' : b >= 1e6 ? Math.round(b / 1e6) + ' МБ' : Math.max(1, Math.round(b / 1e3)) + ' КБ');
const fmtDur = (s) => { s = Math.round(s || 0); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60; return h ? `${h} год ${m} хв` : m ? `${m} хв ${pad(s % 60)} с` : `${s} с`; };
const I = {
  copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  trash: '<path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/>',
  chev: '<path d="m6 9 6 6 6-6"/>',
  file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>',
  scissors: '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4 8.1 15.9M14.5 14.5 20 20M8.1 8.1 12 12"/>',
};
const svg = (k) => `<svg viewBox="0 0 24 24" aria-hidden="true">${I[k]}</svg>`;

function statusOf(it) {
  switch (it.status) {
    case 'pick': return { cls: 'wait', text: ed.it === it ? 'Обираєте шматок' : 'Чекає на шматок' };
    case 'queued': return { cls: 'wait', text: 'У черзі' };
    case 'audio': return { cls: 'run', text: 'Дістаю звук…' };
    case 'run': {
      if (!it.total) return { cls: 'run', text: 'Розпізнаю…' };
      const left = it.done ? (it.elapsed / it.done) * (it.total - it.done) : 0;
      return { cls: 'run', text: `${Math.round((it.done / it.total) * 100)}%${it.done ? ` · ще ≈ ${fmtDur(left)}` : ''}` };
    }
    case 'done': return { cls: 'ok', text: 'Готово' };
    case 'cancelled': return { cls: 'wait', text: 'Зупинено' };
    default: return { cls: 'err', text: 'Помилка' };
  }
}

function rowHtml(it) {
  const st = statusOf(it);
  const pct = it.status === 'run' && it.total ? (it.done / it.total) * 100 : it.status === 'done' ? 100 : 0;
  const hasText = it.segments?.length > 0;
  const live = ['pick', 'queued', 'audio', 'run'].includes(it.status);
  const canPick = (it.status === 'pick' || it.status === 'queued') && it.file && ed.it !== it;
  const langs = it.langSet ? langName(it.langSet) : it.langs?.length ? it.langs.slice(0, 3).map(langName).join(', ') + ' · авто' : 'мова: авто';
  const range = it.range ? `✂ ${tc(it.range[0], ',', true)}–${tc(it.range[1], ',', true)}` : '';
  const meta = [it.duration ? fmtDur(it.duration) : fmtSize(it.size), range, langs, MODELS[it.model]?.label].filter(Boolean);
  return `
    <div class="row-top">
      <span class="row-icon">${svg('file')}</span>
      <div class="row-main">
        <div class="row-name" title="${esc(it.name)}">${esc(it.name)}</div>
        <div class="row-meta mono">${meta.map(esc).join(' · ')}</div>
      </div>
      <span class="pill pill--${st.cls}">${esc(st.text)}</span>
      <div class="row-actions">
        ${hasText ? `<button class="icon-btn" data-act="copy" title="Скопіювати текст">${svg('copy')}</button>
          <button class="chip" data-act="txt" title="Текст">TXT</button><button class="chip" data-act="srt" title="Субтитри">SRT</button><button class="chip hide-sm" data-act="vtt" title="Субтитри для вебу">VTT</button>` : ''}
        ${canPick ? `<button class="icon-btn" data-act="pick" title="${it.range ? 'Змінити шматок' : 'Обрати шматок'}">${svg('scissors')}</button>` : ''}
        ${live ? `<button class="icon-btn icon-btn--danger" data-act="stop" title="${it.status === 'run' || it.status === 'audio' ? 'Зупинити' : 'Прибрати'}">${svg('x')}</button>`
          : `<button class="icon-btn icon-btn--danger" data-act="del" title="Видалити з браузера">${svg('trash')}</button>`}
      </div>
    </div>
    ${it.status === 'run' || it.status === 'audio' ? `<div class="progress"><i style="width:${pct}%"></i></div>` : ''}
    ${it.error ? `<p class="row-err">${esc(it.error)}</p>` : ''}
    ${hasText ? `<details class="row-text" ${it.open ? 'open' : ''}><summary>${svg('chev')}<span>Текст</span><small class="mono">${it.segments.length} фраз</small></summary>
      <ol class="segs">${it.segments.map((s) => `<li><time class="mono">${tc(s.start, ',', true)}</time><span dir="auto">${esc(s.text)}</span></li>`).join('')}</ol></details>` : ''}`;
}

function renderRow(it) {
  const el = document.querySelector(`.row[data-id="${it.id}"]`);
  if (!el) return renderList();
  const segs = el.querySelector('.segs'), stick = segs && segs.scrollHeight - segs.scrollTop - segs.clientHeight < 40;
  el.innerHTML = rowHtml(it);
  el.className = `row row--${it.status}`;
  const ns = el.querySelector('.segs'); if (ns && stick) ns.scrollTop = ns.scrollHeight;
  renderControls();
}
function renderList() {
  $('list').innerHTML = items.map((it) => `<article class="row row--${it.status}" data-id="${it.id}">${rowHtml(it)}</article>`).join('');
  $('empty').hidden = items.length > 0;
  renderControls();
}
function renderControls() {
  const finished = items.filter((x) => x.segments?.length && ['done', 'cancelled'].includes(x.status)).length;
  const pending = items.filter((x) => ['pick', 'queued', 'audio', 'run'].includes(x.status)).length;
  $('zipAll').disabled = !finished;
  $('zipAll').querySelector('span').textContent = finished > 1 ? `Усі тексти (${finished}) · ZIP` : 'Усі тексти · ZIP';
  $('listHead').hidden = items.length === 0;
  $('count').textContent = pending ? `в роботі ${pending} з ${items.length}` : `${items.length} ${plural(items.length, ['файл', 'файли', 'файлів'])}`;
  for (const b of document.querySelectorAll('#modelSeg button')) b.disabled = busy || (b.dataset.v === 'turbo' && !canTurbo());
}
const plural = (n, [one, few, many]) => { const a = n % 10, b = n % 100; return a === 1 && b !== 11 ? one : a >= 2 && a <= 4 && (b < 12 || b > 14) ? few : many; };

function renderModel() {
  const el = $('modelState');
  let text = '', cls = '';
  if (model.state === 'loading') {
    const p = model.total ? Math.min(100, (model.loaded / model.total) * 100) : 0;
    text = model.total ? `Завантажую модель · ${Math.floor(p)}% (${fmtSize(model.loaded)} з ${fmtSize(model.total)})` : 'Готую модель…';
    cls = 'run'; el.style.setProperty('--p', p + '%');
  } else if (model.state === 'warming') { text = 'Прогріваю модель…'; cls = 'run'; el.style.setProperty('--p', '100%'); }
  else if (model.state === 'ready') { text = `Модель готова · ${gpu.webgpu ? 'WebGPU' : 'процесор'}`; cls = 'ok'; }
  else if (model.state === 'error') { text = 'Модель не завантажилась'; cls = 'err'; }
  else { text = `Модель завантажиться з першим файлом (${MODELS[prefs.model].size}, один раз)`; }
  el.className = `model-state mono ${cls ? 'is-' + cls : ''}`;
  el.querySelector('span').textContent = text;
  el.title = model.state === 'error' ? model.error || '' : '';
}

async function refreshStorage() {
  try {
    const { usage = 0 } = (await navigator.storage?.estimate?.()) || {};
    const hasModel = (await caches.keys()).includes('transformers-cache');
    $('storage').innerHTML = `У цьому браузері зайнято <b>${fmtSize(usage)}</b> (модель і тексти). ${hasModel ? '<button class="link-btn" id="dropModel" type="button">Видалити модель з браузера</button>' : ''}`;
  } catch { $('storage').textContent = ''; }
}

let toastT = 0;
function toast(msg, bad = false) { const t = $('toast'); t.textContent = msg; t.classList.toggle('is-bad', bad); t.classList.add('is-on'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('is-on'), 3200); }

/* ---------- settings UI ---------- */
function buildSettings() {
  const sel = $('lang');
  const opt = (c) => `<option value="${c}">${esc(langName(c))}</option>`;
  const rest = ALL.filter((c) => !POPULAR.includes(c)).sort((a, b) => langName(a).localeCompare(langName(b), 'uk'));
  sel.innerHTML = `<option value="">Визначити автоматично</option><optgroup label="Часті">${POPULAR.map(opt).join('')}</optgroup><optgroup label="Усі мови">${rest.map(opt).join('')}</optgroup>`;
  sel.value = prefs.lang;
  sel.onchange = () => { prefs.lang = sel.value; savePrefs(); };

  $('modelSeg').innerHTML = Object.entries(MODELS).map(([k, m]) => `<button type="button" role="radio" data-v="${k}"><b>${m.label}</b><small>${m.sub} · ${m.size}</small></button>`).join('');
  if (!canTurbo()) prefs.model = 'small';
  const paint = () => { for (const b of $('modelSeg').children) { b.classList.toggle('is-active', b.dataset.v === prefs.model); b.setAttribute('aria-checked', b.dataset.v === prefs.model); } };
  $('modelSeg').onclick = (e) => {
    const b = e.target.closest('button'); if (!b || b.disabled) return;
    prefs.model = b.dataset.v; savePrefs(); paint();
    if (model.state !== 'loading' && model.state !== 'warming') { model = { key: '', state: 'idle' }; renderModel(); }
  };
  paint();
  const paintPick = () => { for (const b of $('pickSeg').children) { const on = (b.dataset.v === 'pick') === prefs.pick; b.classList.toggle('is-active', on); b.setAttribute('aria-checked', on); } };
  $('pickSeg').onclick = (e) => { const b = e.target.closest('button'); if (!b) return; prefs.pick = b.dataset.v === 'pick'; savePrefs(); paintPick(); };
  paintPick();
  $('deviceNote').innerHTML = !gpu.webgpu
    ? '<b>Цей браузер не має WebGPU</b>, тому працює легка модель на процесорі. Вона повільніша й менш точна. Краще відкрити сторінку в Chrome або Edge на комп’ютері.'
    : !gpu.f16 ? 'Відеокарта цього пристрою не тягне точну модель, тож працює легка.'
    : 'Точна модель краще розпізнає арабську та змішане мовлення. Легка швидша і менша за розміром.';
}

/* ---------- piece picker (like VoiceSnip): video + waveform, drag a frame over the part to transcribe ---------- */
const ed = { it: null, peaks: null, dur: 0, s: 0, e: 0, url: '', drag: null, stopAt: null, raf: 0 };
const MIN_SEL = 0.5;
const video = $('edVideo'), waveEl = $('edWave'), cv = $('edCanvas'), selEl = $('edSel'), phEl = $('edPh'), hovEl = $('edHover');
const fmtT = (t) => { t = Math.max(0, t); const h = Math.floor(t / 3600), m = Math.floor(t / 60) % 60, s = t % 60; return (h ? h + ':' + pad(m) : m) + ':' + s.toFixed(1).padStart(4, '0'); };

function peaksOf(audio, n = 4000) {
  const out = new Float32Array(n), step = audio.length / n;
  for (let i = 0; i < n; i++) {
    let m = 0; const a = Math.floor(i * step), b = Math.min(audio.length, Math.floor((i + 1) * step));
    for (let j = a; j < b; j += 4) { const v = Math.abs(audio[j]); if (v > m) m = v; } // every 4th sample is plenty for a picture
    out[i] = m;
  }
  let top = 0; for (const v of out) if (v > top) top = v;
  if (top > 0) for (let i = 0; i < n; i++) out[i] = Math.min(1, out[i] / top);
  return out;
}

async function openEditor(it) {
  if (!it?.file) return;
  if (ed.it && ed.it !== it) closeEditor(true);
  if (it.status === 'queued') it.status = 'pick';
  ed.it = it; ed.peaks = null; ed.dur = 0;
  $('editor').hidden = false;
  $('edName').textContent = it.name;
  const waiting = items.filter((x) => x.status === 'pick').length;
  $('edCount').textContent = waiting > 1 ? `ще ${waiting - 1} ${plural(waiting - 1, ['файл чекає', 'файли чекають', 'файлів чекають'])}` : '';
  $('edLoading').hidden = false; $('edLoading').textContent = 'Читаю звук…';
  waveEl.classList.add('is-loading');
  URL.revokeObjectURL(ed.url); ed.url = URL.createObjectURL(it.file);
  video.src = ed.url;
  $('edPlayer').hidden = /^audio\//.test(it.file.type);
  renderList();
  $('editor').scrollIntoView({ behavior: 'smooth', block: 'start' });
  try {
    const audio = await decode(it.file);
    if (ed.it !== it) return;
    ed.dur = audio.length / SR; it.duration = ed.dur;
    ed.peaks = peaksOf(audio);
    [ed.s, ed.e] = it.range || [0, ed.dur];
    $('edLoading').hidden = true; waveEl.classList.remove('is-loading');
    $('edTotal').textContent = fmtT(ed.dur);
    drawWave(); drawSel();
  } catch {
    if (ed.it !== it) return;
    $('edLoading').textContent = 'Браузер не зміг дістати звук з цього файлу (спробуйте MP4 або MP3)';
  }
}

function closeEditor(keepPick) {
  stopPlay(); video.removeAttribute('src'); video.load();
  URL.revokeObjectURL(ed.url); ed.url = '';
  const it = ed.it; ed.it = null; ed.peaks = null;
  $('editor').hidden = true;
  if (it && keepPick && items.includes(it)) renderRow(it);
}
function openNextPick() {
  const next = [...items].reverse().find((x) => x.status === 'pick' && x.file);
  if (next) openEditor(next); else closeEditor(false);
}

function confirmPiece(whole) {
  const it = ed.it; if (!it || !ed.dur) return;
  if (whole) { ed.s = 0; ed.e = ed.dur; }
  it.range = ed.s > 0.05 || ed.e < ed.dur - 0.05 ? [+ed.s.toFixed(2), +ed.e.toFixed(2)] : null;
  it.status = 'queued';
  closeEditor(false);
  renderList();
  openNextPick();
  pump();
}

/* drawing */
function drawWave() {
  const r = waveEl.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  const g = cv.getContext('2d'); g.clearRect(0, 0, cv.width, cv.height);
  if (!ed.peaks) return;
  const cs = getComputedStyle(document.documentElement);
  const on = cs.getPropertyValue('--accent').trim(), off = cs.getPropertyValue('--muted').trim();
  const bar = 2 * dpr, gap = 1 * dpr, n = Math.floor(cv.width / (bar + gap)), mid = cv.height / 2, P = ed.peaks;
  for (let i = 0; i < n; i++) {
    const a = Math.floor((i / n) * P.length), b = Math.max(a + 1, Math.floor(((i + 1) / n) * P.length));
    let m = 0; for (let j = a; j < b; j++) if (P[j] > m) m = P[j];
    const t = ((i + 0.5) / n) * ed.dur, h = Math.max(1.5 * dpr, m * (cv.height - 8 * dpr));
    g.fillStyle = t >= ed.s && t <= ed.e ? on : off;
    g.globalAlpha = t >= ed.s && t <= ed.e ? 1 : 0.45;
    const x = i * (bar + gap);
    g.beginPath(); g.roundRect ? g.roundRect(x, mid - h / 2, bar, h, bar / 2) : g.rect(x, mid - h / 2, bar, h); g.fill();
  }
}
function drawSel() {
  if (!ed.dur) return;
  selEl.style.left = (ed.s / ed.dur) * 100 + '%';
  selEl.style.width = ((ed.e - ed.s) / ed.dur) * 100 + '%';
  selEl.classList.toggle('is-full', ed.e - ed.s >= ed.dur * 0.98);
  const len = ed.e - ed.s;
  $('edRangeT').textContent = `${fmtT(ed.s)} → ${fmtT(ed.e)}`;
  $('edRangeS').textContent = len >= ed.dur - 0.05 ? `весь файл · ${fmtDur(len)}` : `${fmtDur(len)} з ${fmtDur(ed.dur)}`;
  $('edGo').querySelector('span').textContent = len >= ed.dur - 0.05 ? 'Розпізнати весь файл' : 'Розпізнати шматок';
  drawHead();
}
function drawHead() { if (ed.dur) phEl.style.left = (Math.min(ed.dur, video.currentTime || 0) / ed.dur) * 100 + '%'; }
let waveT = 0;
const redraw = () => { cancelAnimationFrame(waveT); waveT = requestAnimationFrame(() => { drawWave(); drawSel(); }); };
addEventListener('resize', () => { if (ed.it) redraw(); });

/* dragging: on a handle → resize, on the frame → move it, elsewhere → draw a new frame (a plain click seeks) */
const tAt = (x) => { const r = waveEl.getBoundingClientRect(); return Math.max(0, Math.min(1, (x - r.left) / r.width)) * ed.dur; };
waveEl.addEventListener('pointerdown', (e) => {
  if (!ed.dur || e.button) return;
  const t = tAt(e.clientX), h = e.target.closest('[data-h]')?.dataset.h;
  const full = ed.e - ed.s >= ed.dur * 0.98; // a frame over the whole file can't move, so dragging on it draws a new one
  ed.drag = h ? { mode: h } : e.target.closest('#edSel') && !full ? { mode: 'move', grab: t - ed.s, len: ed.e - ed.s } : { mode: 'new', from: t, x: e.clientX, prev: [ed.s, ed.e] };
  waveEl.setPointerCapture(e.pointerId); waveEl.classList.add('dragging');
  e.preventDefault();
});
waveEl.addEventListener('pointermove', (e) => {
  if (!ed.dur) return;
  const t = tAt(e.clientX);
  hovEl.style.display = 'block'; hovEl.style.left = (t / ed.dur) * 100 + '%'; hovEl.firstChild.textContent = fmtT(t);
  hovEl.classList.toggle('flip', t / ed.dur > 0.85);
  const d = ed.drag; if (!d) return;
  if (d.mode === 'l') ed.s = Math.min(t, ed.e - MIN_SEL);
  else if (d.mode === 'r') ed.e = Math.max(t, ed.s + MIN_SEL);
  else if (d.mode === 'move') { ed.s = Math.max(0, Math.min(ed.dur - d.len, t - d.grab)); ed.e = ed.s + d.len; }
  else if (Math.abs(e.clientX - d.x) > 3) { d.moved = true; ed.s = Math.min(d.from, t); ed.e = Math.max(d.from, t); }
  ed.s = Math.max(0, ed.s); ed.e = Math.min(ed.dur, ed.e);
  redraw();
});
const endDrag = (e) => {
  const d = ed.drag; if (!d) return;
  ed.drag = null; waveEl.classList.remove('dragging');
  if (d.mode === 'new' && !d.moved) { [ed.s, ed.e] = d.prev; seek(tAt(e.clientX)); }
  else if (d.mode === 'new' && ed.e - ed.s < MIN_SEL) { ed.e = Math.min(ed.dur, ed.s + MIN_SEL); ed.s = ed.e - MIN_SEL; }
  redraw();
};
waveEl.addEventListener('pointerup', endDrag);
waveEl.addEventListener('pointercancel', endDrag);
waveEl.addEventListener('pointerleave', () => { hovEl.style.display = 'none'; });

/* playback: the play button plays the frame (from the playhead if it's inside, else from the frame's start) */
function seek(t) { video.currentTime = Math.max(0, Math.min(ed.dur, t)); drawHead(); }
function tick() { drawHead(); if (ed.stopAt != null && video.currentTime >= ed.stopAt) { video.pause(); seek(ed.stopAt); } if (!video.paused) ed.raf = requestAnimationFrame(tick); }
function playSel() {
  if (!ed.dur) return;
  if (!video.paused) return stopPlay();
  const t = video.currentTime;
  if (t < ed.s || t >= ed.e - 0.05) video.currentTime = ed.s;
  ed.stopAt = ed.e;
  video.play().catch(() => {});
}
function stopPlay() { video.pause(); ed.stopAt = null; }
video.addEventListener('play', () => { $('edPlay').classList.add('is-playing'); selEl.classList.add('playing'); cancelAnimationFrame(ed.raf); ed.raf = requestAnimationFrame(tick); });
video.addEventListener('pause', () => { $('edPlay').classList.remove('is-playing'); selEl.classList.remove('playing'); drawHead(); });
video.addEventListener('seeked', drawHead);

/* typed times: "1:20 - 3:45", "80-225", "1:02:03.5 – 1:05" */
function parseT(x) {
  x = String(x).trim().replace(',', '.'); if (!x) return NaN;
  if (!/^\d+(:\d+){0,2}(\.\d+)?$/.test(x)) return NaN;
  return x.split(':').reduce((a, p) => a * 60 + parseFloat(p), 0);
}
function openRange() {
  if (!ed.dur) return;
  const inp = $('edRangeIn'); inp.value = `${fmtT(ed.s)} - ${fmtT(ed.e)}`;
  $('edRangeT').hidden = true; inp.hidden = false; inp.focus(); inp.select();
}
function closeRange(apply) {
  const inp = $('edRangeIn'); if (inp.hidden) return;
  if (apply) {
    const [a, b] = inp.value.split(/\s*(?:-|–|—|→)\s*/).map(parseT);
    if (Number.isFinite(a) && Number.isFinite(b) && b - a >= MIN_SEL) { ed.s = Math.max(0, a); ed.e = Math.min(ed.dur, b); seek(ed.s); }
    else if (inp.value.trim()) toast('Напишіть час так: 1:20 - 3:45', true);
  }
  inp.hidden = true; $('edRangeT').hidden = false; redraw();
}
$('edRangeT').onclick = openRange;
$('edRangeIn').addEventListener('keydown', (e) => { if (e.key === 'Enter') closeRange(true); if (e.key === 'Escape') closeRange(false); e.stopPropagation(); });
$('edRangeIn').addEventListener('blur', () => closeRange(true));

$('edPlay').onclick = playSel;
$('edAll').onclick = () => { if (!ed.dur) return; ed.s = 0; ed.e = ed.dur; redraw(); };
$('edGo').onclick = () => confirmPiece(false);
$('edDrop').onclick = () => { const it = ed.it; if (!it) return; items.splice(items.indexOf(it), 1); closeEditor(false); renderList(); openNextPick(); };
$('edStart').onclick = () => { if (ed.dur) { ed.s = Math.min(video.currentTime, ed.e - MIN_SEL); redraw(); } };
$('edEnd').onclick = () => { if (ed.dur) { ed.e = Math.max(video.currentTime, ed.s + MIN_SEL); redraw(); } };
// keys while the picker is open: Space play, [ and ] set start / end at the playhead
addEventListener('keydown', (e) => {
  if (!ed.it || !ed.dur || e.target.closest?.('input, textarea, select')) return;
  if (e.key === ' ' && !e.target.closest?.('button')) { e.preventDefault(); playSel(); }
  else if (e.key === '[') $('edStart').click();
  else if (e.key === ']') $('edEnd').click();
});

/* ---------- events ---------- */
$('pick').onclick = () => $('files').click();
$('files').onchange = (e) => { addFiles(e.target.files); e.target.value = ''; };
const drop = $('drop');
let depth = 0;
addEventListener('dragenter', (e) => { if (e.dataTransfer?.types?.includes('Files')) { depth++; drop.classList.add('is-over'); } });
addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; drop.classList.remove('is-over'); } });
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => { e.preventDefault(); depth = 0; drop.classList.remove('is-over'); if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files); });
drop.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('files').click(); } };
drop.onclick = (e) => { if (!e.target.closest('button')) $('files').click(); };

$('list').addEventListener('click', async (e) => {
  const b = e.target.closest('[data-act]'); if (!b) return;
  const id = b.closest('.row').dataset.id, it = items.find((x) => x.id === id); if (!it) return;
  const act = b.dataset.act;
  if (act === 'copy') { try { await navigator.clipboard.writeText(toTxt(it.segments)); toast('Текст скопійовано'); } catch { toast('Не вдалося скопіювати', true); } }
  else if (FORMATS[act]) download(`${base(it.name)}.${act}`, new Blob([FORMATS[act](it.segments)], { type: 'text/plain;charset=utf-8' }));
  else if (act === 'stop') {
    if (it.status === 'queued' || it.status === 'pick') { items.splice(items.indexOf(it), 1); if (ed.it === it) closeEditor(true); renderList(); }
    else if (it.status === 'run') worker?.postMessage({ type: 'cancel', id });
  } else if (act === 'pick') openEditor(it);
  else if (act === 'del') { items.splice(items.indexOf(it), 1); await dbDel(id); renderList(); refreshStorage(); }
});
$('list').addEventListener('toggle', (e) => { const r = e.target.closest?.('.row'); const it = r && items.find((x) => x.id === r.dataset.id); if (it) it.open = e.target.open; }, true);
$('zipAll').onclick = downloadAll;
// clears every video in the queue and every saved text; the downloaded model stays
$('clearAll').onclick = async () => {
  if (!items.length) return;
  if (!confirm('Прибрати всі відео й тексти з цієї сторінки? Модель залишиться в браузері.')) return;
  const running = items.find((x) => x.status === 'run');
  items.length = 0; // the running file's late messages find no item and are dropped (nothing gets saved)
  closeEditor(false);
  if (running) worker?.postMessage({ type: 'cancel', id: running.id });
  await dbClear();
  renderList(); refreshStorage(); toast('Відео й тексти прибрано');
};
$('storage').addEventListener('click', async (e) => {
  if (e.target.id !== 'dropModel') return;
  if (busy) return toast('Спершу дочекайтесь кінця черги', true);
  if (!confirm('Видалити завантажену модель? Наступного разу вона завантажиться знову. Тексти залишаться.')) return;
  worker?.terminate(); worker = null; model = { key: '', state: 'idle' };
  await caches.delete('transformers-cache'); renderModel(); refreshStorage(); toast('Модель видалено з браузера');
});
addEventListener('beforeunload', (e) => { if (busy) { e.preventDefault(); e.returnValue = ''; } });

/* ---------- start ---------- */
(async () => {
  gpu = await detectGpu();
  buildSettings();
  renderModel();
  const saved = (await dbAll()) || [];
  saved.sort((a, b) => b.createdAt - a.createdAt).forEach((x) => items.push(x));
  renderList();
  refreshStorage();
})();
