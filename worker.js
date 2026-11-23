// Punch Text — the speech-to-text worker. Whisper runs here, in the user's browser (WebGPU, or WASM as a fallback).
// Model files are downloaded once from Hugging Face and kept in the browser's Cache Storage ("transformers-cache").
import { pipeline, env, Tensor } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js';

env.allowLocalModels = false;

// the two models the page offers; dtypes per device (fp16 encoder needs WebGPU's shader-f16)
export const MODELS = {
  turbo: { repo: 'onnx-community/whisper-large-v3-turbo', webgpu: { encoder_model: 'fp16', decoder_model_merged: 'q4' } },
  small: { repo: 'onnx-community/whisper-small', webgpu: { encoder_model: 'fp32', decoder_model_merged: 'q4' }, wasm: { encoder_model: 'q8', decoder_model_merged: 'q8' } },
};

const SR = 16000;              // Whisper wants 16 kHz mono
const MAX_WIN = 29.5 * SR;     // one Whisper window is 30 s
const MIN_WIN = 18 * SR;       // cut each window at the quietest moment between 18 and 29.5 s, so words aren't split
const FRAME = 160;             // 10 ms energy frames
const SILENT = 0.004;          // 10 ms frames below this count as silence; an all-silent window is skipped (no "Thank you." hallucinations)
const PAD = 0.3 * SR;

let asr = null, loadedKey = '', device = 'wasm', idToLang = new Map();
const cancelled = new Set();

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'load') await load(m.model, m.device);
    else if (m.type === 'run') await run(m);
    else if (m.type === 'cancel') cancelled.add(m.id);
  } catch (e) {
    console.error(e);
    self.postMessage({ type: 'error', id: m.id ?? null, stage: m.type, message: String(e?.message || e) });
  }
};

async function load(model, dev) {
  const key = `${model}:${dev}`;
  if (asr && loadedKey === key) return self.postMessage({ type: 'ready', model, device });
  if (asr) { try { await asr.dispose(); } catch {} asr = null; }
  const cfg = MODELS[model];
  device = dev;
  const files = new Map(); // file -> [loaded, total], summed into one progress figure
  asr = await pipeline('automatic-speech-recognition', cfg.repo, {
    device: dev,
    dtype: cfg[dev] || cfg.webgpu,
    progress_callback: (p) => {
      if (p.status === 'progress' && p.file) {
        files.set(p.file, [p.loaded || 0, p.total || 0]);
        let loaded = 0, total = 0;
        for (const [l, t] of files.values()) { loaded += l; total += t; }
        self.postMessage({ type: 'load-progress', loaded, total });
      }
    },
  });
  loadedKey = key;
  const gc = asr.model.generation_config;
  idToLang = new Map(Object.entries(gc.lang_to_id || {}).map(([tok, id]) => [id, tok.slice(2, -2)]));
  // warm up the GPU shaders on a second of silence, so the first real window isn't the slow one
  self.postMessage({ type: 'warming' });
  await asr(new Float32Array(SR), { language: 'en', task: 'transcribe' });
  self.postMessage({ type: 'ready', model, device });
}

// split the audio into windows of at most 30 s, ending each at its quietest 10 ms
function windows(audio) {
  const n = Math.ceil(audio.length / FRAME);
  const rms = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    let s = 0; const a = f * FRAME, b = Math.min(audio.length, a + FRAME);
    for (let i = a; i < b; i++) s += audio[i] * audio[i];
    rms[f] = Math.sqrt(s / Math.max(1, b - a));
  }
  const out = [];
  let start = 0;
  while (start < audio.length) {
    let end = audio.length;
    if (end - start > MAX_WIN) {
      let best = Infinity, bestF = 0;
      for (let f = Math.floor((start + MIN_WIN) / FRAME); f < Math.floor((start + MAX_WIN) / FRAME); f++) {
        const v = rms[f] + 0.5 * ((rms[f - 1] || 0) + (rms[f + 1] || 0)); // a little smoothing
        if (v < best) { best = v; bestF = f; }
      }
      end = bestF * FRAME;
    }
    // trim the silence at both ends (Whisper puts the first timestamp at 0 otherwise), keeping 0.3 s around speech
    let a = -1, b = -1;
    for (let f = Math.floor(start / FRAME); f < Math.ceil(end / FRAME); f++) if (rms[f] >= SILENT) { if (a < 0) a = f; b = f; }
    if (a >= 0) out.push({ s: Math.max(start, a * FRAME - PAD), e: Math.min(end, (b + 1) * FRAME + PAD) });
    start = end;
  }
  return out;
}

// Transformers.js has no language detection, so ask the model for the token that follows <|startoftranscript|>.
// Done per window in auto mode, so a video that switches language mid-way is still transcribed, not translated.
async function detectLanguage(chunk) {
  const gc = asr.model.generation_config;
  const { input_features } = await asr.processor(chunk);
  const out = await asr.model.generate({
    inputs: input_features,
    decoder_input_ids: new Tensor('int64', [BigInt(gc.decoder_start_token_id)], [1, 1]),
    max_new_tokens: 1,
  });
  const seq = (out.sequences ?? out).tolist()[0];
  return idToLang.get(Number(seq.at(-1))) || null;
}

// a decoding loop ("de la publicidad de la publicidad …") collapses to one copy
const unloop = (t) => t.replace(/(.{6,80}?)(?:[\s,.;:،-]*\1){3,}/gsu, '$1');

async function run({ id, audio, language, offset = 0 }) {
  cancelled.delete(id);
  const wins = windows(audio);
  if (!wins.length) return self.postMessage({ type: 'done', id, langs: language ? [language] : [], segments: [] });

  const segments = [], langs = new Map(); // language -> seconds of speech
  let done = 0, last = null;
  const t0 = performance.now();
  for (const w of wins) {
    if (cancelled.has(id)) { cancelled.delete(id); return self.postMessage({ type: 'cancelled', id, segments, langs: ranked(langs) }); }
    const chunk = audio.subarray(w.s, w.e), off = offset + w.s / SR, winEnd = offset + w.e / SR; // times stay those of the original video
    const lang = language || (await detectLanguage(chunk)) || last || 'en';
    last = lang;
    langs.set(lang, (langs.get(lang) || 0) + (w.e - w.s) / SR);
    const res = await asr(chunk, { language: lang, task: 'transcribe', return_timestamps: true });
    const fresh = [];
    for (const c of res.chunks || [{ timestamp: [0, null], text: res.text }]) {
      const text = unloop((c.text || '').trim());
      if (!text) continue;
      const [a, b] = c.timestamp || [0, null];
      const start = +(off + (a ?? 0)).toFixed(2);
      const end = +Math.min(winEnd, off + (b ?? winEnd - off)).toFixed(2);
      const prev = segments.at(-1);
      if (prev && prev.text === text && start - prev.end < 2) { prev.end = Math.max(prev.end, end); continue; } // a looped repeat
      const seg = { start, end: Math.max(end, start + 0.2), text, lang };
      segments.push(seg); fresh.push(seg);
    }
    done++;
    self.postMessage({ type: 'progress', id, done, total: wins.length, fresh, langs: ranked(langs), elapsed: (performance.now() - t0) / 1000 });
  }
  self.postMessage({ type: 'done', id, langs: ranked(langs), segments });
}
const ranked = (m) => [...m].sort((a, b) => b[1] - a[1]).map(([l]) => l);
