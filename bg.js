// Punch hub background: the LipSync app's background scenes (sound wave -> lips + glove knockout -> water -> globe),
// all drawn as ASCII characters. Copied from public/bg.js; the hub only adds the all-ASCII mode and a glyph atlas.
// Cheap by design: ~18 fps, only the art's bounding box is computed, paused when the tab is hidden,
// a single still frame when the user prefers reduced motion.

const CELL = 4;   // css px per dot cell
const FPS = 24;   // more frames, so slow motion stays smooth
const SPEED = 0.8; // the whole background runs at 80% speed: smooth, not sleepy
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(v => (v + 0.5) / 16);

// speech-like loudness 0..1: syllables riding on slow phrases, with pauses
function voice(t) {
  const syl = Math.sin(t * 1.9) * 0.5 + Math.sin(t * 4.7 + 1.3) * 0.32 + Math.sin(t * 10.3 + 0.4) * 0.18;
  const phrase = Math.max(0, Math.sin(t * 0.42) * 0.8 + 0.45);
  return Math.min(1, Math.max(0, syl) * phrase * 1.25);
}

const hash = (x, y) => { const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return s - Math.floor(s); };
const smooth = (a, b, x) => { const k = Math.min(1, Math.max(0, (x - a) / (b - a))); return k * k * (3 - 2 * k); };

/* ---------- scenes: each returns a bounding box and a per-cell shader -> [ink 0..1, accent 0..1] ---------- */

// 1) a dotted globe that slowly turns; its latitude lines ripple with the voice
function sphere(W, H) {
  const R = Math.min(Math.max(W * 0.34, 340), 640);  // big, partly beyond the corner
  const cx = Math.max(R * 0.5, W * 0.16), cy = H - R * 0.25;
  const tilt = 0.38, ct = Math.cos(tilt), st = Math.sin(tilt);
  const L = [-0.55, -0.6, 0.58]; // light from the upper left
  return {
    box: [cx - R - 8, cy - R - 8, cx + R + 8, cy + R + 8],
    begin(t) { this.rot = t * 0.07; this.v = voice(t); this.t = t; },
    shade(x, y) {
      const dx = (x - cx) / R, dy = (y - cy) / R, d2 = dx * dx + dy * dy;
      if (d2 > 1) return 0;
      const z = Math.sqrt(1 - d2);
      const yy = dy * ct - z * st, zz = dy * st + z * ct;
      const lat = Math.asin(Math.max(-1, Math.min(1, yy)));
      const lon = Math.atan2(dx, zz) + this.rot;
      const ripple = Math.sin(lat * 9 - this.t * 2.6) * this.v * 0.9;
      const lines = Math.pow(0.5 + 0.5 * Math.cos(lat * 46 + ripple), 3);
      const land = smooth(0.05, 0.35, Math.sin(lon * 2 + Math.sin(lat * 3.1) * 1.6) * Math.cos(lat * 2.4 - lon * 0.9) + 0.12 * Math.sin(lon * 7 + lat * 5));
      const light = 0.18 + 0.82 * Math.max(0, -(dx * L[0] + dy * L[1]) + z * L[2]);
      const rim = Math.pow(1 - z, 6) * 0.5;
      return Math.min(1, light * (0.25 + 0.75 * lines) * (0.45 + 0.55 * land) + rim);
    },
  };
}

// a real 3D boxing glove: a signed-distance model (padded fist, knuckle pad, folded fingers, thumb, cuff, strap),
// ray-marched per ASCII cell and lit like leather (key light, glossy highlight, rim light)
const sdEll = (x, y, z, cx, cy, cz, rx, ry, rz) => {
  const px = (x - cx) / rx, py = (y - cy) / ry, pz = (z - cz) / rz;
  const k0 = Math.hypot(px, py, pz), k1 = Math.hypot(px / rx, py / ry, pz / rz);
  return k1 === 0 ? -Math.min(rx, ry, rz) : k0 * (k0 - 1) / k1;
};
const sdCapX = (x, y, z, x0, x1, r) => { const cx = Math.max(x0, Math.min(x1, x)); return Math.hypot(x - cx, y, z) - r; };
const smin = (a, b, k) => { const h = Math.max(k - Math.abs(a - b), 0) / k; return Math.min(a, b) - h * h * k * 0.25; };
function gloveSDF(x, y, z) {               // glove space: +x points at the target, +y up, +z towards the viewer
  let d = sdEll(x, y, z, 0.12, 0.02, 0, 0.43, 0.37, 0.33);             // padded fist
  d = smin(d, sdEll(x, y, z, 0.38, 0.03, 0, 0.27, 0.32, 0.3), 0.12);    // knuckle pad
  d = smin(d, sdEll(x, y, z, 0.22, 0.2, 0, 0.33, 0.16, 0.28), 0.08);    // folded fingers on top
  d = smin(d, sdEll(x, y, z, 0.16, -0.2, 0.19, 0.27, 0.11, 0.12), 0.06); // thumb along the side
  const r = 0.25 + 0.025 * (x + 0.75);                                  // cuff, slightly tapered
  d = smin(d, sdCapX(x, y, z, -0.75, -0.15, r), 0.12);
  d = Math.min(d, sdCapX(x, y, z, -0.56, -0.43, 0.275));               // strap
  return d;
}
const GL_L = (() => { const v = [0.28, 0.5, 0.82], n = Math.hypot(...v); return v.map(c => c / n); })();
const GL_H = (() => { const v = [GL_L[0], GL_L[1], GL_L[2] + 1], n = Math.hypot(...v); return v.map(c => c / n); })();
// px, py: glove-space point on the image plane; roll: rotation around the glove's own axis (a cross turns over in flight)
function gloveShade(px, py, roll) {
  if (px < -0.95 || px > 0.75 || py < -0.6 || py > 0.6) return null;
  const c = Math.cos(roll), sn = Math.sin(roll);
  const at = (z) => gloveSDF(px, py * c - z * sn, py * sn + z * c);
  let z = 0.7, d = 0, hit = false;
  for (let i = 0; i < 28; i++) { d = at(z); if (d < 0.004) { hit = true; break; } z -= Math.max(d, 0.01); if (z < -0.7) break; }
  if (!hit) return null;
  const e = 0.006;                         // normal from the field (in view space)
  const f = (dx, dy, dz) => gloveSDF(px + dx, (py + dy) * c - (z + dz) * sn, (py + dy) * sn + (z + dz) * c);
  let nx = f(e, 0, 0) - f(-e, 0, 0), ny = f(0, e, 0) - f(0, -e, 0), nz = f(0, 0, e) - f(0, 0, -e);
  const nl = Math.hypot(nx, ny, nz) || 1; nx /= nl; ny /= nl; nz /= nl;
  const diff = Math.max(0, nx * GL_L[0] + ny * GL_L[1] + nz * GL_L[2]);
  const spec = Math.pow(Math.max(0, nx * GL_H[0] + ny * GL_H[1] + nz * GL_H[2]), 24);
  const rim = Math.pow(1 - Math.max(0, nz), 3) * 0.3;
  let v = 0.14 + 0.58 * diff + rim + 0.55 * spec;
  const crease = Math.abs(sdEll(px, py * c - z * sn, py * sn + z * c, 0.22, 0.2, 0, 0.33, 0.16, 0.28) - sdEll(px, py * c - z * sn, py * sn + z * c, 0.12, 0.02, 0, 0.43, 0.37, 0.33));
  if (crease < 0.02) v *= 0.6 + 20 * crease;
  if (Math.abs(px + 0.15) < 0.018 || Math.abs(px + 0.43) < 0.012 || Math.abs(px + 0.56) < 0.012) v *= 0.55; // seams at the cuff
  return { v: Math.min(1, v), tint: Math.max(0, 1 - spec * 2.2) }; // leather is brand orange, the gloss goes white
}

// 2) lips drawn in ASCII characters, full and lit from above, quietly mouthing words;
//    every few seconds the logo's glove flies in: sharp acceleration with a speed trail, impact, a springy squash,
//    a shock ring and an orange flash, then it eases back out
function lips(W, H) {
  const w = Math.min(Math.max(W * 0.15, 180), 280);
  const S = w * 1.45;                            // glove scale: px per glove unit (the fist is ~0.7 units tall)
  const cx = Math.max(w * 2.3, W * 0.3), cy = H - w * 0.8;
  const g = (x, m, s) => Math.exp(-(((x - m) / s) ** 2));
  const PUNCH = 6.8;                            // the lips speak briefly, then one slow-motion knockout carries us into the next scene
  const left = cx - w;                          // the lips' left corner, where the glove lands
  const gEnd = left + w * 0.28 - 0.63 * S;      // glove centre when its front touches the lips
  const gStart = -0.8 * S;
  const gyc = cy + w * 0.02;
  // droplets thrown off on impact (fixed seeds, so every punch looks alike but natural)
  const DROPS = Array.from({ length: 28 }, (_, i) => {
    const a = -1.25 + 2.5 * hash(i, 1.7), sp = 260 + 520 * hash(i, 9.1);
    return { vx: Math.cos(a) * sp * 0.8 + 120, vy: Math.sin(a) * sp - 140, r: 2.5 + 3 * hash(i, 4.4) };
  });
  return {
    ascii: true,
    box: [0, cy - Math.max(w * 1.05, 0.62 * S) - 60, cx + w * 1.8, H],
    begin(t, local) {
      this.t = t;
      const L = local === undefined ? t % 16 : local;
      this.open = voice(t);
      this.gx = null; this.blur = 0; this.sq = 0; this.tau = -9; this.roll = 0; this.scale = 1; this.shx = 0; this.shy = 0;
      this.shatter = 0; this.gloveDust = 0; this.gone = false;
      const tau = L - PUNCH;
      const ease3 = (r) => (r < 0.5 ? 4 * r * r * r : 1 - Math.pow(-2 * r + 2, 3) / 2); // smooth in and out
      if (tau >= 4.6) this.gone = true;           // knocked out: nothing left, the next scene has taken over
      else if (tau >= -4.4) {
        this.tau = tau;
        const ready = gEnd - S * 0.55;          // where the glove gathers, its fist already on screen
        const cocked = ready - S * 0.1;         // drawn back before the strike
        if (tau < -2.2) {                       // it slowly gathers out of drifting dust while drawing back (the wind-up)
          const r = (tau + 4.4) / 2.2, e = ease3(r);
          this.gx = ready + S * 0.06 + (cocked - ready - S * 0.06) * e;
          this.gloveDust = 1 - e;
          this.roll = 0.45; this.scale = 0.88;
        } else if (tau < -1.8) {                // a held breath, barely moving
          this.gx = cocked - S * 0.01 * Math.sin((tau + 2.2) / 0.4 * Math.PI);
          this.roll = 0.45; this.scale = 0.88;
        } else if (tau < 0) {                   // the strike in slow motion: one long push, turning over, coming out of depth
          const r = (tau + 1.8) / 1.8, k = r * r * (3 - 2 * r) * 0.4 + r * r * 0.6; // eases in, keeps its weight to the end
          this.gx = cocked + (gEnd - cocked) * k;
          this.roll = 0.45 * (1 - k);
          this.scale = 0.88 + 0.12 * k;
          this.blur = 0.35 * k;                 // a light smear only
        } else {                                // pressed in, sinking a little further while everything breaks apart
          this.gx = gEnd + w * 0.08 * Math.max(0, Math.exp(-tau * 1.5) * Math.cos(tau * 4.5)) + w * 0.07 * tau;
          this.sq = Math.exp(-tau * 1.5) * Math.cos(tau * 4.5);   // the lips give way, in slow motion
          const sh = 2.5 * Math.exp(-tau * 3);                    // a gentle tremor
          this.shx = sh * Math.sin(tau * 22); this.shy = sh * 0.6 * Math.cos(tau * 18);
          this.shatter = 1;                     // the lips crack from the impact point outward (per character, in shade)
          const b2 = Math.min(1, Math.max(0, (tau - 1.4) / 3)); // then the glove crumbles and carries the change
          this.gloveDust = b2 * b2 * (3 - 2 * b2);
        }
      }
      this.open = Math.max(this.open, 0.85 * Math.max(0, this.sq));
    },
    lipsAt(x, y) {
      const sq = this.sq;
      const shift = 0.5 * w * 0.17 * sq, sx = (1 - this.open * 0.05) * (1 - 0.17 * sq), sy = 1 + 0.13 * sq;
      const u = (x - cx - shift) / (w * sx), v = (y - cy) / (w * 1.4 * sy); // lips are ~2.3:1, not a flat strip
      const au = Math.abs(u);
      // faint "skin" texture around the mouth, like the dotted field in an ASCII portrait
      const halo = Math.max(0, 1 - Math.hypot(u / 1.5, v / 0.9));
      let ink = halo * (0.05 + 0.07 * hash(Math.floor(x / 6), Math.floor(y / 10)));
      if (au >= 1) return ink;
      const s = 1 - u * u;
      const m = 0.025 * u * u - 0.01;                           // mouth line, corners a bit lower
      const gap = (0.006 + this.open * 0.13) * Math.pow(s, 0.9); // opening
      const top = m - gap * 0.5 - (0.27 * Math.pow(s, 0.55) * (1 - 0.32 * g(u, 0, 0.11)) + 0.045 * g(au, 0.27, 0.13) * s);
      const bot = m + gap * 0.5 + 0.34 * Math.pow(s, 0.7);
      const upIn = m - gap * 0.5, loIn = m + gap * 0.5;
      const corner = smooth(1, 0.82, au);
      // volume: each lip is lit as a rounded tube (normal from its cross-section), light from the upper left + a glossy highlight
      const lit = (k, bulge) => {
        const ph = Math.PI * k;
        const ny = -Math.cos(ph), nz = Math.sin(ph) * bulge, nx = u * 0.55;
        const n = Math.hypot(nx, ny, nz) || 1;
        const d = Math.max(0, (nx * -0.32 + ny * -0.55 + nz * 0.77) / n);
        return 0.08 + 0.72 * d + 0.55 * Math.pow(d, 18);
      };
      if (v >= top && v < upIn) {                               // upper lip: lit ridge on top, curls into shadow
        const k = (v - top) / (upIn - top);
        ink = lit(k, 0.85) * corner * (1 - 0.35 * smooth(0.75, 1, k)) + 0.08 * g(au, 0, 0.1) * (1 - k);
      } else if (v >= loIn && v < bot) {                        // lower lip: fuller, catches the main highlight
        const k = (v - loIn) / (bot - loIn);
        ink = lit(k, 1.25) * corner * (1 - 0.3 * smooth(0, 0.12, 0.12 - k)) - 0.06 * g(au, 0, 0.06) * g(k, 0.3, 0.3);
      } else if (v >= upIn && v < loIn) {                       // inside: dark, with a hint of teeth under the upper lip
        const k = (v - upIn) / Math.max(1e-3, loIn - upIn);
        ink = this.open > 0.25 ? 0.42 * g(k, 0.12, 0.14) * smooth(0.75, 0.35, au) : 0;
      } else if (v >= bot && v < bot + 0.16) {                  // soft shadow under the lower lip
        ink = Math.max(ink, 0.03);
      } else if (v < top && v > top - 0.06) {                   // a thin lit rim above the upper lip
        ink = Math.max(ink, 0.18 * corner * (1 - (top - v) / 0.06));
      }
      return Math.max(0, ink);
    },
    gloveAt(x, y, gx) {
      const sc = S * this.scale;
      return gloveShade((x - gx) / sc, -(y - gyc) / sc, this.roll);
    },
    shade(x, y) {
      this.tint = 0;
      x += this.shx; y += this.shy;
      if (this.gone) return 0;
      const tau = this.tau;
      let ink;
      if (this.shatter > 0) {                                   // the break spreads from the impact point across the lips
        const r = hash(Math.floor(x / 6), Math.floor(y / 10));
        const dx = x - left, dy = y - cy, dl = Math.hypot(dx, dy * 1.3) || 1;
        const start = 0.12 + dl / (w * 2.2) * 1.7 + r * 0.35;   // further from the hit = later
        const a0 = Math.min(1, Math.max(0, (tau - start) / 1.9)), a = 1 - (1 - a0) * (1 - a0);
        if (a <= 0) ink = this.lipsAt(x, y);
        else if (r < a * 0.85) ink = 0;
        else {
          const amp = w * 1.9 * a * (0.35 + r);
          const sx = x - dx / dl * amp, sy = y - dy / dl * amp * 0.8 - w * 0.3 * a * a;
          ink = this.lipsAt(sx, sy) * Math.pow(1 - a, 0.8);
          if (ink > 0.05) this.tint = Math.max(this.tint, 0.55 * (1 - a));
        }
      } else ink = this.lipsAt(x, y);
      if (tau >= 0 && tau < 2) {                                // a warm flush spreading over the lips on impact
        const near = Math.exp(-Math.hypot(x - left, (y - cy) * 1.4) / (w * 0.8)) * (1 - tau / 2);
        this.tint = Math.max(this.tint, near * 0.9);
        ink = Math.min(1, ink * (1 + 0.5 * near));
      }
      if (tau >= 0 && tau < 2) {                                // a faint pressure ring
        const d = Math.hypot(x - left, (y - cy) * 1.25), r = 30 + tau * 190;
        const ring = g(d, r, 14) * Math.pow(1 - tau / 2, 2) * 0.35;
        if (ring > ink) ink = ring;
      }
      if (tau >= 0 && tau < 3.2) {                              // droplets flying off, falling under gravity
        const ix = left + w * 0.05, iy = cy;
        for (const d of DROPS) {
          const tt = tau * 0.3, px = ix + d.vx * tt, py = iy + d.vy * tt + 900 * tt * tt; // slow motion
          const dd = Math.hypot(x - px, y - py);
          if (dd < d.r + 3) { const dv = (1 - tau / 3.2) * 0.85; if (dv > ink) { ink = dv; this.tint = 0; } }
        }
      }
      if (this.gx !== null) {
        let hit = null;
        if (this.gloveDust > 0) {                               // the glove crumbles the same way, drifting back and down
          const r = hash(Math.floor(y / 10) + 0.5, Math.floor(x / 6));
          if (r >= this.gloveDust * 0.9) {
            const gcx = this.gx, dx = x - gcx, dy = y - gyc, dl = Math.hypot(dx, dy) || 1;
            const amp = S * 1.3 * this.gloveDust * (0.3 + r);
            const h = this.gloveAt(x - dx / dl * amp + S * 0.25 * this.gloveDust, y - dy / dl * amp - S * 0.3 * this.gloveDust * this.gloveDust, this.gx);
            if (h) hit = { v: h.v * Math.pow(1 - this.gloveDust, 0.7), tint: h.tint };
          }
        } else hit = this.gloveAt(x, y, this.gx);
        if (!hit && this.blur > 0.05) {                         // motion smear: the glove a moment earlier
          for (let k = 1; k <= 3 && !hit; k++) {
            const h2 = this.gloveAt(x, y, this.gx - k * Math.min(this.blur, 3) * 22);
            if (h2) hit = { v: h2.v * (0.55 - k * 0.13), tint: h2.tint };
          }
        }
        if (hit && hit.v >= ink * 0.6) { this.tint = hit.tint; return Math.max(0.1, hit.v); }
      }
      return ink;
    },
  };
}

// 3) a dithered voice waveform scrolling along the bottom edge
function wave(W, H) {
  const y0 = H - Math.min(150, H * 0.17), band = Math.min(110, H * 0.13);
  return {
    box: [0, y0 - band - 30, W, H],
    begin(t) { this.t = t; },
    shade(x, y) {
      const col = Math.floor(x / 6) * 6;                    // 6px bars
      const tt = this.t * 1.1 - col / 140;
      const a = (0.12 + voice(tt) * 0.88) * (0.55 + 0.45 * Math.sin(col * 0.013 + this.t * 0.3));
      const h = a * band;
      const dy = Math.abs(y - y0);
      if (dy > h + 26) return 0;
      const core = dy < h ? 0.55 + 0.45 * (1 - dy / Math.max(1, h)) : 0;
      const mist = smooth(h + 26, h, dy) * 0.3;
      const fade = smooth(0, W * 0.25, x) * smooth(W, W * 0.75, x) * 0.6 + 0.4;
      return Math.max(core, mist) * fade;
    },
  };
}

// 5) digital water: a dotted sea surface in perspective, wave fronts rolling towards the viewer, glints on the crests
function water(W, H) {
  const hy = H * 0.62, D = 260, cx = W / 2;
  return {
    box: [0, hy, W, H],
    begin(t) { this.t = t; },
    shade(x, y) {
      const dy = y - hy;
      if (dy <= 2) return 0;
      const z = 1600 / dy;                          // depth: ~4 at the bottom edge, large near the horizon
      const xw = (x - cx) * z / 1600;               // world x on the surface
      const t = this.t;
      const bend = 1.6 * Math.sin(xw * 0.9 + t * 0.4) + 0.8 * Math.sin(xw * 2.3 - t * 0.7 + z * 0.3);
      const phase = z * 2.2 + bend + t * 1.4;        // fronts roll towards the viewer
      const crest = Math.pow(0.5 + 0.5 * Math.sin(phase), 3);
      const swell = 0.5 + 0.5 * Math.sin(z * 0.45 - t * 0.5 + xw * 0.35);
      const glint = hash(Math.floor(xw * 3), Math.floor(z * 4)) > 0.975 ? crest * (0.5 + 0.5 * Math.sin(t * 2.5 + xw * 7)) : 0;
      const far = smooth(0, 110, dy);               // dissolve into the horizon
      const sides = 0.55 + 0.45 * smooth(0, W * 0.3, Math.min(x, W - x));
      return Math.min(1, (0.06 + 0.8 * crest * (0.55 + 0.45 * swell) + 0.5 * glint) * far * sides);
    },
  };
}

// 6) the Punch "P" mark as a solid slab, spinning around its vertical axis like a coin and resting face-on between turns.
// The logo's paths are rasterised once into a mask; each ASCII cell marches a ray through the rotated slab.
const LOGO_PATHS = [
  'M97.5,237.6l-35.3-35.2l338.2-1.6c0,0,119.4-6.9,115.2,105.7c-4.4,117.4-116.1,147-216.8,147L278.2,420c0,0,141.7-4.8,156.4-88.1c19.7-111.9-55.1-92-157.3-94.3L97.5,237.6z',
  'M273.2,588.5H167l78.7-350.9l53.2,86l-51.6,241.9C247.3,565.5,239.2,588.5,273.2,588.5z',
];
const LOGO_BOX = [60, 198, 518, 590]; // the paths' bounds in their own units
function logoMask() {
  const MW = 240, MH = Math.round(MW * (LOGO_BOX[3] - LOGO_BOX[1]) / (LOGO_BOX[2] - LOGO_BOX[0]));
  const c = document.createElement('canvas'); c.width = MW; c.height = MH;
  const x = c.getContext('2d', { willReadFrequently: true });
  x.scale(MW / (LOGO_BOX[2] - LOGO_BOX[0]), MH / (LOGO_BOX[3] - LOGO_BOX[1]));
  x.translate(-LOGO_BOX[0], -LOGO_BOX[1]);
  for (const d of LOGO_PATHS) x.fill(new Path2D(d));
  const a = x.getImageData(0, 0, MW, MH).data, m = new Uint8Array(MW * MH);
  for (let i = 0; i < m.length; i++) m[i] = a[i * 4 + 3] > 110 ? 1 : 0;
  return { m, MW, MH };
}
function logo(W, H) {
  const { m, MW, MH } = logoMask();
  const S = Math.min(Math.max(W * 0.3, 260), 520);            // logo width in px
  const LH = S * MH / MW;
  const cx = W < 760 ? W * 0.5 : W * 0.76, cy = W < 760 ? H * 0.72 : H * 0.56;
  const D = 0.16;                                             // slab thickness, in logo widths
  const R = Math.hypot(0.5, D / 2) + 0.02;                    // half extent of the spinning slab
  const STEPS = 44, dz = (2 * R) / STEPS;
  const inside = (u, v) => {                                   // u, v: -0.5..0.5 across the logo width / height
    const i = ((u + 0.5) * MW) | 0, j = ((v + 0.5) * MH) | 0;
    return i >= 0 && j >= 0 && i < MW && j < MH && m[j * MW + i] === 1;
  };
  return {
    ascii: true,
    box: [cx - S * R - 8, cy - LH / 2 - 8, cx + S * R + 8, cy + LH / 2 + 8],
    begin(t) {
      const P = 5.2, ph = t / P, f = ph - Math.floor(ph);     // one turn per period: ~60% spinning, then a rest
      const k = smooth(0, 0.62, f);
      this.th = Math.PI * 2 * (Math.floor(ph) + k) + 0.18 * Math.sin(t * 0.7) * (1 - Math.sin(Math.PI * k));
      this.c = Math.cos(this.th); this.s = Math.sin(this.th);
      this.gloss = (k * 2.2 - 1.1);                           // a highlight sweeps across while it turns
    },
    shade(x, y) {
      this.tint = 0;
      const sx = (x - cx) / S, v = (y - cy) / LH;
      if (v < -0.5 || v > 0.5 || Math.abs(sx) > R) return 0;
      const c = this.c, s = this.s, half = D / 2;
      let prevSlab = false;
      for (let n = 0; n <= STEPS; n++) {                      // from the viewer backwards
        const z = R - n * dz;
        const u = sx * c - z * s, w = sx * s + z * c;          // object space: u across the logo, w through its thickness
        const slab = Math.abs(w) <= half;
        if (slab && inside(u, v * 1)) {
          if (!prevSlab || n === 0) {                          // entered through the front or back face
            const face = Math.abs(c);
            const gd = ((u + v * 0.6) - this.gloss) / 0.12, g = Math.exp(-gd * gd) * face;
            this.tint = 0.85 - 0.5 * g;
            return Math.min(1, 0.22 + 0.62 * Math.pow(face, 0.7) + 0.5 * g);
          }
          this.tint = 0.35;                                    // a side wall: lit by how far it has turned towards us
          return 0.16 + 0.5 * Math.abs(s);
        }
        prevSlab = slab;
      }
      return 0;
    },
  };
}

// 4) sound wave -> lips (knockout) -> water -> big globe -> ...: each shape scatters into dust and re-gathers as the next one.
// Globe and wave are dots, the lips are ASCII characters, so a frame can hold both layers during a change.
function cycle(W, H) {
  const list = [wave(W, H), lips(W, H), water(W, H), logo(W, H), sphere(W, H)]; // the big globe closes the loop
  // how long each scene holds before it dissolves into the next; the lips are a short interlude ended by the punch
  const HOLDS = [16, 9.4, 16, 21, 16], MOVE = 4;
  const STARTS = []; let acc = 0; for (const h of HOLDS) { STARTS.push(acc); acc += h + MOVE; }
  const PERIOD = acc;
  const box = list.reduce((b, s) => [Math.min(b[0], s.box[0]), Math.min(b[1], s.box[1]), Math.max(b[2], s.box[2]), Math.max(b[3], s.box[3])], [Infinity, Infinity, -Infinity, -Infinity]);
  const kind = () => 'ascii'; // hub: every scene is ASCII
  return {
    box,
    begin(t) {
      const k = t % PERIOD;
      let i = 0; while (i < list.length - 1 && k >= STARTS[i + 1]) i++;
      const in_ = k - STARTS[i], HOLD = HOLDS[i], SEG = HOLD + MOVE;
      this.from = list[i]; this.to = list[(i + 1) % list.length];
      const p = in_ < HOLD ? 0 : (in_ - HOLD) / MOVE;
      this.p = p * p * (3 - 2 * p);
      this.from.begin(t, in_); if (this.p > 0) this.to.begin(t, in_ - SEG);
      const mid = Math.sin(Math.PI * this.p);   // 0 at both ends, 1 halfway
      this.amp = mid * 46;                      // how far the dots fly apart
      this.dim = 1 - 0.3 * mid;
      this.t = t;
      this.layers = { [kind(this.from)]: 1 };
      if (this.p > 0) this.layers[kind(this.to)] = 1;
    },
    shade(x, y, mode) {
      this.tint = 0;
      if (this.p === 0) { if (kind(this.from) !== mode) return 0; const v = this.from.shade(x, y); this.tint = this.from.tint || 0; return v; }
      const g = hash(y + 3.1, x - 7.7);
      // every dot switches shape at its own moment, so the change dissolves instead of wiping
      const scene = g < this.p ? this.to : this.from;
      if (kind(scene) !== mode) return 0;
      const h = hash(x, y);
      const a = h * 6.2832 + this.t * 0.4, r = this.amp * (0.35 + g);
      const sx = x + Math.cos(a) * r, sy = y + Math.sin(a) * r - this.amp * 0.4; // a little upward drift, like dust
      const v = scene.shade(sx, sy) * this.dim;
      this.tint = scene.tint || 0;
      return v;
    },
  };
}

const SCENES = { sphere, lips, wave, water, logo, cycle };

// play of light: soft orange highlights drifting slowly across the art (0 = plain ink, 1 = full accent)
function glow(x, y, t) {
  const a = Math.sin(x * 0.0042 + y * 0.0027 - t * 0.33);
  const b = Math.sin(x * 0.0023 - y * 0.0049 + t * 0.21 + 1.7);
  return 0.7 * smooth(0.42, 0.9, (a * 0.6 + b * 0.4) * 0.5 + 0.5);
}
const STEPS = 16;

/* ---------- renderer ---------- */

function parseColor(ctx, css, fallback) {
  ctx.fillStyle = fallback; ctx.fillStyle = css || fallback;
  ctx.clearRect(0, 0, 1, 1); ctx.fillRect(0, 0, 1, 1);
  const d = ctx.getImageData(0, 0, 1, 1).data;
  return [d[0], d[1], d[2]];
}

export function startBackground(name) {
  const make = SCENES[name] || cycle;
  const canvas = document.createElement('canvas');
  canvas.className = 'bg-art';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.prepend(canvas);
  const ctx = canvas.getContext('2d');
  const small = document.createElement('canvas');
  const sctx = small.getContext('2d', { willReadFrequently: false });
  const probe = document.createElement('canvas').getContext('2d', { willReadFrequently: true });

  let W = 0, H = 0, dpr = 1, cols = 0, rows = 0, img = null, buf = null, scene = null, mask = null, ink = 0, inkCss = '#fff', PAL = [], PALC = [];
  const still = matchMedia('(prefers-reduced-motion: reduce)');

  function colors() {
    const cs = getComputedStyle(document.documentElement);
    const [r, g, b] = parseColor(probe, cs.getPropertyValue('--text').trim(), '#ffffff');
    ink = (255 << 24) | (b << 16) | (g << 8) | r; // little-endian RGBA
    inkCss = `rgb(${r},${g},${b})`;
    const [ar, ag, ab] = parseColor(probe, cs.getPropertyValue('--accent').trim(), '#ff5a36');
    PAL = []; PALC = [];
    for (let i = 0; i < STEPS; i++) {
      const k = i / (STEPS - 1);
      const R = Math.round(r + (ar - r) * k), G = Math.round(g + (ag - g) * k), B = Math.round(b + (ab - b) * k);
      PAL.push((255 << 24) | (B << 16) | (G << 8) | R); PALC.push(`rgb(${R},${G},${B})`);
    }
  }

  function resize() {
    W = Math.max(1, innerWidth); H = Math.max(1, innerHeight); dpr = Math.min(2, devicePixelRatio || 1);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    cols = Math.ceil(W / CELL); rows = Math.ceil(H / CELL);
    small.width = cols; small.height = rows;
    img = sctx.createImageData(cols, rows); buf = new Uint32Array(img.data.buffer);
    scene = make(W, H);
    // round-dot mask, one dot per cell, in device pixels
    const p = document.createElement('canvas'); const s = CELL * dpr;
    p.width = p.height = Math.round(s);
    const pc = p.getContext('2d'); pc.fillStyle = '#000';
    pc.beginPath(); pc.arc(s / 2, s / 2, 1.05 * dpr, 0, Math.PI * 2); pc.fill();
    mask = ctx.createPattern(p, 'repeat');
  }

  // ASCII scenes: one character per 6x10 cell, picked from a density ramp; brighter cells are also more opaque
  const RAMP = ' .,:;-~=+*xX#%@';
  // glyph atlas: every ramp character pre-rendered in every palette step, so a frame is just drawImage calls
  const CW = 6, CH = 10;
  let atlas = null;
  function buildAtlas() {
    atlas = document.createElement('canvas');
    const w = Math.round(CW * dpr), h = Math.round(CH * dpr);
    atlas.width = w * RAMP.length; atlas.height = h * STEPS;
    const a = atlas.getContext('2d');
    a.setTransform(dpr, 0, 0, dpr, 0, 0);
    a.font = '600 10px ui-monospace, "SF Mono", Consolas, Menlo, monospace';
    a.textAlign = 'center'; a.textBaseline = 'middle';
    for (let s = 0; s < STEPS; s++) {
      a.fillStyle = PALC[s];
      for (let i = 1; i < RAMP.length; i++) a.fillText(RAMP[i], i * CW + CW / 2, s * CH + CH / 2);
    }
  }
  function drawAscii(t) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    const w = Math.round(CW * dpr), h = Math.round(CH * dpr);
    const [x0, y0, x1, y1] = scene.box;
    for (let y = Math.max(0, Math.floor(y0 / CH) * CH) + CH / 2; y < Math.min(H, y1); y += CH) {
      for (let x = Math.max(0, Math.floor(x0 / CW) * CW) + CW / 2; x < Math.min(W, x1); x += CW) {
        const v = scene.shade(x, y, 'ascii');
        if (v < 0.04) continue;
        const i = Math.min(RAMP.length - 1, Math.round(v * (RAMP.length - 1)));
        if (i === 0) continue;
        ctx.globalAlpha = 0.18 + 0.82 * Math.pow(Math.min(1, v), 1.1);
        const st = (Math.max(glow(x, y, t), scene.tint || 0) * (STEPS - 1)) | 0;
        ctx.drawImage(atlas, i * w, st * h, w, h, Math.round((x - CW / 2) * dpr), Math.round((y - CH / 2) * dpr), w, h);
      }
    }
    ctx.globalAlpha = 1;
  }

  function draw(t) {
    scene.begin(t);
    const layers = scene.layers || { ascii: 1 };
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (layers.dot) drawDots(t);
    else { ctx.globalCompositeOperation = 'source-over'; ctx.clearRect(0, 0, canvas.width, canvas.height); }
    if (layers.ascii) drawAscii(t); // on top: the dot pass clears and masks the whole canvas
  }

  function drawDots(t) {
    buf.fill(0);
    const [x0, y0, x1, y1] = scene.box;
    const c0 = Math.max(0, Math.floor(x0 / CELL)), c1 = Math.min(cols, Math.ceil(x1 / CELL));
    const r0 = Math.max(0, Math.floor(y0 / CELL)), r1 = Math.min(rows, Math.ceil(y1 / CELL));
    for (let r = r0; r < r1; r++) {
      const y = r * CELL + CELL / 2, row = r * cols, br = (r & 3) << 2;
      for (let c = c0; c < c1; c++) {
        const x = c * CELL + CELL / 2, v = scene.shade(x, y, 'dot');
        // depth: denser dither, and every dot carries its own opacity, so lit parts read solid and shadows melt away
        if (v > 0.03 && v > BAYER[br | (c & 3)] * 0.55) {
          const a = Math.round(255 * (0.2 + 0.8 * Math.pow(Math.min(1, v), 1.1)));
          buf[row + c] = ((PAL[(glow(x, y, t) * (STEPS - 1)) | 0] & 0xffffff) | (a << 24)) >>> 0;
        }
      }
    }
    sctx.putImageData(img, 0, 0);
    ctx.globalCompositeOperation = 'copy';
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(small, 0, 0, cols * CELL * dpr, rows * CELL * dpr);
    ctx.globalCompositeOperation = 'destination-in';
    ctx.fillStyle = mask; ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  let raf = 0, last = 0;
  // The animation runs on the wall clock, so a reload (or another tab) picks up exactly where it would be instead of
  // starting over. 7340 s = 100 full loops, keeps the numbers small without a visible jump.
  // ?bgt= pins a start time for local previews.
  const pinned = Number(new URLSearchParams(location.search).get('bgt')) || 0;
  const t0 = performance.now();
  const clock = (now) => (pinned ? pinned + (now - t0) / 1000 * SPEED : (Date.now() / 1000 * SPEED) % 7340);
  function loop(now) {
    raf = requestAnimationFrame(loop);
    if (now - last < 1000 / FPS) return;
    last = now; draw(clock(now));
  }
  function run() {
    cancelAnimationFrame(raf); raf = 0;
    if (still.matches) { draw(Number(new URLSearchParams(location.search).get('bgt')) || 8); return; }
    if (!document.hidden) raf = requestAnimationFrame(loop);
  }

  let rt = 0;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { resize(); buildAtlas(); run(); if (raf) last = 0; }, 120); });
  document.addEventListener('visibilitychange', run);
  still.addEventListener?.('change', run);
  // theme switches (data-theme attribute or the OS scheme)
  new MutationObserver(() => { colors(); buildAtlas(); if (!raf) run(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });
  matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => { colors(); buildAtlas(); if (!raf) run(); });

  colors(); resize(); buildAtlas(); run();
}

// self-start; ?bg=sphere|lips|wave is for local previews only
const pick = new URLSearchParams(location.search).get('bg');
const start = () => startBackground(pick || 'cycle');
if ('requestIdleCallback' in window) requestIdleCallback(start, { timeout: 800 }); else setTimeout(start, 200);
