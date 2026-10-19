/**
 * Deterministic synthetic "pipe end" scenes with exact ground truth.
 * Rendered with OpenCV drawing primitives so they run in Node without a canvas.
 */
import type { CV } from '../../src/analysis/cv';
import type { GrayImage } from '../../src/analysis/circles';

export interface GtCircle {
  x: number;
  y: number;
  r: number;
  /** Index into the scene's radius classes. */
  cls: number;
}

export interface SceneSpec {
  name: string;
  width: number;
  height: number;
  /** Radius per class (px). */
  radii: number[];
  /** Relative frequency per class. */
  weights?: number[];
  layout: 'hex' | 'random';
  /** Number of objects for random layout (upper bound, packing may place fewer). */
  count?: number;
  hollow?: boolean;
  /** Per-object radius jitter (relative, 1 sigma). */
  radiusJitter?: number;
  noiseSigma?: number;
  blur?: number;
  /** Strength (0–1) of a left-to-right illumination falloff. */
  lightGradient?: number;
  clutter?: boolean;
  /** Vertical squash (perspective-like ellipses), 1 = none. */
  squash?: number;
  /** Bar occluding part of the stack. */
  occluder?: boolean;
  seed: number;
}

export interface Scene {
  spec: SceneSpec;
  image: GrayImage;
  truth: GtCircle[];
}

export function rng(seed: number) {
  let s = seed >>> 0 || 1;
  const next = () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
  const gauss = () => {
    const u = Math.max(1e-9, next());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
  return { next, gauss };
}

export function makeScene(cv: CV, spec: SceneSpec): Scene {
  const R = rng(spec.seed);
  const { width: w, height: h } = spec;
  const squash = spec.squash ?? 1;
  const truth: GtCircle[] = [];
  const weights = spec.weights ?? spec.radii.map(() => 1);
  const wsum = weights.reduce((a, b) => a + b, 0);
  const pickClass = () => {
    let u = R.next() * wsum;
    for (let i = 0; i < weights.length; i++) if ((u -= weights[i]!) <= 0) return i;
    return weights.length - 1;
  };
  const jitter = (r: number) => r * (1 + (spec.radiusJitter ?? 0.02) * R.gauss());

  if (spec.layout === 'hex') {
    const r0 = spec.radii[0]!;
    const dx = 2 * r0 * 1.04;
    const dy = dx * Math.sqrt(3) / 2;
    const margin = r0 * 1.6;
    for (let row = 0, y = margin; y < h - margin; row++, y += dy * squash) {
      for (let x = margin + (row % 2 ? dx / 2 : 0); x < w - margin; x += dx) {
        truth.push({ x: x + R.gauss() * r0 * 0.03, y: y + R.gauss() * r0 * 0.03, r: jitter(r0), cls: 0 });
      }
    }
  } else {
    const n = spec.count ?? 150;
    const maxR = Math.max(...spec.radii);
    let attempts = 0;
    while (truth.length < n && attempts < n * 400) {
      attempts++;
      const cls = pickClass();
      const r = jitter(spec.radii[cls]!);
      const x = r + 4 + R.next() * (w - 2 * r - 8);
      const y = r * squash + 4 + R.next() * (h - 2 * r * squash - 8);
      let ok = true;
      for (const t of truth) {
        if (Math.abs(t.x - x) > maxR * 2 + r || Math.abs(t.y - y) > maxR * 2 + r) continue;
        if (Math.hypot(t.x - x, (t.y - y) / squash) < (t.r + r) * 1.03) {
          ok = false;
          break;
        }
      }
      if (ok) truth.push({ x, y, r, cls });
    }
  }

  const mat = new cv.Mat(h, w, cv.CV_8UC1, new cv.Scalar(35));
  try {
    if (spec.clutter) {
      for (let i = 0; i < 25; i++) {
        const p1 = new cv.Point(Math.round(R.next() * w), Math.round(R.next() * h));
        const p2 = new cv.Point(Math.round(R.next() * w), Math.round(R.next() * h));
        cv.line(mat, p1, p2, new cv.Scalar(60 + R.next() * 100), 1 + Math.round(R.next() * 4), cv.LINE_AA);
      }
      for (let i = 0; i < 6; i++) {
        const x = Math.round(R.next() * w), y = Math.round(R.next() * h);
        cv.rectangle(mat, new cv.Point(x, y), new cv.Point(x + 20 + R.next() * 120, y + 20 + R.next() * 80), new cv.Scalar(50 + R.next() * 80), -1);
      }
    }
    for (const t of truth) {
      const c = new cv.Point(Math.round(t.x), Math.round(t.y));
      const bright = 150 + R.next() * 80;
      const axes = (rr: number) => new cv.Size(Math.round(rr), Math.round(rr * squash));
      // Dark contact shadow, then the face.
      cv.ellipse(mat, c, axes(t.r + 1.5), 0, 0, 360, new cv.Scalar(20), -1, cv.LINE_AA);
      cv.ellipse(mat, c, axes(t.r), 0, 0, 360, new cv.Scalar(bright), -1, cv.LINE_AA);
      // Mild radial shading of the cut face.
      cv.ellipse(mat, c, axes(t.r * 0.75), 0, 0, 360, new cv.Scalar(Math.min(255, bright + 12)), -1, cv.LINE_AA);
      if (spec.hollow) cv.ellipse(mat, c, axes(t.r * 0.72), 0, 0, 360, new cv.Scalar(25 + R.next() * 30), -1, cv.LINE_AA);
    }
    if (spec.occluder) {
      cv.rectangle(mat, new cv.Point(Math.round(w * 0.55), 0), new cv.Point(Math.round(w * 0.62), h), new cv.Scalar(90), -1);
    }
    if (spec.blur) cv.GaussianBlur(mat, mat, new cv.Size(0, 0), spec.blur, spec.blur);
    const data = new Uint8Array(mat.data);
    const g = spec.lightGradient ?? 0;
    const ns = spec.noiseSigma ?? 0;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        let v = data[i]!;
        if (g) v *= 1 - g * (x / w);
        if (ns) v += R.gauss() * ns;
        data[i] = Math.max(0, Math.min(255, Math.round(v)));
      }
    // Objects hidden behind the occluder are not countable: drop those mostly covered.
    const truthVisible = spec.occluder
      ? truth.filter((t) => {
          const x0 = w * 0.55, x1 = w * 0.62;
          const covered = Math.max(0, Math.min(t.x + t.r, x1) - Math.max(t.x - t.r, x0)) / (2 * t.r);
          return covered < 0.5;
        })
      : truth;
    return { spec, image: { data, width: w, height: h }, truth: truthVisible };
  } finally {
    mat.delete();
  }
}
