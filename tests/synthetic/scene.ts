/**
 * Deterministic synthetic "pipe end" scenes with exact ground truth.
 * Rendered with OpenCV drawing primitives so they run in Node without a canvas.
 */
import type { CV } from '../../src/analysis/cv';
import type { GrayImage } from '../../src/analysis/circles';
import { type Mat3, apply3, localScale3, mul3 } from '../../src/analysis/homography';

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

// --- multi-view rendering (Phase 6: multi-shot fusion) -----------------------

export interface MultiViewSpec {
  /** Number of photos, 2–5. View 0 is the reference. */
  views: number;
  /** Max translation as a fraction of the longer side (default 0.025). */
  maxShift?: number;
  /** Max in-plane rotation in degrees (default 2). */
  maxRotDeg?: number;
  /** Max relative scale change (default 0.03). */
  maxScale?: number;
  /** Max perspective coefficient times the longer side (default 0.02). */
  maxPersp?: number;
  /** Draw one occluding bar at a random place in every view (default true). */
  occluder?: boolean;
  /** Independent per-view noise sigma (default: the scene's noiseSigma). */
  noiseSigma?: number;
  seed: number;
}

export interface SceneView {
  image: GrayImage;
  /** World → this view. */
  H: Mat3;
  /** Objects visible in this view, in view coordinates; `id` indexes `worldTruth`. */
  truth: (GtCircle & { id: number })[];
}

export interface MultiViewScene {
  worldTruth: GtCircle[];
  views: SceneView[];
  /** Objects visible in at least one view, in reference (view 0) coordinates — what fusion must count. */
  fusedTruth: (GtCircle & { id: number })[];
}

/**
 * Renders one scene as 2–5 photos from nearly the same viewpoint: the stack is
 * padded with textured surroundings (as real photos have), then every view gets
 * its own small homography (shift, rotation, scale, slight perspective), its own
 * occluding bar and independent noise.
 */
export function makeMultiViewScene(cv: CV, spec: SceneSpec, mv: MultiViewSpec): MultiViewScene {
  const R = rng(mv.seed * 7919 + 17);
  const base = makeScene(cv, { ...spec, noiseSigma: 0, occluder: false });
  const pad = Math.round(0.08 * Math.max(spec.width, spec.height));
  const W = spec.width + 2 * pad, Hh = spec.height + 2 * pad;
  const worldTruth = base.truth.map((t) => ({ ...t, x: t.x + pad, y: t.y + pad }));

  const world = new cv.Mat(Hh, W, cv.CV_8UC1, new cv.Scalar(35));
  const views: SceneView[] = [];
  try {
    // Surroundings: clutter in the padding so the frame has non-repetitive structure.
    for (let i = 0; i < 70; i++) {
      const p1 = new cv.Point(Math.round(R.next() * W), Math.round(R.next() * Hh));
      const p2 = new cv.Point(Math.round(p1.x + (R.next() - 0.5) * 300), Math.round(p1.y + (R.next() - 0.5) * 300));
      cv.line(world, p1, p2, new cv.Scalar(60 + R.next() * 150), 1 + Math.round(R.next() * 5), cv.LINE_AA);
    }
    for (let i = 0; i < 30; i++) {
      const x = Math.round(R.next() * W), y = Math.round(R.next() * Hh);
      cv.rectangle(world, new cv.Point(x, y), new cv.Point(x + 10 + R.next() * 60, y + 10 + R.next() * 60), new cv.Scalar(40 + R.next() * 180), -1);
    }
    const stack = new cv.Mat(spec.height, spec.width, cv.CV_8UC1);
    stack.data.set(base.image.data);
    const inner = world.roi(new cv.Rect(pad, pad, spec.width, spec.height));
    stack.copyTo(inner);
    inner.delete();
    stack.delete();

    const L = Math.max(W, Hh);
    const cx = W / 2, cy = Hh / 2;
    const maxShift = (mv.maxShift ?? 0.025) * L;
    const maxRot = ((mv.maxRotDeg ?? 2) * Math.PI) / 180;
    const maxScale = mv.maxScale ?? 0.03;
    const maxPersp = (mv.maxPersp ?? 0.02) / L;
    const ns = mv.noiseSigma ?? spec.noiseSigma ?? 0;
    const sym = () => 2 * R.next() - 1;

    for (let v = 0; v < mv.views; v++) {
      const a = sym() * maxRot, s = 1 + sym() * maxScale;
      const tx = sym() * maxShift, ty = sym() * maxShift;
      const T1: Mat3 = [1, 0, -cx, 0, 1, -cy, 0, 0, 1];
      const P: Mat3 = [1, 0, 0, 0, 1, 0, sym() * maxPersp, sym() * maxPersp, 1];
      const RS: Mat3 = [s * Math.cos(a), -s * Math.sin(a), 0, s * Math.sin(a), s * Math.cos(a), 0, 0, 0, 1];
      const T2: Mat3 = [1, 0, cx + tx, 0, 1, cy + ty, 0, 0, 1];
      const H = mul3(T2, mul3(RS, mul3(P, T1)));

      const out = new cv.Mat();
      const Hm = cv.matFromArray(3, 3, cv.CV_64F, H);
      cv.warpPerspective(world, out, Hm, new cv.Size(W, Hh), cv.INTER_LINEAR, cv.BORDER_REPLICATE);
      Hm.delete();

      const truth: SceneView['truth'] = [];
      worldTruth.forEach((t, id) => {
        const [x, y] = apply3(H, t.x, t.y);
        const r = t.r * localScale3(H, t.x, t.y);
        if (x - r < 0 || y - r < 0 || x + r > W || y + r > Hh) return;
        truth.push({ x, y, r, cls: t.cls, id });
      });

      let visible = truth;
      if (mv.occluder ?? true) {
        const bw = 0.07 * spec.width;
        const x0 = pad + R.next() * (spec.width - bw), x1 = x0 + bw;
        cv.rectangle(out, new cv.Point(Math.round(x0), 0), new cv.Point(Math.round(x1), Hh), new cv.Scalar(90), -1);
        visible = truth.filter((t) => Math.max(0, Math.min(t.x + t.r, x1) - Math.max(t.x - t.r, x0)) / (2 * t.r) < 0.5);
      }

      const data = new Uint8Array(out.data);
      out.delete();
      if (ns) for (let i = 0; i < data.length; i++) data[i] = Math.max(0, Math.min(255, Math.round(data[i]! + R.gauss() * ns)));
      views.push({ image: { data, width: W, height: Hh }, H, truth: visible });
    }
  } finally {
    world.delete();
  }

  const seen = new Set(views.flatMap((v) => v.truth.map((t) => t.id)));
  const H0 = views[0]!.H;
  const fusedTruth = worldTruth.flatMap((t, id) => {
    if (!seen.has(id)) return [];
    const [x, y] = apply3(H0, t.x, t.y);
    const r = t.r * localScale3(H0, t.x, t.y);
    if (x < 0 || y < 0 || x > W || y > Hh) return [];
    return [{ x, y, r, cls: t.cls, id }];
  });
  return { worldTruth, views, fusedTruth };
}
