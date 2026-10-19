/**
 * Circular cross-section detector (pipes, rods, tubes, logs seen end-on).
 *
 * Pipeline (see docs/technical-decisions.md, "Circle detection"):
 *  1. CLAHE contrast normalisation (uneven light across a stack is the norm).
 *  2. Candidate generation: Hough gradient transform run per *radius band*.
 *     Large-radius bands run on a downscaled pyramid level, which keeps the cost
 *     roughly constant per band and lets one pass cover a wide range of sizes
 *     (needed for mixed-diameter bundles; a single [0.6r, 1.45r] range cannot).
 *  3. Independent verification of every candidate by *edge support*: fraction of
 *     the perimeter where the image gradient is strong and radially aligned.
 *  4. Local refinement of (x, y, r) to the strongest radial edge, which matters
 *     for diameter classification far more than for counting.
 *  5. Overlap suppression (prefers the outer wall over the bore of a tube) and
 *     removal of "aggregate" circles that enclose several accepted objects.
 */
import { type CV, type Mat, withMats } from './cv';

export interface GrayImage {
  data: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
}

export interface Circle {
  x: number;
  y: number;
  r: number;
  /** Edge-support coverage in [0, 1]. Heuristic, not a probability. */
  score: number;
}

export interface CircleDetectOptions {
  /** 0–100. Maps to the minimum edge-support score. */
  sensitivity: number;
  /** Radius range in pixels of `img`. Omit for automatic (wide) range. */
  minRadius?: number;
  maxRadius?: number;
  onProgress?: (fraction: number) => void;
  /** Collects intermediate stages (tests/diagnostics only). */
  debug?: { raw?: Circle[]; verified?: Circle[]; kept?: Circle[]; field?: Field };
}

export interface CircleDetectResult {
  circles: Circle[];
  candidates: number;
  radiusRange: [number, number];
  durationMs: number;
}

/** Sensitivity (0–100) → minimum edge-support coverage. */
export function scoreThreshold(sensitivity: number): number {
  const s = Math.min(100, Math.max(0, sensitivity)) / 100;
  return 0.78 - 0.46 * s; // 0 → 0.78 (strict), 50 → 0.55, 100 → 0.32 (permissive)
}

const BAND_RATIO = 1.5;
/** Hough runs on a pyramid level where the band's smallest radius is about this many pixels. */
const HOUGH_TARGET_RADIUS = 14;

export function autoRadiusRange(width: number, height: number): [number, number] {
  const minSide = Math.min(width, height);
  return [Math.max(5, Math.round(minSide * 0.006)), Math.max(12, Math.round(minSide * 0.22))];
}

export function detectCircles(cv: CV, img: GrayImage, opts: CircleDetectOptions): CircleDetectResult {
  const t0 = now();
  const { width: w, height: h } = img;
  const [autoMin, autoMax] = autoRadiusRange(w, h);
  const rMin = Math.max(3, opts.minRadius ?? autoMin);
  const rMax = Math.max(rMin + 1, opts.maxRadius ?? autoMax);
  const minScore = scoreThreshold(opts.sensitivity);

  return withMats((track) => {
    const src = track(new cv.Mat(h, w, cv.CV_8UC1));
    src.data.set(img.data);

    const { eq, field } = gradientField(cv, src, rMin, track);

    // Median filtering suppresses texture but must not erase the thin dark seams
    // between tightly packed small objects, so the kernel follows the radius band.
    const houghBases = new Map<number, unknown>();
    const houghBase = (ksize: number) => {
      let m = houghBases.get(ksize);
      if (!m) {
        m = track(new cv.Mat());
        cv.medianBlur(eq, m, ksize);
        houghBases.set(ksize, m);
      }
      return m;
    };

    // --- candidate generation per radius band ---
    const bands: [number, number][] = [];
    for (let a = rMin; a < rMax; a *= BAND_RATIO) bands.push([a, Math.min(rMax, a * BAND_RATIO * 1.08)]);
    const s01 = Math.min(100, Math.max(0, opts.sensitivity)) / 100;
    const raw: Circle[] = [];
    bands.forEach(([ra, rb], i) => {
      // Small radii are upsampled (Hough is unreliable below ~10 px), large ones downsampled.
      const f = Math.min(2, HOUGH_TARGET_RADIUS / ra);
      const base = houghBase(ra < 12 ? 3 : 5);
      let level = base;
      if (f !== 1) {
        level = track(new cv.Mat());
        const size = new cv.Size(Math.max(8, Math.round(w * f)), Math.max(8, Math.round(h * f)));
        cv.resize(base, level, size, 0, 0, f < 1 ? cv.INTER_AREA : cv.INTER_LINEAR);
      }
      const lra = ra * f;
      const lrb = rb * f;
      const out = track(new cv.Mat());
      // Accumulator threshold relative to the band's circumference; permissive on purpose,
      // because every candidate is re-verified below.
      const acc = Math.max(8, Math.round(lra * (1.5 - 0.6 * s01)));
      cv.HoughCircles(level, out, cv.HOUGH_GRADIENT, 1, Math.max(3, lra * 1.2), 100, acc, Math.floor(lra), Math.ceil(lrb));
      const d = out.data32F as Float32Array;
      for (let k = 0; k < out.cols; k++) {
        raw.push({ x: d[k * 3]! / f, y: d[k * 3 + 1]! / f, r: d[k * 3 + 2]! / f, score: 0 });
      }
      opts.onProgress?.(((i + 1) / bands.length) * 0.7);
    });

    // --- verification + refinement ---
    const verified: Scored[] = [];
    const preThreshold = Math.max(0.2, minScore - 0.2);
    for (let i = 0; i < raw.length; i++) {
      const c = raw[i]!;
      const first = edgeSupport(field, c.x, c.y, c.r);
      if (first.inside < 0.6 || first.coverage < preThreshold) continue;
      const refined = polishRadius(field, refine(field, c, rMin, rMax));
      const sup = edgeSupport(field, refined.x, refined.y, refined.r);
      if (sup.inside < 0.6 || sup.coverage < minScore) continue;
      verified.push({ ...refined, score: sup.coverage, strength: sup.strength, polarity: sup.polarity });
      // Tubes: the bore and the outer wall are concentric; Hough often reports only one
      // of them. Look for a stronger-supported outer wall so suppression can prefer it.
      if (sup.polarity === 1) {
        const outer = findOuterWall(field, refined, rMax, minScore);
        if (outer) verified.push(outer);
      }
      if (i % 200 === 0) opts.onProgress?.(0.7 + 0.25 * (i / raw.length));
    }

    const kept = suppressOverlaps(verified);
    if (opts.debug) Object.assign(opts.debug, { raw, verified, kept, field: { ...field, gx: field.gx.slice(), gy: field.gy.slice() } });
    const circles = removeAggregates(kept).map(({ x, y, r, score }) => ({ x, y, r, score }));
    opts.onProgress?.(1);
    return { circles, candidates: raw.length, radiusRange: [rMin, rMax] as [number, number], durationMs: now() - t0 };
  });
}

// ---------------------------------------------------------------------------

/** CLAHE-normalised image and the gradient field used for verification/refinement. */
function gradientField(cv: CV, src: Mat, rMin: number, track: <M extends { delete(): void }>(m: M) => M) {
  const eq = track(new cv.Mat());
  const clahe = track(new cv.CLAHE(2.0, new cv.Size(8, 8)));
  clahe.apply(src, eq);
  // Smoothed relative to the smallest size of interest.
  const smooth = track(new cv.Mat());
  const sigma = Math.max(1, Math.min(4, rMin / 6));
  cv.GaussianBlur(eq, smooth, new cv.Size(0, 0), sigma, sigma, cv.BORDER_REPLICATE);
  const gxM = track(new cv.Mat());
  const gyM = track(new cv.Mat());
  cv.Sobel(smooth, gxM, cv.CV_32F, 1, 0, 3);
  cv.Sobel(smooth, gyM, cv.CV_32F, 0, 1, 3);
  return { eq, field: makeField(gxM.data32F as Float32Array, gyM.data32F as Float32Array, src.cols, src.rows) };
}

/**
 * Edge support of given circles in `img`, after a small local refinement (±15 % radius,
 * a few pixels of centre) that absorbs registration error. Used by multi-shot fusion
 * to re-check, in another photo, an object that photo's detector did not report.
 */
export function measureSupport(cv: CV, img: GrayImage, circles: { x: number; y: number; r: number }[]): number[] {
  if (!circles.length) return [];
  const rMin = Math.max(3, Math.min(...circles.map((c) => c.r)));
  return withMats((track) => {
    const src = track(new cv.Mat(img.height, img.width, cv.CV_8UC1));
    src.data.set(img.data);
    const { field } = gradientField(cv, src, rMin, track);
    return circles.map((c) => {
      const first = edgeSupport(field, c.x, c.y, c.r);
      if (first.inside < 0.6) return 0;
      const ref = refine(field, { ...c, score: 0 }, c.r * 0.85, c.r * 1.15);
      if (Math.hypot(ref.x - c.x, ref.y - c.y) > 0.3 * c.r) return first.coverage;
      const sup = edgeSupport(field, ref.x, ref.y, ref.r);
      return Math.max(first.coverage, sup.inside < 0.6 ? 0 : sup.coverage);
    });
  });
}

export interface Field {
  gx: Float32Array;
  gy: Float32Array;
  w: number;
  h: number;
  /** Gradient magnitude above which a perimeter sample counts as an edge. */
  edgeThreshold: number;
}

function makeField(gx: Float32Array, gy: Float32Array, w: number, h: number): Field {
  // Adaptive edge threshold: a high percentile of the gradient magnitude (sampled).
  const step = Math.max(1, Math.floor((w * h) / 200000));
  const mags: number[] = [];
  for (let i = 0; i < gx.length; i += step) mags.push(Math.hypot(gx[i]!, gy[i]!));
  mags.sort((a, b) => a - b);
  const p70 = mags[Math.floor(mags.length * 0.7)] ?? 0;
  const p99 = mags[Math.floor(mags.length * 0.99)] ?? 1;
  return { gx, gy, w, h, edgeThreshold: Math.max(p70, p99 * 0.12, 8) };
}

interface Support {
  pos: number;
  neg: number;
  /** +1: brighter outside (e.g. a bore); −1: brighter inside (a lit end face). */
  polarity: 1 | -1;
  /** Fraction of in-image perimeter samples with a strong, radial gradient of the dominant polarity. */
  coverage: number;
  /** Fraction of perimeter samples inside the image. */
  inside: number;
  /** Mean absolute radial gradient (refinement objective). */
  strength: number;
}

function sampleCount(r: number): number {
  return Math.max(24, Math.min(144, Math.round((2 * Math.PI * r) / 2.5)));
}

export function edgeSupport(f: Field, x: number, y: number, r: number): Support {
  const n = sampleCount(r);
  let inside = 0;
  let pos = 0;
  let neg = 0;
  let radialSum = 0;
  const rIn = r - Math.max(1.5, r * 0.06);
  for (let k = 0; k < n; k++) {
    const t = (2 * Math.PI * k) / n;
    const c = Math.cos(t);
    const s = Math.sin(t);
    const px = Math.round(x + r * c);
    const py = Math.round(y + r * s);
    if (px < 1 || py < 1 || px >= f.w - 1 || py >= f.h - 1) continue;
    inside++;
    const idx = py * f.w + px;
    const gx = f.gx[idx]!;
    const gy = f.gy[idx]!;
    const m = Math.hypot(gx, gy);
    const signed = gx * c + gy * s;
    const radial = Math.abs(signed);
    radialSum += radial;
    if (m > f.edgeThreshold && radial > 0.8 * m) {
      // Polarity is read just inside the rim: at the seam centre the sign follows
      // whichever neighbour is brighter, while the face→seam transition is consistent.
      const ix = Math.round(x + rIn * c);
      const iy = Math.round(y + rIn * s);
      const ii = iy * f.w + ix;
      const inner = ii >= 0 && ii < f.gx.length ? f.gx[ii]! * c + f.gy[ii]! * s : 0;
      const pol = Math.abs(inner) > 0.5 * f.edgeThreshold ? inner : signed;
      if (pol > 0) pos++;
      else neg++;
    }
  }
  // A real end face has a consistent edge polarity all round (bright face / dark seam,
  // or the reverse). Mixed polarity is typical of circles fitted through clutter.
  // Neighbouring faces of different brightness legitimately flip part of the rim,
  // so only near-balanced polarity (≥ 40 % minority) is treated as clutter.
  const supported = pos + neg;
  const balance = supported ? Math.min(pos, neg) / supported : 0;
  return {
    pos, neg,
    polarity: pos > neg ? 1 : -1,
    coverage: inside && balance < 0.4 ? supported / inside : 0,
    inside: inside / n,
    strength: inside ? radialSum / inside : 0,
  };
}

/**
 * For a bore (dark inside, brighter outside) searches for the concentric outer wall:
 * 1.15–1.7× larger, with good support and the opposite polarity (bright face, dark seam).
 */
function findOuterWall(f: Field, c: Circle, rMax: number, minScore: number): Scored | null {
  let best: Scored | null = null;
  let bestObj = -Infinity;
  for (let k = 1.15; k <= 1.7 && c.r * k <= rMax * 1.1; k += 0.03) {
    const s = edgeSupport(f, c.x, c.y, c.r * k);
    if (s.inside < 0.6 || s.polarity !== -1 || s.coverage < minScore * 0.85) continue;
    const obj = objective(f, s);
    if (obj > bestObj) {
      bestObj = obj;
      best = { x: c.x, y: c.y, r: c.r * k, score: s.coverage, strength: s.strength, polarity: -1 };
    }
  }
  if (!best) return null;
  const refined = polishRadius(f, refine(f, best, best.r * 0.9, best.r * 1.1));
  const s = edgeSupport(f, refined.x, refined.y, refined.r);
  return s.coverage >= minScore * 0.85 && s.polarity === -1 ? { ...refined, score: s.coverage, strength: s.strength, polarity: -1 } : null;
}

/**
 * Final radius polish: with the centre fixed, move the radius (±7 %) to the strongest
 * radial edge as long as coverage does not drop noticeably. This sets the measured
 * diameter, which drives size classification.
 */
function polishRadius(f: Field, c: Circle): Circle {
  const base = edgeSupport(f, c.x, c.y, c.r);
  let best = c.r;
  let bestStrength = base.strength;
  const step = Math.max(0.25, c.r * 0.01);
  for (let r = c.r * 0.93; r <= c.r * 1.07; r += step) {
    const s = edgeSupport(f, c.x, c.y, r);
    if (s.coverage >= base.coverage * 0.9 && s.strength > bestStrength) {
      bestStrength = s.strength;
      best = r;
    }
  }
  return { ...c, r: best };
}

/** Coverage first; gradient strength only breaks ties (pure strength drifts onto neighbours' edges). */
function objective(f: Field, s: Support): number {
  return s.coverage + 0.15 * Math.min(1, s.strength / (4 * f.edgeThreshold));
}

/** Coordinate-descent search for the strongest radial edge near the candidate. */
function refine(f: Field, c: Circle, rMin: number, rMax: number): Circle {
  let { x, y, r } = c;
  let best = objective(f, edgeSupport(f, x, y, r));
  let posStep = Math.max(1, r * 0.08);
  let radStep = Math.max(0.5, r * 0.04);
  const r0 = r;
  for (let iter = 0; iter < 12 && (posStep >= 0.5 || radStep >= 0.25); iter++) {
    let improved = false;
    const tries: [number, number, number][] = [
      [x + posStep, y, r], [x - posStep, y, r], [x, y + posStep, r], [x, y - posStep, r],
      [x, y, r + radStep], [x, y, r - radStep],
    ];
    for (const [tx, ty, tr] of tries) {
      // Stay near the Hough estimate so we do not jump to a neighbouring object or the bore.
      if (tr < Math.max(rMin * 0.9, r0 * 0.88) || tr > Math.min(rMax * 1.1, r0 * 1.12)) continue;
      if (Math.hypot(tx - c.x, ty - c.y) > r0 * 0.25) continue;
      const s = objective(f, edgeSupport(f, tx, ty, tr));
      if (s > best) {
        best = s;
        x = tx;
        y = ty;
        r = tr;
        improved = true;
      }
    }
    if (!improved) {
      posStep /= 2;
      radStep /= 2;
    }
  }
  return { x, y, r, score: 0 };
}

type Scored = Circle & { strength: number; polarity: 1 | -1 };

/**
 * Greedy overlap suppression ordered by score. Two circles conflict when their
 * centres are closer than 0.7·(r1 + r2) (touching neighbours are at 1.0·(r1 + r2)).
 * For (near-)concentric conflicts, a clearly larger circle with comparable support
 * replaces the smaller one: for tubes the outer wall, not the bore, is the object.
 */
export function suppressOverlaps(cands: Scored[]): Scored[] {
  const sorted = [...cands].sort((a, b) => b.score - a.score || b.strength - a.strength);
  const kept: Scored[] = [];
  const grid = new SpatialGrid<Scored>(Math.max(8, median(sorted.map((c) => c.r)) * 2));
  for (const c of sorted) {
    let conflict: Scored | null = null;
    for (const k of grid.near(c.x, c.y, (c.r + maxR(kept)) )) {
      if (Math.hypot(c.x - k.x, c.y - k.y) < 0.7 * (c.r + k.r)) {
        conflict = k;
        break;
      }
    }
    if (!conflict) {
      kept.push(c);
      grid.add(c);
      continue;
    }
    const d = Math.hypot(c.x - conflict.x, c.y - conflict.y);
    const concentric = d < 0.25 * Math.min(c.r, conflict.r);
    // Bore → wall: the wall touches its neighbours, so its coverage is naturally lower.
    const boreToWall = conflict.polarity === 1 && c.polarity === -1 && c.score >= conflict.score * 0.45;
    if (concentric && c.r > conflict.r * 1.12 && (boreToWall || c.score >= conflict.score * 0.95)) {
      // Replace bore by outer wall only if the outer circle conflicts with nothing else.
      const others = grid.near(c.x, c.y, c.r + maxR(kept)).filter(
        (k) => k !== conflict && Math.hypot(c.x - k.x, c.y - k.y) < 0.7 * (c.r + k.r),
      );
      if (others.length === 0) {
        kept.splice(kept.indexOf(conflict), 1);
        grid.remove(conflict);
        kept.push(c);
        grid.add(c);
      }
    }
  }
  return kept;
}

let maxRCache = { arr: null as Scored[] | null, len: -1, value: 0 };
function maxR(arr: Scored[]): number {
  if (maxRCache.arr !== arr || maxRCache.len !== arr.length) {
    let m = 0;
    for (const c of arr) m = Math.max(m, c.r);
    maxRCache = { arr, len: arr.length, value: m };
  }
  return maxRCache.value;
}

/** Removes circles that enclose the centres of ≥ 2 clearly smaller accepted circles (bundle outlines, etc.). */
export function removeAggregates<T extends Circle>(circles: T[]): T[] {
  return circles.filter((big) => {
    let inside = 0;
    for (const s of circles) {
      if (s === big || s.r > big.r * 0.6) continue;
      if (Math.hypot(s.x - big.x, s.y - big.y) < big.r) inside++;
      if (inside >= 2) return false;
    }
    return true;
  });
}

class SpatialGrid<T extends { x: number; y: number }> {
  private cells = new Map<string, T[]>();
  constructor(private size: number) {}
  private key(cx: number, cy: number) {
    return `${cx},${cy}`;
  }
  add(item: T) {
    const k = this.key(Math.floor(item.x / this.size), Math.floor(item.y / this.size));
    const cell = this.cells.get(k);
    if (cell) cell.push(item);
    else this.cells.set(k, [item]);
  }
  remove(item: T) {
    const cell = this.cells.get(this.key(Math.floor(item.x / this.size), Math.floor(item.y / this.size)));
    if (cell) cell.splice(cell.indexOf(item), 1);
  }
  near(x: number, y: number, radius: number): T[] {
    const out: T[] = [];
    const r = Math.ceil(radius / this.size);
    const cx = Math.floor(x / this.size);
    const cy = Math.floor(y / this.size);
    for (let i = cx - r; i <= cx + r; i++)
      for (let j = cy - r; j <= cy + r; j++) {
        const cell = this.cells.get(this.key(i, j));
        if (cell) out.push(...cell);
      }
    return out;
  }
}

function median(v: number[]): number {
  if (!v.length) return 0;
  const s = [...v].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
