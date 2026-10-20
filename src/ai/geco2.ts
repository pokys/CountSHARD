/**
 * GeCo2 (detection-based few-shot counter) pre- and post-processing. Pure, no DOM, so the same
 * code runs in the AI worker and in Node tests.
 *
 * Model: three ONNX graphs (scripts/geco2/): backbone (image → features), exemplars (features +
 * boxes → prototypes), head (features + prototypes → objectness and box offsets on an S/2 grid).
 * Preprocessing and counting follow GeCo2's own evaluation: the longer side is scaled to S and
 * further down so the mean exemplar side is ≤ 80·S/1024 px (small photos are scaled UP to S), padded to
 * S × S with the ImageNet mean; detections are local maxima of the objectness above max/8,
 * boxes from the offsets, greedy NMS at IoU 0.5, centres in the padding dropped.
 */
import { type AiBox, type AiDetection, type CountArea, type CountedObject, inCountArea } from '../model/types';
import { MEAN, STD, type Rgba } from './tensor';

/** Input sizes the app ships: 1024 (GeCo2's own) for WebGPU, 768 for CPU/WASM (phones). */
export type Geco2Size = 768 | 1024;

/** Mean exemplar side GeCo2 was evaluated with, at S = 1024. */
const EXEMPLAR_CAP_1024 = 80;
/** Below this exemplar side (model pixels) the model misses objects; the result says so. */
export const GECO2_MIN_EXEMPLAR_PX = 10;

/** Model-input pixels per photo pixel. */
export function geco2Scale(imgW: number, imgH: number, exemplars: Pick<AiBox, 'w' | 'h'>[], size: Geco2Size): number {
  const s = size / Math.max(imgW, imgH);
  if (!exemplars.length) return s;
  const mw = exemplars.reduce((a, e) => a + e.w, 0) / exemplars.length;
  const mh = exemplars.reduce((a, e) => a + e.h, 0) / exemplars.length;
  const side = ((mw + mh) / 2) * s;
  return side > 0 ? Math.min(1, (EXEMPLAR_CAP_1024 * size) / 1024 / side) * s : s;
}

/** Scaled content size (≤ size × size). */
export function geco2ContentSize(imgW: number, imgH: number, scale: number, size: Geco2Size): { w: number; h: number } {
  return { w: Math.max(1, Math.min(size, Math.floor(imgW * scale))), h: Math.max(1, Math.min(size, Math.floor(imgH * scale))) };
}

/** Mean exemplar side in model pixels (for the "objects too small" warning). */
export function geco2ExemplarSide(exemplars: Pick<AiBox, 'w' | 'h'>[], scale: number): number {
  if (!exemplars.length) return 0;
  return (exemplars.reduce((a, e) => a + (e.w + e.h) / 2, 0) / exemplars.length) * scale;
}

/** RGBA content (top-left) → normalised NCHW [1,3,S,S]; the padding stays 0 (= ImageNet mean). */
export function geco2Input(img: Rgba, size: Geco2Size, out = new Float32Array(3 * size * size)): Float32Array {
  out.fill(0);
  const plane = size * size;
  const k = [1 / (255 * STD[0]), 1 / (255 * STD[1]), 1 / (255 * STD[2])];
  const m = [MEAN[0] / STD[0], MEAN[1] / STD[1], MEAN[2] / STD[2]];
  const h = Math.min(img.height, size);
  const w = Math.min(img.width, size);
  for (let y = 0; y < h; y++) {
    let src = y * img.width * 4;
    let dst = y * size;
    for (let x = 0; x < w; x++, src += 4, dst++) {
      out[dst] = img.data[src]! * k[0]! - m[0]!;
      out[plane + dst] = img.data[src + 1]! * k[1]! - m[1]!;
      out[2 * plane + dst] = img.data[src + 2]! * k[2]! - m[2]!;
    }
  }
  return out;
}

/** Exemplar boxes in model pixels, as the exemplars graph expects ([1, K, 4]: x1, y1, x2, y2). */
export function geco2Boxes(exemplars: Pick<AiBox, 'x' | 'y' | 'w' | 'h'>[], scale: number): Float32Array {
  const out = new Float32Array(exemplars.length * 4);
  exemplars.forEach((e, i) => out.set([e.x * scale, e.y * scale, (e.x + e.w) * scale, (e.y + e.h) * scale], i * 4));
  return out;
}

export interface Geco2Detections {
  /** x1, y1, x2, y2 per detection, in model pixels. */
  boxes: Float32Array;
  scores: Float32Array;
}

/**
 * Objectness [H×W] + box offsets [4×H×W] (left, top, right, bottom as fractions of the input)
 * → detections in model pixels. `content` is the scaled photo size inside the S × S input.
 */
export function geco2Detect(
  obj: ArrayLike<number>,
  ltrb: ArrayLike<number>,
  gridW: number,
  gridH: number,
  size: number,
  content: { w: number; h: number },
  opts: { scoreRatio?: number; iou?: number } = {},
): Geco2Detections {
  const ratio = opts.scoreRatio ?? 0.11;
  const iouMax = opts.iou ?? 0.5;
  const n = gridW * gridH;
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (obj[i]! > max) max = obj[i]!;
  const cand: { v: number; b: [number, number, number, number] }[] = [];
  if (!(max > 0)) return { boxes: new Float32Array(0), scores: new Float32Array(0) };
  const thr = Math.max(max / 8, max * ratio);
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const i = y * gridW + x;
      const v = obj[i]!;
      if (!(v > thr)) continue;
      let peak = true;
      for (let dy = -1; dy <= 1 && peak; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= gridH) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= gridW) continue;
          if (obj[yy * gridW + xx]! > v) {
            peak = false;
            break;
          }
        }
      }
      if (!peak) continue;
      const cx = x / gridW;
      const cy = y / gridH;
      cand.push({ v, b: [cx - ltrb[i]!, cy - ltrb[n + i]!, cx + ltrb[2 * n + i]!, cy + ltrb[3 * n + i]!] });
    }
  }
  cand.sort((a, b) => b.v - a.v);
  const keep = nms(cand, iouMax);
  const out: { v: number; b: number[] }[] = [];
  for (const c of keep) {
    const b = c.b.map((v) => Math.min(1, Math.max(0, v)) * size);
    if ((b[0]! + b[2]!) / 2 < content.w && (b[1]! + b[3]!) / 2 < content.h) out.push({ v: c.v, b });
  }
  const boxes = new Float32Array(out.length * 4);
  const scores = new Float32Array(out.length);
  out.forEach((o, i) => {
    boxes.set(o.b, i * 4);
    scores[i] = o.v;
  });
  return { boxes, scores };
}

/** Greedy NMS over candidates sorted by descending score. */
function nms<T extends { b: [number, number, number, number] }>(sorted: T[], iouMax: number): T[] {
  const keep: T[] = [];
  const area = (b: number[]) => Math.max(0, b[2]! - b[0]!) * Math.max(0, b[3]! - b[1]!);
  for (const c of sorted) {
    let ok = true;
    for (const k of keep) {
      const iw = Math.min(c.b[2], k.b[2]) - Math.max(c.b[0], k.b[0]);
      const ih = Math.min(c.b[3], k.b[3]) - Math.max(c.b[1], k.b[1]);
      if (iw <= 0 || ih <= 0) continue;
      const inter = iw * ih;
      if (inter / (area(c.b) + area(k.b) - inter) > iouMax) {
        ok = false;
        break;
      }
    }
    if (ok) keep.push(c);
  }
  return keep;
}

// --- results in the project ------------------------------------------------------------

let seq = 0;
function detId(): string {
  seq = (seq + 1) % 1e6;
  return `d_${Date.now().toString(36)}${Math.floor(Math.random() * 1e6).toString(36)}${seq.toString(36)}`;
}

/** Model-pixel detections → project-pixel automatic detections. */
export function toProjectDetections(d: Geco2Detections, scale: number): AiDetection[] {
  const out: AiDetection[] = [];
  for (let i = 0; i < d.scores.length; i++) {
    const [x1, y1, x2, y2] = [d.boxes[4 * i]! / scale, d.boxes[4 * i + 1]! / scale, d.boxes[4 * i + 2]! / scale, d.boxes[4 * i + 3]! / scale];
    if (x2 - x1 < 1 || y2 - y1 < 1) continue;
    out.push({ id: detId(), x: x1, y: y1, w: x2 - x1, h: y2 - y1, score: d.scores[i]!, source: 'auto', review: 'unreviewed' });
  }
  return out;
}

/**
 * A new run never discards user work (same rule as circle mode): manual and confirmed detections
 * stay, rejected ones stay rejected (a new detection at the same place is skipped), previous
 * unreviewed automatic detections are replaced.
 */
export function mergeAiDetections(old: AiDetection[], fresh: AiDetection[]): AiDetection[] {
  const kept = old.filter((d) => d.source === 'manual' || d.review !== 'unreviewed');
  const close = (a: AiDetection, b: AiDetection) =>
    Math.hypot(a.x + a.w / 2 - (b.x + b.w / 2), a.y + a.h / 2 - (b.y + b.h / 2)) < 0.3 * (Math.max(a.w, b.w) + Math.max(a.h, b.h));
  return [...kept, ...fresh.filter((f) => !kept.some((k) => close(k, f)))];
}

export function newManualDetection(x: number, y: number, w: number, h: number): AiDetection {
  return { id: detId(), x: x - w / 2, y: y - h / 2, w, h, source: 'manual', review: 'confirmed' };
}

/**
 * Detections as circle-mode objects (centre, r = mean half side) so numbering, hit testing and
 * drawing are shared with Mode A. The box itself is drawn from `boxesOf`.
 */
export function asCountedObjects(list: AiDetection[]): CountedObject[] {
  return list.map((d) => ({
    id: d.id,
    x: d.x + d.w / 2,
    y: d.y + d.h / 2,
    r: (d.w + d.h) / 4,
    source: d.source,
    review: d.review,
    score: d.score,
    groupId: null,
    groupLocked: false,
  }));
}

export function boxesOf(list: AiDetection[]): Map<string, { w: number; h: number }> {
  return new Map(list.map((d) => [d.id, { w: d.w, h: d.h }]));
}

/** Counted detections: not rejected, centre inside the counting area (if any). */
export function countedDetections(list: AiDetection[], area: CountArea | null = null): AiDetection[] {
  return list.filter((d) => d.review !== 'rejected' && inCountArea(area, d.x + d.w / 2, d.y + d.h / 2));
}

/** Median box size of the counted detections nearest to (x, y) (size of a tap-added object). */
export function typicalBox(list: AiDetection[], x: number, y: number, fallback: number): { w: number; h: number } {
  const c = list.filter((d) => d.review !== 'rejected');
  if (!c.length) return { w: fallback, h: fallback };
  const near = [...c].sort((a, b) => Math.hypot(a.x + a.w / 2 - x, a.y + a.h / 2 - y) - Math.hypot(b.x + b.w / 2 - x, b.y + b.h / 2 - y)).slice(0, 6);
  const med = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)]!;
  return { w: med(near.map((d) => d.w)), h: med(near.map((d) => d.h)) };
}

/** Manual detections are deleted; automatic ones are marked rejected (a re-run keeps them out). */
export function removeAiDetection(list: AiDetection[], id: string): AiDetection[] {
  return list.flatMap((d) => (d.id !== id ? [d] : d.source === 'manual' ? [] : [{ ...d, review: 'rejected' as const }]));
}
