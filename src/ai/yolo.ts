/**
 * Steel-profile detector (YOLOv8n, 640 px, classes: round pipe, square tube, angle, channel)
 * pre- and post-processing. Pure, no DOM, so the same code runs in the AI worker and in tests.
 *
 * The model needs no example boxes. One photo runs in up to two passes:
 * 1. the whole photo letterboxed into 640 × 640;
 * 2. when the items found are small in that input (or nothing was found), overlapping tiles so
 *    an item covers about TARGET_SIDE model pixels; tile results replace the overview.
 * Raw output [1, 4 + nc, N] (centre x, centre y, w, h; per-class scores) → best class above
 * SCORE_MIN, class-agnostic NMS, nested boxes removed, oversized boxes (several items) dropped.
 */
import { resizeRgba, type Rgba } from './tensor';

export const YOLO_SIZE = 640;
/** Class order of the model (index = class id in the output). */
export const YOLO_CLASSES = ['circle', 'square', 'angle', 'channel'] as const;
export type YoloClass = (typeof YOLO_CLASSES)[number];

/** Ultralytics' default confidence threshold: the default sensitivity. */
export const SCORE_MIN = 0.25;
/** Lowest score kept from the model, so the sensitivity slider can show more without a re-run. */
export const SCORE_FLOOR = 0.1;
/** Sensitivity slider (0–100) ↔ minimum score: 0 → 0.6, 70 → 0.25 (default), 100 → 0.1. */
export function minScoreFromSensitivity(s: number): number {
  const v = Math.min(100, Math.max(0, s));
  return v <= 70 ? 0.6 - (0.35 * v) / 70 : 0.25 - (0.15 * (v - 70)) / 30;
}
export function sensitivityFromMinScore(m: number): number {
  return m >= 0.25 ? Math.round((70 * (0.6 - m)) / 0.35) : Math.round(70 + (30 * (0.25 - m)) / 0.15);
}
const IOU_MAX = 0.6;
const NESTED_MAX = 0.7;
/** Items smaller than this (model pixels) in the overview trigger the tiled pass. */
export const SMALL_SIDE = 28;
/** Item side the tiled pass aims for (model pixels). */
const TARGET_SIDE = 56;
/** Upper limit of tiles per side (≤ 16 runs per photo). */
export const MAX_GRID = 4;
/** Letterbox padding colour (Ultralytics uses 114 grey). */
const PAD = 114;

export interface YoloBox {
  /** x1, y1, x2, y2 in the coordinates of the image given to the worker. */
  b: [number, number, number, number];
  score: number;
  cls: number;
}

export interface Tile {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Scale and offset of a tile inside the letterboxed 640 × 640 input. */
export interface Letterbox {
  scale: number;
  padX: number;
  padY: number;
}

export function letterbox(w: number, h: number, size = YOLO_SIZE): Letterbox {
  const scale = size / Math.max(w, h);
  return { scale, padX: Math.floor((size - Math.round(w * scale)) / 2), padY: Math.floor((size - Math.round(h * scale)) / 2) };
}

function crop(img: Rgba, t: Tile): Rgba {
  if (t.x === 0 && t.y === 0 && t.w === img.width && t.h === img.height) return img;
  const out = new Uint8ClampedArray(t.w * t.h * 4);
  for (let y = 0; y < t.h; y++) out.set(img.data.subarray(((t.y + y) * img.width + t.x) * 4, ((t.y + y) * img.width + t.x + t.w) * 4), y * t.w * 4);
  return { data: out, width: t.w, height: t.h };
}

/** Tile of `img` → letterboxed NCHW float tensor [1, 3, 640, 640], RGB in 0..1. */
export function yoloInput(img: Rgba, t: Tile, size = YOLO_SIZE, out = new Float32Array(3 * size * size)): { tensor: Float32Array; lb: Letterbox } {
  const lb = letterbox(t.w, t.h, size);
  const cw = Math.max(1, Math.min(size, Math.round(t.w * lb.scale)));
  const ch = Math.max(1, Math.min(size, Math.round(t.h * lb.scale)));
  const src = resizeRgba(crop(img, t), cw, ch);
  out.fill(PAD / 255);
  const plane = size * size;
  for (let y = 0; y < ch; y++) {
    let s = y * cw * 4;
    let d = (y + lb.padY) * size + lb.padX;
    for (let x = 0; x < cw; x++, s += 4, d++) {
      out[d] = src.data[s]! / 255;
      out[plane + d] = src.data[s + 1]! / 255;
      out[2 * plane + d] = src.data[s + 2]! / 255;
    }
  }
  return { tensor: out, lb };
}

/**
 * Raw output [4 + nc, n] (channel-major) of one tile → boxes in image coordinates. `edges` says
 * which tile sides lie inside the photo: boxes touching them are cut by the tile and dropped
 * (the overlapping neighbour tile sees the whole item).
 */
export function yoloDecode(
  raw: ArrayLike<number>,
  n: number,
  nc: number,
  t: Tile,
  lb: Letterbox,
  edges: { left: boolean; top: boolean; right: boolean; bottom: boolean } = { left: false, top: false, right: false, bottom: false },
  scoreMin = SCORE_MIN,
): YoloBox[] {
  const out: YoloBox[] = [];
  const touch = 2 / lb.scale;
  for (let i = 0; i < n; i++) {
    let best = -1;
    let score = scoreMin;
    for (let c = 0; c < nc; c++) {
      const v = raw[(4 + c) * n + i]!;
      if (v > score) {
        score = v;
        best = c;
      }
    }
    if (best < 0) continue;
    const cx = raw[i]!, cy = raw[n + i]!, w = raw[2 * n + i]!, h = raw[3 * n + i]!;
    let x1 = (cx - w / 2 - lb.padX) / lb.scale;
    let y1 = (cy - h / 2 - lb.padY) / lb.scale;
    let x2 = (cx + w / 2 - lb.padX) / lb.scale;
    let y2 = (cy + h / 2 - lb.padY) / lb.scale;
    x1 = Math.max(0, x1);
    y1 = Math.max(0, y1);
    x2 = Math.min(t.w, x2);
    y2 = Math.min(t.h, y2);
    if (x2 - x1 < 1 || y2 - y1 < 1) continue;
    if ((edges.left && x1 < touch) || (edges.top && y1 < touch) || (edges.right && x2 > t.w - touch) || (edges.bottom && y2 > t.h - touch)) continue;
    out.push({ b: [x1 + t.x, y1 + t.y, x2 + t.x, y2 + t.y], score, cls: best });
  }
  return out;
}

const area = (b: number[]) => Math.max(0, b[2]! - b[0]!) * Math.max(0, b[3]! - b[1]!);

/** Class-agnostic greedy NMS + removal of boxes mostly inside a stronger one. */
export function yoloSuppress(boxes: YoloBox[], iouMax = IOU_MAX, nestedMax = NESTED_MAX): YoloBox[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep: YoloBox[] = [];
  for (const c of sorted) {
    let ok = true;
    for (const k of keep) {
      const iw = Math.min(c.b[2], k.b[2]) - Math.max(c.b[0], k.b[0]);
      const ih = Math.min(c.b[3], k.b[3]) - Math.max(c.b[1], k.b[1]);
      if (iw <= 0 || ih <= 0) continue;
      const inter = iw * ih;
      if (inter / (area(c.b) + area(k.b) - inter) > iouMax || inter / Math.max(1e-9, Math.min(area(c.b), area(k.b))) > nestedMax) {
        ok = false;
        break;
      }
    }
    if (ok) keep.push(c);
  }
  return keep;
}

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)]! : 0;
}

/**
 * Drops boxes far larger than the typical item of their class (a box around a whole bundle or
 * several pieces). Only for classes with at least 5 boxes.
 */
export function dropOversized(boxes: YoloBox[], maxRatio = 4): YoloBox[] {
  const byCls = new Map<number, number>();
  for (const c of new Set(boxes.map((b) => b.cls))) {
    const a = boxes.filter((b) => b.cls === c).map((b) => area(b.b));
    if (a.length >= 5) byCls.set(c, median(a));
  }
  return boxes.filter((b) => {
    const m = byCls.get(b.cls);
    return m === undefined || area(b.b) <= maxRatio * m;
  });
}

/** Median item side (√(w·h)) in model pixels of a pass whose input scale is `scale`. */
export function medianSide(boxes: YoloBox[], scale: number): number {
  return median(boxes.map((b) => Math.sqrt(area(b.b)))) * scale;
}

/**
 * Tiles for the second pass, or null when the overview is good enough. Tiles are square-ish,
 * overlap by at least 1.5 item sides, and number at most MAX_GRID per side.
 */
export function tilePlan(imgW: number, imgH: number, overview: YoloBox[]): Tile[] | null {
  const s0 = YOLO_SIZE / Math.max(imgW, imgH);
  const side = medianSide(overview, s0);
  let grid: number;
  if (!overview.length) grid = 2;
  else if (side >= SMALL_SIDE) return null;
  else grid = Math.min(MAX_GRID, Math.max(2, Math.ceil(TARGET_SIDE / Math.max(1, side))));
  const itemPx = overview.length ? side / s0 : Math.max(imgW, imgH) / 20;
  const overlap = Math.ceil(Math.max(32, 1.5 * itemPx));
  const nx = imgW >= imgH ? grid : Math.max(1, Math.round((grid * imgW) / imgH));
  const ny = imgH >= imgW ? grid : Math.max(1, Math.round((grid * imgH) / imgW));
  const tiles: Tile[] = [];
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const x0 = Math.max(0, Math.floor((i * imgW) / nx) - overlap);
      const y0 = Math.max(0, Math.floor((j * imgH) / ny) - overlap);
      const x1 = Math.min(imgW, Math.ceil(((i + 1) * imgW) / nx) + overlap);
      const y1 = Math.min(imgH, Math.ceil(((j + 1) * imgH) / ny) + overlap);
      tiles.push({ x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
    }
  return tiles.length > 1 ? tiles : null;
}

/** Which sides of a tile lie inside the photo (boxes touching them are cut by the tile). */
export function innerEdges(t: Tile, imgW: number, imgH: number) {
  return { left: t.x > 0, top: t.y > 0, right: t.x + t.w < imgW, bottom: t.y + t.h < imgH };
}

export interface YoloDetections {
  /** x1, y1, x2, y2 per detection, image pixels. */
  boxes: Float32Array;
  scores: Float32Array;
  classes: Uint8Array;
  /** Number of model runs (1 = overview only). */
  runs: number;
}

/** Final list → transferable arrays. */
export function packDetections(list: YoloBox[], runs: number): YoloDetections {
  const boxes = new Float32Array(list.length * 4);
  const scores = new Float32Array(list.length);
  const classes = new Uint8Array(list.length);
  list.forEach((d, i) => {
    boxes.set(d.b, i * 4);
    scores[i] = d.score;
    classes[i] = d.cls;
  });
  return { boxes, scores, classes, runs };
}

/**
 * Whole detection for one photo, given a function that runs the model on one input tensor and
 * returns the raw output with its dimensions.
 */
export async function yoloDetect(
  img: Rgba,
  run: (input: Float32Array) => Promise<{ data: ArrayLike<number>; nc: number; n: number }>,
  progress: (f: number) => void = () => undefined,
): Promise<YoloDetections> {
  const full: Tile = { x: 0, y: 0, w: img.width, h: img.height };
  const buf = new Float32Array(3 * YOLO_SIZE * YOLO_SIZE);
  const pass = async (t: Tile) => {
    const { tensor, lb } = yoloInput(img, t, YOLO_SIZE, buf);
    const o = await run(tensor);
    return yoloDecode(o.data, o.n, o.nc, t, lb, innerEdges(t, img.width, img.height), SCORE_FLOOR);
  };
  const overview = dropOversized(yoloSuppress(await pass(full)));
  // Tiling is decided on the items shown at the default sensitivity.
  const tiles = tilePlan(img.width, img.height, overview.filter((b) => b.score >= SCORE_MIN));
  if (!tiles) {
    progress(1);
    return packDetections(overview, 1);
  }
  progress(1 / (tiles.length + 1));
  const all: YoloBox[] = [];
  for (let i = 0; i < tiles.length; i++) {
    all.push(...(await pass(tiles[i]!)));
    progress((i + 2) / (tiles.length + 1));
  }
  const tiled = dropOversized(yoloSuppress(all));
  // Tiling found nothing more (e.g. a photo with a few large items): keep the overview.
  const sure = (l: YoloBox[]) => l.filter((b) => b.score >= SCORE_MIN).length;
  return packDetections(sure(tiled) >= sure(overview) ? tiled : overview, tiles.length + 1);
}
