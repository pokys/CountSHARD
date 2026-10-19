/**
 * Mode B (generic AI counting) pre-processing: how a photo of any size and aspect ratio is
 * fed to LOCA, whose input is a fixed 512 × 512 square.
 *
 * LOCA's own evaluation squashes every image to 512 × 512. That distorts shapes and makes
 * small objects in big photos a few pixels wide. Instead the photo is scaled uniformly (aspect
 * ratio kept) so that the exemplar boxes get a size the model saw in training, then covered by
 * overlapping 512 × 512 tiles (letterboxed with the ImageNet mean at the borders). Each tile
 * "owns" the part of the image closest to its centre, so every pixel's density is counted
 * exactly once when the tiles are stitched.
 *
 * Pure functions, no DOM; all coordinates are pixels.
 */

export const MODEL_INPUT = 512;
/** Context kept on each side of a tile's owned area (8 feature cells at LOCA's stride 8). */
export const TILE_MARGIN = 32;
/**
 * Exemplar size band (geometric mean of box width and height, in model-input pixels):
 * p25–p75 of the FSC-147 *training* exemplars after LOCA's 512 × 512 resize
 * (measured from annotation_FSC147_384.json: p25 35.8, median 58.7, p75 92.2).
 * Inside the band the photo is just fitted into one tile; outside it is rescaled.
 */
export const EXEMPLAR_BAND: readonly [number, number] = [36, 92];
/** Upper bound for the number of tiles (≈ inference time and memory). */
export const DEFAULT_MAX_TILES = 16;
/** Never upscale a photo by more than this. */
export const MAX_UPSCALE = 4;
/** LOCA always takes exactly three exemplar boxes. */
export const MODEL_EXEMPLARS = 3;

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Tile {
  /** Top-left corner in scaled-image pixels (the 512 window may extend past the image). */
  x: number;
  y: number;
  /** Owned area in scaled-image pixels, [x0, x1) × [y0, y1), always inside the image. */
  own: { x0: number; y0: number; x1: number; y1: number };
}

export interface TilePlan {
  /** Scaled-image pixels per input-image pixel. */
  scale: number;
  /** Scale the exemplar band asked for (differs from `scale` when the tile limit applied). */
  desiredScale: number;
  limitedByTiles: boolean;
  /** Scaled image size. */
  width: number;
  height: number;
  tiles: Tile[];
  /** 512 window (scaled-image pixels) whose features define the exemplar prototypes. */
  exemplarWindow: { x: number; y: number };
  /** Index of the tile identical to the exemplar window, if any (its features are reused). */
  exemplarTileIndex: number;
  /** Exactly MODEL_EXEMPLARS boxes [x1, y1, x2, y2] in exemplar-window pixels (repeated if fewer were usable). */
  exemplarBoxes: [number, number, number, number][];
  /** Indices (into the input exemplars) of the boxes that fit into the exemplar window. */
  exemplarsUsed: number[];
}

export interface PlanOptions {
  maxTiles?: number;
  band?: readonly [number, number];
}

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

/** Characteristic exemplar size: median of sqrt(w·h). */
export function exemplarSize(exemplars: Box[]): number {
  return median(exemplars.map((b) => Math.sqrt(Math.max(1, b.w) * Math.max(1, b.h))));
}

/** Tile start positions along one axis of length `len` (scaled px). */
export function axisStarts(len: number, tile = MODEL_INPUT, margin = TILE_MARGIN): number[] {
  if (len <= tile) return [0];
  const stride = tile - 2 * margin;
  const n = 1 + Math.ceil((len - tile) / stride);
  return Array.from({ length: n }, (_, i) => Math.round((i * (len - tile)) / (n - 1)));
}

/** Owned intervals along one axis: boundaries halfway through each overlap. */
function axisOwn(len: number, starts: number[], tile = MODEL_INPUT): [number, number][] {
  return starts.map((s, i) => {
    const a = i === 0 ? 0 : Math.round((starts[i - 1]! + tile + s) / 2);
    const b = i === starts.length - 1 ? len : Math.round((s + tile + starts[i + 1]!) / 2);
    return [a, b];
  });
}

export function tileCount(width: number, height: number): number {
  return axisStarts(width).length * axisStarts(height).length;
}

export function makeTiles(width: number, height: number): Tile[] {
  const xs = axisStarts(width);
  const ys = axisStarts(height);
  const ox = axisOwn(width, xs);
  const oy = axisOwn(height, ys);
  const tiles: Tile[] = [];
  ys.forEach((y, j) =>
    xs.forEach((x, i) => tiles.push({ x, y, own: { x0: ox[i]![0], x1: ox[i]![1], y0: oy[j]![0], y1: oy[j]![1] } })),
  );
  return tiles;
}

/** Scale chosen from the exemplar size (before the tile limit). */
export function desiredScale(imgW: number, imgH: number, exemplars: Box[], band: readonly [number, number] = EXEMPLAR_BAND): number {
  const fit = MODEL_INPUT / Math.max(imgW, imgH);
  if (!exemplars.length) return fit;
  const g = exemplarSize(exemplars);
  const at = fit * g;
  let s = fit;
  if (at < band[0]) s = band[0] / g;
  else if (at > band[1]) s = band[1] / g;
  return Math.min(s, MAX_UPSCALE);
}

export function planTiles(imgW: number, imgH: number, exemplars: Box[], opts: PlanOptions = {}): TilePlan {
  const maxTiles = opts.maxTiles ?? DEFAULT_MAX_TILES;
  const want = desiredScale(imgW, imgH, exemplars, opts.band);
  let scale = want;
  const dims = (s: number) => [Math.max(1, Math.round(imgW * s)), Math.max(1, Math.round(imgH * s))] as const;
  while (tileCount(...dims(scale)) > maxTiles) scale *= 0.97;
  const [width, height] = dims(scale);
  const tiles = makeTiles(width, height);

  // Exemplar window: the 512 window holding as many (scaled) exemplars as possible.
  const sb = exemplars.map((b) => ({ x1: b.x * scale, y1: b.y * scale, x2: (b.x + b.w) * scale, y2: (b.y + b.h) * scale }));
  const clampStart = (c: number, len: number) => (len <= MODEL_INPUT ? 0 : Math.round(Math.min(len - MODEL_INPUT, Math.max(0, c - MODEL_INPUT / 2))));
  const candidates: { x: number; y: number }[] = [];
  if (sb.length) {
    const ux = (Math.min(...sb.map((b) => b.x1)) + Math.max(...sb.map((b) => b.x2))) / 2;
    const uy = (Math.min(...sb.map((b) => b.y1)) + Math.max(...sb.map((b) => b.y2))) / 2;
    candidates.push({ x: clampStart(ux, width), y: clampStart(uy, height) });
    for (const b of sb) candidates.push({ x: clampStart((b.x1 + b.x2) / 2, width), y: clampStart((b.y1 + b.y2) / 2, height) });
  } else {
    candidates.push({ x: 0, y: 0 });
  }
  const inside = (w: { x: number; y: number }) =>
    sb.map((b, i) => (b.x1 >= w.x - 0.5 && b.y1 >= w.y - 0.5 && b.x2 <= w.x + MODEL_INPUT + 0.5 && b.y2 <= w.y + MODEL_INPUT + 0.5 ? i : -1)).filter((i) => i >= 0);
  let best = candidates[0]!;
  let bestIn = inside(best);
  for (const c of candidates.slice(1)) {
    const n = inside(c);
    if (n.length > bestIn.length) {
      best = c;
      bestIn = n;
    }
  }
  const used = bestIn.length ? bestIn : sb.length ? [0] : [];
  const boxes = used.map((i) => {
    const b = sb[i]!;
    const c = (v: number) => Math.min(MODEL_INPUT, Math.max(0, v));
    return [c(b.x1 - best.x), c(b.y1 - best.y), c(b.x2 - best.x), c(b.y2 - best.y)] as [number, number, number, number];
  });
  const exemplarBoxes = padExemplars(boxes);
  const exemplarTileIndex = tiles.findIndex((t) => t.x === best.x && t.y === best.y);
  return {
    scale,
    desiredScale: want,
    limitedByTiles: scale < want,
    width,
    height,
    tiles,
    exemplarWindow: best,
    exemplarTileIndex,
    exemplarBoxes,
    exemplarsUsed: used,
  };
}

/** LOCA needs exactly three boxes; fewer are repeated cyclically (1 → A A A, 2 → A B A). */
export function padExemplars<T>(boxes: T[]): T[] {
  if (!boxes.length) throw new Error('at least one exemplar is required');
  return Array.from({ length: MODEL_EXEMPLARS }, (_, i) => boxes[i % boxes.length]!);
}
