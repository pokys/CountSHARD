/**
 * Mode B result arithmetic: density storage, region sums and user corrections. Pure, no DOM.
 *
 * The model returns a density map whose sum is an ESTIMATE of the number of objects. The user
 * can correct it region by region: inside a corrected region the user's count replaces the
 * model's density sum. Regions may not overlap, so every pixel is counted by exactly one source.
 */
import { type AiBox, type AiDensityResult, type AiRegionCorrection, type AiState, type CountArea, inCountArea, isDensityResult, pointInPolygon } from '../model/types';

export const MAX_EXEMPLARS = 3;

// --- storage ---------------------------------------------------------------

export function encodeDensity(data: Float32Array): string {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

const decoded = new Map<string, Float32Array>();

/** Decodes (and memoises the last few) base64 density grids. */
export function decodeDensity(b64: string): Float32Array {
  const hit = decoded.get(b64);
  if (hit) return hit;
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const out = new Float32Array(bytes.buffer);
  decoded.set(b64, out);
  if (decoded.size > 4) decoded.delete(decoded.keys().next().value!);
  return out;
}

// --- geometry --------------------------------------------------------------

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Normalises a rectangle drawn in any direction and clips it to the image. */
export function normalizeRect(x1: number, y1: number, x2: number, y2: number, imgW: number, imgH: number): Rect {
  const a = Math.max(0, Math.min(x1, x2));
  const b = Math.max(0, Math.min(y1, y2));
  const c = Math.min(imgW, Math.max(x1, x2));
  const d = Math.min(imgH, Math.max(y1, y2));
  return { x: a, y: b, w: Math.max(0, c - a), h: Math.max(0, d - b) };
}

export function rectsOverlap(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

/**
 * Density summed over a rectangle in project pixels. Cells partly inside contribute in
 * proportion to the covered area (density assumed uniform within a cell).
 */
export function sumInRect(res: Pick<AiDensityResult, 'cellPx' | 'gridW' | 'gridH'>, data: ArrayLike<number>, r: Rect): number {
  const c = res.cellPx;
  const cx0 = Math.max(0, Math.floor(r.x / c));
  const cy0 = Math.max(0, Math.floor(r.y / c));
  const cx1 = Math.min(res.gridW, Math.ceil((r.x + r.w) / c));
  const cy1 = Math.min(res.gridH, Math.ceil((r.y + r.h) / c));
  let s = 0;
  for (let cy = cy0; cy < cy1; cy++) {
    const fy = Math.max(0, Math.min((cy + 1) * c, r.y + r.h) - Math.max(cy * c, r.y)) / c;
    if (fy <= 0) continue;
    for (let cx = cx0; cx < cx1; cx++) {
      const fx = Math.max(0, Math.min((cx + 1) * c, r.x + r.w) - Math.max(cx * c, r.x)) / c;
      if (fx > 0) s += data[cy * res.gridW + cx]! * fx * fy;
    }
  }
  return s;
}

/**
 * Density summed over a lasso polygon (optionally also limited to `clip`). Each cell is weighted
 * by the share of a 4 × 4 grid of sample points inside the polygon.
 */
export function sumInPolygon(res: Pick<AiDensityResult, 'cellPx' | 'gridW' | 'gridH'>, data: ArrayLike<number>, poly: [number, number][], box: Rect, clip?: Rect): number {
  const r = clip ? { x: Math.max(box.x, clip.x), y: Math.max(box.y, clip.y), w: 0, h: 0 } : { ...box };
  if (clip) {
    r.w = Math.min(box.x + box.w, clip.x + clip.w) - r.x;
    r.h = Math.min(box.y + box.h, clip.y + clip.h) - r.y;
    if (r.w <= 0 || r.h <= 0) return 0;
  }
  const c = res.cellPx;
  const S = 4;
  let s = 0;
  for (let cy = Math.max(0, Math.floor(r.y / c)); cy < Math.min(res.gridH, Math.ceil((r.y + r.h) / c)); cy++)
    for (let cx = Math.max(0, Math.floor(r.x / c)); cx < Math.min(res.gridW, Math.ceil((r.x + r.w) / c)); cx++) {
      let n = 0;
      for (let j = 0; j < S; j++)
        for (let i = 0; i < S; i++) {
          const px = (cx + (i + 0.5) / S) * c;
          const py = (cy + (j + 0.5) / S) * c;
          if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h && pointInPolygon(poly, px, py)) n++;
        }
      if (n) s += (data[cy * res.gridW + cx]! * n) / (S * S);
    }
  return s;
}

// --- corrections -----------------------------------------------------------

export type CorrectionError = 'overlap' | 'too-small' | 'bad-count';

/** Smallest correction region side (project px). */
export const MIN_REGION_PX = 8;

export function validateCorrection(existing: AiRegionCorrection[], c: Omit<AiRegionCorrection, 'id'> & { id?: string }): CorrectionError | null {
  if (c.w < MIN_REGION_PX || c.h < MIN_REGION_PX) return 'too-small';
  if (!Number.isInteger(c.userCount) || c.userCount < 0) return 'bad-count';
  if (existing.some((e) => e.id !== c.id && rectsOverlap(e, c))) return 'overlap';
  return null;
}

export interface RegionSummary {
  id: string;
  /** Model's density sum inside the region. */
  model: number;
  user: number;
}

export interface EstimateSummary {
  /** Model estimate for the whole photo. */
  raw: number;
  /** Raw estimate with every corrected region's model sum replaced by the user's count. */
  corrected: number;
  /** Part of `corrected` that comes from user counts. */
  userCounted: number;
  regions: RegionSummary[];
}

export function summarize(ai: AiState, area: CountArea | null = null): EstimateSummary | null {
  const res = ai.result;
  if (!isDensityResult(res)) return null;
  const data = decodeDensity(res.density);
  // With a counting area only the density inside it counts; corrections whose centre lies
  // outside are ignored, and inside ones replace the model's sum within the area.
  const raw = !area ? res.rawTotal : area.poly ? sumInPolygon(res, data, area.poly, area) : sumInRect(res, data, area);
  const inside = (c: Rect) => inCountArea(area, c.x + c.w / 2, c.y + c.h / 2);
  const modelIn = (c: Rect): number => {
    if (!area) return sumInRect(res, data, c);
    if (area.poly) return sumInPolygon(res, data, area.poly, area, c);
    const x = Math.max(c.x, area.x), y = Math.max(c.y, area.y);
    return sumInRect(res, data, { x, y, w: Math.max(0, Math.min(c.x + c.w, area.x + area.w) - x), h: Math.max(0, Math.min(c.y + c.h, area.y + area.h) - y) });
  };
  const regions = ai.corrections.filter(inside).map((c) => ({ id: c.id, model: modelIn(c), user: c.userCount }));
  const replaced = regions.reduce((a, r) => a + r.model, 0);
  const userCounted = regions.reduce((a, r) => a + r.user, 0);
  return { raw, corrected: Math.max(0, raw - replaced + userCounted), userCounted, regions };
}

/** Display rounding of an estimate (never shown with false precision). */
export function roundEstimate(v: number): number {
  return Math.max(0, Math.round(v));
}

// --- state operations (pure) -----------------------------------------------

export function emptyAiState(): AiState {
  return { exemplars: [], result: null, corrections: [] };
}

export function addExemplar(ai: AiState, box: AiBox): AiState {
  return { ...ai, exemplars: [...ai.exemplars, box].slice(-MAX_EXEMPLARS) };
}

export function removeExemplar(ai: AiState, id: string): AiState {
  return { ...ai, exemplars: ai.exemplars.filter((e) => e.id !== id) };
}

export function addCorrection(ai: AiState, c: AiRegionCorrection): { ai: AiState; error: CorrectionError | null } {
  const error = validateCorrection(ai.corrections, c);
  return error ? { ai, error } : { ai: { ...ai, corrections: [...ai.corrections, c] }, error: null };
}

export function setCorrectionCount(ai: AiState, id: string, userCount: number): AiState {
  if (!Number.isInteger(userCount) || userCount < 0) return ai;
  return { ...ai, corrections: ai.corrections.map((c) => (c.id === id ? { ...c, userCount } : c)) };
}

export function removeCorrection(ai: AiState, id: string): AiState {
  return { ...ai, corrections: ai.corrections.filter((c) => c.id !== id) };
}

/** Exemplars changed since the result was computed → the result is stale. */
export function resultIsStale(ai: AiState): boolean {
  const res = ai.result;
  if (!res) return false;
  const now = ai.exemplars.map((e) => e.id).sort().join(',');
  if (now !== [...res.inputExemplarIds].sort().join(',')) return true;
  return !res.exemplarsUsed.every((u) => ai.exemplars.some((e) => e.id === u.id && e.x === u.x && e.y === u.y && e.w === u.w && e.h === u.h));
}

// --- reliability warnings ----------------------------------------------------

/** Exemplars whose sizes differ more than this (ratio of sqrt(w·h)) confuse the model. */
export const EXEMPLAR_SIZE_RATIO_WARN = 1.8;
/**
 * Below this estimate the model is unreliable: LOCA was trained on images with dozens to
 * hundreds of small, similar objects; a few large objects (e.g. 8 profiles in perspective)
 * make it spread density over sides and background.
 */
export const LOW_COUNT_WARN = 15;

/** Ratio of the largest to the smallest exemplar size (1 when fewer than two exemplars). */
export function exemplarSizeRatio(exemplars: Pick<AiBox, 'w' | 'h'>[]): number {
  if (exemplars.length < 2) return 1;
  const sizes = exemplars.map((e) => Math.sqrt(Math.max(1, e.w) * Math.max(1, e.h)));
  return Math.max(...sizes) / Math.min(...sizes);
}

export function exemplarsTooDifferent(exemplars: Pick<AiBox, 'w' | 'h'>[]): boolean {
  return exemplarSizeRatio(exemplars) > EXEMPLAR_SIZE_RATIO_WARN;
}

/** True when the shown estimate is small enough that manual counting is the better choice. */
export function estimateTooLow(summary: Pick<EstimateSummary, 'corrected'> | null): boolean {
  return !!summary && summary.corrected < LOW_COUNT_WARN;
}
