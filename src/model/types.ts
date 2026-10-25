/**
 * Core data model. All coordinates are in *project image* pixels
 * (the normalised image stored with the project), never in screen pixels.
 */

export type DetectionSource = 'auto' | 'manual';

/**
 * - `unreviewed`: produced by the detector, user has not touched it.
 * - `confirmed`: user explicitly confirmed, moved or resized it; re-analysis never replaces it.
 * - `rejected`: user removed an automatic detection. Kept (hidden) so re-analysis
 *   does not resurrect the same false positive.
 */
export type ReviewState = 'unreviewed' | 'confirmed' | 'rejected';

export interface CountedObject {
  id: string;
  x: number;
  y: number;
  /** Radius in project-image pixels. */
  r: number;
  source: DetectionSource;
  review: ReviewState;
  /**
   * Edge-support score in [0, 1] from the detector: the fraction of the circle's
   * perimeter with a strong, radially aligned image gradient. This is a heuristic
   * quality measure, NOT a calibrated probability.
   */
  score?: number;
  /** Size group id. `null` = not yet classified. */
  groupId: string | null;
  /** True when the user assigned the group manually; automatic clustering keeps it. */
  groupLocked: boolean;
  /** Present when the object comes from multi-shot fusion: in how many photos it was seen. */
  multiShot?: MultiShotObservation;
}

/** Cross-photo agreement of one fused object (see src/analysis/fusion.ts). */
export interface MultiShotObservation {
  seenIn: number;
  inFrame: number;
  /** `uncertain` objects are review candidates and do not count until confirmed. */
  agreement: 'consistent' | 'majority' | 'minority' | 'uncertain';
}

export interface SizeGroup {
  id: string;
  /** Display label, e.g. "A". */
  label: string;
  color: string;
  /** Optional nominal diameter (mm) the user attached to this group. */
  nominalMm?: number;
}

export interface DetectionParams {
  /** 0–100; higher finds more (and more doubtful) circles. */
  sensitivity: number;
  /** `auto` estimates the size range; `manual` uses min/max diameters below. */
  sizeMode: 'auto' | 'manual';
  /** Diameters in project-image pixels, used when sizeMode = manual. */
  minDiameterPx: number;
  maxDiameterPx: number;
  /** Longest side of the image used for analysis (detections are mapped back). */
  analysisMaxSide: number;
}

export interface GroupingParams {
  /** `auto` lets the clustering decide the number of groups; a number forces k groups. */
  mode: 'auto' | 'fixed';
  k: number;
  /** Minimum relative diameter difference between two groups in auto mode (e.g. 0.08 = 8 %). */
  minSeparation: number;
}

export interface Calibration {
  /** Millimetres per project-image pixel. */
  mmPerPx: number;
  referenceObjectId: string | null;
  referenceDiameterMm: number;
}

export interface ImageInfo {
  name: string;
  width: number;
  height: number;
  /** Original file dimensions before normalisation (downscaling). */
  originalWidth: number;
  originalHeight: number;
  mime: string;
}

export interface AnalysisInfo {
  algorithm: string;
  engine: string;
  durationMs: number;
  analysisWidth: number;
  analysisHeight: number;
  estimatedDiameterPx: [number, number] | null;
  at: string;
  /** Set when the result comes from fusing several photos. */
  multiShot?: MultiShotInfo;
}

export interface MultiShotPhotoInfo {
  name: string;
  status: 'reference' | 'used' | 'refused-registration' | 'refused-inconsistent';
  reason?: string;
  detections: number;
  inliers: number;
  residualPx: number;
  agreement: number | null;
  added: number;
}

/** Summary of the last multi-shot fusion (also kept when fusion was refused). */
export interface MultiShotInfo {
  status: 'fused' | 'refused';
  photos: MultiShotPhotoInfo[];
  referenceCount: number;
  countedTotal: number;
  addedCount: number;
  flaggedCount: number;
  uncertainCount: number;
  droppedFromReference: number;
}

/**
 * Counting mode. `circles` = Mode A (round-end detector, individually editable objects).
 * `ai` = Mode B (generic objects from exemplars). Its result is either a LOCA density map with a
 * total ESTIMATE (no individual objects), or GeCo2 detections (individual, editable boxes).
 * Circle objects and AI results never mix.
 */
export type CountMode = 'circles' | 'ai';

/** Axis-aligned box in project-image pixels. */
export interface AiBox {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

/** The user's count for a region; replaces the model's estimate inside it. */
export interface AiRegionCorrection extends AiBox {
  userCount: number;
}

export interface AiDensityResult {
  kind: 'density-estimate';
  model: string;
  engine: string;
  /** Project-image pixels per density cell (the grid starts at the image origin). */
  cellPx: number;
  gridW: number;
  gridH: number;
  /** Density per cell, Float32 little-endian, base64. Sums to `rawTotal`. */
  density: string;
  /** Model's estimate for the whole photo (sum of the density), before corrections. */
  rawTotal: number;
  tiles: number;
  /** Model-input pixels per project pixel. */
  scale: number;
  /** True when the tile limit forced a smaller scale than the exemplars asked for. */
  limitedByTiles: boolean;
  /** Exemplars the model actually used (those fitting into one 512 px window). */
  exemplarsUsed: AiBox[];
  /** Ids of all exemplars drawn when the result was computed (to detect a stale result). */
  inputExemplarIds: string[];
  durationMs: number;
  at: string;
}

/**
 * One object found by a detection model (GeCo2), project pixels. Edited like circle-mode
 * objects: removing an automatic one marks it rejected, a tap adds a manual one.
 */
export interface AiDetection extends AiBox {
  /** Objectness score of the model (not a probability). Absent for manual ones. */
  score?: number;
  /** Class index of a multi-class model (steel profiles: see AiDetectionResult.classes). */
  cls?: number;
  source: DetectionSource;
  review: ReviewState;
}

/** Result of a detection model: individual objects, counted (not an estimate). */
export interface AiDetectionResult {
  kind: 'detections';
  /** Model id, e.g. 'geco2'. */
  modelId: string;
  /** Model description recorded with the result (name, export, input size). */
  model: string;
  engine: string;
  /** Model input size (S × S). */
  inputSize: number;
  /** Model-input pixels per project pixel. */
  scale: number;
  objects: AiDetection[];
  /** Class names of a multi-class model, indexed by AiDetection.cls. */
  classes?: string[];
  exemplarsUsed: AiBox[];
  inputExemplarIds: string[];
  durationMs: number;
  at: string;
}

export type AiResult = AiDensityResult | AiDetectionResult;

export function isDetectionResult(r: AiResult | null | undefined): r is AiDetectionResult {
  return !!r && r.kind === 'detections';
}
export function isDensityResult(r: AiResult | null | undefined): r is AiDensityResult {
  return !!r && r.kind !== 'detections';
}

export interface AiState {
  /** 1–3 exemplar boxes drawn by the user. */
  exemplars: AiBox[];
  result: AiResult | null;
  /** Non-overlapping regions with a user count. */
  corrections: AiRegionCorrection[];
}

export interface Project {
  format: 'countshard-project';
  version: 1;
  /** Missing in projects saved before Mode B existed (= `circles`). */
  mode?: CountMode;
  /** Mode B state; missing/null until the user starts AI counting. */
  ai?: AiState | null;
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  image: ImageInfo;
  params: DetectionParams;
  grouping: GroupingParams;
  objects: CountedObject[];
  groups: SizeGroup[];
  calibration: Calibration | null;
  /**
   * Optional counting area (project-image pixels). Objects whose centre lies outside, and AI
   * density outside, are ignored. Nothing is deleted: enlarging the area brings them back.
   */
  countArea?: CountArea | null;
  analysis: AnalysisInfo | null;
  appVersion: string;
}

export const DEFAULT_DETECTION_PARAMS: DetectionParams = {
  sensitivity: 50,
  sizeMode: 'auto',
  minDiameterPx: 20,
  maxDiameterPx: 120,
  analysisMaxSide: 1600,
};

export const DEFAULT_GROUPING: GroupingParams = { mode: 'auto', k: 2, minSeparation: 0.1 };

/** A fused object the photos disagree about; shown for review, not counted until confirmed. */
export function isCandidate(o: CountedObject): boolean {
  return o.review === 'unreviewed' && o.multiShot?.agreement === 'uncertain';
}

export interface CountArea {
  /** Bounding box (for a rectangle the area itself). */
  x: number;
  y: number;
  w: number;
  h: number;
  /** Freehand outline (lasso), closed implicitly. Absent for a rectangle. */
  poly?: [number, number][];
}

/** Even–odd point-in-polygon test. */
export function pointInPolygon(poly: [number, number][], x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]!;
    const [xj, yj] = poly[j]!;
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** True when the point lies inside the counting area (or there is no area). */
export function inCountArea(area: CountArea | null | undefined, x: number, y: number): boolean {
  if (!area) return true;
  if (x < area.x || x > area.x + area.w || y < area.y || y > area.y + area.h) return false;
  return !area.poly || pointInPolygon(area.poly, x, y);
}

/** Objects that count towards the total. */
export function isCounted(o: CountedObject): boolean {
  return o.review !== 'rejected' && !isCandidate(o);
}

/** Objects the user can select and edit: counted ones and review candidates. */
export function isSelectable(o: CountedObject): boolean {
  return o.review !== 'rejected';
}

/** Counted object seen in at most half of the photos covering it (multi-shot). */
export function isMultiShotFlagged(o: CountedObject): boolean {
  return o.review === 'unreviewed' && o.multiShot?.agreement === 'minority';
}
