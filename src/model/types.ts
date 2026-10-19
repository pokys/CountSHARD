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
}

export interface Project {
  format: 'countit-project';
  version: 1;
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

/** Objects that count towards the total. */
export function isCounted(o: CountedObject): boolean {
  return o.review !== 'rejected';
}
