import type { GtCircle } from './scene';

export interface MatchResult {
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
  countError: number;
  /** Mean absolute relative radius error over true positives. */
  radiusMape: number;
  pairs: [number, number][];
}

/** Greedy one-to-one matching by centre distance (< 0.5·r_gt). */
export function matchCircles(truth: GtCircle[], det: { x: number; y: number; r: number }[]): MatchResult {
  const cand: [number, number, number][] = [];
  truth.forEach((t, i) =>
    det.forEach((d, j) => {
      const dist = Math.hypot(t.x - d.x, t.y - d.y);
      if (dist < 0.5 * t.r) cand.push([dist, i, j]);
    }),
  );
  cand.sort((a, b) => a[0] - b[0]);
  const usedT = new Set<number>();
  const usedD = new Set<number>();
  const pairs: [number, number][] = [];
  for (const [, i, j] of cand) {
    if (usedT.has(i) || usedD.has(j)) continue;
    usedT.add(i);
    usedD.add(j);
    pairs.push([i, j]);
  }
  const tp = pairs.length;
  const fp = det.length - tp;
  const fn = truth.length - tp;
  const precision = det.length ? tp / det.length : 1;
  const recall = truth.length ? tp / truth.length : 1;
  const radiusMape = tp ? pairs.reduce((s, [i, j]) => s + Math.abs(det[j]!.r - truth[i]!.r) / truth[i]!.r, 0) / tp : 0;
  return {
    tp, fp, fn, precision, recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    countError: det.length - truth.length,
    radiusMape,
    pairs,
  };
}
