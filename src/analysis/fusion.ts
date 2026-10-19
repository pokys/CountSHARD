/**
 * Multi-shot fusion (Phase 6): detections from 2–5 photos of the same stack, taken
 * from nearly the same viewpoint, are mapped into the reference photo and matched
 * one-to-one, so every physical object counts once.
 *
 * - Every view must have a reliable registration (see registration.ts); unreliable
 *   views are left out and reported, never fused "approximately".
 * - Matching is greedy one-to-one by centre distance with a gate relative to the
 *   radius (0.5·r) and a radius-ratio gate. Real end faces do not overlap, so their
 *   centres are ≥ 2r apart and a correct match is unambiguous; greedy gives the same
 *   assignment as Hungarian here at a fraction of the cost.
 * - A view whose mapped detections mostly disagree with what is already known is
 *   refused as well (a wrong-but-plausible homography would otherwise duplicate the
 *   whole stack).
 * - Each fused object carries "seen in k of n" (n = photos in which its position is
 *   inside the frame); objects with k < n are flagged for review because the photos
 *   disagree about them (occlusion, a missed detection, or a false positive).
 */
import type { Circle } from './circles';
import { type Mat3, apply3, invert3, localScale3 } from './homography';
import type { RegistrationResult } from './registration';

export interface FusionView {
  width: number;
  height: number;
  circles: Circle[];
  /** Registration onto the reference; `null` for the reference itself (view 0). */
  registration: RegistrationResult | null;
}

export interface FusedObject {
  x: number;
  y: number;
  r: number;
  /** Best edge-support score over the observations. */
  score: number;
  /** Number of photos with a detection of this object. */
  seenIn: number;
  /** Number of used photos whose frame contains this object. */
  inFrame: number;
  /**
   * - `consistent`: seen in every photo that covers it,
   * - `majority`: seen in more than half of them (typically hidden in the others),
   * - `minority`: seen in at most half of them, but strongly supported and not
   *   overlapping anything the majority confirms — counted, flagged for review,
   * - `uncertain`: seen in at most half of them and weak, or overlapping an object the
   *   majority confirms — NOT counted unless the user confirms it.
   */
  agreement: 'consistent' | 'majority' | 'minority' | 'uncertain';
  /** Counted in the fused total. */
  counted: boolean;
  /** Should be looked at by the user (minority or uncertain, or a cross-photo conflict). */
  flagged: boolean;
  /** Indices of the views that observed it. */
  views: number[];
}

export type ViewStatus = 'reference' | 'used' | 'refused-registration' | 'refused-inconsistent';

export interface FusionViewReport {
  index: number;
  status: ViewStatus;
  /** Registration failure reason, or 'inconsistent' / 'residual-vs-size'. */
  reason?: string;
  detections: number;
  inliers: number;
  residualPx: number;
  /** Fraction of the view's detections (in the overlap) that matched an already known object. */
  agreement: number | null;
  /** Objects this view added that no earlier view had. */
  added: number;
  /** Observations overlapping a known object without matching it (not counted again). */
  conflicts: number;
}

export interface FusionResult {
  status: 'fused' | 'refused';
  /** Why fusion as a whole was refused. */
  reason?: 'no-usable-view';
  objects: FusedObject[];
  views: FusionViewReport[];
  referenceCount: number;
  /** Counted fused objects. */
  countedTotal: number;
  /** Counted fused objects not detected in the reference photo. */
  addedCount: number;
  /** Counted objects the photos disagree about (minority, or a cross-photo conflict). */
  flaggedCount: number;
  /** Candidates not counted (see FusedObject.agreement). */
  uncertainCount: number;
  /** Reference detections that the other photos do not support (now uncertain). */
  droppedFromReference: number;
  usedViews: number;
}

export interface FusionOptions {
  /** Centre-distance gate as a fraction of the smaller radius. */
  gate?: number;
  /** Min fraction of a view's detections that must agree with known objects. */
  minAgreement?: number;
  /** Max registration residual relative to the median object radius. */
  maxResidualVsRadius?: number;
  /** A minority object is counted only with at least this edge-support score. */
  minorityMinScore?: number;
  /** A minority object overlapping a majority object (d < f·(r1 + r2)) is uncertain. */
  overlapFactor?: number;
}

const DEFAULTS: Required<FusionOptions> = { gate: 0.5, minAgreement: 0.5, maxResidualVsRadius: 0.15, minorityMinScore: 0.8, overlapFactor: 0.9 };

interface Track {
  /** Score-weighted centre sums. */
  sx: number;
  sy: number;
  sw: number;
  /** Radius of the best-supported observation (a partly hidden face or a bore fits worse). */
  r: number;
  score: number;
  views: number[];
  /** An observation from another photo overlapped this object but could not be matched to it. */
  conflict: boolean;
}

const tx = (t: Track) => t.sx / t.sw;
const ty = (t: Track) => t.sy / t.sw;
const tr = (t: Track) => t.r;
const newTrack = (c: Circle, view: number): Track => {
  const w = Math.max(0.05, c.score);
  return { sx: c.x * w, sy: c.y * w, sw: w, r: c.r, score: c.score, views: [view], conflict: false };
};

function inside(x: number, y: number, r: number, w: number, h: number) {
  return x - r >= 0 && y - r >= 0 && x + r <= w && y + r <= h;
}

/**
 * Greedy one-to-one matching of tracks and observations within the gate. There is
 * deliberately no radius gate: a concentric circle of another size (the bore of a
 * tube, the shading ring of a rod, a partly hidden face) is the same object, and
 * treating it as a new one is exactly how duplicates arise.
 */
function match(tracks: Track[], obs: Circle[], o: Required<FusionOptions>): [number, number][] {
  const cand: [number, number, number][] = [];
  tracks.forEach((t, i) => {
    const x = tx(t), y = ty(t), r = tr(t);
    obs.forEach((d, j) => {
      const dist = Math.hypot(x - d.x, y - d.y);
      if (dist >= o.gate * Math.min(r, d.r)) return;
      cand.push([dist / Math.min(r, d.r), i, j]);
    });
  });
  cand.sort((a, b) => a[0] - b[0]);
  const ut = new Set<number>(), uo = new Set<number>();
  const pairs: [number, number][] = [];
  for (const [, i, j] of cand) {
    if (ut.has(i) || uo.has(j)) continue;
    ut.add(i);
    uo.add(j);
    pairs.push([i, j]);
  }
  return pairs;
}

export function fuseDetections(views: FusionView[], opts: FusionOptions = {}): FusionResult {
  const o = { ...DEFAULTS, ...opts };
  const ref = views[0]!;
  const tracks: Track[] = ref.circles.map((c) => newTrack(c, 0));
  const medianR = (() => {
    const s = ref.circles.map((c) => c.r).sort((a, b) => a - b);
    return s.length ? s[Math.floor(s.length / 2)]! : 0;
  })();

  const reports: FusionViewReport[] = [
    { index: 0, status: 'reference', detections: ref.circles.length, inliers: 0, residualPx: 0, agreement: null, added: ref.circles.length, conflicts: 0 },
  ];
  /** Used views with their view → reference and reference → view transforms. */
  const used: { index: number; toRef: Mat3 | null; fromRef: Mat3 | null; w: number; h: number }[] = [
    { index: 0, toRef: null, fromRef: null, w: ref.width, h: ref.height },
  ];

  for (let vi = 1; vi < views.length; vi++) {
    const v = views[vi]!;
    const reg = v.registration;
    const rep: FusionViewReport = {
      index: vi,
      status: 'used',
      detections: v.circles.length,
      inliers: reg?.inliers ?? 0,
      residualPx: reg?.residualPx ?? Infinity,
      agreement: null,
      added: 0,
      conflicts: 0,
    };
    reports.push(rep);
    const H = reg?.H ?? null;
    const Hinv = H ? invert3(H) : null;
    if (!reg || !reg.ok || !H || !Hinv) {
      rep.status = 'refused-registration';
      rep.reason = reg?.reason ?? 'no-registration';
      continue;
    }
    if (medianR > 0 && reg.residualPx > o.maxResidualVsRadius * medianR) {
      rep.status = 'refused-registration';
      rep.reason = 'residual-vs-size';
      continue;
    }

    // Detections in reference coordinates; those outside the reference frame cannot be shown there.
    const mapped: Circle[] = v.circles.flatMap((c) => {
      const [x, y] = apply3(H, c.x, c.y);
      const r = c.r * localScale3(H, c.x, c.y);
      return inside(x, y, 0, ref.width, ref.height) ? [{ x, y, r, score: c.score }] : [];
    });
    const pairs = match(tracks, mapped, o);

    // Agreement in the overlap of this view with what is known.
    const knownInView = tracks.filter((t) => {
      const [x, y] = apply3(Hinv, tx(t), ty(t));
      return inside(x, y, tr(t), v.width, v.height);
    }).length;
    const denom = Math.min(mapped.length, knownInView);
    rep.agreement = denom ? pairs.length / denom : 0;
    if (denom > 0 && rep.agreement < o.minAgreement) {
      rep.status = 'refused-inconsistent';
      rep.reason = 'inconsistent';
      continue;
    }

    const matched = new Set<number>();
    for (const [i, j] of pairs) {
      const t = tracks[i]!, d = mapped[j]!;
      const w = Math.max(0.05, d.score);
      t.sx += d.x * w;
      t.sy += d.y * w;
      t.sw += w;
      if (d.score > t.score) {
        t.score = d.score;
        t.r = d.r;
      }
      t.views.push(vi);
      matched.add(j);
    }
    const known = tracks.length;
    mapped.forEach((d, j) => {
      if (matched.has(j)) return;
      // Distinct end faces never overlap; an unmatched observation whose centre lies
      // inside a known object (or vice versa) is that object seen badly, not a new one.
      const clash = tracks.slice(0, known).find((t) => Math.hypot(tx(t) - d.x, ty(t) - d.y) < Math.max(tr(t), d.r));
      if (clash) {
        clash.conflict = true;
        rep.conflicts++;
        return;
      }
      tracks.push(newTrack(d, vi));
      rep.added++;
    });
    used.push({ index: vi, toRef: H, fromRef: Hinv, w: v.width, h: v.height });
  }

  const base = tracks.map((t) => {
    const x = tx(t), y = ty(t), r = tr(t);
    let inFrame = 0;
    for (const u of used) {
      if (!u.fromRef) {
        if (inside(x, y, r, u.w, u.h)) inFrame++;
        continue;
      }
      const [vx, vy] = apply3(u.fromRef, x, y);
      if (inside(vx, vy, r / localScale3(u.toRef!, vx, vy), u.w, u.h)) inFrame++;
    }
    // An object cut by the frame edge in some photo can still be observed there.
    inFrame = Math.max(inFrame, t.views.length);
    return { x, y, r, score: t.score, seenIn: t.views.length, inFrame, views: t.views, conflict: t.conflict };
  });
  const isMajority = (b: { seenIn: number; inFrame: number }) => 2 * b.seenIn > b.inFrame;
  const majority = base.filter((b) => isMajority(b) && b.seenIn >= 2);
  const objects: FusedObject[] = base.map(({ conflict, ...b }) => {
    let agreement: FusedObject['agreement'];
    if (b.seenIn === b.inFrame) agreement = 'consistent';
    else if (isMajority(b)) agreement = 'majority';
    else {
      const overlaps = majority.some((m) => m !== b && Math.hypot(m.x - b.x, m.y - b.y) < o.overlapFactor * (m.r + b.r));
      agreement = !overlaps && b.score >= o.minorityMinScore ? 'minority' : 'uncertain';
    }
    const counted = agreement !== 'uncertain';
    return { ...b, agreement, counted, flagged: agreement === 'minority' || agreement === 'uncertain' || conflict };
  });

  const usedViews = used.length;
  const status = usedViews >= 2 ? 'fused' : 'refused';
  const counted = objects.filter((ob) => ob.counted);
  return {
    status,
    reason: status === 'refused' ? 'no-usable-view' : undefined,
    objects,
    views: reports,
    referenceCount: ref.circles.length,
    countedTotal: counted.length,
    addedCount: counted.filter((ob) => !ob.views.includes(0)).length,
    flaggedCount: counted.filter((ob) => ob.flagged).length,
    uncertainCount: objects.length - counted.length,
    droppedFromReference: objects.filter((ob) => !ob.counted && ob.views.includes(0)).length,
    usedViews,
  };
}
