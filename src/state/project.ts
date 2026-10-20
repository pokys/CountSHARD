/**
 * Pure project operations. UI code calls these; they never mutate their inputs.
 */
import { clusterRadii, median, relativeSpread } from '../analysis/clustering';
import {
  type Calibration,
  type CountArea,
  inCountArea,
  type CountedObject,
  type GroupingParams,
  type Project,
  type SizeGroup,
  DEFAULT_DETECTION_PARAMS,
  type MultiShotObservation,
  DEFAULT_GROUPING,
  isCandidate,
  isCounted,
  isMultiShotFlagged,
  isSelectable,
} from '../model/types';

export const APP_VERSION = '1.1.0';

/** Colours with a fixed meaning in the overlay; size groups must never use them (or look-alikes). */
export const RESERVED_COLORS = { review: '#f59e0b', rejected: '#ef4444' } as const;
const RESERVED_LOOKALIKES = new Set<string>([RESERVED_COLORS.review, RESERVED_COLORS.rejected, '#f97316']);

export const GROUP_COLORS = ['#22c55e', '#3b82f6', '#ec4899', '#14b8a6', '#a855f7', '#84cc16', '#06b6d4', '#6366f1', '#d946ef', '#cbd5e1'];

let idCounter = 0;
export function newId(prefix: string): string {
  idCounter = (idCounter + 1) % 1e6;
  return `${prefix}_${Date.now().toString(36)}${Math.floor(Math.random() * 1e6).toString(36)}${idCounter.toString(36)}`;
}

/** Camera apps name every photo "image.jpg", "IMG_1234.JPG", …; such names are replaced by the date. */
export function projectName(fileName: string, now = new Date()): string {
  const base = fileName.replace(/\.[^.]+$/, '').trim();
  if (base && !/^(image|photo|img|dsc|pxl|dcim|foto)[\s_-]?\d*([_-]\d+)*$/i.test(base)) return base;
  const p = (n: number) => String(n).padStart(2, '0');
  return `Foto ${now.getDate()}. ${now.getMonth() + 1}. ${p(now.getHours())}:${p(now.getMinutes())}`;
}

export function createProject(image: Project['image']): Project {
  const now = new Date().toISOString();
  return {
    format: 'countshard-project',
    version: 1,
    id: newId('p'),
    name: projectName(image.name),
    createdAt: now,
    updatedAt: now,
    image,
    params: { ...DEFAULT_DETECTION_PARAMS },
    grouping: { ...DEFAULT_GROUPING },
    objects: [],
    groups: [],
    calibration: null,
    analysis: null,
    appVersion: APP_VERSION,
  };
}

export interface DetectedCircle {
  x: number;
  y: number;
  r: number;
  score: number;
  /** Set for multi-shot fusion results. */
  multiShot?: MultiShotObservation;
}

/**
 * Merges a fresh detection run into existing objects without discarding user work:
 * - manual and confirmed objects are kept; new detections overlapping them are skipped,
 * - rejected detections are kept (hidden); a new detection at the same place stays rejected,
 * - previous *unreviewed* automatic detections are replaced by the new run.
 */
export function mergeDetections(objects: CountedObject[], detected: DetectedCircle[]): CountedObject[] {
  const kept = objects.filter((o) => o.source === 'manual' || o.review !== 'unreviewed');
  const blocks = (d: DetectedCircle) =>
    kept.some((o) => Math.hypot(o.x - d.x, o.y - d.y) < 0.6 * Math.max(o.r, d.r));
  const fresh: CountedObject[] = detected
    .filter((d) => !blocks(d))
    .map((d) => ({
      id: newId('o'),
      x: d.x,
      y: d.y,
      r: d.r,
      score: d.score,
      source: 'auto',
      review: 'unreviewed',
      groupId: null,
      groupLocked: false,
      ...(d.multiShot ? { multiShot: d.multiShot } : {}),
    }));
  return [...kept, ...fresh];
}

/** Re-clusters unlocked counted objects; keeps group identities (ids, colours, nominal sizes) stable. */
export function applyGrouping(
  objects: CountedObject[],
  groups: SizeGroup[],
  grouping: GroupingParams,
  area: CountArea | null = null,
): { objects: CountedObject[]; groups: SizeGroup[] } {
  if (area) {
    // Objects outside the counting area take no part in grouping and hold no group.
    objects = objects.map((o) => (inCountArea(area, o.x, o.y) || !o.groupId ? o : { ...o, groupId: null, groupLocked: false }));
  }
  const counted = objects.filter((o) => isCounted(o) && inCountArea(area, o.x, o.y));
  const free = counted.filter((o) => !o.groupLocked);
  if (counted.length === 0) return { objects, groups: [] };

  // Current median radius of every existing group (used to keep identities stable).
  const groupMedian = new Map(groups.map((g) => [g.id, median(counted.filter((o) => o.groupId === g.id).map((o) => o.r))]));

  const res = clusterRadii(free.map((o) => o.r), {
    mode: grouping.mode,
    k: grouping.k,
    minSeparation: grouping.minSeparation,
  });
  const used = new Set<string>();
  const tol = Math.log(1 + Math.max(0.05, grouping.minSeparation));
  const lockedGroupIds = new Set(counted.filter((o) => o.groupLocked && o.groupId).map((o) => o.groupId!));
  const nextGroups: SizeGroup[] = groups.filter((g) => lockedGroupIds.has(g.id)).map((g) => ({ ...g }));
  lockedGroupIds.forEach((id) => used.add(id));

  const clusterToGroup = res.centers.map((c) => {
    let best: SizeGroup | null = null;
    let bestD = Infinity;
    for (const g of groups) {
      const gm = groupMedian.get(g.id);
      if (gm === undefined || Number.isNaN(gm)) continue;
      const d = Math.abs(Math.log(c) - Math.log(gm));
      if (d < tol && d < bestD && (!used.has(g.id) || lockedGroupIds.has(g.id))) {
        best = g;
        bestD = d;
      }
    }
    if (best) {
      if (!nextGroups.some((g) => g.id === best!.id)) nextGroups.push({ ...best });
      used.add(best.id);
      return best.id;
    }
    const g: SizeGroup = { id: newId('g'), label: '', color: unusedColor([...nextGroups, ...groups]) };
    nextGroups.push(g);
    used.add(g.id);
    return g.id;
  });

  const assign = new Map<string, string>();
  free.forEach((o, i) => assign.set(o.id, clusterToGroup[res.labels[i]!]!));
  const nextObjects = objects.map((o) => (assign.has(o.id) ? { ...o, groupId: assign.get(o.id)! } : o));
  return { objects: nextObjects, groups: relabel(nextObjects, nextGroups) };
}

function unusedColor(groups: SizeGroup[]): string {
  const taken = new Set(groups.map((g) => g.color.toLowerCase()));
  return GROUP_COLORS.find((c) => !taken.has(c)) ?? GROUP_COLORS[groups.length % GROUP_COLORS.length]!;
}

/** Drops empty groups and labels the rest A, B, C… by ascending median size. */
export function relabel(objects: CountedObject[], groups: SizeGroup[]): SizeGroup[] {
  const counted = objects.filter(isCounted);
  const withSize = groups
    .map((g) => ({ g, m: median(counted.filter((o) => o.groupId === g.id).map((o) => o.r)) }))
    .filter((x) => !Number.isNaN(x.m))
    .sort((a, b) => a.m - b.m);
  // Groups saved before the palette change may still carry a reserved colour; recolour them.
  const out = withSize.map(({ g }, i) => ({ ...g, label: groupLabel(i) }));
  for (const g of out) if (RESERVED_LOOKALIKES.has(g.color.toLowerCase())) g.color = unusedColor(out);
  return out;
}

export function groupLabel(i: number): string {
  return i < 26 ? String.fromCharCode(65 + i) : `G${i + 1}`;
}

// --- editing operations ----------------------------------------------------

export function addManualObject(objects: CountedObject[], x: number, y: number, r: number, groupId: string | null): { objects: CountedObject[]; id: string } {
  const id = newId('o');
  return {
    id,
    objects: [...objects, { id, x, y, r, source: 'manual', review: 'confirmed', groupId, groupLocked: groupId !== null }],
  };
}

/** Manual objects are deleted; automatic ones are marked rejected so re-analysis keeps them out. */
export function removeObject(objects: CountedObject[], id: string): CountedObject[] {
  return objects.flatMap((o) => {
    if (o.id !== id) return [o];
    if (o.source === 'manual') return [];
    return [{ ...o, review: 'rejected' as const }];
  });
}

export function updateObject(objects: CountedObject[], id: string, patch: Partial<Pick<CountedObject, 'x' | 'y' | 'r'>>): CountedObject[] {
  return objects.map((o) => (o.id === id ? { ...o, ...patch, review: 'confirmed' } : o));
}

export function setReview(objects: CountedObject[], ids: string[], review: 'confirmed' | 'unreviewed'): CountedObject[] {
  const set = new Set(ids);
  return objects.map((o) => (set.has(o.id) && o.review !== 'rejected' ? { ...o, review } : o));
}

export function setObjectGroup(objects: CountedObject[], ids: string[], groupId: string): CountedObject[] {
  const set = new Set(ids);
  return objects.map((o) => (set.has(o.id) ? { ...o, groupId, groupLocked: true } : o));
}

export function createGroup(groups: SizeGroup[]): SizeGroup {
  return { id: newId('g'), label: groupLabel(groups.length), color: unusedColor(groups) };
}

export function mergeGroups(objects: CountedObject[], groups: SizeGroup[], fromId: string, intoId: string) {
  const nextObjects = objects.map((o) => (o.groupId === fromId ? { ...o, groupId: intoId, groupLocked: true } : o));
  return { objects: nextObjects, groups: relabel(nextObjects, groups.filter((g) => g.id !== fromId)) };
}

/**
 * Discards a whole size group: its automatic detections become `rejected` (so re-analysis
 * does not bring them back), manually added members are deleted. Undo restores them.
 */
export function removeGroup(objects: CountedObject[], groups: SizeGroup[], groupId: string) {
  const nextObjects = objects.flatMap((o) => {
    if (o.groupId !== groupId || !isCounted(o)) return [o];
    if (o.source === 'manual') return [];
    return [{ ...o, review: 'rejected' as const }];
  });
  return { objects: nextObjects, groups: relabel(nextObjects, groups.filter((g) => g.id !== groupId)) };
}

/** Splits one group into two by size (k-means, k = 2); members become locked to their new group. */
export function splitGroup(objects: CountedObject[], groups: SizeGroup[], groupId: string) {
  const members = objects.filter((o) => isCounted(o) && o.groupId === groupId);
  if (members.length < 2) return { objects, groups };
  const res = clusterRadii(members.map((o) => o.r), { mode: 'fixed', k: 2 });
  if (res.centers.length < 2) return { objects, groups };
  const ng = createGroup(groups);
  const big = new Set(members.filter((_, i) => res.labels[i] === 1).map((o) => o.id));
  const nextObjects = objects.map((o) =>
    o.groupId === groupId && isCounted(o) ? { ...o, groupId: big.has(o.id) ? ng.id : groupId, groupLocked: true } : o,
  );
  return { objects: nextObjects, groups: relabel(nextObjects, [...groups, ng]) };
}

export function unlockAllGroups(objects: CountedObject[]): CountedObject[] {
  return objects.map((o) => (o.groupLocked ? { ...o, groupLocked: false } : o));
}

// --- calibration -----------------------------------------------------------

export function calibrateFromObject(obj: CountedObject, diameterMm: number): Calibration {
  return { mmPerPx: diameterMm / (2 * obj.r), referenceObjectId: obj.id, referenceDiameterMm: diameterMm };
}

/** Calibrates from a whole group's median diameter — more robust than one object. */
export function calibrateFromGroup(objects: CountedObject[], groupId: string, diameterMm: number): Calibration | null {
  const m = median(objects.filter((o) => isCounted(o) && o.groupId === groupId).map((o) => o.r));
  if (!m || Number.isNaN(m)) return null;
  return { mmPerPx: diameterMm / (2 * m), referenceObjectId: null, referenceDiameterMm: diameterMm };
}

// --- statistics ------------------------------------------------------------

export interface GroupStats {
  group: SizeGroup;
  count: number;
  medianDiameterPx: number;
  /** Robust relative spread of diameters (MAD/median). */
  spread: number;
  medianDiameterMm: number | null;
}

export interface ProjectStats {
  total: number;
  auto: number;
  manual: number;
  rejected: number;
  confirmed: number;
  unreviewed: number;
  ungrouped: number;
  /** Multi-shot review candidates (not counted). */
  candidates: number;
  /** Counted multi-shot objects seen in at most half of the photos (unreviewed). */
  multiShotFlagged: number;
  groups: GroupStats[];
}

export function computeStats(p: Pick<Project, 'objects' | 'groups' | 'calibration'> & { countArea?: CountArea | null }): ProjectStats {
  const counted = p.objects.filter((o) => isCounted(o) && inCountArea(p.countArea, o.x, o.y));
  const groups = p.groups.map((g) => {
    const radii = counted.filter((o) => o.groupId === g.id).map((o) => o.r);
    const m = median(radii);
    return {
      group: g,
      count: radii.length,
      medianDiameterPx: 2 * m,
      spread: relativeSpread(radii),
      medianDiameterMm: p.calibration ? 2 * m * p.calibration.mmPerPx : null,
    };
  }).filter((g) => !p.countArea || g.count > 0);
  return {
    total: counted.length,
    auto: counted.filter((o) => o.source === 'auto').length,
    manual: counted.filter((o) => o.source === 'manual').length,
    rejected: p.objects.filter((o) => o.review === 'rejected' && inCountArea(p.countArea, o.x, o.y)).length,
    confirmed: counted.filter((o) => o.review === 'confirmed').length,
    unreviewed: counted.filter((o) => o.review === 'unreviewed').length,
    ungrouped: counted.filter((o) => !o.groupId || !p.groups.some((g) => g.id === o.groupId)).length,
    candidates: p.objects.filter(isCandidate).length,
    multiShotFlagged: counted.filter(isMultiShotFlagged).length,
    groups,
  };
}

/** Objects in reading order (top-to-bottom rows, then left-to-right) for stable numbering. */
export function numbering(objects: CountedObject[], area: CountArea | null = null): Map<string, number> {
  const counted = objects.filter((o) => isCounted(o) && inCountArea(area, o.x, o.y));
  const rowH = Math.max(4, median(counted.map((o) => o.r)) * 1.2 || 10);
  const sorted = [...counted].sort((a, b) => Math.floor(a.y / rowH) - Math.floor(b.y / rowH) || a.x - b.x);
  return new Map(sorted.map((o, i) => [o.id, i + 1]));
}

/** The project as counted: objects outside the counting area removed (for display and export). */
export function visibleInArea<T extends Pick<Project, 'objects'> & { countArea?: CountArea | null }>(p: T): T {
  return p.countArea ? { ...p, objects: p.objects.filter((o) => inCountArea(p.countArea, o.x, o.y)) } : p;
}

/** Normalised rectangle from two corners, clipped to the image; null when (nearly) empty. */
export function makeCountArea(x1: number, y1: number, x2: number, y2: number, w: number, h: number): CountArea | null {
  const ax = Math.max(0, Math.min(x1, x2));
  const ay = Math.max(0, Math.min(y1, y2));
  const bx = Math.min(w, Math.max(x1, x2));
  const by = Math.min(h, Math.max(y1, y2));
  if (bx - ax < 8 || by - ay < 8) return null;
  return { x: ax, y: ay, w: bx - ax, h: by - ay };
}

/**
 * Counting area from a drawn stroke (image pixels, clipped to the image). A stroke that encloses
 * a real shape becomes a lasso polygon; a roughly straight drag becomes the rectangle spanned by
 * its two ends (the diagonal), so one tool serves both.
 */
export function makeCountAreaFromPath(points: [number, number][], w: number, h: number): CountArea | null {
  if (points.length < 2) return null;
  const pts = points.map(([x, y]) => [Math.min(w, Math.max(0, x)), Math.min(h, Math.max(0, y))] as [number, number]);
  const first = pts[0]!;
  const last = pts[pts.length - 1]!;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const bx = Math.min(...xs), by = Math.min(...ys);
  const bw = Math.max(...xs) - bx, bh = Math.max(...ys) - by;
  let area2 = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) area2 += (pts[j]![0] + pts[i]![0]) * (pts[j]![1] - pts[i]![1]);
  const enclosed = Math.abs(area2) / 2;
  if (pts.length < 8 || enclosed < 0.25 * bw * bh) return makeCountArea(first[0], first[1], last[0], last[1], w, h);
  if (bw < 8 || bh < 8) return null;
  return { x: bx, y: by, w: bw, h: bh, poly: simplifyPath(pts, Math.max(bw, bh) / 300) };
}

/** Drops points closer than `minDist` to the previous kept one (keeps the outline small). */
function simplifyPath(pts: [number, number][], minDist: number): [number, number][] {
  const out: [number, number][] = [pts[0]!];
  for (const p of pts) {
    const q = out[out.length - 1]!;
    if (Math.hypot(p[0] - q[0], p[1] - q[1]) >= minDist) out.push([Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]);
  }
  return out;
}

/** Hit test in image coordinates; returns the nearest selectable object containing the point. */
export function hitTest(objects: CountedObject[], x: number, y: number, slack = 1.05): CountedObject | null {
  let best: CountedObject | null = null;
  let bestD = Infinity;
  for (const o of objects) {
    if (!isSelectable(o)) continue;
    const d = Math.hypot(o.x - x, o.y - y);
    if (d < Math.max(o.r * slack, 6) && d < bestD) {
      best = o;
      bestD = d;
    }
  }
  return best;
}
