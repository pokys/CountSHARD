/**
 * Pure project operations. UI code calls these; they never mutate their inputs.
 */
import { clusterRadii, median, relativeSpread } from '../analysis/clustering';
import {
  type Calibration,
  type CountedObject,
  type GroupingParams,
  type Project,
  type SizeGroup,
  DEFAULT_DETECTION_PARAMS,
  DEFAULT_GROUPING,
  isCounted,
} from '../model/types';

export const APP_VERSION = '0.1.0';

export const GROUP_COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#ec4899', '#14b8a6', '#a855f7', '#ef4444', '#84cc16', '#06b6d4', '#f97316'];

let idCounter = 0;
export function newId(prefix: string): string {
  idCounter = (idCounter + 1) % 1e6;
  return `${prefix}_${Date.now().toString(36)}${Math.floor(Math.random() * 1e6).toString(36)}${idCounter.toString(36)}`;
}

export function createProject(image: Project['image']): Project {
  const now = new Date().toISOString();
  return {
    format: 'countshard-project',
    version: 1,
    id: newId('p'),
    name: image.name.replace(/\.[^.]+$/, '') || 'Projekt',
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
    }));
  return [...kept, ...fresh];
}

/** Re-clusters unlocked counted objects; keeps group identities (ids, colours, nominal sizes) stable. */
export function applyGrouping(objects: CountedObject[], groups: SizeGroup[], grouping: GroupingParams): { objects: CountedObject[]; groups: SizeGroup[] } {
  const counted = objects.filter(isCounted);
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
  const taken = new Set(groups.map((g) => g.color));
  return GROUP_COLORS.find((c) => !taken.has(c)) ?? GROUP_COLORS[groups.length % GROUP_COLORS.length]!;
}

/** Drops empty groups and labels the rest A, B, C… by ascending median size. */
export function relabel(objects: CountedObject[], groups: SizeGroup[]): SizeGroup[] {
  const counted = objects.filter(isCounted);
  const withSize = groups
    .map((g) => ({ g, m: median(counted.filter((o) => o.groupId === g.id).map((o) => o.r)) }))
    .filter((x) => !Number.isNaN(x.m))
    .sort((a, b) => a.m - b.m);
  return withSize.map(({ g }, i) => ({ ...g, label: groupLabel(i) }));
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
  groups: GroupStats[];
}

export function computeStats(p: Pick<Project, 'objects' | 'groups' | 'calibration'>): ProjectStats {
  const counted = p.objects.filter(isCounted);
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
  });
  return {
    total: counted.length,
    auto: counted.filter((o) => o.source === 'auto').length,
    manual: counted.filter((o) => o.source === 'manual').length,
    rejected: p.objects.filter((o) => o.review === 'rejected').length,
    confirmed: counted.filter((o) => o.review === 'confirmed').length,
    unreviewed: counted.filter((o) => o.review === 'unreviewed').length,
    ungrouped: counted.filter((o) => !o.groupId || !p.groups.some((g) => g.id === o.groupId)).length,
    groups,
  };
}

/** Objects in reading order (top-to-bottom rows, then left-to-right) for stable numbering. */
export function numbering(objects: CountedObject[]): Map<string, number> {
  const counted = objects.filter(isCounted);
  const rowH = Math.max(4, median(counted.map((o) => o.r)) * 1.2 || 10);
  const sorted = [...counted].sort((a, b) => Math.floor(a.y / rowH) - Math.floor(b.y / rowH) || a.x - b.x);
  return new Map(sorted.map((o, i) => [o.id, i + 1]));
}

/** Hit test in image coordinates; returns the nearest counted object containing the point. */
export function hitTest(objects: CountedObject[], x: number, y: number, slack = 1.05): CountedObject | null {
  let best: CountedObject | null = null;
  let bestD = Infinity;
  for (const o of objects) {
    if (!isCounted(o)) continue;
    const d = Math.hypot(o.x - x, o.y - y);
    if (d < Math.max(o.r * slack, 6) && d < bestD) {
      best = o;
      bestD = d;
    }
  }
  return best;
}
