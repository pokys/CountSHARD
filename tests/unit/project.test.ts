import { describe, expect, it } from 'vitest';
import {
  addManualObject,
  applyGrouping,
  calibrateFromGroup,
  calibrateFromObject,
  computeStats,
  GROUP_COLORS,
  RESERVED_COLORS,
  createProject,
  mergeDetections,
  mergeGroups,
  numbering,
  projectName,
  hitTest,
  removeGroup,
  removeObject,
  setObjectGroup,
  setReview,
  splitGroup,
  updateObject,
} from '../../src/state/project';
import type { CountedObject } from '../../src/model/types';
import { DEFAULT_GROUPING } from '../../src/model/types';

const circle = (x: number, y: number, r = 10) => ({ x, y, r, score: 0.9 });

function base() {
  return mergeDetections([], [circle(10, 10), circle(40, 10), circle(70, 10)]);
}

describe('mergeDetections', () => {
  it('replaces unreviewed detections on re-analysis', () => {
    const first = base();
    const second = mergeDetections(first, [circle(10, 10), circle(40, 10)]);
    expect(second).toHaveLength(2);
    expect(second.every((o) => o.source === 'auto')).toBe(true);
  });

  it('keeps manual objects and does not double-count detections at their position', () => {
    const { objects } = addManualObject(base(), 100, 10, 10, null);
    const next = mergeDetections(objects, [circle(10, 10), circle(101, 11)]);
    expect(next.filter((o) => o.source === 'manual')).toHaveLength(1);
    expect(computeStats({ objects: next, groups: [], calibration: null }).total).toBe(2);
  });

  it('does not resurrect a rejected false positive', () => {
    const first = base();
    const rejected = removeObject(first, first[0]!.id);
    const next = mergeDetections(rejected, [circle(10, 10), circle(40, 10)]);
    const stats = computeStats({ objects: next, groups: [], calibration: null });
    expect(stats.total).toBe(1);
    expect(stats.rejected).toBe(1);
  });

  it('keeps a moved (confirmed) object where the user put it', () => {
    const first = base();
    const moved = updateObject(first, first[1]!.id, { x: 45 });
    const next = mergeDetections(moved, [circle(40, 10)]);
    expect(next.find((o) => o.id === first[1]!.id)?.x).toBe(45);
    expect(next.filter((o) => o.review !== 'rejected')).toHaveLength(1);
  });
});

describe('editing and counting', () => {
  it('total follows additions and removals', () => {
    let objects = base();
    const added = addManualObject(objects, 200, 200, 10, null);
    objects = added.objects;
    expect(computeStats({ objects, groups: [], calibration: null }).total).toBe(4);
    objects = removeObject(objects, added.id);
    objects = removeObject(objects, objects[0]!.id);
    const s = computeStats({ objects, groups: [], calibration: null });
    expect(s.total).toBe(2);
    expect(s.manual).toBe(0);
    expect(s.rejected).toBe(1);
  });

  it('numbers objects in reading order', () => {
    const objects = mergeDetections([], [circle(70, 10), circle(10, 50), circle(10, 10)]);
    const n = numbering(objects);
    const byNum = [...n.entries()].sort((a, b) => a[1] - b[1]).map(([id]) => objects.find((o) => o.id === id)!);
    expect(byNum.map((o) => [o.x, o.y])).toEqual([[10, 10], [70, 10], [10, 50]]);
  });
});

describe('grouping', () => {
  const mixed = (): CountedObject[] =>
    mergeDetections(
      [],
      [...Array.from({ length: 20 }, (_, i) => circle(i * 30, 0, 10 + (i % 3) * 0.1)), ...Array.from({ length: 10 }, (_, i) => circle(i * 50, 100, 20 + (i % 2) * 0.2))],
    );

  it('creates groups ordered by size and labels them A, B', () => {
    const g = applyGrouping(mixed(), [], DEFAULT_GROUPING);
    expect(g.groups.map((x) => x.label)).toEqual(['A', 'B']);
    const stats = computeStats({ ...g, calibration: null });
    expect(stats.groups.map((s) => s.count)).toEqual([20, 10]);
    expect(stats.ungrouped).toBe(0);
  });

  it('keeps group identity and manual assignment across re-grouping', () => {
    const g1 = applyGrouping(mixed(), [], DEFAULT_GROUPING);
    const small = g1.groups[0]!;
    const big = g1.groups[1]!;
    const victim = g1.objects.find((o) => o.groupId === small.id)!;
    const reassigned = setObjectGroup(g1.objects, [victim.id], big.id);
    const g2 = applyGrouping(reassigned, g1.groups, DEFAULT_GROUPING);
    expect(g2.groups.map((g) => g.id).sort()).toEqual([small.id, big.id].sort());
    expect(g2.objects.find((o) => o.id === victim.id)!.groupId).toBe(big.id);
  });

  it('discards a whole group and keeps it out after re-analysis', () => {
    const withManual = addManualObject(mixed(), 900, 100, 20, null).objects;
    const g1 = applyGrouping(withManual, [], DEFAULT_GROUPING);
    const big = g1.groups[1]!;
    const d = removeGroup(g1.objects, g1.groups, big.id);
    const stats = computeStats({ ...d, calibration: null });
    expect(d.groups.map((g) => g.label)).toEqual(['A']);
    expect(stats.total).toBe(20);
    expect(stats.rejected).toBe(10);
    expect(d.objects.some((o) => o.source === 'manual')).toBe(false);
    // Re-detecting the same large circles must not resurrect them.
    const again = mergeDetections(d.objects, [
      ...Array.from({ length: 20 }, (_, i) => circle(i * 30, 0, 10)),
      ...Array.from({ length: 10 }, (_, i) => circle(i * 50, 100, 20)),
    ]);
    expect(computeStats({ objects: again, groups: d.groups, calibration: null }).total).toBe(20);
  });

  it('merges and splits groups', () => {
    const g1 = applyGrouping(mixed(), [], DEFAULT_GROUPING);
    const merged = mergeGroups(g1.objects, g1.groups, g1.groups[1]!.id, g1.groups[0]!.id);
    expect(merged.groups).toHaveLength(1);
    const split = splitGroup(merged.objects, merged.groups, merged.groups[0]!.id);
    expect(split.groups).toHaveLength(2);
    expect(computeStats({ ...split, calibration: null }).groups.map((s) => s.count).sort()).toEqual([10, 20]);
  });
});

describe('group colours', () => {
  it('never use the reserved review/rejected colours', () => {
    const reserved = [RESERVED_COLORS.review, RESERVED_COLORS.rejected, '#f97316'];
    expect(GROUP_COLORS.some((c) => reserved.includes(c))).toBe(false);
    // 12 distinct size groups: every one still avoids the reserved colours.
    const objects = mergeDetections([], Array.from({ length: 36 }, (_, i) => circle(i * 200, 0, 10 * Math.pow(1.3, Math.floor(i / 3)))));
    const g = applyGrouping(objects, [], DEFAULT_GROUPING);
    expect(g.groups.length).toBeGreaterThan(5);
    expect(g.groups.some((x) => reserved.includes(x.color))).toBe(false);
  });

  it('recolours a group saved with a reserved colour', () => {
    const g = applyGrouping(mergeDetections([], [circle(0, 0), circle(30, 0), circle(60, 0)]), [], DEFAULT_GROUPING);
    const legacy = g.groups.map((x) => ({ ...x, color: '#f59e0b' }));
    const again = applyGrouping(g.objects, legacy, DEFAULT_GROUPING);
    expect(again.groups[0]!.id).toBe(legacy[0]!.id);
    expect(again.groups[0]!.color).not.toBe('#f59e0b');
  });
});

describe('calibration', () => {
  it('derives mm per pixel from one object or a group median', () => {
    const objects = mergeDetections([], [circle(0, 0, 10), circle(30, 0, 10), circle(60, 0, 11)]);
    expect(calibrateFromObject(objects[0]!, 20).mmPerPx).toBeCloseTo(1);
    const g = applyGrouping(objects, [], DEFAULT_GROUPING);
    const cal = calibrateFromGroup(g.objects, g.groups[0]!.id, 26.9)!;
    expect(cal.mmPerPx).toBeCloseTo(26.9 / 20);
    const stats = computeStats({ ...g, calibration: cal });
    expect(stats.groups[0]!.medianDiameterMm).toBeCloseTo(26.9);
  });
});

describe('projectName', () => {
  it('keeps meaningful file names and replaces camera defaults by the date', () => {
    const d = new Date(2026, 9, 20, 7, 5);
    expect(projectName('sklad A – trubky 26,9.jpg', d)).toBe('sklad A – trubky 26,9');
    for (const n of ['image.jpg', 'IMG_1234.JPG', 'PXL_20261019_123456.jpg', 'DSC01234.jpeg', 'photo.png'])
      expect(projectName(n, d)).toBe('Foto 20. 10. 07:05');
  });
});

describe('createProject', () => {
  it('produces a versioned, empty project', () => {
    const p = createProject({ name: 'trubky.jpg', width: 100, height: 80, originalWidth: 100, originalHeight: 80, mime: 'image/jpeg' });
    expect(p.format).toBe('countshard-project');
    expect(p.name).toBe('trubky');
    expect(p.objects).toEqual([]);
  });
});

describe('multi-shot results in the project', () => {
  const fused = (x: number, agreement: 'consistent' | 'majority' | 'minority' | 'uncertain') => ({
    ...circle(x, 10),
    multiShot: { seenIn: agreement === 'consistent' ? 3 : 1, inFrame: 3, agreement },
  });

  it('keeps corrections when a fused result replaces a single-photo one', () => {
    let objs = base();
    objs = removeObject(objs, objs[0]!.id); // rejected at (10, 10)
    objs = addManualObject(objs, 100, 10, 10, null).objects;
    const next = mergeDetections(objs, [fused(10, 'consistent'), fused(40, 'consistent'), fused(101, 'minority'), fused(130, 'uncertain')]);
    expect(next.filter((o) => o.review === 'rejected')).toHaveLength(1);
    expect(next.filter((o) => o.source === 'manual')).toHaveLength(1);
    const stats = computeStats({ objects: next, groups: [], calibration: null });
    // 40 (fused) + manual 100; the uncertain one at 130 is a candidate only.
    expect(stats.total).toBe(2);
    expect(stats.candidates).toBe(1);
  });

  it('counts an uncertain candidate only after the user confirms it, and it stays selectable', () => {
    const objs = mergeDetections([], [fused(10, 'consistent'), fused(40, 'uncertain'), fused(70, 'minority')]);
    const stats = computeStats({ objects: objs, groups: [], calibration: null });
    expect(stats.total).toBe(2);
    expect(stats.multiShotFlagged).toBe(1);
    const cand = hitTest(objs, 40, 10)!;
    expect(cand.multiShot?.agreement).toBe('uncertain');
    const confirmed = setReview(objs, [cand.id], 'confirmed');
    expect(computeStats({ objects: confirmed, groups: [], calibration: null }).total).toBe(3);
    // Removing a candidate rejects it, so re-fusion keeps it out.
    const removed = removeObject(objs, cand.id);
    const again = mergeDetections(removed, [fused(10, 'consistent'), fused(40, 'uncertain')]);
    expect(again.filter((o) => Math.abs(o.x - 40) < 1 && o.review !== 'rejected')).toHaveLength(0);
  });
});
