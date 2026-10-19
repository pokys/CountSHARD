/**
 * Multi-shot (Phase 6): registration and fusion on synthetic multi-view scenes.
 * Acceptance: fused count equals ground truth with zero duplicates; fusion refuses
 * a photo of a different scene.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { CV } from '../../src/analysis/cv';
import { type Circle, detectCircles } from '../../src/analysis/circles';
import { registerImages } from '../../src/analysis/registration';
import { type FusionView, fuseDetections } from '../../src/analysis/fusion';
import { type Mat3, IDENTITY3, apply3, invert3, mul3 } from '../../src/analysis/homography';
import { loadCv } from '../loadCv';
import { makeMultiViewScene } from '../synthetic/scene';
import { matchCircles } from '../synthetic/metrics';
import { SCENES } from '../eval/scenes';

let cv: CV;
beforeAll(async () => {
  cv = await loadCv();
});

const spec = (name: string) => SCENES.find((s) => s.name === name)!;

function runFusion(name: string, views: number, seed: number) {
  const mv = makeMultiViewScene(cv, spec(name), { views, seed });
  const det = mv.views.map((v) => detectCircles(cv, v.image, { sensitivity: 50 }).circles);
  const regs = mv.views.map((v, i) => (i === 0 ? null : registerImages(cv, mv.views[0]!.image, v.image)));
  const fused = fuseDetections(mv.views.map((v, i) => ({ width: v.image.width, height: v.image.height, circles: det[i]!, registration: regs[i]! })));
  return { mv, det, regs, fused };
}

/** Counted fused objects that land on a ground-truth object another fused object already took. */
function duplicates(truth: { x: number; y: number; r: number }[], objs: { x: number; y: number }[]): number {
  const onTruth = objs.filter((o) => truth.some((t) => Math.hypot(t.x - o.x, t.y - o.y) < 0.5 * t.r)).length;
  return onTruth - matchCircles(truth.map((t) => ({ ...t, cls: 0 })), objs.map((o) => ({ ...o, r: 1 }))).tp;
}

describe('registration', () => {
  it('recovers the true homography between two views to about a pixel', () => {
    const mv = makeMultiViewScene(cv, spec('uniform-rods'), { views: 2, seed: 3 });
    const reg = registerImages(cv, mv.views[0]!.image, mv.views[1]!.image);
    expect(reg.ok).toBe(true);
    expect(reg.inliers).toBeGreaterThan(200);
    expect(reg.residualPx).toBeLessThan(2);
    // True view1 → view0 transform: H0 · H1⁻¹.
    const truth = mul3(mv.views[0]!.H, invert3(mv.views[1]!.H)!);
    const { width: w, height: h } = mv.views[1]!.image;
    for (const [x, y] of [[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2]] as const) {
      const [ax, ay] = apply3(reg.H!, x, y);
      const [bx, by] = apply3(truth, x, y);
      expect(Math.hypot(ax - bx, ay - by)).toBeLessThan(2);
    }
  });

  it('refuses to register a photo of a different scene', () => {
    const a = makeMultiViewScene(cv, spec('uniform-rods'), { views: 1, seed: 1 }).views[0]!;
    const b = makeMultiViewScene(cv, { ...spec('mixed-4-sizes') }, { views: 1, seed: 9 }).views[0]!;
    const reg = registerImages(cv, a.image, b.image);
    expect(reg.ok).toBe(false);
    expect(reg.reason).toBeDefined();
  });
});

describe('fusion on synthetic multi-view sets', () => {
  it.each([
    ['uniform-rods', 5, 1],
    ['uniform-tubes', 3, 1],
    ['mixed-4-sizes', 3, 2],
    ['mixed-4-sizes', 2, 1],
  ] as const)('%s × %d views (seed %d): fused count = ground truth, zero duplicates', (name, n, seed) => {
    const { mv, det, fused } = runFusion(name, n, seed);
    expect(fused.status).toBe('fused');
    expect(fused.usedViews).toBe(n);
    const counted = fused.objects.filter((o) => o.counted);
    expect(duplicates(mv.fusedTruth, counted)).toBe(0);
    const m = matchCircles(mv.fusedTruth, counted);
    expect(m.fp).toBe(0);
    expect(fused.countedTotal).toBe(mv.fusedTruth.length);
    // The single reference photo misses the objects its occluder hides.
    expect(matchCircles(mv.fusedTruth, det[0]!).recall).toBeLessThan(0.95);
  });

  it('never duplicates objects, even when the fused count is not exact', () => {
    for (const [name, n, seed] of [['uniform-tubes', 5, 2], ['clutter', 3, 1], ['noisy-blurred', 3, 1]] as const) {
      const { mv, fused } = runFusion(name, n, seed);
      expect(duplicates(mv.fusedTruth, fused.objects.filter((o) => o.counted))).toBe(0);
    }
  });

  it('reports "seen in k of n" per object', () => {
    const { fused } = runFusion('uniform-rods', 3, 1);
    for (const o of fused.objects) {
      expect(o.seenIn).toBeGreaterThanOrEqual(1);
      expect(o.seenIn).toBeLessThanOrEqual(o.inFrame);
      expect(o.inFrame).toBeLessThanOrEqual(3);
    }
    expect(fused.objects.filter((o) => o.agreement === 'consistent').length).toBeGreaterThan(200);
  });

  it('refuses fusion when the only other photo shows a different scene', () => {
    const mv = makeMultiViewScene(cv, spec('uniform-rods'), { views: 1, seed: 1 });
    const other = makeMultiViewScene(cv, spec('mixed-4-sizes'), { views: 1, seed: 9 }).views[0]!;
    const views = [mv.views[0]!, other];
    const det = views.map((v) => detectCircles(cv, v.image, { sensitivity: 50 }).circles);
    const reg = registerImages(cv, views[0]!.image, other.image);
    const fused = fuseDetections([
      { width: views[0]!.image.width, height: views[0]!.image.height, circles: det[0]!, registration: null },
      { width: other.image.width, height: other.image.height, circles: det[1]!, registration: reg },
    ]);
    expect(fused.status).toBe('refused');
    expect(fused.reason).toBe('no-usable-view');
    expect(fused.views[1]!.status).toBe('refused-registration');
  });

  it('leaves out a different-scene photo among good ones and says so', () => {
    const mv = makeMultiViewScene(cv, spec('uniform-rods'), { views: 3, seed: 1 });
    const intruder = makeMultiViewScene(cv, spec('mixed-4-sizes'), { views: 1, seed: 9 }).views[0]!;
    const views = [mv.views[0]!, intruder, mv.views[1]!, mv.views[2]!];
    const det = views.map((v) => detectCircles(cv, v.image, { sensitivity: 50 }).circles);
    const fused = fuseDetections(
      views.map((v, i) => ({
        width: v.image.width,
        height: v.image.height,
        circles: det[i]!,
        registration: i === 0 ? null : registerImages(cv, views[0]!.image, v.image),
      })),
    );
    expect(fused.status).toBe('fused');
    expect(fused.usedViews).toBe(3);
    expect(fused.views[1]!.status).toBe('refused-registration');
    expect(duplicates(mv.fusedTruth, fused.objects.filter((o) => o.counted))).toBe(0);
    // The intruder has no effect at all: same result as fusing the three good photos.
    const without = runFusion('uniform-rods', 3, 1).fused;
    expect(fused.countedTotal).toBe(without.countedTotal);
    expect(fused.uncertainCount).toBe(without.uncertainCount);
  });
});

describe('fusion logic (no images)', () => {
  const grid = (dx = 0, dy = 0, r = 10): Circle[] => {
    const out: Circle[] = [];
    for (let i = 0; i < 10; i++) for (let j = 0; j < 8; j++) out.push({ x: 30 + i * 21 + dx, y: 30 + j * 21 + dy, r, score: 1 });
    return out;
  };
  const okReg = (H: Mat3) => ({ ok: true, H, keypoints: [500, 500] as [number, number], matches: 400, inliers: 300, inlierRatio: 0.75, residualPx: 1, coverage: 0.8, cornerShift: 0.02 });
  const view = (circles: Circle[], H: Mat3 | null): FusionView => ({ width: 260, height: 220, circles, registration: H ? okReg(H) : null });

  it('counts identical observations once and marks them consistent', () => {
    const f = fuseDetections([view(grid(), null), view(grid(), IDENTITY3), view(grid(), IDENTITY3)]);
    expect(f.countedTotal).toBe(80);
    expect(f.objects.every((o) => o.agreement === 'consistent' && o.seenIn === 3)).toBe(true);
  });

  it('maps through the homography before matching', () => {
    // View 1 is shifted by (+7, −4): its detections sit at (x+7, y−4); H maps back.
    const H: Mat3 = [1, 0, -7, 0, 1, 4, 0, 0, 1];
    const f = fuseDetections([view(grid(), null), view(grid(7, -4), H)]);
    expect(f.countedTotal).toBe(80);
    expect(f.addedCount).toBe(0);
  });

  it('treats a concentric smaller circle (bore, partly hidden face) as the same object', () => {
    const inner = grid().map((c) => ({ ...c, r: c.r * 0.7, score: 0.6 }));
    const f = fuseDetections([view(grid(), null), view(inner, IDENTITY3)]);
    expect(f.countedTotal).toBe(80);
    expect(f.objects.every((o) => o.r === 10)).toBe(true);
  });

  it('refuses a view whose registration is plausible but wrong (half a pitch off)', () => {
    const f = fuseDetections([view(grid(), null), view(grid(), [1, 0, 10.5, 0, 1, 0, 0, 0, 1])]);
    expect(f.views[1]!.status).toBe('refused-inconsistent');
    expect(f.status).toBe('refused');
    expect(f.countedTotal).toBe(80);
  });

  it('refuses a view whose registration residual is large relative to the objects', () => {
    const bad = { ...okReg(IDENTITY3), residualPx: 4 };
    const f = fuseDetections([view(grid(), null), { width: 260, height: 220, circles: grid(), registration: bad }]);
    expect(f.views[1]!.status).toBe('refused-registration');
    expect(f.views[1]!.reason).toBe('residual-vs-size');
  });

  it('counts a strong object seen in one of two photos but flags it; a weak one stays uncertain', () => {
    const extraStrong: Circle = { x: 240, y: 200, r: 10, score: 0.95 };
    const extraWeak: Circle = { x: 240, y: 30, r: 10, score: 0.6 };
    const f = fuseDetections([view([...grid(), extraStrong], null), view([...grid(), extraWeak], IDENTITY3)]);
    expect(f.countedTotal).toBe(81);
    expect(f.flaggedCount).toBe(1);
    expect(f.uncertainCount).toBe(1);
    const weak = f.objects.find((o) => o.y < 40 && o.x > 230)!;
    expect(weak.agreement).toBe('uncertain');
    expect(weak.counted).toBe(false);
  });
});
