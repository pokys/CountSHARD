/**
 * Regression thresholds for the circle detector on the synthetic suite.
 * Thresholds sit slightly below measured values (see docs/testing.md) so that
 * real regressions fail while harmless numeric drift does not.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { CV } from '../../src/analysis/cv';
import { detectCircles, removeAggregates, scoreThreshold } from '../../src/analysis/circles';
import { clusterRadii } from '../../src/analysis/clustering';
import { loadCv } from '../loadCv';
import { makeScene } from '../synthetic/scene';
import { matchCircles } from '../synthetic/metrics';
import { SCENES } from '../eval/scenes';

let cv: CV;
beforeAll(async () => {
  cv = await loadCv();
});

const spec = (name: string) => SCENES.find((s) => s.name === name)!;

function run(name: string, sensitivity = 50) {
  const scene = makeScene(cv, spec(name));
  const det = detectCircles(cv, scene.image, { sensitivity });
  return { scene, det, m: matchCircles(scene.truth, det.circles) };
}

describe('detectCircles on synthetic scenes', () => {
  it.each([
    ['uniform-rods', 0.98, 0.97],
    ['uniform-tubes', 0.98, 0.98],
    ['mixed-4-sizes', 0.98, 0.98],
    ['mixed-tubes', 0.98, 0.98],
    ['uneven-light', 0.98, 0.98],
    ['clutter', 0.9, 0.98],
    ['perspective-0.88', 0.98, 0.95],
    ['occluded', 0.98, 0.9],
  ])('%s: precision ≥ %d, recall ≥ %d', (name, p, r) => {
    const { m } = run(name);
    expect(m.precision).toBeGreaterThanOrEqual(p);
    expect(m.recall).toBeGreaterThanOrEqual(r);
  });

  it('measures radii to within 5 % on mixed sizes (needed for size classes)', () => {
    expect(run('mixed-4-sizes').m.radiusMape).toBeLessThan(0.05);
  });

  it('measures tubes by their outer wall, not the bore', () => {
    const { det } = run('uniform-tubes');
    const bores = det.circles.filter((c) => c.r < 26 * 0.85);
    expect(bores.length).toBeLessThanOrEqual(3);
  });

  it('recovers the four size classes of a mixed bundle', () => {
    const { det } = run('mixed-4-sizes');
    const cl = clusterRadii(det.circles.map((c) => c.r), { mode: 'auto', minSeparation: 0.1 });
    expect(cl.centers).toHaveLength(4);
  });

  it('higher sensitivity never finds fewer objects', () => {
    const lo = run('noisy-blurred', 30).det.circles.length;
    const hi = run('noisy-blurred', 70).det.circles.length;
    expect(hi).toBeGreaterThanOrEqual(lo);
  });

  it('returns nothing on an empty image', () => {
    const img = { data: new Uint8Array(400 * 300).fill(90), width: 400, height: 300 };
    expect(detectCircles(cv, img, { sensitivity: 100 }).circles).toHaveLength(0);
  });
});

describe('helpers', () => {
  it('scoreThreshold decreases with sensitivity', () => {
    expect(scoreThreshold(0)).toBeGreaterThan(scoreThreshold(50));
    expect(scoreThreshold(50)).toBeGreaterThan(scoreThreshold(100));
  });

  it('removeAggregates drops a circle enclosing several smaller ones', () => {
    const small = [
      { x: 0, y: 0, r: 5, score: 1 },
      { x: 10, y: 0, r: 5, score: 1 },
    ];
    const big = { x: 5, y: 0, r: 20, score: 1 };
    expect(removeAggregates([...small, big])).toEqual(small);
  });
});
