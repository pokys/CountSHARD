import { describe, expect, it } from 'vitest';
import { clusterRadii, relativeSpread } from '../../src/analysis/clustering';
import { rng } from '../synthetic/scene';

function sample(groups: { r: number; n: number }[], noise: number, seed = 1) {
  const R = rng(seed);
  const radii: number[] = [];
  const truth: number[] = [];
  groups.forEach((g, i) => {
    for (let k = 0; k < g.n; k++) {
      radii.push(g.r * (1 + noise * R.gauss()));
      truth.push(i);
    }
  });
  return { radii, truth };
}

function accuracy(labels: number[], truth: number[]) {
  return labels.filter((l, i) => l === truth[i]).length / labels.length;
}

describe('clusterRadii auto', () => {
  it('finds a single group for uniform objects', () => {
    const { radii } = sample([{ r: 20, n: 300 }], 0.03);
    expect(clusterRadii(radii, { mode: 'auto' }).centers).toHaveLength(1);
  });

  it('separates the four EN 10255 sizes (20 / 26.9 / 33.7 / 42.4 mm) with 3 % noise', () => {
    const { radii, truth } = sample(
      [{ r: 10, n: 57 }, { r: 13.45, n: 124 }, { r: 16.85, n: 39 }, { r: 21.2, n: 18 }],
      0.03,
    );
    const res = clusterRadii(radii, { mode: 'auto' });
    expect(res.centers).toHaveLength(4);
    expect(accuracy(res.labels, truth)).toBeGreaterThan(0.97);
  });

  it('does not split one group into several under 6 % noise', () => {
    const { radii } = sample([{ r: 30, n: 500 }], 0.06, 7);
    expect(clusterRadii(radii, { mode: 'auto' }).centers).toHaveLength(1);
  });

  it('keeps a small but distinct group', () => {
    const { radii, truth } = sample([{ r: 15, n: 200 }, { r: 25, n: 6 }], 0.03, 3);
    const res = clusterRadii(radii, { mode: 'auto' });
    expect(res.centers).toHaveLength(2);
    expect(accuracy(res.labels, truth)).toBe(1);
  });

  it('orders clusters by ascending radius', () => {
    const { radii } = sample([{ r: 30, n: 50 }, { r: 10, n: 50 }], 0.02);
    const res = clusterRadii(radii, { mode: 'auto' });
    expect(res.centers[0]).toBeLessThan(res.centers[1]!);
  });
});

describe('clusterRadii fixed', () => {
  it('splits into exactly k groups', () => {
    const { radii, truth } = sample([{ r: 10, n: 40 }, { r: 12, n: 40 }], 0.02);
    const res = clusterRadii(radii, { mode: 'fixed', k: 2 });
    expect(res.centers).toHaveLength(2);
    expect(accuracy(res.labels, truth)).toBeGreaterThan(0.9);
  });
});

describe('relativeSpread', () => {
  it('approximates the coefficient of variation', () => {
    const { radii } = sample([{ r: 20, n: 2000 }], 0.05);
    expect(relativeSpread(radii)).toBeGreaterThan(0.04);
    expect(relativeSpread(radii)).toBeLessThan(0.06);
  });
});
