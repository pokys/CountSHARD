/**
 * Detector evaluation on the synthetic suite. Prints a comparison table
 * (CountIT detector vs. single-range baseline). Run: npm run eval
 */
import { describe, it } from 'vitest';
import { loadCv } from '../loadCv';
import { makeScene } from '../synthetic/scene';
import { matchCircles } from '../synthetic/metrics';
import { detectCircles } from '../../src/analysis/circles';
import { baselineDetect } from './baseline';
import { SCENES } from './scenes';

describe('detector evaluation (synthetic)', () => {
  it('prints metrics table', async () => {
    const cv = await loadCv();
    const rows: string[] = [];
    rows.push('| scene | GT | CountIT det | P | R | count err | r MAPE | ms | baseline det | P | R | count err |');
    rows.push('|---|---|---|---|---|---|---|---|---|---|---|---|');
    for (const spec of SCENES) {
      const scene = makeScene(cv, spec);
      const ours = detectCircles(cv, scene.image, { sensitivity: 50 });
      const m = matchCircles(scene.truth, ours.circles);
      const base = baselineDetect(cv, scene.image);
      const b = matchCircles(scene.truth, base.circles);
      const f = (v: number) => v.toFixed(3);
      rows.push(
        `| ${spec.name} | ${scene.truth.length} | ${ours.circles.length} | ${f(m.precision)} | ${f(m.recall)} | ${m.countError} | ${(m.radiusMape * 100).toFixed(1)} % | ${Math.round(ours.durationMs)} | ${base.circles.length} | ${f(b.precision)} | ${f(b.recall)} | ${b.countError} |`,
      );
    }
    console.log('\n' + rows.join('\n'));
  });
});

import { clusterRadii } from '../../src/analysis/clustering';

describe('size classification (synthetic, detect → cluster)', () => {
  it('prints class accuracy', async () => {
    const cv = await loadCv();
    for (const name of ['mixed-4-sizes', 'mixed-tubes', 'uniform-rods', 'uniform-tubes']) {
      const spec = SCENES.find((s) => s.name === name)!;
      const scene = makeScene(cv, spec);
      const det = detectCircles(cv, scene.image, { sensitivity: 50 });
      const m = matchCircles(scene.truth, det.circles);
      const cl = clusterRadii(det.circles.map((c) => c.r), { mode: 'auto', minSeparation: 0.1 });
      // Map each cluster to its majority true class, then score matched detections.
      const votes = new Map<number, Map<number, number>>();
      for (const [ti, dj] of m.pairs) {
        const c = cl.labels[dj]!;
        const v = votes.get(c) ?? new Map<number, number>();
        v.set(scene.truth[ti]!.cls, (v.get(scene.truth[ti]!.cls) ?? 0) + 1);
        votes.set(c, v);
      }
      const major = new Map([...votes].map(([c, v]) => [c, [...v].sort((a, b) => b[1] - a[1])[0]![0]]));
      const correct = m.pairs.filter(([ti, dj]) => major.get(cl.labels[dj]!) === scene.truth[ti]!.cls).length;
      console.log(`${name}: true classes ${spec.radii.length}, found ${cl.centers.length} (centres ${cl.centers.map((c) => c.toFixed(1)).join(', ')} vs true ${spec.radii.map((r) => r.toFixed(1)).join(', ')}), class accuracy ${(correct / Math.max(1, m.pairs.length) * 100).toFixed(1)} %`);
    }
  });
});
