/**
 * Multi-shot evaluation table (prints metrics; run with `npm run eval`).
 * Single reference photo vs. fusion of n photos of the same synthetic scene.
 */
import { beforeAll, describe, it } from 'vitest';
import type { CV } from '../../src/analysis/cv';
import { detectCircles } from '../../src/analysis/circles';
import { registerImages } from '../../src/analysis/registration';
import { fuseDetections } from '../../src/analysis/fusion';
import { loadCv } from '../loadCv';
import { makeMultiViewScene } from '../synthetic/scene';
import { matchCircles } from '../synthetic/metrics';
import { SCENES } from './scenes';

let cv: CV;
beforeAll(async () => {
  cv = await loadCv();
});

export const MV_SCENES = ['uniform-rods', 'uniform-tubes', 'mixed-4-sizes', 'clutter', 'noisy-blurred'];

describe('multi-shot evaluation', () => {
  it('prints table', () => {
    const rows: string[] = [];
    for (const name of MV_SCENES) {
      for (const n of [2, 3, 5]) {
        for (const seed of [1, 2]) {
          const spec = SCENES.find((s) => s.name === name)!;
          const mv = makeMultiViewScene(cv, spec, { views: n, seed });
          const t0 = performance.now();
          const det = mv.views.map((v) => detectCircles(cv, v.image, { sensitivity: 50 }).circles);
          const t1 = performance.now();
          const regs = mv.views.map((v, i) => (i === 0 ? null : registerImages(cv, mv.views[0]!.image, v.image)));
          const t2 = performance.now();
          const fused = fuseDetections(mv.views.map((v, i) => ({ width: v.image.width, height: v.image.height, circles: det[i]!, registration: regs[i]! })));
          const single = matchCircles(mv.fusedTruth, det[0]!);
          const counted = fused.objects.filter((o) => o.counted);
          const fm = matchCircles(mv.fusedTruth, counted);
          // duplicates: fused objects whose nearest GT is already taken by another fused object
          const dup = counted.filter((ob) => mv.fusedTruth.some((t) => Math.hypot(t.x - ob.x, t.y - ob.y) < 0.5 * t.r)).length - fm.tp;
          rows.push(
            `| ${name} | ${n} | ${seed} | ${mv.fusedTruth.length} | ${det[0]!.length} (R ${single.recall.toFixed(3)}, FP ${single.fp}) | ${fused.countedTotal} | ${fm.recall.toFixed(3)} | ${fm.fp} | ${dup} | ${fused.flaggedCount} | ${fused.uncertainCount} (${matchCircles(mv.fusedTruth, fused.objects.filter((o) => !o.counted)).tp} real) | ${regs.slice(1).map((r) => (r!.ok ? `${r!.inliers}/${r!.residualPx.toFixed(2)}` : r!.reason)).join(' ')} | ${((t1 - t0) / n).toFixed(0)} | ${((t2 - t1) / Math.max(1, n - 1)).toFixed(0)} |`,
          );
        }
      }
    }
    console.log(['', '| scene | n | seed | GT | single ref | fused count | R | FP | dup | flagged | uncertain (not counted) | registration inliers/residual | det ms/view | reg ms/view |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|', ...rows].join('\n'));
  }, 600_000);
});
