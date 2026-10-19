/**
 * Diagnostics for the multi-shot consistency check: per-view agreement under the
 * found homography, across scenes and sensitivities (unstable detections).
 */
import { it } from 'vitest';
import { loadCv } from '../loadCv';
import { detectCircles } from '../../src/analysis/circles';
import { registerImages } from '../../src/analysis/registration';
import { fuseDetections } from '../../src/analysis/fusion';
import { makeMultiViewScene } from '../synthetic/scene';
import { SCENES } from './scenes';

it('prints agreement per view', async () => {
  const cv = await loadCv();
  const rows: string[] = [];
  for (const name of ['uniform-rods', 'noisy-blurred', 'clutter', 'dense-small']) {
    for (const sens of [30, 50, 80, 95]) {
      const mv = makeMultiViewScene(cv, SCENES.find((s) => s.name === name)!, { views: 3, seed: 1 });
      const det = mv.views.map((v) => detectCircles(cv, v.image, { sensitivity: sens }).circles);
      const regs = mv.views.map((v, i) => (i === 0 ? null : registerImages(cv, mv.views[0]!.image, v.image)));
      const f = fuseDetections(mv.views.map((v, i) => ({ width: v.image.width, height: v.image.height, circles: det[i]!, registration: regs[i]! })));
      rows.push(`${name} s${sens}: det ${det.map((d) => d.length).join('/')} | ${f.views.slice(1).map((v) => `${v.status} agr ${(v.agreement ?? 0).toFixed(2)}`).join(' ; ')} | ${f.status}`);
    }
  }
  console.log('\n' + rows.join('\n'));
}, 600_000);
