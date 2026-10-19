import { it } from 'vitest';
import { loadCv } from '../loadCv';
import { makeScene } from '../synthetic/scene';
import { matchCircles } from '../synthetic/metrics';
import { detectCircles, edgeSupport } from '../../src/analysis/circles';
import { SCENES } from './scenes';
import { writePng } from '../synthetic/png';

it.skipIf(!process.env.DEBUG_SCENE)('debug one scene', async () => {
  const cv = await loadCv();
  const spec = SCENES.find((s) => s.name === process.env.DEBUG_SCENE)!;
  const scene = makeScene(cv, spec);
  const dbg: { raw?: any[]; verified?: any[]; kept?: any[]; field?: any } = {};
  const res = detectCircles(cv, scene.image, { sensitivity: Number(process.env.SENS ?? 50), debug: dbg });
  for (const k of ['raw', 'verified', 'kept'] as const) {
    const mm = matchCircles(scene.truth, dbg[k]!);
    console.log(k, dbg[k]!.length, 'recall', mm.recall.toFixed(3));
  }
  const m = matchCircles(scene.truth, res.circles);
  const matchedD = new Set(m.pairs.map((p) => p[1]));
  const matchedT = new Set(m.pairs.map((p) => p[0]));
  const fps = res.circles.filter((_, j) => !matchedD.has(j));
  const fns = scene.truth.filter((_, i) => !matchedT.has(i));
  console.log('candidates', res.candidates, 'range', res.radiusRange, 'det', res.circles.length, 'P/R', m.precision, m.recall);
  console.log('FP sample', fps.slice(0, 15).map((c) => `${c.x.toFixed(0)},${c.y.toFixed(0)} r${c.r.toFixed(1)} s${c.score.toFixed(2)}`).join(' | '));
  console.log('FN sample', fns.slice(0, 15).map((c) => `${c.x.toFixed(0)},${c.y.toFixed(0)} r${c.r.toFixed(1)}`).join(' | '));
  const rs = m.pairs.map(([i, j]) => res.circles[j]!.r / scene.truth[i]!.r);
  rs.sort((a, b) => a - b);
  console.log('r ratio quantiles', [0.05, 0.25, 0.5, 0.75, 0.95].map((q) => rs[Math.floor(q * (rs.length - 1))]?.toFixed(3)));
  console.log('FN support', fns.slice(0, 10).map((c) => { const e = edgeSupport(dbg.field, c.x, c.y, c.r); return `cov${e.coverage.toFixed(2)} +${e.pos}/-${e.neg} str${e.strength.toFixed(0)}`; }).join(' | '));
  for (const c of fns.slice(0, 4)) {
    const near = dbg.raw!.filter((d: any) => Math.hypot(d.x - c.x, d.y - c.y) < c.r * 0.6);
    console.log('GT', c.x.toFixed(0), c.y.toFixed(0), c.r.toFixed(1), 'raw:', near.map((d: any) => { const e = edgeSupport(dbg.field, d.x, d.y, d.r); return `(${(d.x - c.x).toFixed(1)},${(d.y - c.y).toFixed(1)}) r${d.r.toFixed(1)} cov${e.coverage.toFixed(2)} +${e.pos}/-${e.neg}`; }).join(' ; '));
  }
  console.log('edgeThr', dbg.field.edgeThreshold);
  if (process.env.DEBUG_PNG) {
    const { width: w, height: h, data } = scene.image;
    const src = new cv.Mat(h, w, cv.CV_8UC1); src.data.set(data);
    const rgb = new cv.Mat(); cv.cvtColor(src, rgb, cv.COLOR_GRAY2RGB);
    for (const c of fns) cv.circle(rgb, new cv.Point(Math.round(c.x), Math.round(c.y)), Math.round(c.r), new cv.Scalar(0, 120, 255), 2);
    for (const c of fps) cv.circle(rgb, new cv.Point(Math.round(c.x), Math.round(c.y)), Math.round(c.r), new cv.Scalar(255, 0, 0), 2);
    writePng(process.env.DEBUG_PNG, new Uint8Array(rgb.data), w, h);
    src.delete(); rgb.delete();
  }
});
