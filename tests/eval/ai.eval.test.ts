/**
 * Mode B (LOCA) evaluation with the shipped int8 model through onnxruntime-web (WASM, Node).
 * Compares LOCA's own protocol (squash to 512 × 512, plain sum; and the reference site's
 * "sum above 5 % of max") with CountSHARD's aspect-preserving scale + tiling.
 *
 * - Synthetic pipe/rod scenes (tests/eval/scenes.ts), exemplars = the 3 true objects nearest the
 *   image centre (their bounding boxes).
 * - FSC-147 sample (20 test images, 3 annotated exemplars each) if `node scripts/fetch-fsc147-sample.mjs`
 *   has been run; otherwise skipped.
 *
 * Run: npm run eval -- ai     (takes several minutes: ≈ 5 s per model pass, single thread)
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'vitest';
import * as ort from 'onnxruntime-web';
import jpeg from 'jpeg-js';
import { loadCv } from '../loadCv';
import { makeScene } from '../synthetic/scene';
import { SCENES } from './scenes';
import { type LocaBackend, runPlan, runSquashed, thresholdedSum } from '../../src/ai/runner';
import { type Box, planTiles } from '../../src/ai/tiling';
import { type Rgba, resizeRgba } from '../../src/ai/tensor';

const MODEL_DIR = join(__dirname, '..', '..', 'public', 'models', 'loca');
const FSC_DIR = join(__dirname, '.data', 'fsc147');
const LONG = 60 * 60 * 1000;

async function backend(): Promise<LocaBackend> {
  ort.env.wasm.numThreads = 1;
  const opts = { executionProviders: ['wasm'] };
  const f = await ort.InferenceSession.create(readFileSync(join(MODEL_DIR, 'loca-features-w8.onnx')), opts);
  const h = await ort.InferenceSession.create(readFileSync(join(MODEL_DIR, 'loca-head-w8.onnx')), opts);
  return {
    async features(image) {
      const o = await f.run({ image: new ort.Tensor('float32', image, [1, 3, 512, 512]) });
      return o.features!.data as Float32Array;
    },
    async head(features, ex, boxes) {
      const o = await h.run({
        features: new ort.Tensor('float32', features, [1, 256, 64, 64]),
        exemplar_features: new ort.Tensor('float32', ex, [1, 256, 64, 64]),
        bboxes: new ort.Tensor('float32', boxes, [1, 3, 4]),
      });
      return o.density!.data as Float32Array;
    },
  };
}

interface Case {
  name: string;
  img: Rgba;
  exemplars: Box[];
  gt: number;
}

async function evaluate(cases: Case[], title: string) {
  const be = await backend();
  const rows = [
    `| ${title} | GT | size px | exemplar px | squash 512: sum | squash: >5 % max | CountSHARD tiles | scale | tiles: sum | ms |`,
    '|---|---|---|---|---|---|---|---|---|---|',
  ];
  const err = { sq: [] as number[], th: [] as number[], ours: [] as number[] };
  for (const c of cases) {
    const sq = await runSquashed(be, c.img, c.exemplars, resizeRgba);
    const th = thresholdedSum(sq.density);
    const plan = planTiles(c.img.width, c.img.height, c.exemplars);
    const scaled = resizeRgba(c.img, plan.width, plan.height);
    const t0 = performance.now();
    const res = await runPlan(be, scaled, plan);
    const ms = performance.now() - t0;
    err.sq.push(sq.total - c.gt);
    err.th.push(th - c.gt);
    err.ours.push(res.total - c.gt);
    const ex = Math.round(Math.sqrt(c.exemplars.reduce((a, b) => a + b.w * b.h, 0) / c.exemplars.length));
    rows.push(
      `| ${c.name} | ${c.gt} | ${c.img.width}×${c.img.height} | ${ex} | ${sq.total.toFixed(1)} | ${th.toFixed(1)} | ${plan.tiles.length}${plan.limitedByTiles ? ' (limit)' : ''} | ${plan.scale.toFixed(2)} | ${res.total.toFixed(1)} | ${Math.round(ms)} |`,
    );
  }
  const mae = (v: number[]) => v.reduce((a, b) => a + Math.abs(b), 0) / v.length;
  const mape = (v: number[]) => (v.reduce((a, b, i) => a + Math.abs(b) / cases[i]!.gt, 0) / v.length) * 100;
  rows.push(`| **MAE** | | | | ${mae(err.sq).toFixed(1)} | ${mae(err.th).toFixed(1)} | | | ${mae(err.ours).toFixed(1)} | |`);
  rows.push(`| **mean abs. % error** | | | | ${mape(err.sq).toFixed(1)} % | ${mape(err.th).toFixed(1)} % | | | ${mape(err.ours).toFixed(1)} % | |`);
  console.log('\n' + rows.join('\n'));
}

describe('LOCA evaluation (int8 ONNX, onnxruntime-web WASM)', () => {
  it('synthetic pipe/rod scenes', async () => {
    const cv = await loadCv();
    const cases: Case[] = [];
    for (const spec of SCENES) {
      const s = makeScene(cv, spec);
      const { width, height, data } = s.image;
      const rgba = new Uint8ClampedArray(width * height * 4);
      for (let i = 0; i < data.length; i++) {
        rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = data[i]!;
        rgba[4 * i + 3] = 255;
      }
      const near = [...s.truth].sort((a, b) => Math.hypot(a.x - width / 2, a.y - height / 2) - Math.hypot(b.x - width / 2, b.y - height / 2)).slice(0, 3);
      cases.push({
        name: spec.name,
        img: { data: rgba, width, height },
        exemplars: near.map((c) => ({ x: c.x - c.r, y: c.y - c.r, w: 2 * c.r, h: 2 * c.r })),
        gt: s.truth.length,
      });
    }
    await evaluate(cases, 'synthetic scene');
  }, LONG);

  it.skipIf(!existsSync(join(FSC_DIR, 'annotations.json')))('FSC-147 sample', async () => {
    const ann = JSON.parse(readFileSync(join(FSC_DIR, 'annotations.json'), 'utf8')) as Record<string, { class: string; count: number; boxes: number[][][] }>;
    const cases: Case[] = Object.entries(ann).map(([n, a]) => {
      const d = jpeg.decode(readFileSync(join(FSC_DIR, n)), { useTArray: true });
      return {
        name: `${n} (${a.class})`,
        img: { data: d.data, width: d.width, height: d.height },
        // FSC-147 boxes: 4 corner points [[x1,y1],[x1,y2],[x2,y2],[x2,y1]] in this image's pixels.
        exemplars: a.boxes.map((b) => {
          const xs = b.map((p) => p[0]!), ys = b.map((p) => p[1]!);
          return { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
        }),
        gt: a.count,
      };
    });
    await evaluate(cases, 'FSC-147 image');
  }, LONG);

  // FSC-147 images are only 384 px tall, so the photo-size problem squashing has never shows.
  // A 2 × 2 mosaic of the same photo has 4× the objects at a quarter of the relative size —
  // like a wider shot of a bigger pile. GT = 4 × count, exemplars from the top-left copy.
  it.skipIf(!existsSync(join(FSC_DIR, 'annotations.json')))('FSC-147 "tree logs", 2 × 2 mosaic', async () => {
    const ann = JSON.parse(readFileSync(join(FSC_DIR, 'annotations.json'), 'utf8')) as Record<string, { class: string; count: number; boxes: number[][][] }>;
    const cases: Case[] = Object.entries(ann)
      .filter(([, a]) => a.class === 'tree logs')
      .map(([n, a]) => {
        const d = jpeg.decode(readFileSync(join(FSC_DIR, n)), { useTArray: true });
        const W = d.width * 2, H = d.height * 2;
        const data = new Uint8ClampedArray(W * H * 4);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const s = ((y % d.height) * d.width + (x % d.width)) * 4, t = (y * W + x) * 4;
          data[t] = d.data[s]!; data[t + 1] = d.data[s + 1]!; data[t + 2] = d.data[s + 2]!; data[t + 3] = 255;
        }
        return {
          name: `${n} ×4`,
          img: { data, width: W, height: H },
          exemplars: a.boxes.map((b) => {
            const xs = b.map((p) => p[0]!), ys = b.map((p) => p[1]!);
            return { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
          }),
          gt: 4 * a.count,
        };
      });
    await evaluate(cases, 'FSC-147 mosaic');
  }, LONG);
});
