/**
 * The shipped Mode B files: the download manifest matches the files on disk, and the int8
 * model actually runs through onnxruntime-web (WASM) on one tile of a synthetic scene.
 */
import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-web';
import { aiAssets } from '../../src/ai/modelStore';
import { type LocaBackend, runPlan } from '../../src/ai/runner';
import { planTiles } from '../../src/ai/tiling';
import { resizeRgba } from '../../src/ai/tensor';
import { loadCv } from '../loadCv';
import { makeScene } from '../synthetic/scene';

const ROOT = join(__dirname, '..', '..');

describe('Mode B model files', () => {
  it('match the size and SHA-256 the app verifies after download', () => {
    const assets = aiAssets('https://example.test/app/');
    const files: Record<string, string> = {
      features: join(ROOT, 'public', 'models', 'loca', 'loca-features-w8.onnx'),
      head: join(ROOT, 'public', 'models', 'loca', 'loca-head-w8.onnx'),
      wasm: join(ROOT, 'node_modules', 'onnxruntime-web', 'dist', 'ort-wasm-simd-threaded.jsep.wasm'),
    };
    for (const a of assets) {
      expect(statSync(files[a.key]!).size, a.key).toBe(a.bytes);
      if (a.sha256) expect(createHash('sha256').update(readFileSync(files[a.key]!)).digest('hex'), a.key).toBe(a.sha256);
      if (a.key !== 'wasm') expect(a.url).toBe(`https://example.test/app/models/loca/loca-${a.key}-w8.onnx`);
    }
  });

  it('counts a small synthetic rod scene within 10 % (one tile, WASM)', async () => {
    ort.env.wasm.numThreads = 1;
    const opts = { executionProviders: ['wasm'] };
    const f = await ort.InferenceSession.create(readFileSync(join(ROOT, 'public', 'models', 'loca', 'loca-features-w8.onnx')), opts);
    const h = await ort.InferenceSession.create(readFileSync(join(ROOT, 'public', 'models', 'loca', 'loca-head-w8.onnx')), opts);
    const backend: LocaBackend = {
      features: async (image) => (await f.run({ image: new ort.Tensor('float32', image, [1, 3, 512, 512]) })).features!.data as Float32Array,
      head: async (feat, ex, boxes) =>
        (
          await h.run({
            features: new ort.Tensor('float32', feat, [1, 256, 64, 64]),
            exemplar_features: new ort.Tensor('float32', ex, [1, 256, 64, 64]),
            bboxes: new ort.Tensor('float32', boxes, [1, 3, 4]),
          })
        ).density!.data as Float32Array,
    };
    const cv = await loadCv();
    const s = makeScene(cv, { name: 'smoke', width: 480, height: 360, radii: [20], layout: 'hex', noiseSigma: 6, blur: 1, seed: 11 });
    const { width, height, data } = s.image;
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < data.length; i++) {
      rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = data[i]!;
      rgba[4 * i + 3] = 255;
    }
    const ex = [...s.truth].sort((a, b) => Math.hypot(a.x - 240, a.y - 180) - Math.hypot(b.x - 240, b.y - 180)).slice(0, 3);
    const plan = planTiles(width, height, ex.map((c) => ({ x: c.x - c.r, y: c.y - c.r, w: 2 * c.r, h: 2 * c.r })));
    expect(plan.tiles).toHaveLength(1);
    expect(plan.width).toBe(512); // 40 px exemplars are inside the band: the photo is just fitted into one tile
    const res = await runPlan(backend, resizeRgba({ data: rgba, width, height }, plan.width, plan.height), plan);
    expect(Math.abs(res.total - s.truth.length) / s.truth.length).toBeLessThan(0.1);
  });
});
