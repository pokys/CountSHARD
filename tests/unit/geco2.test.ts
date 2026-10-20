/**
 * GeCo2 (Mode B detection model): pre/post-processing, editing of detections, automatic model
 * choice, exports. Optional end-to-end run of the real int8 graphs when GECO2_MODELS points to a
 * folder with geco2-{backbone,exemplars,head}-768-w8.onnx (scripts/geco2/, not in the repo).
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  asCountedObjects,
  EXEMPLAR_FIT_BIG,
  EXEMPLAR_FIT_SMALL,
  countedDetections,
  exemplarFit,
  geco2Boxes,
  geco2ContentSize,
  geco2Detect,
  geco2ExemplarSide,
  geco2Input,
  geco2Scale,
  mergeAiDetections,
  newManualDetection,
  removeAiDetection,
  toProjectDetections,
  typicalBox,
} from '../../src/ai/geco2';
import { MEAN, STD, resizeRgba } from '../../src/ai/tensor';
import { AUTO_MAX_SECONDS, type DeviceProfile, geco2Plan, heavyModelWarning, resolveModel, updateMeasured } from '../../src/ai/device';
import { GECO2_BASE, geco2Assets } from '../../src/ai/modelStore';
import { type AiDetection, type AiDetectionResult, type Project, isDensityResult, isDetectionResult } from '../../src/model/types';
import { createProject, numbering } from '../../src/state/project';
import { detectionSummary, parseProjectFile, toAiDetectionsCsv, toProjectFile } from '../../src/io/serialize';
import { draggedExemplar, summarize } from '../../src/ai/estimate';
import { loadCv } from '../loadCv';
import { makeScene } from '../synthetic/scene';

const det = (x: number, y: number, s: number, extra: Partial<AiDetection> = {}): AiDetection => ({
  id: `d${x}_${y}`,
  x,
  y,
  w: s,
  h: s,
  source: 'auto',
  review: 'unreviewed',
  score: 1,
  ...extra,
});

describe('GeCo2 pre-processing', () => {
  it('scales the longer side to S and caps the mean exemplar side at 80·S/1024', () => {
    expect(geco2Scale(2000, 1000, [], 1024)).toBeCloseTo(0.512);
    // Small exemplars: the photo is only fitted into S.
    expect(geco2Scale(2000, 1000, [{ w: 40, h: 40 }], 1024)).toBeCloseTo(0.512);
    // Large exemplars (200 px → 102 px at 0.512) are scaled down to 80 px.
    expect(geco2Scale(2000, 1000, [{ w: 200, h: 200 }], 1024)).toBeCloseTo(0.4);
    // 768: the cap is 60 px.
    expect(geco2Scale(1000, 500, [{ w: 100, h: 100 }], 768) * 100).toBeCloseTo(60);
    const s = geco2Scale(1000, 700, [{ w: 30, h: 30 }], 768);
    expect(geco2ContentSize(1000, 700, s, 768)).toEqual({ w: 768, h: 537 });
    expect(geco2ExemplarSide([{ w: 30, h: 50 }], 0.5)).toBe(20);
  });

  it('normalises with ImageNet statistics, NCHW, and pads with 0 (the mean)', () => {
    const img = { data: new Uint8ClampedArray([255, 0, 128, 255, 0, 0, 0, 255]), width: 2, height: 1 };
    const t = geco2Input(img, 768);
    const plane = 768 * 768;
    expect(t[0]).toBeCloseTo((1 - MEAN[0]) / STD[0], 5);
    expect(t[plane]).toBeCloseTo((0 - MEAN[1]) / STD[1], 5);
    expect(t[2 * plane]).toBeCloseTo((128 / 255 - MEAN[2]) / STD[2], 5);
    expect(t[2]).toBe(0);
    expect(t[768]).toBe(0);
  });

  it('exemplar boxes become x1, y1, x2, y2 in model pixels', () => {
    expect([...geco2Boxes([{ x: 10, y: 20, w: 30, h: 40 }], 0.5)]).toEqual([5, 10, 20, 30]);
  });
});

/** Objectness/ltrb maps with Gaussian-free single-pixel peaks. */
function maps(g: number, peaks: { x: number; y: number; v: number; half: number }[]) {
  const obj = new Float32Array(g * g);
  const ltrb = new Float32Array(4 * g * g);
  for (const p of peaks) {
    const i = p.y * g + p.x;
    obj[i] = p.v;
    for (let c = 0; c < 4; c++) ltrb[c * g * g + i] = p.half;
  }
  return { obj, ltrb };
}

describe('GeCo2 post-processing (as in GeCo2 evaluation)', () => {
  it('keeps local maxima above max/8, boxes from offsets, drops duplicates by NMS', () => {
    const { obj, ltrb } = maps(64, [
      { x: 10, y: 10, v: 1, half: 0.05 },
      { x: 40, y: 20, v: 0.5, half: 0.02 },
      // Same object again two cells away (IoU 0.52), weaker → removed by NMS.
      { x: 12, y: 10, v: 0.6, half: 0.05 },
      // Below max/8 → ignored.
      { x: 50, y: 50, v: 0.1, half: 0.02 },
    ]);
    const d = geco2Detect(obj, ltrb, 64, 64, 128, { w: 128, h: 128 });
    expect(d.scores.length).toBe(2);
    expect(d.scores[0]).toBe(1);
    // Centre (10/64, 10/64) ± 0.05, times S = 128.
    expect(d.boxes[0]).toBeCloseTo((10 / 64 - 0.05) * 128, 4);
    expect(d.boxes[2]).toBeCloseTo((10 / 64 + 0.05) * 128, 4);
  });

  it('drops detections whose centre lies in the padding', () => {
    const { obj, ltrb } = maps(64, [
      { x: 10, y: 10, v: 1, half: 0.02 },
      { x: 60, y: 10, v: 1, half: 0.02 },
    ]);
    const d = geco2Detect(obj, ltrb, 64, 64, 128, { w: 100, h: 128 });
    expect(d.scores.length).toBe(1);
  });

  it('returns nothing for an empty map', () => {
    expect(geco2Detect(new Float32Array(16), new Float32Array(64), 4, 4, 8, { w: 8, h: 8 }).scores.length).toBe(0);
  });

  it('converts to project pixels', () => {
    const list = toProjectDetections({ boxes: new Float32Array([10, 20, 30, 50]), scores: new Float32Array([0.7]) }, 0.5);
    expect(list).toHaveLength(1);
    expect(list[0]).toMatchObject({ x: 20, y: 40, w: 40, h: 60, score: expect.closeTo(0.7, 5), source: 'auto', review: 'unreviewed' });
  });
});

describe('editing detections (same rules as circle mode)', () => {
  it('removing an automatic detection rejects it, a manual one is deleted', () => {
    const m = newManualDetection(100, 100, 20, 20);
    const list = removeAiDetection(removeAiDetection([det(0, 0, 10), m], 'd0_0'), m.id);
    expect(list).toEqual([{ ...det(0, 0, 10), review: 'rejected' }]);
    expect(countedDetections(list)).toHaveLength(0);
  });

  it('a re-run keeps manual, confirmed and rejected ones and replaces the rest', () => {
    const old = [det(0, 0, 10), det(100, 0, 10, { review: 'rejected' }), newManualDetection(205, 5, 10, 10)];
    const fresh = [det(1, 1, 10), det(101, 1, 10), det(201, 1, 10), det(300, 0, 10)];
    const merged = mergeAiDetections(old, fresh);
    // Rejected one stays rejected (fresh duplicate skipped), manual one kept (duplicate skipped).
    expect(merged.filter((d) => d.review === 'rejected')).toHaveLength(1);
    expect(merged.filter((d) => d.source === 'manual')).toHaveLength(1);
    expect(countedDetections(merged).map((d) => d.x).sort((a, b) => a - b)).toEqual([1, 200, 300]);
  });

  it('respects the counting area and numbers like circle mode', () => {
    const list = [det(0, 0, 10), det(50, 0, 10), det(100, 100, 10)];
    const area = { x: 0, y: 0, w: 70, h: 20 };
    expect(countedDetections(list, area)).toHaveLength(2);
    const nums = numbering(asCountedObjects(list), area);
    expect([...nums.values()]).toEqual([1, 2]);
  });

  it('a tap-added object gets the typical size of its neighbours', () => {
    const list = [det(0, 0, 10), det(20, 0, 12), det(40, 0, 14)];
    expect(typicalBox(list, 25, 5, 99)).toEqual({ w: 12, h: 12 });
    expect(typicalBox([], 0, 0, 33)).toEqual({ w: 33, h: 33 });
  });
});

describe('adjusting exemplars and checking their size', () => {
  const box = { x: 100, y: 100, w: 50, h: 40 };
  it('moves inside the image and resizes by any corner, normalised', () => {
    expect(draggedExemplar(box, 'move', 10, -20, 400, 300)).toEqual({ x: 110, y: 80, w: 50, h: 40 });
    expect(draggedExemplar(box, 'move', 1000, 1000, 400, 300)).toEqual({ x: 350, y: 260, w: 50, h: 40 });
    expect(draggedExemplar(box, 'se', -20, -10, 400, 300)).toEqual({ x: 100, y: 100, w: 30, h: 30 });
    expect(draggedExemplar(box, 'nw', 10, 5, 400, 300)).toEqual({ x: 110, y: 105, w: 40, h: 35 });
    // Corner dragged past the opposite one: the box flips instead of going negative.
    expect(draggedExemplar(box, 'ne', -80, 0, 400, 300)).toEqual({ x: 70, y: 100, w: 30, h: 40 });
    expect(draggedExemplar(box, 'sw', 50, 0, 400, 300).w).toBe(4);
  });

  it('compares the exemplar size with the items found', () => {
    const found = [10, 11, 12, 10, 9, 11].map((s, i) => det(i * 20, 0, s));
    expect(exemplarFit([{ w: 20, h: 20 }], found)).toBeCloseTo(20 / 11, 5);
    expect(exemplarFit([{ w: 20, h: 20 }], found)!).toBeGreaterThan(EXEMPLAR_FIT_BIG);
    expect(exemplarFit([{ w: 5, h: 5 }], found)!).toBeLessThan(EXEMPLAR_FIT_SMALL);
    expect(exemplarFit([{ w: 11, h: 11 }], found)).toBeCloseTo(1, 5);
    // Too few detections, or only manual / rejected ones: no verdict.
    expect(exemplarFit([{ w: 20, h: 20 }], found.slice(0, 4))).toBeNull();
    expect(exemplarFit([{ w: 20, h: 20 }], found.map((d) => ({ ...d, review: 'rejected' as const })))).toBeNull();
  });
});

describe('automatic model choice', () => {
  const base: DeviceProfile = { mobile: false, webgpu: true, deviceMemoryGB: 8, cpuBenchMs: 30, measured: {}, geco2ForceWasm: false };

  it('desktop with WebGPU → GeCo2 1024 on WebGPU', () => {
    expect(geco2Plan(base)).toMatchObject({ backend: 'webgpu', size: 1024, measured: false });
    expect(resolveModel('auto', base)).toEqual({ model: 'geco2', reason: 'geco2-fast' });
  });

  it('phones never run GeCo2 on WebGPU (it crashed both test phones): WASM 768', () => {
    expect(geco2Plan({ ...base, mobile: true })).toMatchObject({ backend: 'wasm', size: 768 });
    expect(geco2Plan({ ...base, geco2ForceWasm: true })).toMatchObject({ backend: 'wasm', size: 768 });
  });

  it('slow or memory-poor devices get LOCA automatically, measured times override the estimate', () => {
    const slowPhone: DeviceProfile = { ...base, mobile: true, cpuBenchMs: 100 };
    expect(resolveModel('auto', slowPhone)).toEqual({ model: 'loca', reason: 'geco2-slow' });
    expect(resolveModel('auto', { ...slowPhone, measured: { 'wasm-768': 16 } })).toEqual({ model: 'geco2', reason: 'geco2-fast' });
    expect(resolveModel('auto', { ...base, deviceMemoryGB: 2 })).toEqual({ model: 'loca', reason: 'low-memory' });
    expect(resolveModel('auto', { ...base, mobile: true, cpuBenchMs: null })).toEqual({ model: 'loca', reason: 'unknown-speed' });
    expect(resolveModel('geco2', slowPhone)).toEqual({ model: 'geco2', reason: null });
    expect(resolveModel('loca', base).model).toBe('loca');
  });

  it('warns when GeCo2 is chosen by hand on a slow or memory-poor device', () => {
    expect(heavyModelWarning(base)).toBeNull();
    const w = heavyModelWarning({ ...base, mobile: true, measured: { 'wasm-768': 60 } });
    expect(w).toEqual({ kind: 'slow', seconds: 60 });
    expect(heavyModelWarning({ ...base, deviceMemoryGB: 1 })).toEqual({ kind: 'memory' });
    expect(AUTO_MAX_SECONDS).toBeGreaterThan(16);
  });

  it('averages measured run times', () => {
    expect(updateMeasured(undefined, 20)).toBe(20);
    expect(updateMeasured(20, 10)).toBe(15);
  });
});

describe('GeCo2 model files (download manifest)', () => {
  it('lists three graphs per input size from the GeCo2 site plus the shared runtime', () => {
    for (const s of [768, 1024] as const) {
      const a = geco2Assets(s, 'https://example.test/app/');
      expect(a.map((x) => x.key)).toEqual(['backbone', 'exemplars', 'head', 'wasm']);
      for (const x of a.slice(0, 3)) {
        expect(x.url).toBe(`${GECO2_BASE}geco2-${x.key}-${s}-w8.onnx`);
        expect(x.sha256).toMatch(/^[0-9a-f]{64}$/);
        expect(x.bytes).toBeGreaterThan(50000);
      }
      expect(a[3]!.url.startsWith('https://example.test/')).toBe(true);
    }
  });
});

function detProject(): Project {
  const p = createProject({ name: 'x.jpg', width: 400, height: 200, originalWidth: 400, originalHeight: 200, mime: 'image/jpeg' });
  const result: AiDetectionResult = {
    kind: 'detections',
    modelId: 'geco2',
    model: 'GeCo2 test',
    engine: 'onnxruntime-web 1.30.0 · WASM',
    inputSize: 768,
    scale: 1,
    objects: [det(10, 10, 20), det(100, 10, 20), det(10, 100, 20, { review: 'rejected' }), newManualDetection(300, 150, 20, 20)],
    exemplarsUsed: [{ id: 'e1', x: 10, y: 10, w: 20, h: 20 }],
    inputExemplarIds: ['e1'],
    durationMs: 1000,
    at: 'now',
  };
  return { ...p, mode: 'ai', ai: { exemplars: [{ id: 'e1', x: 10, y: 10, w: 20, h: 20 }], result, corrections: [] } };
}

describe('GeCo2 results in exports', () => {
  it('CSV lists the counted objects numbered, with the model on every row', () => {
    const csv = toAiDetectionsCsv(detProject(), {
      decimal: '.',
      headers: { number: 'n', x: 'x', y: 'y', w: 'w', h: 'h', source: 's', review: 'r', score: 'sc', model: 'm' },
      sourceLabels: { auto: 'auto', manual: 'manual' },
      reviewLabels: { unreviewed: 'u', confirmed: 'c' },
    });
    const rows = csv.replace('﻿', '').trim().split('\r\n');
    expect(rows).toHaveLength(4);
    expect(rows[1]).toBe('1,20.0,20.0,20.0,20.0,auto,u,1.00,GeCo2 test');
    expect(rows[3]!.endsWith('manual,c,,GeCo2 test')).toBe(true);
  });

  it('JSON summary records the model and that it is a count, not an estimate; the file round-trips', () => {
    const p = detProject();
    expect(detectionSummary(p)).toEqual({ kind: 'detections', isEstimate: false, model: 'GeCo2 test', engine: 'onnxruntime-web 1.30.0 · WASM', count: 3, auto: 2, manual: 1 });
    const file = toProjectFile(p, 'data:image/jpeg;base64,AAAA');
    expect(file.summary.detections?.count).toBe(3);
    expect(file.summary.estimate).toBeUndefined();
    const back = parseProjectFile(JSON.stringify(file));
    expect(isDetectionResult(back.project.ai?.result)).toBe(true);
    expect(isDensityResult(back.project.ai?.result)).toBe(false);
    expect(summarize(back.project.ai!)).toBeNull();
  });

  it('rejects a detection result with malformed objects', () => {
    const p = detProject();
    (p.ai!.result as AiDetectionResult).objects.push({ id: 'bad', x: 1, y: 2, w: 'x', h: 3 } as unknown as AiDetection);
    expect(() => parseProjectFile(JSON.stringify(toProjectFile(p, 'data:image/jpeg;base64,AAAA')))).toThrow();
  });
});

const MODELS = process.env.GECO2_MODELS ?? '';
describe.skipIf(!MODELS || !existsSync(join(MODELS, 'geco2-backbone-768-w8.onnx')))('GeCo2 int8 graphs end to end (onnxruntime-web WASM)', () => {
  it('counts a synthetic rod scene exactly, with individually correct boxes', async () => {
    const ort = await import('onnxruntime-web');
    ort.env.wasm.numThreads = 1;
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'disabled' as const };
    const load = (k: string) => ort.InferenceSession.create(readFileSync(join(MODELS, `geco2-${k}-768-w8.onnx`)), opts);
    const [b, e, h] = [await load('backbone'), await load('exemplars'), await load('head')];
    const cv = await loadCv();
    const s = makeScene(cv, { name: 'g2', width: 480, height: 360, radii: [18], layout: 'hex', noiseSigma: 6, blur: 1, seed: 21 });
    const { width, height, data } = s.image;
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < data.length; i++) {
      rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = data[i]!;
      rgba[4 * i + 3] = 255;
    }
    const ex = [...s.truth].sort((a, c) => Math.hypot(a.x - 240, a.y - 180) - Math.hypot(c.x - 240, c.y - 180)).slice(0, 3).map((c, i) => ({ id: `e${i}`, x: c.x - c.r, y: c.y - c.r, w: 2 * c.r, h: 2 * c.r }));
    const scale = geco2Scale(width, height, ex, 768);
    expect(scale).toBeCloseTo(1.6); // longer side 480 → 768; 36 px exemplars → 58 px, below the 60 px cap
    // As the app does: resize to the content size, use the scale actually applied.
    const content = geco2ContentSize(width, height, scale, 768);
    const sx = content.w / width;
    const scaled = resizeRgba({ data: rgba, width, height }, content.w, content.h);
    const f = await b.run({ image: new ort.Tensor('float32', geco2Input(scaled, 768), [1, 3, 768, 768]) });
    const p = await e.run({ feat: f.feat!, fpn1: f.fpn1!, fpn0: f.fpn0!, boxes: new ort.Tensor('float32', geco2Boxes(ex, sx), [1, 3, 4]) });
    const o = await h.run({ feat: f.feat!, fpn1: f.fpn1!, fpn0: f.fpn0!, proto0: p.proto0!, proto1: p.proto1!, proto2: p.proto2! });
    const d = geco2Detect(o.objectness!.data as Float32Array, o.ltrb!.data as Float32Array, 384, 384, 768, content);
    const dets = toProjectDetections(d, sx);
    expect(dets.length).toBe(s.truth.length);
    const matched = s.truth.filter((t) => dets.some((q) => Math.hypot(q.x + q.w / 2 - t.x, q.y + q.h / 2 - t.y) < 0.5 * t.r)).length;
    expect(matched).toBe(s.truth.length);
  }, 300000);
});
