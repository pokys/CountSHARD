/**
 * Steel-profile detector: pre/post-processing (pure) and, when the model file is present, an end
 * to end run with onnxruntime-web (WASM) in Node. YOLO_SAMPLE=<jpeg> adds a run on a real photo
 * (prints the count per class; not asserted, photos are not part of the repository).
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import jpeg from 'jpeg-js';
import { projectTotal } from '../../src/io/db';
import {
  type YoloBox,
  YOLO_CLASSES,
  YOLO_SIZE,
  dropOversized,
  innerEdges,
  minScoreFromSensitivity,
  sensitivityFromMinScore,
  letterbox,
  tilePlan,
  yoloDecode,
  yoloDetect,
  yoloInput,
  yoloSuppress,
} from '../../src/ai/yolo';
import { classGrouping, classGroups, detectionClassAt, isUncertain, shownDetections, toProjectDetections } from '../../src/ai/geco2';
import { autoProjectName, createProject, migrateMode, switchMode } from '../../src/state/project';
import { detectionSummary, parseProjectFile, toAiDetectionsCsv, toProjectFile } from '../../src/io/serialize';

/** Builds a raw [4 + nc, n] output from (cx, cy, w, h, cls, score) rows. */
function raw(rows: [number, number, number, number, number, number][], nc = 4): Float32Array {
  const n = rows.length;
  const out = new Float32Array((4 + nc) * n);
  rows.forEach(([cx, cy, w, h, c, s], i) => {
    out[i] = cx;
    out[n + i] = cy;
    out[2 * n + i] = w;
    out[3 * n + i] = h;
    out[(4 + c) * n + i] = s;
  });
  return out;
}

const box = (x1: number, y1: number, x2: number, y2: number, score = 0.9, cls = 0): YoloBox => ({ b: [x1, y1, x2, y2], score, cls });

describe('YOLO pre/post-processing', () => {
  it('letterboxes a landscape tile centred vertically', () => {
    const lb = letterbox(1280, 640);
    expect(lb.scale).toBe(0.5);
    expect(lb.padX).toBe(0);
    expect(lb.padY).toBe(160);
  });

  it('fills the letterbox with grey and the content with RGB / 255', () => {
    const img = { data: new Uint8ClampedArray([255, 0, 0, 255, 255, 0, 0, 255]), width: 2, height: 1 };
    const { tensor, lb } = yoloInput(img, { x: 0, y: 0, w: 2, h: 1 });
    const plane = YOLO_SIZE * YOLO_SIZE;
    expect(tensor[0]).toBeCloseTo(114 / 255);
    const d = (lb.padY + 1) * YOLO_SIZE + 10;
    expect(tensor[d]).toBeCloseTo(1);
    expect(tensor[plane + d]).toBeCloseTo(0);
  });

  it('decodes boxes back to image pixels, best class above the threshold', () => {
    const lb = letterbox(1280, 640);
    const out = yoloDecode(raw([[320, 320, 40, 20, 2, 0.8], [100, 330, 10, 10, 1, 0.1]]), 2, 4, { x: 0, y: 0, w: 1280, h: 640 }, lb);
    expect(out).toHaveLength(1);
    expect(out[0]!.cls).toBe(2);
    expect(out[0]!.b.map((v) => Math.round(v))).toEqual([600, 300, 680, 340]);
  });

  it('drops boxes touching an inner tile edge and shifts the rest by the tile offset', () => {
    const t = { x: 500, y: 0, w: 640, h: 640 };
    const lb = letterbox(640, 640);
    const out = yoloDecode(raw([[5, 300, 20, 20, 0, 0.9], [300, 300, 20, 20, 0, 0.9]]), 2, 4, t, lb, innerEdges(t, 2000, 640));
    expect(out).toHaveLength(1);
    expect(out[0]!.b[0]).toBeCloseTo(790);
  });

  it('suppresses duplicates and boxes nested in a stronger one', () => {
    const kept = yoloSuppress([box(0, 0, 100, 100, 0.9), box(2, 2, 101, 99, 0.8), box(10, 10, 60, 60, 0.7), box(200, 0, 300, 100, 0.6)]);
    expect(kept.map((k) => k.score)).toEqual([0.9, 0.6]);
  });

  it('drops a box around several items of the same class', () => {
    const items = Array.from({ length: 6 }, (_, i) => box(i * 30, 0, i * 30 + 25, 25));
    expect(dropOversized([...items, box(0, 0, 180, 60, 0.5)])).toHaveLength(6);
  });

  it('tiles only when the items are small in the overview', () => {
    // 4000 px photo: 640/4000 = 0.16; 100 px items → 16 model px → tiled.
    const small = Array.from({ length: 10 }, (_, i) => box(i * 120, 0, i * 120 + 100, 100));
    const tiles = tilePlan(4000, 3000, small)!;
    expect(tiles.length).toBeGreaterThan(1);
    expect(tiles.length).toBeLessThanOrEqual(16);
    // Tiles cover the photo.
    expect(Math.min(...tiles.map((t) => t.x))).toBe(0);
    expect(Math.max(...tiles.map((t) => t.x + t.w))).toBe(4000);
    const big = Array.from({ length: 10 }, (_, i) => box(i * 400, 0, i * 400 + 380, 380));
    expect(tilePlan(4000, 3000, big)).toBeNull();
  });

  it('keeps the class of each detection and groups by class', () => {
    const d = toProjectDetections({ boxes: new Float32Array([0, 0, 10, 10, 20, 0, 30, 10]), scores: new Float32Array([0.9, 0.8]), classes: new Uint8Array([3, 0]) }, 1);
    expect(d.map((x) => x.cls)).toEqual([3, 0]);
    expect(detectionClassAt(d, 26, 4)).toBe(0);
    const g = classGroups(d);
    expect([...g.keys()].sort()).toEqual([0, 3]);
    const cg = classGrouping(d, d.map((x) => x.id), (c) => `c${c}`, ['red', 'blue']);
    expect(cg.groups.map((x) => [x.label, x.color])).toEqual([['c0', 'red'], ['c3', 'blue']]);
    expect(classGrouping(d.slice(0, 1), [d[0]!.id], String, ['red']).groups).toHaveLength(0);
  });

  it('exports the profile type in the CSV and the project summary, and reads it back', () => {
    const p = createProject({ name: 'x.jpg', width: 400, height: 200, originalWidth: 400, originalHeight: 200, mime: 'image/jpeg' });
    const objects = toProjectDetections({ boxes: new Float32Array([0, 0, 10, 10, 20, 0, 30, 10, 50, 0, 60, 10]), scores: new Float32Array([0.9, 0.8, 0.7]), classes: new Uint8Array([3, 0, 3]) }, 1);
    p.mode = 'ai';
    p.ai = {
      exemplars: [],
      corrections: [],
      result: { kind: 'detections', modelId: 'profiles', model: 'profiles test', engine: 'x', inputSize: 640, scale: 1, objects, classes: [...YOLO_CLASSES], exemplarsUsed: [], inputExemplarIds: [], durationMs: 1, at: '' },
    };
    expect(detectionSummary(p)?.byClass).toEqual({ channel: 2, circle: 1 });
    const csv = toAiDetectionsCsv(p, {
      decimal: '.',
      headers: { number: 'n', x: 'x', y: 'y', w: 'w', h: 'h', source: 's', review: 'r', score: 'sc', model: 'm', type: 'type' },
      sourceLabels: { auto: 'auto', manual: 'manual' },
      reviewLabels: { unreviewed: 'u', confirmed: 'c' },
      classLabels: ['Pipes', 'Square tubes', 'Angles', 'U channels'],
    });
    const lines = csv.trim().split('\r\n');
    expect(lines[0]).toContain(',type,');
    expect(lines.filter((l) => l.includes('U channels'))).toHaveLength(2);
    const back = parseProjectFile(JSON.stringify(toProjectFile(p, 'data:image/jpeg;base64,AAAA')));
    expect((back.project.ai!.result as { objects: { cls?: number }[] }).objects.map((o) => o.cls)).toEqual([3, 0, 3]);
  });

  it('keeps separate state for the steel-profiles and generic AI modes', () => {
    const p = createProject({ name: 'x.jpg', width: 400, height: 200, originalWidth: 400, originalHeight: 200, mime: 'image/jpeg' });
    const prof = { exemplars: [], corrections: [], result: null };
    const gen = { exemplars: [{ id: 'e', x: 1, y: 1, w: 5, h: 5 }], corrections: [], result: null };
    let q = switchMode({ ...p, mode: 'profiles', aiFor: 'profiles', ai: prof }, 'circles');
    expect(q.ai).toBe(prof);
    q = switchMode(q, 'ai');
    expect(q.ai).toBeNull();
    expect(q.aiOther).toBe(prof);
    q = switchMode({ ...q, ai: gen }, 'profiles');
    expect(q.ai).toBe(prof);
    expect(q.aiOther).toBe(gen);
    expect(q.aiFor).toBe('profiles');
    // A new project in the profiles mode without an explicit owner.
    const n = switchMode({ ...p, mode: 'profiles', ai: prof }, 'ai');
    expect(n.ai).toBeNull();
    expect(n.aiOther).toBe(prof);
  });

  it('moves a steel-profiles result saved in the generic AI mode to its own mode', () => {
    const p = createProject({ name: 'x.jpg', width: 400, height: 200, originalWidth: 400, originalHeight: 200, mime: 'image/jpeg' });
    const result = { kind: 'detections' as const, modelId: 'profiles', model: 'm', engine: 'x', inputSize: 640, scale: 1, objects: [], exemplarsUsed: [], inputExemplarIds: [], durationMs: 1, at: '' };
    const m = migrateMode({ ...p, mode: 'ai', ai: { exemplars: [], corrections: [], result } });
    expect(m.mode).toBe('profiles');
    expect(m.aiFor).toBe('profiles');
    expect(migrateMode({ ...p, mode: 'ai' }).mode).toBe('ai');
  });
});

const MODEL = join(__dirname, '../../public/models/profiles/steel-profiles-yolov8n.onnx');
describe.skipIf(!existsSync(MODEL))('steel-profile model end to end (onnxruntime-web WASM)', () => {
  async function session() {
    const ort = await import('onnxruntime-web');
    ort.env.wasm.numThreads = 1;
    const s = await ort.InferenceSession.create(readFileSync(MODEL), { executionProviders: ['wasm'] });
    return async (input: Float32Array) => {
      const o = await s.run({ [s.inputNames[0]!]: new ort.Tensor('float32', input, [1, 3, YOLO_SIZE, YOLO_SIZE]) });
      const t = o[s.outputNames[0]!]!;
      const [, c, n] = t.dims as number[];
      return { data: t.data as Float32Array, nc: c! - 4, n: n! };
    };
  }

  it('has the expected output shape and finds nothing in a blank image', async () => {
    const run = await session();
    const img = { data: new Uint8ClampedArray(800 * 600 * 4).fill(128), width: 800, height: 600 };
    const r = await run(yoloInput(img, { x: 0, y: 0, w: 800, h: 600 }).tensor);
    expect(r.nc).toBe(YOLO_CLASSES.length);
    expect(r.n).toBe(8400);
    const d = await yoloDetect(img, run);
    expect(d.scores.length).toBe(0);
  }, 120000);

  const SAMPLE = process.env.YOLO_SAMPLE ?? '';
  it.skipIf(!SAMPLE || !existsSync(SAMPLE))('runs on a real photo', async () => {
    const run = await session();
    const j = jpeg.decode(readFileSync(SAMPLE), { useTArray: true, maxMemoryUsageInMB: 1024 });
    const t0 = performance.now();
    const d = await yoloDetect({ data: j.data, width: j.width, height: j.height }, run);
    const per = YOLO_CLASSES.map((c, i) => `${c}=${d.classes.filter((k) => k === i).length}`).join(' ');
    console.log(`${SAMPLE}: ${d.scores.length} items (${per}), ${d.runs} runs, ${Math.round(performance.now() - t0)} ms`);
    if (process.env.YOLO_DUMP) console.log(JSON.stringify(Array.from(d.boxes).map((v) => Math.round(v))));
    expect(d.scores.length).toBeGreaterThan(0);
  }, 300000);
});

describe('generated project names', () => {
  it('reads what, how many and when', () => {
    const iso = new Date(2026, 9, 25, 8, 5).toISOString();
    expect(autoProjectName('U profily 27 ks', iso)).toBe('U profily 27 ks – 25.10. 8:05');
  });
});

describe('project list count', () => {
  it('counts the mode the project was left in', () => {
    const p = createProject({ name: 'x.jpg', width: 400, height: 200, originalWidth: 400, originalHeight: 200, mime: 'image/jpeg' });
    expect(projectTotal(p)).toEqual({ total: null, estimate: false });
    const objects = toProjectDetections({ boxes: new Float32Array([0, 0, 10, 10, 20, 0, 30, 10]), scores: new Float32Array([0.9, 0.8]), classes: new Uint8Array([3, 3]) }, 1);
    const result = { kind: 'detections' as const, modelId: 'profiles', model: 'm', engine: 'x', inputSize: 640, scale: 1, objects, exemplarsUsed: [], inputExemplarIds: [], durationMs: 1, at: '' };
    const q = { ...p, mode: 'profiles' as const, ai: { exemplars: [], corrections: [], result } };
    expect(projectTotal(q)).toEqual({ total: 2, estimate: false });
    expect(projectTotal({ ...q, mode: 'circles' })).toEqual({ total: null, estimate: false });
  });
});

describe('steel-profile sensitivity and review', () => {
  it('maps the slider to a score threshold and back, default 0.25', () => {
    expect(minScoreFromSensitivity(70)).toBeCloseTo(0.25);
    expect(minScoreFromSensitivity(0)).toBeCloseTo(0.6);
    expect(minScoreFromSensitivity(100)).toBeCloseTo(0.1);
    for (const s of [0, 20, 55, 70, 85, 100]) expect(sensitivityFromMinScore(minScoreFromSensitivity(s))).toBe(s);
  });

  it('hides unsure automatic items below the threshold but keeps the user’s ones', () => {
    const d = toProjectDetections({ boxes: new Float32Array([0, 0, 10, 10, 20, 0, 30, 10, 40, 0, 50, 10]), scores: new Float32Array([0.9, 0.3, 0.15]), classes: new Uint8Array([0, 0, 0]) }, 1);
    d[2] = { ...d[2]!, review: 'confirmed' };
    expect(shownDetections({ objects: d, minScore: 0.25 }).map((x) => Math.round(x.score! * 100))).toEqual([90, 30, 15]);
    expect(shownDetections({ objects: d, minScore: 0.5 }).map((x) => Math.round(x.score! * 100))).toEqual([90, 15]);
    expect(shownDetections({ objects: d }).length).toBe(3);
    expect(d.map(isUncertain)).toEqual([false, true, false]);
  });
});
