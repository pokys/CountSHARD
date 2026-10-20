import { describe, expect, it } from 'vitest';
import { EXEMPLAR_BAND, MODEL_INPUT, TILE_MARGIN, axisStarts, desiredScale, makeTiles, padExemplars, planTiles, tileCount } from '../../src/ai/tiling';
import { MEAN, STD, accumulateTile, fillTileTensor, gridSum, makeGrid, resizeRgba } from '../../src/ai/tensor';
import { type LocaBackend, runPlan, thresholdedSum } from '../../src/ai/runner';
import {
  addCorrection,
  addExemplar,
  decodeDensity,
  emptyAiState,
  encodeDensity,
  estimateTooLow,
  exemplarSizeRatio,
  exemplarsTooDifferent,
  normalizeRect,
  removeCorrection,
  resultIsStale,
  roundEstimate,
  setCorrectionCount,
  sumInRect,
  summarize,
  validateCorrection,
} from '../../src/ai/estimate';
import { heatColor, heatRgba } from '../../src/ai/heatmap';
import type { AiDensityResult, AiState, Project } from '../../src/model/types';
import { createProject } from '../../src/state/project';
import { estimateSummary, parseProjectFile, toAiCsv, toProjectFile } from '../../src/io/serialize';

const box = (x: number, y: number, s: number, id = `e${x}_${y}`) => ({ id, x, y, w: s, h: s });

describe('tiling (pre-processing)', () => {
  it('owned areas of the tiles partition the scaled image exactly', () => {
    for (const [w, h] of [[300, 200], [512, 512], [513, 400], [1500, 900], [2000, 1600], [700, 3000]] as const) {
      const tiles = makeTiles(w, h);
      const cover = new Uint8Array(w * h);
      for (const t of tiles) {
        expect(t.own.x0).toBeGreaterThanOrEqual(t.x);
        expect(t.own.x1).toBeLessThanOrEqual(t.x + MODEL_INPUT);
        expect(t.own.y0).toBeGreaterThanOrEqual(t.y);
        expect(t.own.y1).toBeLessThanOrEqual(t.y + MODEL_INPUT);
        for (let y = t.own.y0; y < t.own.y1; y++) for (let x = t.own.x0; x < t.own.x1; x++) cover[y * w + x]!++;
      }
      expect(cover.every((c) => c === 1)).toBe(true);
    }
  });

  it('interior tile borders keep at least the margin of context', () => {
    const tiles = makeTiles(2000, 1200);
    for (const t of tiles) {
      if (t.own.x0 > 0) expect(t.own.x0 - t.x).toBeGreaterThanOrEqual(TILE_MARGIN);
      if (t.own.x1 < 2000) expect(t.x + MODEL_INPUT - t.own.x1).toBeGreaterThanOrEqual(TILE_MARGIN);
      if (t.own.y0 > 0) expect(t.own.y0 - t.y).toBeGreaterThanOrEqual(TILE_MARGIN);
    }
  });

  it('a small image is one letterboxed tile; tile starts stay inside the image', () => {
    expect(axisStarts(400)).toEqual([0]);
    expect(axisStarts(512)).toEqual([0]);
    const s = axisStarts(1000);
    expect(s[0]).toBe(0);
    expect(s[s.length - 1]).toBe(1000 - MODEL_INPUT);
  });

  it('keeps the aspect ratio (uniform scale), unlike squashing to 512 × 512', () => {
    const p = planTiles(3000, 1000, [box(100, 100, 60)]);
    expect(p.width / p.height).toBeCloseTo(3, 1);
  });

  it('scales so that the exemplar size falls into the training band', () => {
    // 3000 px photo, 30 px objects: fitting into 512 would make them 5 px → upscale to band[0].
    const s = desiredScale(3000, 2000, [box(0, 0, 30)]);
    expect(30 * s).toBeCloseTo(EXEMPLAR_BAND[0], 5);
    // Huge objects are shrunk to band[1].
    const s2 = desiredScale(800, 600, [box(0, 0, 500)]);
    expect(500 * s2).toBeCloseTo(EXEMPLAR_BAND[1], 5);
    // Within the band: just fit the photo into one tile.
    expect(desiredScale(1024, 768, [box(0, 0, 100)])).toBeCloseTo(MODEL_INPUT / 1024, 6);
  });

  it('respects the tile limit and reports it', () => {
    const p = planTiles(4000, 3000, [box(10, 10, 12)], { maxTiles: 9 });
    expect(p.tiles.length).toBeLessThanOrEqual(9);
    expect(tileCount(p.width, p.height)).toBe(p.tiles.length);
    expect(p.limitedByTiles).toBe(true);
    expect(p.scale).toBeLessThan(p.desiredScale);
  });

  it('puts the exemplars into one window and pads them to three boxes', () => {
    const ex = [box(900, 500, 40, 'a'), box(960, 520, 40, 'b')];
    const p = planTiles(2000, 1500, ex);
    expect(p.exemplarsUsed).toEqual([0, 1]);
    expect(p.exemplarBoxes).toHaveLength(3);
    expect(p.exemplarBoxes[2]).toEqual(p.exemplarBoxes[0]);
    for (const [x1, y1, x2, y2] of p.exemplarBoxes) {
      expect(x1).toBeGreaterThanOrEqual(0);
      expect(y2).toBeLessThanOrEqual(MODEL_INPUT);
      expect(x2 - x1).toBeCloseTo(40 * p.scale, 5);
      expect(y2 - y1).toBeCloseTo(40 * p.scale, 5);
    }
    // Box position inside the window matches its position in the scaled image.
    expect(p.exemplarBoxes[0]![0] + p.exemplarWindow.x).toBeCloseTo(900 * p.scale, 5);
  });

  it('drops exemplars that cannot share one window with the others', () => {
    const p = planTiles(6000, 1000, [box(100, 100, 30, 'a'), box(5800, 800, 30, 'b'), box(150, 120, 30, 'c')], { maxTiles: 64 });
    expect(p.exemplarsUsed.sort()).toEqual([0, 2]);
  });

  it('padExemplars repeats cyclically and refuses zero boxes', () => {
    expect(padExemplars(['A'])).toEqual(['A', 'A', 'A']);
    expect(padExemplars(['A', 'B'])).toEqual(['A', 'B', 'A']);
    expect(() => padExemplars([])).toThrow();
  });
});

describe('tensors and stitching', () => {
  it('normalises with ImageNet statistics, NCHW, and letterboxes outside the image with 0', () => {
    const img = { data: new Uint8ClampedArray(4 * 4 * 4).fill(255), width: 4, height: 4 };
    const t = fillTileTensor(img, -2, 0);
    const plane = MODEL_INPUT * MODEL_INPUT;
    expect(t[0]).toBe(0); // x = -2 → padding
    expect(t[2]).toBeCloseTo((1 - MEAN[0]) / STD[0], 5); // x = 0 of the image, red
    expect(t[plane + 2]).toBeCloseTo((1 - MEAN[1]) / STD[1], 5);
    expect(t[2 * plane + 2]).toBeCloseTo((1 - MEAN[2]) / STD[2], 5);
    expect(t[6]).toBe(0); // x = 4 is past the right edge
    expect(t[4 * MODEL_INPUT + 2]).toBe(0); // y = 4 is past the bottom
  });

  it('stitching keeps the sum of every owned density value exactly once', async () => {
    const plan = planTiles(1300, 900, [box(100, 100, 20)]);
    expect(plan.tiles.length).toBeGreaterThan(1);
    // Backend that puts density 1/1000 on every pixel of every tile, letterbox included.
    const backend: LocaBackend = {
      features: async () => new Float32Array(256 * 64 * 64),
      head: async () => new Float32Array(MODEL_INPUT * MODEL_INPUT).fill(0.001),
    };
    const scaled = { data: new Uint8ClampedArray(plan.width * plan.height * 4), width: plan.width, height: plan.height };
    const progress: number[] = [];
    const res = await runPlan(backend, scaled, plan, (f) => progress.push(f));
    expect(res.total).toBeCloseTo(plan.width * plan.height * 0.001, 1);
    expect(gridSum(res.grid.data)).toBeCloseTo(res.total, 1);
    expect(progress[progress.length - 1]).toBe(1);
  });

  it('accumulateTile adds only the owned area', () => {
    const grid = makeGrid(600, 512, 4);
    const tile = { x: 0, y: 0, own: { x0: 0, y0: 0, x1: 300, y1: 512 } };
    const d = new Float32Array(MODEL_INPUT * MODEL_INPUT).fill(1);
    expect(accumulateTile(grid, tile, d)).toBe(300 * 512);
    expect(gridSum(grid.data)).toBe(300 * 512);
  });

  it('resizeRgba keeps a flat colour and the requested size', () => {
    const img = { data: new Uint8ClampedArray(10 * 6 * 4).fill(200), width: 10, height: 6 };
    for (const [w, h] of [[5, 3], [20, 12], [7, 9]] as const) {
      const r = resizeRgba(img, w, h);
      expect(r.width).toBe(w);
      expect(r.data.length).toBe(w * h * 4);
      expect(Math.min(...r.data)).toBe(200);
    }
  });

  it('thresholdedSum drops values at or below 5 % of the maximum', () => {
    expect(thresholdedSum([1, 0.05, 0.04, 0.5])).toBeCloseTo(1.5, 6);
  });
});

function fakeResult(gridW: number, gridH: number, cellPx: number, fill: (x: number, y: number) => number, extra: Partial<AiDensityResult> = {}): AiDensityResult {
  const d = new Float32Array(gridW * gridH);
  for (let y = 0; y < gridH; y++) for (let x = 0; x < gridW; x++) d[y * gridW + x] = fill(x, y);
  return {
    kind: 'density-estimate',
    model: 'test',
    engine: 'test',
    cellPx,
    gridW,
    gridH,
    density: encodeDensity(d),
    rawTotal: d.reduce((a, b) => a + b, 0),
    tiles: 1,
    scale: 1,
    limitedByTiles: false,
    exemplarsUsed: [],
    inputExemplarIds: [],
    durationMs: 1,
    at: '2026-10-20T00:00:00Z',
    ...extra,
  };
}

describe('estimate and corrections', () => {
  it('density survives base64 round trip', () => {
    const d = new Float32Array([0, 1.5, -0.25, 3e-6, 1234.5]);
    expect([...decodeDensity(encodeDensity(d))]).toEqual([...d]);
  });

  it('sumInRect weights partly covered cells by area', () => {
    const res = fakeResult(10, 10, 10, () => 1); // 100 × 100 px, 1 object per 10 × 10 cell
    const data = decodeDensity(res.density);
    expect(sumInRect(res, data, { x: 0, y: 0, w: 100, h: 100 })).toBeCloseTo(100, 6);
    expect(sumInRect(res, data, { x: 5, y: 5, w: 10, h: 10 })).toBeCloseTo(1, 6);
    expect(sumInRect(res, data, { x: 0, y: 0, w: 25, h: 10 })).toBeCloseTo(2.5, 6);
    expect(sumInRect(res, data, { x: 95, y: 95, w: 50, h: 50 })).toBeCloseTo(0.25, 6); // clipped at the edge
  });

  it('a corrected region replaces the model sum with the user count', () => {
    const res = fakeResult(10, 10, 10, (x) => (x < 5 ? 2 : 0)); // all density in the left half: 100
    let ai: AiState = { exemplars: [], result: res, corrections: [] };
    expect(summarize(ai)!.corrected).toBeCloseTo(100, 6);
    const r1 = addCorrection(ai, { id: 'r1', x: 0, y: 0, w: 50, h: 50, userCount: 30 }); // model says 50 there
    expect(r1.error).toBeNull();
    ai = r1.ai;
    const s = summarize(ai)!;
    expect(s.raw).toBeCloseTo(100, 6);
    expect(s.regions[0]!.model).toBeCloseTo(50, 6);
    expect(s.corrected).toBeCloseTo(80, 6);
    expect(s.userCounted).toBe(30);
    // Editing the count and removing the region.
    expect(summarize(setCorrectionCount(ai, 'r1', 50))!.corrected).toBeCloseTo(100, 6);
    expect(setCorrectionCount(ai, 'r1', -1)).toBe(ai);
    expect(summarize(removeCorrection(ai, 'r1'))!.corrected).toBeCloseTo(100, 6);
  });

  it('overlapping, tiny and non-integer corrections are refused', () => {
    const existing = [{ id: 'a', x: 0, y: 0, w: 50, h: 50, userCount: 3 }];
    expect(validateCorrection(existing, { x: 40, y: 40, w: 30, h: 30, userCount: 1 })).toBe('overlap');
    expect(validateCorrection(existing, { x: 50, y: 0, w: 30, h: 30, userCount: 1 })).toBeNull(); // touching is fine
    expect(validateCorrection(existing, { x: 60, y: 0, w: 4, h: 30, userCount: 1 })).toBe('too-small');
    expect(validateCorrection(existing, { x: 60, y: 0, w: 30, h: 30, userCount: 1.5 })).toBe('bad-count');
    const ai: AiState = { exemplars: [], result: null, corrections: existing };
    expect(addCorrection(ai, { id: 'b', x: 10, y: 10, w: 20, h: 20, userCount: 0 }).ai).toBe(ai);
  });

  it('the corrected estimate is never negative', () => {
    const res = fakeResult(4, 4, 10, () => 0.5, { rawTotal: 2 }); // inconsistent on purpose
    const ai: AiState = { exemplars: [], result: res, corrections: [{ id: 'r', x: 0, y: 0, w: 40, h: 40, userCount: 0 }] };
    expect(summarize(ai)!.corrected).toBe(0);
  });

  it('normalizeRect handles any drag direction and clips to the image', () => {
    expect(normalizeRect(50, 40, 10, -5, 45, 100)).toEqual({ x: 10, y: 0, w: 35, h: 40 });
  });

  it('keeps at most three exemplars and detects a stale result', () => {
    let ai = emptyAiState();
    for (let i = 0; i < 4; i++) ai = addExemplar(ai, box(i * 10, 0, 5, `e${i}`));
    expect(ai.exemplars.map((e) => e.id)).toEqual(['e1', 'e2', 'e3']);
    ai = { ...ai, result: fakeResult(2, 2, 1, () => 1, { exemplarsUsed: ai.exemplars, inputExemplarIds: ai.exemplars.map((e) => e.id) }) };
    expect(resultIsStale(ai)).toBe(false);
    expect(resultIsStale({ ...ai, exemplars: ai.exemplars.slice(1) })).toBe(true);
    expect(resultIsStale({ ...ai, exemplars: ai.exemplars.map((e, i) => (i ? e : { ...e, x: e.x + 1 })) })).toBe(true);
  });

  it('rounds estimates for display only', () => {
    expect(roundEstimate(162.49)).toBe(162);
    expect(roundEstimate(-0.4)).toBe(0);
  });
});

describe('heat map', () => {
  it('colour ramp is monotone in brightness and alpha follows density', () => {
    const lum = (t: number) => heatColor(t).reduce((a, b) => a + b, 0);
    for (let t = 0; t < 1; t += 0.1) expect(lum(t + 0.1)).toBeGreaterThanOrEqual(lum(t));
    const px = heatRgba([0, 0.5, 1, 0], 4, 1);
    expect(px[3]).toBe(0);
    expect(px[7]!).toBeGreaterThan(0);
    expect(px[11]!).toBeGreaterThan(px[7]!);
  });
});

describe('Mode B export', () => {
  const project = (): Project => {
    const p = createProject({ name: 'x.jpg', width: 100, height: 100, originalWidth: 100, originalHeight: 100, mime: 'image/jpeg' });
    const res = fakeResult(10, 10, 10, () => 1.25);
    return { ...p, mode: 'ai', ai: { exemplars: [box(10, 10, 8, 'e1')], result: { ...res, exemplarsUsed: [box(10, 10, 8, 'e1')], inputExemplarIds: ['e1'] }, corrections: [{ id: 'r', x: 0, y: 0, w: 20, h: 20, userCount: 7 }] } };
  };

  it('CSV marks every number as an estimate and lists corrections', () => {
    const csv = toAiCsv(project(), {
      decimal: ',',
      headers: { type: 'Typ', region: 'Oblast', x: 'X', y: 'Y', w: 'W', h: 'H', model: 'Odhad modelu', user: 'Zadaný počet', result: 'Výsledek' },
      rows: { whole: 'ODHAD – celý snímek (AI)', corrected: 'ODHAD po opravách', regionRow: 'Oprava oblasti', exemplarRow: 'Vzor' },
    });
    const lines = csv.replace('﻿', '').trim().split('\r\n').map((l) => l.split(';'));
    expect(lines[1]![0]).toBe('ODHAD – celý snímek (AI)');
    expect(lines[1]![6]).toBe('125,0');
    expect(lines[2]).toEqual(['Oprava oblasti', '1', '0', '0', '20', '20', '5,0', '7', '']);
    expect(lines[3]![0]).toBe('ODHAD po opravách');
    expect(lines[3]![8]).toBe('127'); // 125 − 5 + 7
    expect(lines[4]![0]).toBe('Vzor');
  });

  it('JSON summary says it is an estimate and the file round-trips', () => {
    const p = project();
    const s = estimateSummary(p)!;
    expect(s.isEstimate).toBe(true);
    expect(s.correctedEstimate).toBeCloseTo(127, 6);
    const file = toProjectFile(p, 'data:image/jpeg;base64,AAAA');
    expect(file.summary.mode).toBe('ai');
    expect(file.summary.total).toBe(0); // no individually counted objects
    const back = parseProjectFile(JSON.stringify(file));
    expect(summarize(back.project.ai!)!.corrected).toBeCloseTo(127, 6);
  });

  it('rejects a project file whose density grid does not match its size', () => {
    const p = project();
    const bad = { ...p, ai: { ...p.ai!, result: { ...p.ai!.result!, gridW: 11 } } };
    expect(() => parseProjectFile(JSON.stringify(toProjectFile(bad, 'data:image/jpeg;base64,AAAA')))).toThrow('ai');
  });

  it('old projects without a mode still parse as round-object projects', () => {
    const p = createProject({ name: 'x.jpg', width: 10, height: 10, originalWidth: 10, originalHeight: 10, mime: 'image/jpeg' });
    const file = toProjectFile(p, 'data:image/jpeg;base64,AAAA');
    expect(parseProjectFile(JSON.stringify(file)).summary.mode).toBe('circles');
  });
});

describe('Mode B crash detection', () => {
  it('a run marker left behind is reported once, and a WebGPU crash switches the device to WASM', async () => {
    const store = new Map<string, string>();
    (globalThis as { localStorage?: unknown }).localStorage = {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
    };
    const { takeAiCrashReport } = await import('../../src/ai/client');
    expect(takeAiCrashReport()).toBeNull();
    store.set('countshard.ai.running', JSON.stringify({ backend: 'wasm', at: 'x' }));
    expect(takeAiCrashReport()).toEqual({ backend: 'wasm', at: 'x' });
    expect(store.has('countshard.ai.forceWasm')).toBe(false);
    expect(takeAiCrashReport()).toBeNull();
    store.set('countshard.ai.running', JSON.stringify({ backend: 'webgpu', at: 'y' }));
    expect(takeAiCrashReport()?.backend).toBe('webgpu');
    expect(store.get('countshard.ai.forceWasm')).toBe('1');
    delete (globalThis as { localStorage?: unknown }).localStorage;
  });
});

describe('reliability warnings', () => {
  it('flags exemplars of very different sizes', () => {
    // From a user test: 48 × 64 px vs. a large box including the side of the profile.
    const a = { w: 48, h: 64 };
    const b = { w: 240, h: 170 };
    expect(exemplarSizeRatio([a, b])).toBeGreaterThan(3);
    expect(exemplarsTooDifferent([a, b])).toBe(true);
    expect(exemplarsTooDifferent([a, { w: 52, h: 60 }, { w: 60, h: 70 }])).toBe(false);
    expect(exemplarsTooDifferent([a])).toBe(false);
  });

  it('flags small estimates as unreliable', () => {
    expect(estimateTooLow({ corrected: 8.4 })).toBe(true);
    expect(estimateTooLow({ corrected: 14.9 })).toBe(true);
    expect(estimateTooLow({ corrected: 37 })).toBe(false);
    expect(estimateTooLow(null)).toBe(false);
  });
});

describe('summarize with a counting area', () => {
  it('counts only the density inside the area and ignores corrections outside it', () => {
    // 4×4 grid of 10 px cells, density 1 per cell → total 16.
    const density = encodeDensity(new Float32Array(16).fill(1));
    const ai = {
      exemplars: [],
      corrections: [
        { id: 'in', x: 0, y: 0, w: 10, h: 10, userCount: 5, modelAtCreation: 1 },
        { id: 'out', x: 30, y: 30, w: 10, h: 10, userCount: 9, modelAtCreation: 1 },
      ],
      result: { density, gridW: 4, gridH: 4, cellPx: 10, rawTotal: 16 },
    } as unknown as Parameters<typeof summarize>[0];
    const all = summarize(ai)!;
    expect(all.raw).toBeCloseTo(16);
    const left = summarize(ai, { x: 0, y: 0, w: 20, h: 40 })!; // left half: 8 cells
    expect(left.raw).toBeCloseTo(8);
    expect(left.regions.map((r) => r.id)).toEqual(['in']);
    expect(left.corrected).toBeCloseTo(8 - 1 + 5);
  });
});

describe('summarize with a lasso area', () => {
  it('counts density inside the polygon only', () => {
    // 4×4 grid of 10 px cells, density 1 per cell; triangle = lower-left half of the 40×40 image.
    const density = encodeDensity(new Float32Array(16).fill(1));
    const ai = { exemplars: [], corrections: [], result: { density, gridW: 4, gridH: 4, cellPx: 10, rawTotal: 16 } } as unknown as Parameters<typeof summarize>[0];
    const tri = { x: 0, y: 0, w: 40, h: 40, poly: [[0, 0], [0, 40], [40, 40]] as [number, number][] };
    const half = summarize(ai, tri)!;
    expect(half.raw).toBeGreaterThan(7);
    expect(half.raw).toBeLessThan(9);
  });
});
