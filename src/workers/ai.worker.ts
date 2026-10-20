/// <reference lib="webworker" />
/**
 * Mode B worker: owns onnxruntime-web and the sessions of ONE model at a time (LOCA: two graphs;
 * GeCo2: three). Loading another model releases the previous sessions first (phones cannot hold
 * both). Model files and the WASM runtime are read from Cache Storage (downloaded on demand),
 * so this works offline. Cancellation = the client terminates the worker.
 */
import * as ort from 'onnxruntime-web';
import { readCached } from '../ai/modelStore';
import { type LocaBackend, runPlan } from '../ai/runner';
import { type Geco2Size, geco2Detect, geco2Input } from '../ai/geco2';
import type { AiModelSpec, AiWorkerRequest, AiWorkerResponse } from './aiProtocol';

declare const self: DedicatedWorkerGlobalScope;

type Backend = 'webgpu' | 'wasm';
type Loaded =
  | { model: 'loca'; key: string; features: ort.InferenceSession; head: ort.InferenceSession; backend: Backend }
  | { model: 'geco2'; key: string; size: Geco2Size; backbone: ort.InferenceSession; exemplars: ort.InferenceSession; head: ort.InferenceSession; backend: Backend };

let loaded: Loaded | null = null;

const post = (msg: AiWorkerResponse, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

async function create(model: ArrayBuffer, ep: Backend) {
  // No graph optimisation: the int8 weights are not constant-folded into fp32 copies (WASM in
  // Node: 307 MB RSS after a LOCA features run vs 347 MB with 'all', same speed). WebGPU then runs
  // DequantizeLinear itself; checked to give the same features as WASM (max diff 1.3e-5).
  return ort.InferenceSession.create(new Uint8Array(model), { executionProviders: [ep], graphOptimizationLevel: 'disabled' });
}

async function release() {
  if (!loaded) return;
  const l = loaded;
  loaded = null;
  const all = l.model === 'loca' ? [l.features, l.head] : [l.backbone, l.exemplars, l.head];
  for (const s of all) await s.release().catch(() => undefined);
}

async function gpuAvailable(): Promise<boolean> {
  if (!('gpu' in navigator)) return false;
  try {
    return !!(await (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function load(spec: AiModelSpec, preferGpu: boolean): Promise<Loaded> {
  const key = spec.model === 'loca' ? 'loca' : `geco2-${spec.size}`;
  if (loaded && loaded.key === key) return loaded;
  await release();
  ort.env.wasm.wasmBinary = await readCached(spec.urls.wasm);
  // Multi-threaded WASM needs cross-origin isolation, which GitHub Pages cannot enable.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.proxy = false;
  const ep: Backend = preferGpu && (await gpuAvailable()) ? 'webgpu' : 'wasm';
  if (spec.model === 'loca') {
    const [features, head] = await Promise.all([readCached(spec.urls.features), readCached(spec.urls.head)]);
    const make = async (b: Backend): Promise<Loaded> => ({ model: 'loca', key, features: await create(features, b), head: await create(head, b), backend: b });
    try {
      loaded = await make(ep);
    } catch (e) {
      if (ep !== 'webgpu') throw e;
      loaded = await make('wasm');
    }
    return loaded;
  }
  const backbone = await readCached(spec.urls.backbone);
  const head = await readCached(spec.urls.head);
  const exemplars = await readCached(spec.urls.exemplars);
  const make = async (b: Backend): Promise<Loaded> => ({
    model: 'geco2',
    key,
    size: spec.size,
    backbone: await create(backbone, b),
    // Tiny graph with RoIAlign: always on WASM.
    exemplars: await create(exemplars, 'wasm'),
    head: await create(head, b),
    backend: b,
  });
  try {
    loaded = await make(ep);
  } catch (e) {
    if (ep !== 'webgpu') throw e;
    loaded = await make('wasm');
  }
  return loaded;
}

function locaBackend(s: Extract<Loaded, { model: 'loca' }>): LocaBackend {
  return {
    async features(image) {
      const o = await s.features.run({ image: new ort.Tensor('float32', image, [1, 3, 512, 512]) });
      return (await o.features!.getData()) as Float32Array;
    },
    async head(f, ex, boxes) {
      const o = await s.head.run({
        features: new ort.Tensor('float32', f, [1, 256, 64, 64]),
        exemplar_features: new ort.Tensor('float32', ex, [1, 256, 64, 64]),
        bboxes: new ort.Tensor('float32', boxes, [1, 3, 4]),
      });
      return (await o.density!.getData()) as Float32Array;
    },
  };
}

async function detect(s: Extract<Loaded, { model: 'geco2' }>, req: Extract<AiWorkerRequest, { type: 'detect' }>, progress: (f: number) => void) {
  const S = s.size;
  const img = { data: new Uint8ClampedArray(req.rgba), width: req.width, height: req.height };
  const f = await s.backbone.run({ image: new ort.Tensor('float32', geco2Input(img, S), [1, 3, S, S]) });
  progress(0.6);
  // Read back to CPU: the exemplar graph runs on WASM, and plain tensors keep both graphs independent.
  const feat: Record<'feat' | 'fpn1' | 'fpn0', ort.Tensor> = {} as never;
  for (const k of ['feat', 'fpn1', 'fpn0'] as const) {
    feat[k] = new ort.Tensor('float32', (await f[k]!.getData()) as Float32Array, f[k]!.dims);
    f[k]!.dispose();
  }
  const boxes = new Float32Array(req.boxes);
  const p = await s.exemplars.run({ ...feat, boxes: new ort.Tensor('float32', boxes, [1, boxes.length / 4, 4]) });
  const o = await s.head.run({ ...feat, proto0: p.proto0!, proto1: p.proto1!, proto2: p.proto2! });
  const obj = (await o.objectness!.getData()) as Float32Array;
  const ltrb = (await o.ltrb!.getData()) as Float32Array;
  const [, , gh, gw] = o.objectness!.dims as number[];
  progress(1);
  return geco2Detect(obj, ltrb, gw!, gh!, S, { w: req.width, h: req.height });
}

self.onmessage = async (e: MessageEvent<AiWorkerRequest>) => {
  const req = e.data;
  try {
    if (req.type === 'load') {
      const t0 = performance.now();
      const s = await load(req.spec, req.preferGpu);
      post({ type: 'ready', id: req.id, backend: s.backend, loadMs: performance.now() - t0 });
      return;
    }
    if (req.type === 'estimate') {
      if (!loaded || loaded.model !== 'loca') throw new Error('model not loaded');
      const t0 = performance.now();
      const scaled = { data: new Uint8ClampedArray(req.rgba), width: req.width, height: req.height };
      const res = await runPlan(locaBackend(loaded), scaled, req.plan, (f) => post({ type: 'progress', id: req.id, fraction: f }));
      const grid = res.grid.data;
      post(
        { type: 'estimate', id: req.id, grid: grid.buffer as ArrayBuffer, gridW: res.grid.width, gridH: res.grid.height, cell: res.grid.cell, total: res.total, durationMs: performance.now() - t0, backend: loaded.backend },
        [grid.buffer as ArrayBuffer],
      );
      return;
    }
    if (req.type === 'detect') {
      if (!loaded || loaded.model !== 'geco2') throw new Error('model not loaded');
      const t0 = performance.now();
      const d = await detect(loaded, req, (f) => post({ type: 'progress', id: req.id, fraction: f }));
      post(
        { type: 'detections', id: req.id, boxes: d.boxes.buffer as ArrayBuffer, scores: d.scores.buffer as ArrayBuffer, durationMs: performance.now() - t0, backend: loaded.backend },
        [d.boxes.buffer as ArrayBuffer, d.scores.buffer as ArrayBuffer],
      );
    }
  } catch (err) {
    post({ type: 'error', id: req.id, message: err instanceof Error ? err.message : String(err) });
  }
};
