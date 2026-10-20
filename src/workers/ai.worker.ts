/// <reference lib="webworker" />
/**
 * Mode B worker: owns onnxruntime-web and the two LOCA graphs. WebGPU when the browser offers
 * it, otherwise WASM. Model files and the WASM runtime are read from Cache Storage (they were
 * downloaded on demand), so this works offline. Cancellation = the client terminates the worker.
 */
import * as ort from 'onnxruntime-web';
import { readCached } from '../ai/modelStore';
import { type LocaBackend, runPlan } from '../ai/runner';
import type { AiWorkerRequest, AiWorkerResponse } from './aiProtocol';

declare const self: DedicatedWorkerGlobalScope;

let sessions: { features: ort.InferenceSession; head: ort.InferenceSession; backend: 'webgpu' | 'wasm' } | null = null;

const post = (msg: AiWorkerResponse, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

async function create(model: ArrayBuffer, ep: 'webgpu' | 'wasm') {
  // No graph optimisation: the int8 weights are not constant-folded into fp32 copies (WASM in
  // Node: 307 MB RSS after a features run vs 347 MB with 'all', same speed). WebGPU then runs
  // DequantizeLinear itself; checked to give the same features as WASM (max diff 1.3e-5).
  return ort.InferenceSession.create(new Uint8Array(model), { executionProviders: [ep], graphOptimizationLevel: 'disabled' });
}

async function load(urls: { features: string; head: string; wasm: string }, preferGpu: boolean) {
  if (sessions) return sessions;
  const [features, head, wasm] = await Promise.all([readCached(urls.features), readCached(urls.head), readCached(urls.wasm)]);
  ort.env.wasm.wasmBinary = wasm;
  // Multi-threaded WASM needs cross-origin isolation, which GitHub Pages cannot enable.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.proxy = false;
  if (preferGpu && 'gpu' in navigator) {
    try {
      const adapter = await (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter();
      if (adapter) {
        sessions = { features: await create(features, 'webgpu'), head: await create(head, 'webgpu'), backend: 'webgpu' };
        return sessions;
      }
    } catch {
      /* fall back to WASM */
    }
  }
  sessions = { features: await create(features, 'wasm'), head: await create(head, 'wasm'), backend: 'wasm' };
  return sessions;
}

function backendOf(s: NonNullable<typeof sessions>): LocaBackend {
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

self.onmessage = async (e: MessageEvent<AiWorkerRequest>) => {
  const req = e.data;
  try {
    if (req.type === 'load') {
      const t0 = performance.now();
      const s = await load(req.urls, req.preferGpu);
      post({ type: 'ready', id: req.id, backend: s.backend, loadMs: performance.now() - t0 });
      return;
    }
    if (req.type === 'estimate') {
      if (!sessions) throw new Error('model not loaded');
      const t0 = performance.now();
      const scaled = { data: new Uint8ClampedArray(req.rgba), width: req.width, height: req.height };
      const res = await runPlan(backendOf(sessions), scaled, req.plan, (f) => post({ type: 'progress', id: req.id, fraction: f }));
      const grid = res.grid.data;
      post(
        { type: 'estimate', id: req.id, grid: grid.buffer as ArrayBuffer, gridW: res.grid.width, gridH: res.grid.height, cell: res.grid.cell, total: res.total, durationMs: performance.now() - t0, backend: sessions.backend },
        [grid.buffer as ArrayBuffer],
      );
    }
  } catch (err) {
    post({ type: 'error', id: req.id, message: err instanceof Error ? err.message : String(err) });
  }
};
