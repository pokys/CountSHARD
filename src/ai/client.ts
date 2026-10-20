/**
 * Main-thread client of the Mode B worker, plus the AI readiness status shown in the
 * offline badge ("AI model downloaded" and "AI ready" are separate facts).
 */
import { CancelledError } from '../analysis/client';
import type { AiWorkerRequest, AiWorkerResponse } from '../workers/aiProtocol';
import { aiAssets, aiDownloaded, downloadAi } from './modelStore';
import type { TilePlan } from './tiling';

export interface AiStatus {
  downloaded: boolean;
  downloading: { loaded: number; total: number } | null;
  /** Sessions created in the worker. */
  ready: boolean;
  backend: 'webgpu' | 'wasm' | null;
}

export interface EstimateOutput {
  grid: Float32Array;
  gridW: number;
  gridH: number;
  /** Grid cell in scaled (model-input) pixels. */
  cell: number;
  total: number;
  durationMs: number;
  backend: 'webgpu' | 'wasm';
}

/**
 * Crash detection. iOS Safari kills a tab that uses too much memory and reloads the page, so a
 * failed estimate leaves no error behind. A marker is written before every estimate and removed
 * afterwards; a marker found at start-up means the previous run killed the page.
 */
const RUN_MARKER = 'countshard.ai.running';
/** Set after a crash on WebGPU: later runs go straight to WASM on this device. */
const FORCE_WASM = 'countshard.ai.forceWasm';

export interface AiCrashReport {
  backend: 'webgpu' | 'wasm';
  at: string;
}

function storageGet(k: string): string | null {
  try {
    return localStorage.getItem(k);
  } catch {
    return null;
  }
}
function storageSet(k: string, v: string | null) {
  try {
    if (v === null) localStorage.removeItem(k);
    else localStorage.setItem(k, v);
  } catch {
    /* storage unavailable: no crash detection */
  }
}

/** Reads and clears the crash marker (call once at start-up). */
export function takeAiCrashReport(): AiCrashReport | null {
  const raw = storageGet(RUN_MARKER);
  if (!raw) return null;
  storageSet(RUN_MARKER, null);
  try {
    const r = JSON.parse(raw) as AiCrashReport;
    if (r.backend === 'webgpu') storageSet(FORCE_WASM, '1');
    return r;
  } catch {
    return null;
  }
}

type Pending = { resolve: (v: AiWorkerResponse) => void; reject: (e: Error) => void; onProgress?: (f: number) => void };

class AiClient {
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private loading: Promise<'webgpu' | 'wasm'> | null = null;
  private listeners = new Set<(s: AiStatus) => void>();
  status: AiStatus = { downloaded: false, downloading: null, ready: false, backend: null };

  subscribe(fn: (s: AiStatus) => void): () => void {
    this.listeners.add(fn);
    fn(this.status);
    return () => this.listeners.delete(fn);
  }

  private set(patch: Partial<AiStatus>) {
    this.status = { ...this.status, ...patch };
    this.listeners.forEach((l) => l(this.status));
  }

  async refresh(): Promise<AiStatus> {
    this.set({ downloaded: await aiDownloaded() });
    return this.status;
  }

  async download(signal?: AbortSignal): Promise<void> {
    try {
      await downloadAi((loaded, total) => this.set({ downloading: { loaded, total } }), signal);
      this.set({ downloaded: true });
    } finally {
      this.set({ downloading: null });
    }
  }

  private ensure(): Worker {
    if (this.worker) return this.worker;
    const w = new Worker(new URL('../workers/ai.worker.ts', import.meta.url), { type: 'module' });
    w.onmessage = (e: MessageEvent<AiWorkerResponse>) => {
      const msg = e.data;
      const p = this.pending.get(msg.id);
      if (!p) return;
      if (msg.type === 'progress') return p.onProgress?.(msg.fraction);
      this.pending.delete(msg.id);
      if (msg.type === 'error') p.reject(new Error(msg.message));
      else p.resolve(msg);
    };
    w.onerror = (e) => {
      const err = new Error(e.message || 'AI worker failed');
      this.pending.forEach((p) => p.reject(err));
      this.pending.clear();
      this.reset();
    };
    this.worker = w;
    return w;
  }

  private send(req: AiWorkerRequest, transfer: Transferable[] = [], onProgress?: (f: number) => void): Promise<AiWorkerResponse> {
    const w = this.ensure();
    return new Promise((resolve, reject) => {
      this.pending.set(req.id, { resolve, reject, onProgress });
      w.postMessage(req, transfer);
    });
  }

  private reset() {
    this.worker?.terminate();
    this.worker = null;
    this.loading = null;
    this.set({ ready: false, backend: null });
  }

  /** Creates the inference sessions (files must be downloaded). */
  load(): Promise<'webgpu' | 'wasm'> {
    this.loading ??= (async () => {
      const a = Object.fromEntries(aiAssets().map((x) => [x.key, x.url])) as Record<'features' | 'head' | 'wasm', string>;
      const res = await this.send({ type: 'load', id: ++this.seq, urls: a, preferGpu: storageGet(FORCE_WASM) !== '1' });
      if (res.type !== 'ready') throw new Error('Unexpected AI worker response');
      this.set({ ready: true, backend: res.backend });
      return res.backend;
    })().catch((e) => {
      this.loading = null;
      throw e;
    });
    return this.loading;
  }

  private async estimateOnce(scaled: ImageData, plan: TilePlan, onProgress?: (f: number) => void): Promise<EstimateOutput> {
    const buf = scaled.data.buffer.slice(0) as ArrayBuffer;
    storageSet(RUN_MARKER, JSON.stringify({ backend: this.status.backend ?? 'wasm', at: new Date().toISOString() } satisfies AiCrashReport));
    try {
      const res = await this.send({ type: 'estimate', id: ++this.seq, rgba: buf, width: scaled.width, height: scaled.height, plan }, [buf], onProgress);
      if (res.type !== 'estimate') throw new Error('Unexpected AI worker response');
      return { grid: new Float32Array(res.grid), gridW: res.gridW, gridH: res.gridH, cell: res.cell, total: res.total, durationMs: res.durationMs, backend: res.backend };
    } finally {
      storageSet(RUN_MARKER, null);
    }
  }

  /** Runs the estimate; a failure on WebGPU is retried once on WASM (and WASM is kept for this device). */
  async estimate(scaled: ImageData, plan: TilePlan, onProgress?: (f: number) => void, onFallback?: () => void): Promise<EstimateOutput> {
    try {
      return await this.estimateOnce(scaled, plan, onProgress);
    } catch (e) {
      if (e instanceof CancelledError || this.status.backend !== 'webgpu') throw e;
      storageSet(FORCE_WASM, '1');
      this.reset();
      onFallback?.();
      await this.load();
      return this.estimateOnce(scaled, plan, onProgress);
    }
  }

  cancel() {
    const err = new CancelledError();
    this.pending.forEach((p) => p.reject(err));
    this.pending.clear();
    this.reset();
  }
}

export const aiClient = new AiClient();
