/**
 * Main-thread client of the Mode B worker, plus the AI status shown in the AI panel and the
 * offline menu: which model is chosen (per device), whether its files are downloaded, whether
 * the sessions are ready ("downloaded" and "ready" are separate facts).
 */
import { CancelledError } from '../analysis/client';
import type { AiModelSpec, AiWorkerRequest, AiWorkerResponse } from '../workers/aiProtocol';
import { type AiAsset, GECO2_CACHE, type Geco2Size, AI_CACHE, aiAssets, aiDownloaded, deleteAi, deleteGeco2, downloadAi, geco2Assets, missingBytes } from './modelStore';
import type { TilePlan } from './tiling';
import {
  type AiModelChoice,
  type DeviceProfile,
  type Geco2Plan,
  type ModelDecision,
  type ResolvedModel,
  deviceProfile,
  geco2Plan,
  loadChoice,
  recordGeco2Seconds,
  resolveModel,
  saveChoice,
  setGeco2ForceWasm,
} from './device';

export interface AiStatus {
  /** The user's per-device choice ('auto' by default). */
  choice: AiModelChoice;
  /** The model the choice resolves to on this device (null until the device profile is known). */
  decision: ModelDecision | null;
  /** GeCo2 backend and input size for this device. */
  geco2: Geco2Plan | null;
  profile: DeviceProfile | null;
  /** Files of the chosen model are downloaded. */
  downloaded: boolean;
  /** Bytes the chosen model still needs. */
  missingBytes: number;
  /** Per model (offline menu). */
  models: { loca: boolean; geco2_768: boolean; geco2_1024: boolean };
  downloading: { loaded: number; total: number } | null;
  /** Sessions created in the worker (for `readyModel`). */
  ready: boolean;
  readyModel: ResolvedModel | null;
  backend: 'webgpu' | 'wasm' | null;
  /** A previous run killed the page (read once at start-up). */
  crash: AiCrashReport | null;
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

export interface DetectOutput {
  /** x1, y1, x2, y2 in model pixels. */
  boxes: Float32Array;
  scores: Float32Array;
  durationMs: number;
  backend: 'webgpu' | 'wasm';
  size: Geco2Size;
}

/**
 * Crash detection. iOS Safari kills a tab that uses too much memory and reloads the page, so a
 * failed run leaves no error behind. A marker is written before every run and removed
 * afterwards; a marker found at start-up means the previous run killed the page.
 */
const RUN_MARKER = 'countshard.ai.running';
/** Set after a LOCA crash on WebGPU: later LOCA runs go straight to WASM on this device. */
const FORCE_WASM = 'countshard.ai.forceWasm';

export interface AiCrashReport {
  backend: 'webgpu' | 'wasm';
  /** Missing in markers written before GeCo2 existed (= LOCA). */
  model?: ResolvedModel;
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
    if (r.backend === 'webgpu') {
      if (r.model === 'geco2') setGeco2ForceWasm();
      else storageSet(FORCE_WASM, '1');
    }
    aiClient.noteCrash(r);
    return r;
  } catch {
    return null;
  }
}

type Pending = { resolve: (v: AiWorkerResponse) => void; reject: (e: Error) => void; onProgress?: (f: number) => void };

function assetsFor(model: ResolvedModel, plan: Geco2Plan | null): { assets: AiAsset[]; cache: string } {
  return model === 'loca' ? { assets: aiAssets(), cache: AI_CACHE } : { assets: geco2Assets(plan?.size ?? 768), cache: GECO2_CACHE };
}

function specFor(model: ResolvedModel, size: Geco2Size): AiModelSpec {
  if (model === 'loca') {
    const a = Object.fromEntries(aiAssets().map((x) => [x.key, x.url])) as Record<'features' | 'head' | 'wasm', string>;
    return { model: 'loca', urls: a };
  }
  const a = Object.fromEntries(geco2Assets(size).map((x) => [x.key, x.url])) as Record<'backbone' | 'exemplars' | 'head' | 'wasm', string>;
  return { model: 'geco2', size, urls: a };
}

class AiClient {
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private loading: Promise<'webgpu' | 'wasm'> | null = null;
  private loadingKey = '';
  private listeners = new Set<(s: AiStatus) => void>();
  status: AiStatus = {
    choice: 'auto',
    decision: null,
    geco2: null,
    profile: null,
    downloaded: false,
    missingBytes: 0,
    models: { loca: false, geco2_768: false, geco2_1024: false },
    downloading: null,
    ready: false,
    readyModel: null,
    backend: null,
    crash: null,
  };

  subscribe(fn: (s: AiStatus) => void): () => void {
    this.listeners.add(fn);
    fn(this.status);
    return () => this.listeners.delete(fn);
  }

  private set(patch: Partial<AiStatus>) {
    this.status = { ...this.status, ...patch };
    this.listeners.forEach((l) => l(this.status));
  }

  noteCrash(r: AiCrashReport) {
    this.set({ crash: r });
  }

  dismissCrash() {
    this.set({ crash: null });
  }

  /** The model the next run uses (LOCA until the device profile is known). */
  get model(): ResolvedModel {
    return this.status.decision?.model ?? 'loca';
  }

  async refresh(): Promise<AiStatus> {
    const choice = loadChoice();
    let profile = this.status.profile;
    try {
      profile = await deviceProfile();
    } catch {
      /* keep the previous profile */
    }
    const decision = profile ? resolveModel(choice, profile) : { model: choice === 'geco2' ? 'geco2' : 'loca', reason: null } as ModelDecision;
    const plan = profile ? geco2Plan(profile) : null;
    const { assets } = assetsFor(decision.model, plan);
    const [downloaded, missing, loca, g768, g1024] = await Promise.all([
      aiDownloaded(assets),
      missingBytes(assets),
      aiDownloaded(aiAssets()),
      aiDownloaded(geco2Assets(768)),
      aiDownloaded(geco2Assets(1024)),
    ]);
    this.set({ choice, profile, decision, geco2: plan, downloaded, missingBytes: missing, models: { loca, geco2_768: g768, geco2_1024: g1024 } });
    return this.status;
  }

  /** Remembers the model choice on this device. */
  async setChoice(choice: AiModelChoice): Promise<void> {
    saveChoice(choice);
    await this.refresh();
  }

  async download(signal?: AbortSignal): Promise<void> {
    const { assets, cache } = assetsFor(this.model, this.status.geco2);
    try {
      await downloadAi((loaded, total) => this.set({ downloading: { loaded, total } }), signal, assets, cache);
    } finally {
      this.set({ downloading: null });
      await this.refresh();
    }
  }

  /** Deletes one model's files (offline menu / AI panel). */
  async deleteModel(which: 'loca' | 'geco2_768' | 'geco2_1024'): Promise<void> {
    this.cancel();
    if (which === 'loca') await deleteAi();
    else await deleteGeco2(which === 'geco2_768' ? 768 : 1024);
    await this.refresh();
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
    this.loadingKey = '';
    this.set({ ready: false, readyModel: null, backend: null });
  }

  /**
   * Creates the sessions of the chosen model (files must be downloaded). GeCo2 on WebGPU only
   * where the device plan says so (never on phones); `forceWasm` for the retry after a failure.
   */
  load(forceWasm = false): Promise<'webgpu' | 'wasm'> {
    const model = this.model;
    const size = this.status.geco2?.size ?? 768;
    const key = `${model}-${size}-${forceWasm}`;
    if (this.loading && this.loadingKey !== key) this.loading = null;
    this.loadingKey = key;
    this.loading ??= (async () => {
      const preferGpu = forceWasm ? false : model === 'loca' ? storageGet(FORCE_WASM) !== '1' : this.status.geco2?.backend === 'webgpu';
      const res = await this.send({ type: 'load', id: ++this.seq, spec: specFor(model, size), preferGpu });
      if (res.type !== 'ready') throw new Error('Unexpected AI worker response');
      this.set({ ready: true, readyModel: model, backend: res.backend });
      return res.backend;
    })().catch((e) => {
      this.loading = null;
      throw e;
    });
    return this.loading;
  }

  private mark(model: ResolvedModel) {
    storageSet(RUN_MARKER, JSON.stringify({ backend: this.status.backend ?? 'wasm', model, at: new Date().toISOString() } satisfies AiCrashReport));
  }

  private async estimateOnce(scaled: ImageData, plan: TilePlan, onProgress?: (f: number) => void): Promise<EstimateOutput> {
    const buf = scaled.data.buffer.slice(0) as ArrayBuffer;
    this.mark('loca');
    try {
      const res = await this.send({ type: 'estimate', id: ++this.seq, rgba: buf, width: scaled.width, height: scaled.height, plan }, [buf], onProgress);
      if (res.type !== 'estimate') throw new Error('Unexpected AI worker response');
      return { grid: new Float32Array(res.grid), gridW: res.gridW, gridH: res.gridH, cell: res.cell, total: res.total, durationMs: res.durationMs, backend: res.backend };
    } finally {
      storageSet(RUN_MARKER, null);
    }
  }

  /** LOCA: runs the estimate; a failure on WebGPU is retried once on WASM (and WASM is kept for this device). */
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

  private async detectOnce(scaled: ImageData, boxes: Float32Array, onProgress?: (f: number) => void): Promise<DetectOutput> {
    const buf = scaled.data.buffer.slice(0) as ArrayBuffer;
    const b = boxes.slice().buffer as ArrayBuffer;
    this.mark('geco2');
    try {
      const res = await this.send({ type: 'detect', id: ++this.seq, rgba: buf, width: scaled.width, height: scaled.height, boxes: b }, [buf, b], onProgress);
      if (res.type !== 'detections') throw new Error('Unexpected AI worker response');
      const out = { boxes: new Float32Array(res.boxes), scores: new Float32Array(res.scores), durationMs: res.durationMs, backend: res.backend, size: this.status.geco2?.size ?? 768 };
      // Remembered per variant; feeds the automatic model choice (a WASM retry of the 1024 files is not a plan variant).
      if (out.backend === 'webgpu' && out.size === 1024) recordGeco2Seconds('webgpu-1024', out.durationMs / 1000);
      else if (out.backend === 'wasm' && out.size === 768) recordGeco2Seconds('wasm-768', out.durationMs / 1000);
      return out;
    } finally {
      storageSet(RUN_MARKER, null);
    }
  }

  /**
   * GeCo2: detects the objects. A failure on WebGPU is retried once on WASM with the same files,
   * and later runs on this device use WASM (768 px).
   */
  async detect(scaled: ImageData, boxes: Float32Array, onProgress?: (f: number) => void, onFallback?: () => void): Promise<DetectOutput> {
    try {
      return await this.detectOnce(scaled, boxes, onProgress);
    } catch (e) {
      if (e instanceof CancelledError || this.status.backend !== 'webgpu') throw e;
      setGeco2ForceWasm();
      this.reset();
      onFallback?.();
      await this.load(true);
      const out = await this.detectOnce(scaled, boxes, onProgress);
      void this.refresh();
      return out;
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
