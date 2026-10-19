/**
 * Main-thread client for the analysis worker: progress, cancellation, warm-up.
 */
import type { Circle } from './circles';
import type { WorkerRequest, WorkerResponse } from '../workers/protocol';

export class CancelledError extends Error {
  constructor() {
    super('cancelled');
  }
}

export interface DetectInput {
  image: ImageData;
  sensitivity: number;
  minRadius?: number;
  maxRadius?: number;
}

export interface DetectOutput {
  circles: Circle[];
  radiusRange: [number, number];
  durationMs: number;
  candidates: number;
}

type Pending = {
  resolve: (v: WorkerResponse) => void;
  reject: (e: Error) => void;
  onProgress?: (f: number) => void;
};

class AnalysisClient {
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private readyPromise: Promise<number> | null = null;

  private ensure(): Worker {
    if (this.worker) return this.worker;
    const w = new Worker(new URL('../workers/analysis.worker.ts', import.meta.url), { type: 'module' });
    w.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      const p = this.pending.get(msg.id);
      if (!p) return;
      if (msg.type === 'progress') {
        p.onProgress?.(msg.fraction);
        return;
      }
      this.pending.delete(msg.id);
      if (msg.type === 'error') p.reject(new Error(msg.message));
      else p.resolve(msg);
    };
    w.onerror = (e) => {
      const err = new Error(e.message || 'Worker failed');
      this.pending.forEach((p) => p.reject(err));
      this.pending.clear();
      this.reset();
    };
    this.worker = w;
    return w;
  }

  private send(req: WorkerRequest, transfer: Transferable[], onProgress?: (f: number) => void): Promise<WorkerResponse> {
    const w = this.ensure();
    return new Promise((resolve, reject) => {
      this.pending.set(req.id, { resolve, reject, onProgress });
      w.postMessage(req, transfer);
    });
  }

  private reset() {
    this.worker?.terminate();
    this.worker = null;
    this.readyPromise = null;
  }

  /** Loads OpenCV in the worker; resolves with the load time in ms. */
  warmup(): Promise<number> {
    this.readyPromise ??= this.send({ type: 'warmup', id: ++this.seq }, []).then((r) => (r.type === 'ready' ? r.loadMs : 0));
    return this.readyPromise;
  }

  async detect(input: DetectInput, onProgress?: (f: number) => void): Promise<DetectOutput> {
    const buf = input.image.data.buffer.slice(0) as ArrayBuffer;
    const res = await this.send(
      {
        type: 'detect',
        id: ++this.seq,
        rgba: buf,
        width: input.image.width,
        height: input.image.height,
        sensitivity: input.sensitivity,
        minRadius: input.minRadius,
        maxRadius: input.maxRadius,
      },
      [buf],
      onProgress,
    );
    if (res.type !== 'result') throw new Error('Unexpected worker response');
    return { circles: res.circles, radiusRange: res.radiusRange, durationMs: res.durationMs, candidates: res.candidates };
  }

  /** Aborts all running work by terminating the worker. The next call starts a fresh one. */
  cancel() {
    const err = new CancelledError();
    this.pending.forEach((p) => p.reject(err));
    this.pending.clear();
    this.reset();
  }
}

export const analysisClient = new AnalysisClient();
