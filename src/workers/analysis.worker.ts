/// <reference lib="webworker" />
/**
 * Analysis worker: owns the OpenCV.js runtime so heavy work never blocks the UI.
 * Cancellation = the client terminates this worker and starts a new one.
 */
import cvModule from '@techstark/opencv-js';
import { detectCircles } from '../analysis/circles';
import type { CV } from '../analysis/cv';
import type { WorkerRequest, WorkerResponse } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

let cvPromise: Promise<CV> | null = null;
function getCv(): Promise<CV> {
  // The package's default export is a Promise of the initialised runtime (OpenCV.js ≥ 4.10 builds).
  cvPromise ??= Promise.resolve(cvModule as unknown as Promise<CV>).then((cv: CV) => {
    if (cv?.Mat) return cv;
    return new Promise<CV>((resolve) => {
      cv.onRuntimeInitialized = () => resolve(cv);
    });
  });
  return cvPromise;
}

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const req = e.data;
  try {
    if (req.type === 'warmup') {
      const t0 = performance.now();
      await getCv();
      post({ type: 'ready', id: req.id, loadMs: performance.now() - t0 });
      return;
    }
    if (req.type === 'detect') {
      const cv = await getCv();
      const { width, height } = req;
      const rgba = new Uint8ClampedArray(req.rgba);
      // RGBA → luminance (Rec. 601), done here to keep the main thread free.
      const gray = new Uint8Array(width * height);
      for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
        gray[i] = (rgba[j]! * 299 + rgba[j + 1]! * 587 + rgba[j + 2]! * 114) / 1000;
      }
      const res = detectCircles(cv, { data: gray, width, height }, {
        sensitivity: req.sensitivity,
        minRadius: req.minRadius,
        maxRadius: req.maxRadius,
        onProgress: (p) => post({ type: 'progress', id: req.id, fraction: p }),
      });
      post({ type: 'result', id: req.id, circles: res.circles, radiusRange: res.radiusRange, durationMs: res.durationMs, candidates: res.candidates });
    }
  } catch (err) {
    post({ type: 'error', id: req.id, message: err instanceof Error ? err.message : String(err) });
  }
};
