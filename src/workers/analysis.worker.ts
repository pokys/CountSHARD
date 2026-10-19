/// <reference lib="webworker" />
/**
 * Analysis worker: owns the OpenCV.js runtime so heavy work never blocks the UI.
 * Cancellation = the client terminates this worker and starts a new one.
 */
import cvModule from '@techstark/opencv-js';
import { type GrayImage, detectCircles } from '../analysis/circles';
import { fuseDetections } from '../analysis/fusion';
import { registerImages } from '../analysis/registration';
import type { CV } from '../analysis/cv';
import type { PhotoPixels, WorkerRequest, WorkerResponse } from './protocol';

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

/** RGBA → luminance (Rec. 601), done here to keep the main thread free. */
function toGray(p: PhotoPixels): GrayImage {
  const rgba = new Uint8ClampedArray(p.rgba);
  const gray = new Uint8Array(p.width * p.height);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    gray[i] = (rgba[j]! * 299 + rgba[j + 1]! * 587 + rgba[j + 2]! * 114) / 1000;
  }
  return { data: gray, width: p.width, height: p.height };
}

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
      const res = detectCircles(cv, toGray(req), {
        sensitivity: req.sensitivity,
        minRadius: req.minRadius,
        maxRadius: req.maxRadius,
        onProgress: (p) => post({ type: 'progress', id: req.id, fraction: p }),
      });
      post({ type: 'result', id: req.id, circles: res.circles, radiusRange: res.radiusRange, durationMs: res.durationMs, candidates: res.candidates });
    }
    if (req.type === 'fuse') {
      const cv = await getCv();
      const t0 = performance.now();
      const grays = req.photos.map(toGray);
      const n = grays.length;
      // Progress: detection ≈ 60 %, registration ≈ 40 % of the work.
      const step = (i: number, f: number, phase: 0 | 1) => post({ type: 'progress', id: req.id, fraction: phase === 0 ? (0.6 * (i + f)) / n : 0.6 + (0.4 * (i + f)) / Math.max(1, n - 1) });
      const circles = grays.map((g, i) =>
        detectCircles(cv, g, {
          sensitivity: req.sensitivity,
          minRadius: req.minRadius,
          maxRadius: req.maxRadius,
          onProgress: (p) => step(i, p, 0),
        }).circles,
      );
      const registrations = grays.map((g, i) => {
        if (i === 0) return null;
        const r = registerImages(cv, grays[0]!, g);
        step(i - 1, 1, 1);
        return r;
      });
      const fusion = fuseDetections(grays.map((g, i) => ({ width: g.width, height: g.height, circles: circles[i]!, registration: registrations[i]! })));
      post({ type: 'fused', id: req.id, fusion, durationMs: performance.now() - t0 });
    }
  } catch (err) {
    post({ type: 'error', id: req.id, message: err instanceof Error ? err.message : String(err) });
  }
};
