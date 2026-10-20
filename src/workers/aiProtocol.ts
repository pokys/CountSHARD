import type { TilePlan } from '../ai/tiling';
import type { Geco2Size } from '../ai/geco2';

/** Which model the worker loads, and the cached file URLs it reads. */
export type AiModelSpec =
  | { model: 'loca'; urls: { features: string; head: string; wasm: string } }
  | { model: 'geco2'; size: Geco2Size; urls: { backbone: string; exemplars: string; head: string; wasm: string } };

export type AiWorkerRequest =
  | { type: 'load'; id: number; spec: AiModelSpec; preferGpu: boolean }
  | {
      type: 'estimate';
      id: number;
      /** RGBA of the photo resized to plan.width × plan.height (transferred). */
      rgba: ArrayBuffer;
      width: number;
      height: number;
      plan: TilePlan;
    }
  | {
      type: 'detect';
      id: number;
      /** RGBA of the photo scaled for GeCo2 (≤ size × size, top-left of the input; transferred). */
      rgba: ArrayBuffer;
      width: number;
      height: number;
      /** Exemplar boxes x1, y1, x2, y2 in model pixels (Float32). */
      boxes: ArrayBuffer;
    };

export type AiWorkerResponse =
  | { type: 'ready'; id: number; backend: 'webgpu' | 'wasm'; loadMs: number }
  | { type: 'progress'; id: number; fraction: number }
  | { type: 'estimate'; id: number; grid: ArrayBuffer; gridW: number; gridH: number; cell: number; total: number; durationMs: number; backend: 'webgpu' | 'wasm' }
  | { type: 'detections'; id: number; boxes: ArrayBuffer; scores: ArrayBuffer; durationMs: number; backend: 'webgpu' | 'wasm' }
  | { type: 'error'; id: number; message: string };
