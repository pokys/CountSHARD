import type { TilePlan } from '../ai/tiling';

export type AiWorkerRequest =
  | { type: 'load'; id: number; urls: { features: string; head: string; wasm: string }; preferGpu: boolean }
  | {
      type: 'estimate';
      id: number;
      /** RGBA of the photo resized to plan.width × plan.height (transferred). */
      rgba: ArrayBuffer;
      width: number;
      height: number;
      plan: TilePlan;
    };

export type AiWorkerResponse =
  | { type: 'ready'; id: number; backend: 'webgpu' | 'wasm'; loadMs: number }
  | { type: 'progress'; id: number; fraction: number }
  | { type: 'estimate'; id: number; grid: ArrayBuffer; gridW: number; gridH: number; cell: number; total: number; durationMs: number; backend: 'webgpu' | 'wasm' }
  | { type: 'error'; id: number; message: string };
