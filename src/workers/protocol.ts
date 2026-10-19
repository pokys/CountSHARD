import type { Circle } from '../analysis/circles';

export type WorkerRequest =
  | { type: 'warmup'; id: number }
  | {
      type: 'detect';
      id: number;
      /** RGBA pixels (transferred). */
      rgba: ArrayBuffer;
      width: number;
      height: number;
      sensitivity: number;
      minRadius?: number;
      maxRadius?: number;
    };

export type WorkerResponse =
  | { type: 'ready'; id: number; loadMs: number }
  | { type: 'progress'; id: number; fraction: number }
  | { type: 'result'; id: number; circles: Circle[]; radiusRange: [number, number]; durationMs: number; candidates: number }
  | { type: 'error'; id: number; message: string };
