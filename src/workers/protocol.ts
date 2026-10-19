import type { Circle } from '../analysis/circles';
import type { FusionResult } from '../analysis/fusion';

export interface PhotoPixels {
  /** RGBA pixels (transferred). */
  rgba: ArrayBuffer;
  width: number;
  height: number;
}

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
    }
  | {
      /** Multi-shot: detect in every photo, register each onto photos[0], fuse. */
      type: 'fuse';
      id: number;
      photos: PhotoPixels[];
      sensitivity: number;
      minRadius?: number;
      maxRadius?: number;
    };

export type WorkerResponse =
  | { type: 'ready'; id: number; loadMs: number }
  | { type: 'progress'; id: number; fraction: number }
  | { type: 'result'; id: number; circles: Circle[]; radiusRange: [number, number]; durationMs: number; candidates: number }
  | { type: 'fused'; id: number; fusion: FusionResult; durationMs: number }
  | { type: 'error'; id: number; message: string };
