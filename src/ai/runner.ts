/**
 * Runs LOCA over a tile plan. The inference backend is abstract so the same code runs in the
 * browser worker (onnxruntime-web, WebGPU or WASM) and in the Node evaluation.
 */
import { MODEL_INPUT, type TilePlan } from './tiling';
import { type DensityGrid, type Rgba, accumulateTile, fillTileTensor, makeGrid } from './tensor';

export interface LocaBackend {
  /** image [1,3,512,512] → features [1,256,64,64] */
  features(image: Float32Array): Promise<Float32Array>;
  /** tile features + exemplar-window features + boxes [1,3,4] → density [1,1,512,512] */
  head(features: Float32Array, exemplarFeatures: Float32Array, boxes: Float32Array): Promise<Float32Array>;
}

export interface RunResult {
  grid: DensityGrid;
  /** Sum of the density over the image (the model's raw estimate). */
  total: number;
  /** Density sum per tile (owned area), for diagnostics. */
  tileTotals: number[];
}

/** Grid cell in scaled pixels: 4 keeps the heat map smooth and the stored grid small. */
export const GRID_CELL = 4;

/**
 * `scaled` must be the input image resized to plan.width × plan.height.
 * `onProgress` receives the fraction of model passes done.
 */
export async function runPlan(backend: LocaBackend, scaled: Rgba, plan: TilePlan, onProgress?: (f: number) => void): Promise<RunResult> {
  if (scaled.width !== plan.width || scaled.height !== plan.height) throw new Error('scaled image does not match the plan');
  const passes = plan.tiles.length + (plan.exemplarTileIndex < 0 ? 1 : 0);
  let done = 0;
  const step = () => onProgress?.(++done / passes);
  const buf = new Float32Array(3 * MODEL_INPUT * MODEL_INPUT);

  let exFeatures: Float32Array;
  if (plan.exemplarTileIndex >= 0) {
    const t = plan.tiles[plan.exemplarTileIndex]!;
    exFeatures = await backend.features(fillTileTensor(scaled, t.x, t.y, buf));
  } else {
    exFeatures = await backend.features(fillTileTensor(scaled, plan.exemplarWindow.x, plan.exemplarWindow.y, buf));
    step();
  }
  const boxes = new Float32Array(plan.exemplarBoxes.flat());
  const grid = makeGrid(plan.width, plan.height, GRID_CELL);
  const tileTotals: number[] = [];
  for (let i = 0; i < plan.tiles.length; i++) {
    const t = plan.tiles[i]!;
    const f = i === plan.exemplarTileIndex ? exFeatures : await backend.features(fillTileTensor(scaled, t.x, t.y, buf));
    const d = await backend.head(f, exFeatures, boxes);
    tileTotals.push(accumulateTile(grid, t, d));
    step();
  }
  const total = tileTotals.reduce((a, b) => a + b, 0);
  return { grid, total, tileTotals };
}

/**
 * LOCA's original protocol (also what the reference site does): squash the whole image to
 * 512 × 512, one pass, sum the density. Used only by the evaluation for comparison.
 */
export async function runSquashed(backend: LocaBackend, img: Rgba, exemplars: { x: number; y: number; w: number; h: number }[], resize: (img: Rgba, w: number, h: number) => Rgba): Promise<{ density: Float32Array; total: number }> {
  const sq = resize(img, MODEL_INPUT, MODEL_INPUT);
  const kx = MODEL_INPUT / img.width;
  const ky = MODEL_INPUT / img.height;
  const b = exemplars.map((e) => [e.x * kx, e.y * ky, (e.x + e.w) * kx, (e.y + e.h) * ky]);
  const boxes = new Float32Array(Array.from({ length: 3 }, (_, i) => b[i % b.length]!).flat());
  const f = await backend.features(fillTileTensor(sq, 0, 0));
  const density = await backend.head(f, f, boxes);
  let total = 0;
  for (let i = 0; i < density.length; i++) total += density[i]!;
  return { density, total };
}

/** The reference site's post-processing: sum only density above `fraction` of the maximum. */
export function thresholdedSum(density: ArrayLike<number>, fraction = 0.05): number {
  let max = 0;
  for (let i = 0; i < density.length; i++) max = Math.max(max, density[i]!);
  let s = 0;
  for (let i = 0; i < density.length; i++) if (density[i]! > fraction * max) s += density[i]!;
  return s;
}
