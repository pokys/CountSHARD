/**
 * Pixel ↔ tensor conversions for LOCA and stitching of tile density maps. Pure, no DOM.
 */
import { MODEL_INPUT, type Tile } from './tiling';

/** ImageNet normalisation used by LOCA (torchvision defaults). */
export const MEAN = [0.485, 0.456, 0.406] as const;
export const STD = [0.229, 0.224, 0.225] as const;

export interface Rgba {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}

/**
 * Writes the 512 × 512 window at (x0, y0) of `img` into `out` as a normalised NCHW tensor.
 * Pixels outside the image are letterbox padding with the ImageNet mean (0 after normalisation).
 */
export function fillTileTensor(img: Rgba, x0: number, y0: number, out = new Float32Array(3 * MODEL_INPUT * MODEL_INPUT)): Float32Array {
  const N = MODEL_INPUT;
  const plane = N * N;
  out.fill(0);
  const ys = Math.max(0, -y0);
  const ye = Math.min(N, img.height - y0);
  const xs = Math.max(0, -x0);
  const xe = Math.min(N, img.width - x0);
  const k = [1 / (255 * STD[0]), 1 / (255 * STD[1]), 1 / (255 * STD[2])];
  const m = [MEAN[0] / STD[0], MEAN[1] / STD[1], MEAN[2] / STD[2]];
  for (let y = ys; y < ye; y++) {
    let src = ((y0 + y) * img.width + (x0 + xs)) * 4;
    let dst = y * N + xs;
    for (let x = xs; x < xe; x++, src += 4, dst++) {
      out[dst] = img.data[src]! * k[0]! - m[0]!;
      out[plane + dst] = img.data[src + 1]! * k[1]! - m[1]!;
      out[2 * plane + dst] = img.data[src + 2]! * k[2]! - m[2]!;
    }
  }
  return out;
}

/** Area-averaging / bilinear resize of RGBA pixels (used in Node and as a worker fallback). */
export function resizeRgba(img: Rgba, w: number, h: number): Rgba {
  const out = new Uint8ClampedArray(w * h * 4);
  const sx = img.width / w;
  const sy = img.height / h;
  if (sx <= 1 && sy <= 1) {
    // Upscale / same size: bilinear.
    for (let y = 0; y < h; y++) {
      const fy = Math.min(img.height - 1, Math.max(0, (y + 0.5) * sy - 0.5));
      const y1 = Math.floor(fy), y2 = Math.min(img.height - 1, y1 + 1), wy = fy - y1;
      for (let x = 0; x < w; x++) {
        const fx = Math.min(img.width - 1, Math.max(0, (x + 0.5) * sx - 0.5));
        const x1 = Math.floor(fx), x2 = Math.min(img.width - 1, x1 + 1), wx = fx - x1;
        for (let c = 0; c < 4; c++) {
          const a = img.data[(y1 * img.width + x1) * 4 + c]!, b = img.data[(y1 * img.width + x2) * 4 + c]!;
          const d = img.data[(y2 * img.width + x1) * 4 + c]!, e = img.data[(y2 * img.width + x2) * 4 + c]!;
          out[(y * w + x) * 4 + c] = (a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy;
        }
      }
    }
    return { data: out, width: w, height: h };
  }
  // Downscale: box filter over the source footprint.
  for (let y = 0; y < h; y++) {
    const ya = Math.floor(y * sy), yb = Math.max(ya + 1, Math.min(img.height, Math.floor((y + 1) * sy)));
    for (let x = 0; x < w; x++) {
      const xa = Math.floor(x * sx), xb = Math.max(xa + 1, Math.min(img.width, Math.floor((x + 1) * sx)));
      const acc = [0, 0, 0, 0];
      for (let yy = ya; yy < yb; yy++) for (let xx = xa; xx < xb; xx++) for (let c = 0; c < 4; c++) acc[c]! += img.data[(yy * img.width + xx) * 4 + c]!;
      const n = (yb - ya) * (xb - xa);
      for (let c = 0; c < 4; c++) out[(y * w + x) * 4 + c] = acc[c]! / n;
    }
  }
  return { data: out, width: w, height: h };
}

/** Density grid over the scaled image: one cell = `cell` × `cell` scaled pixels. */
export interface DensityGrid {
  width: number;
  height: number;
  /** Cell size in the pixels the grid was accumulated in. */
  cell: number;
  data: Float32Array;
}

export function makeGrid(imageW: number, imageH: number, cell: number): DensityGrid {
  const width = Math.ceil(imageW / cell);
  const height = Math.ceil(imageH / cell);
  return { width, height, cell, data: new Float32Array(width * height) };
}

/**
 * Adds the part of a tile's 512 × 512 density map that the tile owns into the grid.
 * Negative values (LOCA ends with a LeakyReLU) are kept so the total equals the model's sum.
 */
export function accumulateTile(grid: DensityGrid, tile: Tile, density: Float32Array): number {
  const N = MODEL_INPUT;
  let added = 0;
  const { x0, x1, y0, y1 } = tile.own;
  for (let y = y0; y < y1; y++) {
    const ty = y - tile.y;
    const row = Math.floor(y / grid.cell) * grid.width;
    for (let x = x0; x < x1; x++) {
      const v = density[ty * N + (x - tile.x)]!;
      grid.data[row + Math.floor(x / grid.cell)]! += v;
      added += v;
    }
  }
  return added;
}

export function gridSum(data: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < data.length; i++) s += data[i]!;
  return s;
}
