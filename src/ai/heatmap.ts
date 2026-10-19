/**
 * Density → heat-map colours. Pure: returns RGBA for a grid, drawn scaled over the photo.
 * The scale is relative (the densest cell = full colour); the map shows WHERE the model
 * puts objects, its values are not object counts.
 */

/** Colour stops (t, r, g, b): dark violet → magenta → orange → yellow ("inferno"-like). */
const STOPS: [number, number, number, number][] = [
  [0, 40, 11, 84],
  [0.35, 145, 32, 120],
  [0.65, 237, 105, 37],
  [1, 252, 255, 164],
];

export function heatColor(t: number): [number, number, number] {
  const v = Math.min(1, Math.max(0, t));
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, r1, g1, b1] = STOPS[i]!;
    const [t0, r0, g0, b0] = STOPS[i - 1]!;
    if (v <= t1) {
      const f = (v - t0) / (t1 - t0);
      return [Math.round(r0 + (r1 - r0) * f), Math.round(g0 + (g1 - g0) * f), Math.round(b0 + (b1 - b0) * f)];
    }
  }
  return [STOPS[3]![1], STOPS[3]![2], STOPS[3]![3]];
}

/** Robust maximum: the 99.5th percentile of positive cells (a single hot cell does not wash the map out). */
export function heatScale(data: ArrayLike<number>): number {
  const pos: number[] = [];
  for (let i = 0; i < data.length; i++) if (data[i]! > 0) pos.push(data[i]!);
  if (!pos.length) return 1;
  pos.sort((a, b) => a - b);
  return pos[Math.min(pos.length - 1, Math.floor(pos.length * 0.995))]! || 1;
}

/** RGBA pixels (one per cell); alpha grows with density so empty areas stay see-through. */
export function heatRgba(data: ArrayLike<number>, width: number, height: number, maxAlpha = 0.75): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 4);
  const s = heatScale(data);
  for (let i = 0; i < width * height; i++) {
    const t = Math.min(1, Math.max(0, data[i]! / s));
    if (t < 0.02) continue;
    const [r, g, b] = heatColor(t);
    out[4 * i] = r;
    out[4 * i + 1] = g;
    out[4 * i + 2] = b;
    out[4 * i + 3] = Math.round(255 * maxAlpha * Math.sqrt(t));
  }
  return out;
}
