/** Small 3×3 homography helpers (row-major), shared by registration, fusion and tests. */

export type Mat3 = [number, number, number, number, number, number, number, number, number];

export const IDENTITY3: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function mul3(a: Mat3, b: Mat3): Mat3 {
  const r: number[] = [];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let v = 0;
      for (let k = 0; k < 3; k++) v += a[i * 3 + k]! * b[k * 3 + j]!;
      r.push(v);
    }
  return r as Mat3;
}

export function invert3(m: Mat3): Mat3 | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;
  const inv: Mat3 = [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d];
  return inv.map((v) => v / det) as Mat3;
}

export function apply3(H: Mat3, x: number, y: number): [number, number] {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

/** Local isotropic scale of H at (x, y): sqrt(|det J|). Maps radii between views. */
export function localScale3(H: Mat3, x: number, y: number): number {
  const w = H[6] * x + H[7] * y + H[8];
  const [u, v] = apply3(H, x, y);
  const a = (H[0] - u * H[6]) / w, b = (H[1] - u * H[7]) / w;
  const c = (H[3] - v * H[6]) / w, d = (H[4] - v * H[7]) / w;
  return Math.sqrt(Math.abs(a * d - b * c));
}
