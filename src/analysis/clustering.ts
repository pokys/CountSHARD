/**
 * 1-D size clustering of measured radii (see docs/technical-decisions.md, "Size classes").
 *
 * Works in log-radius space, where a fixed relative measurement error has constant
 * width regardless of object size (20 mm vs 42 mm pipes are measured with similar
 * *relative* error).
 *
 * - `auto`: kernel density estimate + mode finding. Modes closer than the minimum
 *   separation, or separated only by a shallow valley, are merged. This decides the
 *   number of groups without assuming it. In 1-D, density-mode finding covers what
 *   DBSCAN/HDBSCAN would do, with one interpretable parameter (minimum separation).
 * - `fixed`: k groups via 1-D k-means (Lloyd with quantile initialisation).
 */

/** Smallest group the automatic mode will report. */
export const MIN_GROUP_SIZE = 3;

export interface ClusterResult {
  /** Cluster index per input value; clusters are ordered by ascending radius. */
  labels: number[];
  /** Median radius per cluster. */
  centers: number[];
}

export interface ClusterOptions {
  mode: 'auto' | 'fixed';
  k?: number;
  /** Minimum relative difference between group centres in auto mode (0.1 = 10 %). */
  minSeparation?: number;
}

export function clusterRadii(radii: number[], opts: ClusterOptions): ClusterResult {
  const n = radii.length;
  if (n === 0) return { labels: [], centers: [] };
  const logs = radii.map((r) => Math.log(Math.max(r, 1e-6)));
  const boundaries = opts.mode === 'fixed' ? kmeansBoundaries(logs, Math.max(1, Math.min(opts.k ?? 2, n))) : kdeBoundaries(logs, opts.minSeparation ?? 0.1);
  const assign = () =>
    logs.map((v) => {
      let i = 0;
      while (i < boundaries.length && v > boundaries[i]!) i++;
      return i;
    });
  let labels = assign();
  if (opts.mode === 'auto') {
    // A handful of outliers in a tail must not become a "size class": groups need ≥ 3 members.
    for (;;) {
      const counts = new Array(boundaries.length + 1).fill(0);
      for (const l of labels) counts[l]++;
      const small = counts.findIndex((c) => c > 0 && c < MIN_GROUP_SIZE);
      if (small < 0 || boundaries.length === 0) break;
      // Remove the boundary towards the closer neighbouring cluster.
      const center = (i: number) => median(logs.filter((_, j) => labels[j] === i));
      const left = small > 0 ? Math.abs(center(small) - center(small - 1)) : Infinity;
      const right = small < boundaries.length ? Math.abs(center(small + 1) - center(small)) : Infinity;
      boundaries.splice(left <= right ? small - 1 : small, 1);
      labels = assign();
    }
  }
  // Drop empty clusters and renumber.
  const used = [...new Set(labels)].sort((a, b) => a - b);
  const remap = new Map(used.map((c, i) => [c, i]));
  const finalLabels = labels.map((l) => remap.get(l)!);
  const centers = used.map((_, i) => median(radii.filter((_, j) => finalLabels[j] === i)));
  return { labels: finalLabels, centers };
}

/** Returns ascending decision boundaries (in log space) between clusters. */
function kdeBoundaries(logs: number[], minSeparation: number): number[] {
  const n = logs.length;
  if (n < 3) return [];
  const sorted = [...logs].sort((a, b) => a - b);
  const sep = Math.log(1 + minSeparation);
  // Bandwidth: Silverman's rule is tuned for unimodal data and over-smooths mixtures,
  // so it is capped by the separation we want to resolve.
  const sd = stddev(sorted);
  const iqr = (quantile(sorted, 0.75) - quantile(sorted, 0.25)) / 1.34;
  const silverman = 0.9 * Math.min(sd, iqr > 0 ? iqr : sd) * Math.pow(n, -0.2);
  const h = Math.max(0.012, Math.min(silverman, sep / 2.5));
  const lo = sorted[0]! - 3 * h;
  const hi = sorted[n - 1]! + 3 * h;
  const G = 400;
  const step = (hi - lo) / (G - 1);
  const dens = new Float64Array(G);
  for (const v of sorted) {
    const g0 = Math.max(0, Math.floor((v - 4 * h - lo) / step));
    const g1 = Math.min(G - 1, Math.ceil((v + 4 * h - lo) / step));
    for (let g = g0; g <= g1; g++) {
      const z = (lo + g * step - v) / h;
      dens[g] = dens[g]! + Math.exp(-0.5 * z * z);
    }
  }
  // Local maxima.
  let modes: number[] = [];
  for (let g = 1; g < G - 1; g++) if (dens[g]! > dens[g - 1]! && dens[g]! >= dens[g + 1]!) modes.push(g);
  if (modes.length <= 1) return [];
  const valley = (a: number, b: number) => {
    let m = a;
    for (let g = a; g <= b; g++) if (dens[g]! < dens[m]!) m = g;
    return m;
  };
  // Merge the weakest adjacent pair until all pairs are distinct enough.
  for (;;) {
    let worst = -1;
    let worstScore = Infinity;
    for (let i = 0; i < modes.length - 1; i++) {
      const a = modes[i]!;
      const b = modes[i + 1]!;
      const dist = (b - a) * step;
      const v = dens[valley(a, b)]!;
      const depth = v / Math.min(dens[a]!, dens[b]!); // 1 = no valley
      const tooClose = dist < sep;
      const tooShallow = depth > 0.75;
      if (tooClose || tooShallow) {
        const score = dist / sep - depth; // smaller = merge first
        if (score < worstScore) {
          worstScore = score;
          worst = i;
        }
      }
    }
    if (worst < 0) break;
    const a = modes[worst]!;
    const b = modes[worst + 1]!;
    modes = [...modes.slice(0, worst), dens[a]! >= dens[b]! ? a : b, ...modes.slice(worst + 2)];
  }
  const out: number[] = [];
  for (let i = 0; i < modes.length - 1; i++) out.push(lo + valley(modes[i]!, modes[i + 1]!) * step);
  return out;
}

function kmeansBoundaries(logs: number[], k: number): number[] {
  if (k <= 1) return [];
  const sorted = [...logs].sort((a, b) => a - b);
  let centers = Array.from({ length: k }, (_, i) => quantile(sorted, (i + 0.5) / k));
  for (let iter = 0; iter < 100; iter++) {
    const sums = new Array(k).fill(0);
    const counts = new Array(k).fill(0);
    for (const v of sorted) {
      let best = 0;
      for (let c = 1; c < k; c++) if (Math.abs(v - centers[c]!) < Math.abs(v - centers[best]!)) best = c;
      sums[best] += v;
      counts[best]++;
    }
    const next = centers.map((c, i) => (counts[i] ? sums[i] / counts[i] : c));
    const moved = next.some((c, i) => Math.abs(c - centers[i]!) > 1e-9);
    centers = next.sort((a, b) => a - b);
    if (!moved) break;
  }
  const out: number[] = [];
  for (let i = 0; i < k - 1; i++) out.push((centers[i]! + centers[i + 1]!) / 2);
  return out;
}

export function median(v: number[]): number {
  if (!v.length) return NaN;
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const i = Math.floor(pos);
  const f = pos - i;
  return sorted[i]! + (sorted[Math.min(i + 1, sorted.length - 1)]! - sorted[i]!) * f;
}

function stddev(v: number[]): number {
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, v.length - 1));
}

/** Robust relative spread of a group: MAD / median (≈ coefficient of variation for normal data). */
export function relativeSpread(radii: number[]): number {
  if (radii.length < 2) return 0;
  const m = median(radii);
  return (1.4826 * median(radii.map((r) => Math.abs(r - m)))) / m;
}
