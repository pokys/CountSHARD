/**
 * Which AI model to use on this device. Measured on real devices (docs/testing.md, "On phones"):
 * - PC with WebGPU: GeCo2 1024 px on WebGPU ≈ 6 s per photo.
 * - Phones: GeCo2 on WebGPU crashes the tab (iPhone, Galaxy A34: GPU memory), on WASM it works:
 *   iPhone ≈ 16 s (768 px), Galaxy A34 ≈ 57–72 s.
 * So GeCo2 uses WebGPU only on non-mobile devices, otherwise WASM at 768 px. "Automaticky" picks
 * GeCo2 only where it is expected to finish within AUTO_MAX_SECONDS; the expectation comes from
 * run times measured on this device, or before the first run from a short CPU benchmark.
 */
import type { Geco2Size } from './geco2';

export type AiModelChoice = 'auto' | 'loca' | 'geco2';
export type ResolvedModel = 'loca' | 'geco2';
export type Geco2Variant = 'webgpu-1024' | 'wasm-768';

export interface DeviceProfile {
  mobile: boolean;
  webgpu: boolean;
  /** navigator.deviceMemory (GB, capped by browsers at 8), null when not reported (Safari, Firefox). */
  deviceMemoryGB: number | null;
  /** Duration of cpuBenchmark() in ms (lower = faster), null when not measured. */
  cpuBenchMs: number | null;
  /** Measured GeCo2 seconds per photo on this device. */
  measured: Partial<Record<Geco2Variant, number>>;
  /** GeCo2 crashed on WebGPU here before. */
  geco2ForceWasm: boolean;
}

export interface Geco2Plan {
  variant: Geco2Variant;
  backend: 'webgpu' | 'wasm';
  size: Geco2Size;
  /** Expected seconds per photo (measured or estimated). */
  seconds: number | null;
  measured: boolean;
}

/** "Automaticky" picks GeCo2 only below this expected time per photo. */
export const AUTO_MAX_SECONDS = 25;
/** Devices reporting less memory than this never get GeCo2 automatically (≈ 0.7 GB per run on WASM). */
export const MIN_MEMORY_GB = 3;
/** First-run estimate for GeCo2 1024 on a desktop GPU (Intel Gen12 iGPU measured 5.4–6.7 s). */
const WEBGPU_ESTIMATE_S = 8;
/**
 * GeCo2 768 on WASM, seconds per benchmark millisecond. Calibrated on one PC (Intel i5-12500,
 * Node 24: benchmark 29.7 ms, GeCo2 768 WASM 20.8 s); on other CPUs it is only an estimate until
 * the first run on the device is measured (measured times replace it).
 */
export const WASM_768_S_PER_BENCH_MS = 0.7;

export function geco2Plan(p: DeviceProfile): Geco2Plan {
  const gpu = p.webgpu && !p.mobile && !p.geco2ForceWasm;
  const variant: Geco2Variant = gpu ? 'webgpu-1024' : 'wasm-768';
  const m = p.measured[variant];
  const est = gpu ? WEBGPU_ESTIMATE_S : p.cpuBenchMs !== null ? p.cpuBenchMs * WASM_768_S_PER_BENCH_MS : null;
  return { variant, backend: gpu ? 'webgpu' : 'wasm', size: gpu ? 1024 : 768, seconds: m ?? est, measured: m !== undefined };
}

export interface ModelDecision {
  model: ResolvedModel;
  /** Why "Automaticky" chose it (for the UI); null for a manual choice. */
  reason: 'geco2-fast' | 'geco2-slow' | 'low-memory' | 'unknown-speed' | null;
}

export function resolveModel(choice: AiModelChoice, p: DeviceProfile): ModelDecision {
  if (choice !== 'auto') return { model: choice, reason: null };
  if (p.deviceMemoryGB !== null && p.deviceMemoryGB < MIN_MEMORY_GB) return { model: 'loca', reason: 'low-memory' };
  const plan = geco2Plan(p);
  if (plan.seconds === null) return { model: 'loca', reason: 'unknown-speed' };
  return plan.seconds <= AUTO_MAX_SECONDS ? { model: 'geco2', reason: 'geco2-fast' } : { model: 'loca', reason: 'geco2-slow' };
}

/** Warning when GeCo2 is chosen manually on a device where it is slow or short of memory. */
export function heavyModelWarning(p: DeviceProfile): { kind: 'slow'; seconds: number } | { kind: 'memory' } | null {
  if (p.deviceMemoryGB !== null && p.deviceMemoryGB < MIN_MEMORY_GB) return { kind: 'memory' };
  const plan = geco2Plan(p);
  return plan.seconds !== null && plan.seconds > AUTO_MAX_SECONDS ? { kind: 'slow', seconds: plan.seconds } : null;
}

/** Exponential average of measured run times (the first run includes warm-up, later ones dominate). */
export function updateMeasured(prev: number | undefined, seconds: number): number {
  return prev === undefined ? seconds : 0.5 * prev + 0.5 * seconds;
}

/**
 * Short single-thread CPU benchmark (≈ 50–300 ms): a dense float matrix product, the core of the
 * model's work. Returns the best of three runs in ms.
 */
export function cpuBenchmark(n = 180): number {
  const a = new Float32Array(n * n), b = new Float32Array(n * n), c = new Float32Array(n * n);
  for (let i = 0; i < a.length; i++) {
    a[i] = (i % 13) * 0.01;
    b[i] = (i % 7) * 0.02;
  }
  let best = Infinity;
  for (let r = 0; r < 3; r++) {
    const t0 = performance.now();
    for (let rep = 0; rep < 4; rep++) {
      for (let i = 0; i < n; i++)
        for (let k = 0; k < n; k++) {
          const v = a[i * n + k]!;
          for (let j = 0; j < n; j++) c[i * n + j]! += v * b[k * n + j]!;
        }
    }
    best = Math.min(best, performance.now() - t0);
  }
  if (c[0] === -1) console.log(c[0]); // keep the work observable
  return best;
}

// --- browser glue ------------------------------------------------------------------------

const KEY_CHOICE = 'countshard.ai.model';
const KEY_BENCH = 'countshard.ai.cpuBench.v1';
const KEY_MEASURED = 'countshard.ai.geco2.seconds';
export const KEY_GECO2_FORCE_WASM = 'countshard.ai.geco2.forceWasm';

function get(k: string): string | null {
  try {
    return localStorage.getItem(k);
  } catch {
    return null;
  }
}
function set(k: string, v: string | null) {
  try {
    if (v === null) localStorage.removeItem(k);
    else localStorage.setItem(k, v);
  } catch {
    /* storage unavailable: choices are not remembered */
  }
}

export function loadChoice(): AiModelChoice {
  const v = get(KEY_CHOICE);
  return v === 'loca' || v === 'geco2' ? v : 'auto';
}
export function saveChoice(c: AiModelChoice) {
  set(KEY_CHOICE, c === 'auto' ? null : c);
}

export function isMobileDevice(nav: Navigator = navigator): boolean {
  const uad = (nav as Navigator & { userAgentData?: { mobile?: boolean } }).userAgentData;
  if (uad && typeof uad.mobile === 'boolean' && uad.mobile) return true;
  if (/Android|iPhone|iPad|iPod|Mobile/i.test(nav.userAgent)) return true;
  // iPadOS reports itself as a Mac.
  return /Macintosh/.test(nav.userAgent) && (nav.maxTouchPoints ?? 0) > 1;
}

function loadMeasured(): DeviceProfile['measured'] {
  try {
    const o = JSON.parse(get(KEY_MEASURED) ?? '{}') as Record<string, unknown>;
    const out: DeviceProfile['measured'] = {};
    for (const k of ['webgpu-1024', 'wasm-768'] as const) if (typeof o[k] === 'number' && Number.isFinite(o[k])) out[k] = o[k] as number;
    return out;
  } catch {
    return {};
  }
}

export function recordGeco2Seconds(variant: Geco2Variant, seconds: number) {
  const m = loadMeasured();
  m[variant] = updateMeasured(m[variant], seconds);
  set(KEY_MEASURED, JSON.stringify(m));
}

export function setGeco2ForceWasm() {
  set(KEY_GECO2_FORCE_WASM, '1');
}

let profilePromise: Promise<DeviceProfile> | null = null;

/** Device profile; WebGPU adapter and CPU benchmark are measured once per page load / device. */
export function deviceProfile(refresh = false): Promise<DeviceProfile> {
  if (profilePromise && !refresh) return profilePromise.then((p) => ({ ...p, measured: loadMeasured(), geco2ForceWasm: get(KEY_GECO2_FORCE_WASM) === '1' }));
  profilePromise = (async () => {
    let webgpu = false;
    try {
      const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
      webgpu = !!(gpu && (await gpu.requestAdapter()));
    } catch {
      webgpu = false;
    }
    let bench = Number(get(KEY_BENCH));
    if (!(bench > 0)) {
      bench = cpuBenchmark();
      set(KEY_BENCH, String(Math.round(bench * 10) / 10));
    }
    const dm = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    return {
      mobile: isMobileDevice(),
      webgpu,
      deviceMemoryGB: typeof dm === 'number' ? dm : null,
      cpuBenchMs: bench,
      measured: loadMeasured(),
      geco2ForceWasm: get(KEY_GECO2_FORCE_WASM) === '1',
    };
  })();
  return profilePromise;
}
