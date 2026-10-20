/**
 * Lazy download of the Mode B files (LOCA int8 model + onnxruntime-web WASM) into Cache
 * Storage. They are deliberately NOT in the service-worker precache: ~63 MB that only users of
 * AI counting need. After the first download they work offline like the rest of the app.
 */
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';
import type { Geco2Size } from './geco2';

export type { Geco2Size };

export const AI_CACHE = 'countshard-ai-v2';
export const AI_MODEL_ID = 'LOCA few-shot (Đukić et al., ICCV 2023), int8 weight-only ONNX export v2 (chunked attention)';
export const ORT_VERSION = '1.30.0';

export type AiModelId = 'loca' | 'geco2';

export interface AiAsset {
  key: 'features' | 'head' | 'wasm' | 'backbone' | 'exemplars';
  url: string;
  bytes: number;
  /** SHA-256 (hex) for the model files; the WASM runtime is checked by size. */
  sha256?: string;
}

/** Absolute URLs resolved against the page (the app is served from a sub-path). */
export function aiAssets(base: string = document.baseURI): AiAsset[] {
  return [
    {
      key: 'features',
      url: new URL('models/loca/loca-features-w8.onnx', base).href,
      bytes: 28951277,
      sha256: 'd4b181cf01f5453e6637f2090fdf92cacd5d80d212ed1bcf8be67396be0a7a84',
    },
    {
      key: 'head',
      url: new URL('models/loca/loca-head-w8.onnx', base).href,
      bytes: 6001344,
      sha256: '65208043a15430c2248170efe9d18c1d6d8bf08d78f58b000636d6dab98048f3',
    },
    { key: 'wasm', url: new URL(ortWasmUrl, base).href, bytes: 28312028 },
  ];
}

// --- GeCo2 -------------------------------------------------------------------------------

/**
 * GeCo2 files are not in this repository (≈ 80 MB per input size): they are served from the
 * separate GitHub Pages site of the GeCo2 export (CORS allowed), verified by size + SHA-256 and
 * kept in their own cache. Export: scripts/geco2/ (README there).
 */
export const GECO2_BASE = 'https://pokys.github.io/geco2-bench/models/';
export const GECO2_CACHE = 'countshard-geco2-v1';
export const GECO2_SIZES: Geco2Size[] = [768, 1024];

export function geco2ModelId(size: Geco2Size): string {
  return `GeCo2 (Pelhan, Lukežič, Kristan, AAAI 2026), int8 weight-only ONNX export, ${size} px input`;
}

const GECO2_FILES: Record<Geco2Size, Record<'backbone' | 'exemplars' | 'head', { bytes: number; sha256: string }>> = {
  768: {
    backbone: { bytes: 71078314, sha256: '70d86e7fcd51e8ef4a8cadddf7e578bf2f78f15340489c30faef0e9911fdcec3' },
    exemplars: { bytes: 96625, sha256: '9c867a5d08746733a9d709b4ccb95bd71d51eace39b7f48d410971c7ec312f21' },
    head: { bytes: 11659739, sha256: '80a5529749d228f5a2757c4b4f8a3091275dd7f8625498ff7771f005b18e6615' },
  },
  1024: {
    backbone: { bytes: 71303903, sha256: '7b084367f376181f8a19e249ba232e4a7efb9fc44dc830c920cba933ba27271e' },
    exemplars: { bytes: 96627, sha256: '5cdf1f8c00a237cf03e2d37ed6d7ff56fbb1c45d767203c2070ce297a4e312e1' },
    head: { bytes: 16042034, sha256: '31782c1a7cfe45a577bcebf2ffd28a9460f3014ee923641314ccd41493ff175c' },
  },
};

/** GeCo2 graphs of one input size + the shared onnxruntime-web WASM runtime. */
export function geco2Assets(size: Geco2Size, base: string = document.baseURI): AiAsset[] {
  const f = GECO2_FILES[size];
  return [
    ...(['backbone', 'exemplars', 'head'] as const).map((key) => ({ key, url: `${GECO2_BASE}geco2-${key}-${size}-w8.onnx`, ...f[key] })),
    { key: 'wasm' as const, url: new URL(ortWasmUrl, base).href, bytes: 28312028 },
  ];
}

/** Model files only (without the shared runtime), e.g. for the size shown in the offline menu. */
export function modelBytes(assets: AiAsset[]): number {
  return assets.filter((a) => a.key !== 'wasm').reduce((a, b) => a + b.bytes, 0);
}

export function totalBytes(assets = aiAssets()): number {
  return assets.reduce((a, b) => a + b.bytes, 0);
}

/** Removes caches of earlier model versions (the file URLs stay the same, the content changed). */
async function dropOldCaches() {
  for (const k of await caches.keys()) {
    if ((k.startsWith('countshard-ai-') && k !== AI_CACHE) || (k.startsWith('countshard-geco2-') && k !== GECO2_CACHE)) await caches.delete(k);
  }
}

/**
 * True when every file is cached. Looked up across all caches: the WASM runtime is shared and
 * may have been stored with either model.
 */
export async function aiDownloaded(assets = aiAssets()): Promise<boolean> {
  if (!('caches' in self)) return false;
  try {
    await dropOldCaches();
    for (const a of assets) if (!(await caches.match(a.url))) return false;
    return true;
  } catch {
    return false;
  }
}

/** Bytes still to download (files not cached yet). */
export async function missingBytes(assets: AiAsset[]): Promise<number> {
  if (!('caches' in self)) return totalBytes(assets);
  let n = 0;
  for (const a of assets) if (!(await caches.match(a.url).catch(() => undefined))) n += a.bytes;
  return n;
}

export class AiDownloadError extends Error {
  constructor(public code: 'offline' | 'http' | 'integrity' | 'storage' | 'unsupported', detail = '') {
    super(`${code}${detail ? `: ${detail}` : ''}`);
  }
}

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const d = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Downloads every missing file with byte-level progress, verifies it and stores it.
 * `onProgress(loaded, total)` counts bytes of all files, including ones already cached.
 */
export async function downloadAi(onProgress?: (loaded: number, total: number) => void, signal?: AbortSignal, assets = aiAssets(), cacheName = AI_CACHE): Promise<void> {
  if (!('caches' in self)) throw new AiDownloadError('unsupported');
  const cache = await caches.open(cacheName);
  const total = totalBytes(assets);
  let loaded = 0;
  for (const a of assets) {
    if (await caches.match(a.url)) {
      loaded += a.bytes;
      onProgress?.(loaded, total);
      continue;
    }
    let res: Response;
    try {
      res = await fetch(a.url, { cache: 'no-store', signal });
    } catch (e) {
      if (signal?.aborted) throw e;
      throw new AiDownloadError('offline', a.url);
    }
    if (!res.ok || !res.body) throw new AiDownloadError('http', `${res.status} ${a.url}`);
    const reader = res.body.getReader();
    const buf = new Uint8Array(a.bytes);
    let off = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (off + value.length > buf.length) throw new AiDownloadError('integrity', `${a.key}: larger than expected`);
      buf.set(value, off);
      off += value.length;
      onProgress?.(loaded + off, total);
    }
    if (off !== a.bytes) throw new AiDownloadError('integrity', `${a.key}: ${off} of ${a.bytes} bytes`);
    if (a.sha256 && (await sha256Hex(buf.buffer)) !== a.sha256) throw new AiDownloadError('integrity', `${a.key}: checksum`);
    const type = a.key === 'wasm' ? 'application/wasm' : 'application/octet-stream';
    try {
      await cache.put(a.url, new Response(buf, { headers: { 'Content-Type': type, 'Content-Length': String(a.bytes) } }));
    } catch (e) {
      throw new AiDownloadError('storage', e instanceof Error ? e.message : String(e));
    }
    loaded += a.bytes;
  }
  onProgress?.(total, total);
}

/**
 * The WASM runtime is stored once, in the cache of whichever model was downloaded first. Before
 * that cache is deleted, the runtime moves to the other model's cache if that model still has
 * its files (so deleting one model never breaks the other).
 */
async function keepRuntimeFor(fromCache: string, toCache: string, otherModelFiles: AiAsset[]) {
  const wasm = aiAssets().find((a) => a.key === 'wasm')!;
  const from = await caches.open(fromCache);
  const r = await from.match(wasm.url);
  if (!r) return;
  const to = await caches.open(toCache);
  for (const a of otherModelFiles) if (a.key !== 'wasm' && !(await to.match(a.url))) return;
  await to.put(wasm.url, r);
}

export async function deleteAi(): Promise<void> {
  if (!('caches' in self)) return;
  for (const s of GECO2_SIZES) await keepRuntimeFor(AI_CACHE, GECO2_CACHE, geco2Assets(s));
  await caches.delete(AI_CACHE);
}

/** Deletes one GeCo2 input size; the whole GeCo2 cache (incl. its copy of the runtime) when none is left. */
export async function deleteGeco2(size: Geco2Size): Promise<void> {
  if (!('caches' in self)) return;
  const cache = await caches.open(GECO2_CACHE);
  for (const a of geco2Assets(size)) if (a.key !== 'wasm') await cache.delete(a.url);
  const others = GECO2_SIZES.filter((s) => s !== size);
  let left = false;
  for (const s of others) for (const a of geco2Assets(s)) if (a.key !== 'wasm' && (await cache.match(a.url))) left = true;
  if (!left) {
    await keepRuntimeFor(GECO2_CACHE, AI_CACHE, aiAssets());
    await caches.delete(GECO2_CACHE);
  }
}

/** Reads a cached asset (also used inside the worker); any cache, see aiDownloaded. */
export async function readCached(url: string): Promise<ArrayBuffer> {
  const r = await caches.match(url);
  if (!r) throw new AiDownloadError('storage', `missing ${url}`);
  return r.arrayBuffer();
}
