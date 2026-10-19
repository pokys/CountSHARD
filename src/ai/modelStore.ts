/**
 * Lazy download of the Mode B files (LOCA int8 model + onnxruntime-web WASM) into Cache
 * Storage. They are deliberately NOT in the service-worker precache: ~63 MB that only users of
 * AI counting need. After the first download they work offline like the rest of the app.
 */
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';

export const AI_CACHE = 'countshard-ai-v1';
export const AI_MODEL_ID = 'LOCA few-shot (Đukić et al., ICCV 2023), int8 weight-only ONNX export v1';
export const ORT_VERSION = '1.30.0';

export interface AiAsset {
  key: 'features' | 'head' | 'wasm';
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
      bytes: 28891798,
      sha256: '11383053c6d34834ad44a2268b735b43e9d9b32520aada249b6f271261b0d8af',
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

export function totalBytes(assets = aiAssets()): number {
  return assets.reduce((a, b) => a + b.bytes, 0);
}

export async function aiDownloaded(assets = aiAssets()): Promise<boolean> {
  if (!('caches' in self)) return false;
  try {
    const cache = await caches.open(AI_CACHE);
    for (const a of assets) if (!(await cache.match(a.url))) return false;
    return true;
  } catch {
    return false;
  }
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
export async function downloadAi(onProgress?: (loaded: number, total: number) => void, signal?: AbortSignal, assets = aiAssets()): Promise<void> {
  if (!('caches' in self)) throw new AiDownloadError('unsupported');
  const cache = await caches.open(AI_CACHE);
  const total = totalBytes(assets);
  let loaded = 0;
  for (const a of assets) {
    if (await cache.match(a.url)) {
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

export async function deleteAi(): Promise<void> {
  if ('caches' in self) await caches.delete(AI_CACHE);
}

/** Reads a cached asset (also used inside the worker). */
export async function readCached(url: string): Promise<ArrayBuffer> {
  const cache = await caches.open(AI_CACHE);
  const r = await cache.match(url);
  if (!r) throw new AiDownloadError('storage', `missing ${url}`);
  return r.arrayBuffer();
}
