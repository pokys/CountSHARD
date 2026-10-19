/**
 * Offline readiness. Each layer is checked separately instead of assuming that a
 * registered service worker means everything is available offline.
 */
export interface OfflineStatus {
  /** Service workers unavailable (dev server, insecure context, unsupported browser). */
  unsupported: boolean;
  shellCached: boolean;
  engineCached: boolean;
  engineLoaded: boolean;
  persisted: boolean;
  usage: number | null;
  quota: number | null;
}

async function cachedUrls(): Promise<string[]> {
  if (!('caches' in window)) return [];
  const urls: string[] = [];
  for (const name of await caches.keys()) {
    const c = await caches.open(name);
    for (const req of await c.keys()) urls.push(req.url);
  }
  return urls;
}

export async function checkOffline(engineLoaded: boolean): Promise<OfflineStatus> {
  const unsupported = import.meta.env.DEV || !('serviceWorker' in navigator) || !window.isSecureContext;
  let shellCached = false;
  let engineCached = false;
  if (!unsupported) {
    try {
      const urls = await cachedUrls();
      shellCached = urls.some((u) => /\/index\.html(\?|$)/.test(u)) && !!navigator.serviceWorker.controller;
      engineCached = urls.some((u) => /analysis\.worker[^/]*\.js/.test(u));
    } catch {
      /* caches unavailable */
    }
  }
  let persisted = false;
  let usage: number | null = null;
  let quota: number | null = null;
  try {
    persisted = (await navigator.storage?.persisted?.()) ?? false;
    const e = await navigator.storage?.estimate?.();
    usage = e?.usage ?? null;
    quota = e?.quota ?? null;
  } catch {
    /* ignore */
  }
  return { unsupported, shellCached, engineCached, engineLoaded, persisted, usage, quota };
}

export function isFullyOffline(s: OfflineStatus): boolean {
  return s.shellCached && s.engineCached;
}
