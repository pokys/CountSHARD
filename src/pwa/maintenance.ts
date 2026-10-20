/**
 * Maintenance actions offered in the offline popover. Normally never needed: old app versions
 * and old AI models are removed automatically. They exist for a stuck update or a full reset.
 */
import { deleteAllData } from '../io/db';

const AI_CACHE_PREFIX = 'countshard-ai-';

async function unregisterServiceWorkers() {
  if (!('serviceWorker' in navigator)) return;
  for (const r of await navigator.serviceWorker.getRegistrations()) await r.unregister();
}

/** Re-downloads the app on the next load. Projects and the AI model are kept. */
export async function refreshApp(): Promise<void> {
  await unregisterServiceWorkers();
  if ('caches' in window) for (const k of await caches.keys()) if (!k.startsWith(AI_CACHE_PREFIX)) await caches.delete(k);
  location.reload();
}

/** Deletes projects, the AI model, caches and settings stored by this app on this device. */
export async function wipeEverything(): Promise<void> {
  await deleteAllData();
  if ('caches' in window) for (const k of await caches.keys()) await caches.delete(k);
  try {
    for (const k of Object.keys(localStorage)) if (k.startsWith('countshard')) localStorage.removeItem(k);
  } catch {
    /* storage unavailable */
  }
  await unregisterServiceWorkers();
  location.reload();
}

/** iPhone/iPad Safari outside the Home Screen app: storage may be evicted after 7 days unused. */
export function iosBrowserTab(): boolean {
  const ua = navigator.userAgent;
  const ios = /iPhone|iPad|iPod/.test(ua) || (ua.includes('Macintosh') && navigator.maxTouchPoints > 1);
  const standalone = matchMedia('(display-mode: standalone)').matches || (navigator as unknown as { standalone?: boolean }).standalone === true;
  return ios && !standalone;
}
