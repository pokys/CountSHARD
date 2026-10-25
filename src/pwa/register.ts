/**
 * Service worker registration (vite-plugin-pwa). Updates are applied only after the
 * user agrees, so an analysis in progress is never interrupted by a reload.
 */
let registered = false;

/**
 * The browser looks for a new service worker only when the app is (re)loaded. An installed app
 * that stays open, or is resumed from the background on a phone, would never see an update, so
 * it is checked on return to the app (at most every CHECK_MIN_MS) and every CHECK_EVERY_MS.
 */
const CHECK_EVERY_MS = 30 * 60 * 1000;
const CHECK_MIN_MS = 60 * 1000;

function watchForUpdates(r: ServiceWorkerRegistration) {
  let last = Date.now();
  const check = () => {
    if (Date.now() - last < CHECK_MIN_MS || !navigator.onLine) return;
    last = Date.now();
    r.update().catch(() => undefined);
  };
  setInterval(check, CHECK_EVERY_MS);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') check();
  });
  window.addEventListener('online', check);
}

export function registerServiceWorker(cb: { onReady(): void; onNeedRefresh(apply: () => void): void }) {
  if (registered || import.meta.env.DEV || !('serviceWorker' in navigator)) return;
  registered = true;
  import('virtual:pwa-register')
    .then(({ registerSW }) => {
      const update = registerSW({
        immediate: true,
        onOfflineReady: cb.onReady,
        onRegisteredSW: (_url, r) => {
          cb.onReady();
          if (r) watchForUpdates(r);
        },
        onNeedRefresh: () => cb.onNeedRefresh(() => void update(true)),
      });
    })
    .catch(() => {
      /* SW unavailable */
    });
}
