/**
 * Service worker registration (vite-plugin-pwa). Updates are applied only after the
 * user agrees, so an analysis in progress is never interrupted by a reload.
 */
let registered = false;

export function registerServiceWorker(cb: { onReady(): void; onNeedRefresh(apply: () => void): void }) {
  if (registered || import.meta.env.DEV || !('serviceWorker' in navigator)) return;
  registered = true;
  import('virtual:pwa-register')
    .then(({ registerSW }) => {
      const update = registerSW({
        immediate: true,
        onOfflineReady: cb.onReady,
        onRegisteredSW: () => cb.onReady(),
        onNeedRefresh: () => cb.onNeedRefresh(() => void update(true)),
      });
    })
    .catch(() => {
      /* SW unavailable */
    });
}
