# Third-party software shipped in the application bundle

| Component | Version | Licence | Where |
|---|---|---|---|
| OpenCV (via `@techstark/opencv-js`) | 5.0.0-release.1 | Apache-2.0 | embedded in `assets/analysis.worker-*.js` |
| React, React DOM | 18.3.1 | MIT | main bundle |
| Workbox (window, service worker runtime via `vite-plugin-pwa`) | 7.4.1 | MIT | `sw.js`, `workbox-*.js` |

Build/test-only tools (not shipped): Vite 5.4 (MIT), Vitest 2.1 (MIT), TypeScript 5.6
(Apache-2.0), `@vitejs/plugin-react` (MIT), `vite-plugin-pwa` (MIT).

The Apache-2.0 licence requires that redistributions include the licence text and
any NOTICE file. The OpenCV licence is available at <https://github.com/opencv/opencv/blob/4.x/LICENSE>
and in `node_modules/@techstark/opencv-js/LICENSE` after `npm install`; include it
when deploying the built application.

No AI model weights are included. See `docs/research/loreatec-analysis.md` §4 for
the licensing situation of candidate models (LOCA, GeCo).
