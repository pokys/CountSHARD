# Architecture

```
src/
  analysis/      pure algorithms (no DOM) — run in the worker and in Node tests
    circles.ts     multi-band Hough + edge-support verification + refinement
    clustering.ts  size classes (KDE modes / 1-D k-means)
    client.ts      main-thread client of the worker (progress, cancel, warm-up)
    cv.ts          OpenCV typing shim + Mat lifetime helper
  workers/       analysis.worker.ts (owns OpenCV.js), protocol.ts
  model/         types.ts — Project, CountedObject, SizeGroup, params
  state/         project.ts (pure operations), history.ts (undo/redo)
  io/            image.ts (decode/normalise), draw.ts (shared renderer),
                 export.ts (PNG/download), serialize.ts (CSV/JSON), db.ts (IndexedDB)
  pwa/           register.ts (service worker), offline.ts (readiness checks)
  i18n/          cs.ts (primary), en.ts, provider
  ui/            App, StartScreen, Workspace, Viewer, Panels, OfflineBadge, styles
  demo/          synthetic demo photo
tests/
  synthetic/     scene generator, metrics, PNG writer
  unit/          vitest suites (Node + real OpenCV.js)
  eval/          evaluation tables, baseline, diagnostics
```

## Data flow

1. **Import** (`io/image.ts`): decode with EXIF orientation, cap at 3072 px → *project
   image* (stored as a Blob in IndexedDB). All annotation coordinates are in this
   image's pixels.
2. **Analyse** (`ui/Workspace.tsx` → `analysis/client.ts` → worker): the project
   image is downscaled to `analysisMaxSide` (1600/2400), RGBA is transferred to the
   worker, converted to grey there, and `detectCircles` runs. Results are scaled
   back by `toProject`. Cancel = terminate the worker.
3. **Merge** (`state/project.ts#mergeDetections`): replaces only unreviewed automatic
   objects; manual/confirmed/rejected objects are preserved.
4. **Group** (`applyGrouping`): clusters unlocked objects, keeps group identities.
5. **Edit**: every change is a pure function `Project → Project`, committed into the
   undo history (`state/history.ts`); drags are coalesced into one undo step.
6. **Persist**: debounced autosave of the project JSON to IndexedDB.
7. **Export**: PNG (`renderAnnotatedPng`, same renderer as the viewer), CSV, JSON
   (project + embedded image, re-importable and fully editable).

## Object model

`CountedObject { id, x, y, r, source: auto|manual, review: unreviewed|confirmed|rejected,
score?, groupId, groupLocked }` — `rejected` objects are kept but not counted;
`groupLocked` marks manual group assignment.

`Project` also stores detection params, grouping params, calibration (mm/px with its
reference), analysis metadata (algorithm id, engine version, duration, resolution)
and the app version — so an exported file documents how its numbers were produced.

## Threads and memory

- Main thread: UI, canvas rendering, image decode/scale.
- Worker: OpenCV.js (WASM, single thread). Every OpenCV `Mat` is released via
  `withMats` (WASM heap has no GC).
- The project image is capped at 3072 px (≈ 38 MB RGBA worst case incl. copies).

## Offline

`vite-plugin-pwa` (`generateSW`) precaches `index.html`, JS/CSS (incl. the 14.5 MB
worker with embedded OpenCV WASM), icons and the manifest. No runtime network access
is needed after install. `pwa/offline.ts` verifies each layer separately.

## Extension points

- New detectors (e.g. AI) → new worker request type in `workers/protocol.ts`, results
  merged through the same `mergeDetections` (instance models) or a separate
  estimate-only result type (density models).
- Multi-shot → a `MultiShotProject` holding several `ImageInfo`s with homographies to
  a reference; fusion produces ordinary `CountedObject`s in reference coordinates plus
  per-object observation counts.
