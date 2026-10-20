# Architecture

```
src/
  analysis/      pure algorithms (no DOM) — run in the worker and in Node tests
    circles.ts     multi-band Hough + edge-support verification + refinement
    clustering.ts  size classes (KDE modes / 1-D k-means)
    client.ts      main-thread client of the worker (progress, cancel, warm-up)
    cv.ts          OpenCV typing shim + Mat lifetime helper
  ai/            Mode B (generic objects: LOCA density estimate or GeCo2 detections) — pure except client/modelStore/device glue
    geco2.ts       GeCo2 scale/input/boxes, detection post-processing (peaks + NMS), editing of detections
    device.ts      device profile (mobile, WebGPU, deviceMemory, CPU benchmark, measured run times) → model choice
    tiling.ts      aspect-preserving scale from exemplar size, 512 px tiles with ownership
    tensor.ts      RGBA → normalised NCHW (letterbox), resize, stitching into a density grid
    runner.ts      runs a tile plan on an abstract backend (worker: ORT-Web; eval: ORT-Web/Node)
    estimate.ts    density storage, region sums, user corrections, pure state operations
    heatmap.ts     density → heat-map RGBA
    modelStore.ts  lazy download of model + ORT WASM into Cache Storage (progress, SHA-256)
    client.ts      main-thread client of the AI worker + "downloaded / ready" status
  workers/       analysis.worker.ts (owns OpenCV.js), protocol.ts,
                 ai.worker.ts (owns onnxruntime-web + the sessions of one model: LOCA's two graphs or
                 GeCo2's three), aiProtocol.ts
  model/         types.ts — Project, CountedObject, SizeGroup, params
  state/         project.ts (pure operations), history.ts (undo/redo)
  io/            image.ts (decode/normalise), draw.ts (shared renderer),
                 export.ts (PNG/download), serialize.ts (CSV/JSON), db.ts (IndexedDB)
  pwa/           register.ts (service worker), offline.ts (readiness checks)
public/models/loca/  loca-features-w8.onnx, loca-head-w8.onnx (served, not precached)
scripts/loca/        how the ONNX files were exported, quantised and verified
  i18n/          cs.ts (primary), en.ts, provider
  ui/            App, StartScreen, Workspace, Viewer, Panels, AiPanels, OfflineBadge, styles
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

## Mode B data flow (generic objects, AI estimate)

1. **Mode switch** (`Project.mode`: `circles` | `ai`) in the workspace. The two modes share
   the photo, undo history, persistence and export panel; their results never mix.
2. **Download on demand** (`ai/modelStore.ts`): model graphs (34.9 MB) and the ORT-Web WASM
   (28.3 MB) are fetched with progress, size- and hash-checked and stored in Cache Storage.
3. **Exemplars**: the user draws 1–3 boxes (`AiState.exemplars`, project pixels).
4. **Plan** (`ai/tiling.ts`, main thread): scale from the exemplar size, tiles, exemplar
   window; the photo is resized to the plan size on a canvas and transferred to the worker.
5. **Inference** (`ai.worker.ts` → `ai/runner.ts`): `features` for the exemplar window and
   every tile, `head` per tile with the exemplar features; owned parts of each 512 × 512
   density map are summed into a grid with 4 × 4 model-pixel cells. WebGPU or WASM.
6. **Result** (`AiDensityResult`): grid (Float32, base64), raw total, scale, tiles, exemplars
   used, engine, duration — stored in the project; `AiState.corrections` holds the user's
   region counts. `summarize()` gives raw and corrected estimates.
7. **Display/export**: heat map + exemplar boxes + region boxes (`drawAiOverlay`, shared by
   viewer and PNG), CSV with explicit estimate rows, JSON with `summary.estimate.isEstimate`.

### GeCo2 (second Mode B model, individual detections)

1. **Model choice** (`ai/device.ts`, `ai/client.ts`): per-device choice in localStorage
   (`auto` | `loca` | `geco2`). `auto` resolves from the device profile: GeCo2 only when the
   expected time per photo is ≤ 25 s (measured on this device, otherwise WebGPU ≈ 8 s or a CPU
   benchmark × calibration) and the device does not report < 3 GB memory. GeCo2 runs on WebGPU
   at 1024 px only on non-mobile devices; phones always use WASM at 768 px (WebGPU crashed both
   test phones).
2. **Download** (`geco2Assets(size)`): three graphs per input size from
   `https://pokys.github.io/geco2-bench/models/` (CORS allowed), size + SHA-256 checked, cache
   `countshard-geco2-v1`. The ORT WASM runtime is shared: found via `caches.match` in either
   model's cache, moved to the other cache before a delete.
3. **Inference** (`ai.worker.ts` `detect`): photo scaled on the main thread (`geco2Scale`:
   longer side → S, mean exemplar side ≤ 80·S/1024), padded to S × S in the worker; backbone and
   head on WebGPU or WASM, the tiny exemplar graph (RoIAlign) always on WASM; `geco2Detect`
   (local maxima > max/8, NMS 0.5, centres in the padding dropped) → boxes in model pixels.
4. **Result** (`AiDetectionResult`): detections in project pixels (`AiDetection`: box, score,
   source, review), model id/name, engine, input size, scale. A re-run merges like circle mode
   (`mergeAiDetections`: manual/confirmed kept, rejected stay rejected).
5. **Display/edit**: detections are drawn and edited through the circle-mode path
   (`asCountedObjects` + `boxes` in `drawAnnotations`): numbered boxes, tap removes (auto →
   rejected) or adds (typical neighbour size), counting area respected, undo/redo.
6. **Export**: PNG with numbered boxes and the model in the footer, CSV one row per counted
   object with the model on every row, JSON `summary.detections` (`isEstimate: false`).
7. **Crash guard**: the run marker records model + backend; after a GeCo2 crash on WebGPU the
   device uses WASM for GeCo2, and the AI panel offers switching to LOCA.

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
worker with embedded OpenCV WASM and the 0.4 MB AI worker), icons and the manifest. No
runtime network access is needed after install for Mode A. Mode B files (`.onnx`, ORT
`.wasm`) are excluded from the precache by its glob patterns and stored in a separate
caches (`countshard-ai-v2` for LOCA, `countshard-geco2-v1` for GeCo2) after the user downloads them. `pwa/offline.ts` and
`ai/client.ts` verify each layer separately.

## Extension points

- New detectors: instance models (individually correctable points/boxes) → results
  merged through `mergeDetections`; density models → the Mode B estimate type
  (`AiDensityResult`), as LOCA does now. A different ONNX model only needs new files in
  `public/models/`, `ai/modelStore.ts` and the backend in `ai.worker.ts`.
- Multi-shot (implemented, `analysis/registration.ts`, `analysis/fusion.ts`, worker
  request `fuse`): the project photo is the reference; extra photos are registered
  onto it (ORB + RANSAC homography), detections are fused into ordinary
  `CountedObject`s in reference coordinates with `multiShot` = seen in k of n photos.
  Extra photos are held in memory only; the fused result goes through the same
  `mergeDetections`, so corrections and undo behave as for a single photo.
