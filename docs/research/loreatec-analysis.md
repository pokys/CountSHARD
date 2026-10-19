# Reference analysis: LoreaTec "Count objects in a photo"

URL: <https://tools.loreatec.jp/image/count-objects/> · Inspected 2026-10-19.

**Method.** Only publicly served resources were inspected: the page in a browser, the
JavaScript modules and manifest the page itself loads, and HTTP response headers.
Tools used:

- REA `analyze_web_bundle` against a local headless Chrome over CDP (evidence id
  `ev_848dcb9e60d5b978e1c01af093451e03466382ee52e8b5537ab390fe79875e46`): 10 scripts
  parsed, 0 parse failures, no source maps served, bundler fingerprint *Vite*.
- A built-in browser pane (DOM, `performance` resource timing, FAQ text).
- `curl` for the same public assets (to read the minified module) and headers.

No access restriction was bypassed. No code, text, model file or asset from the
site was copied into this repository. The bundle is minified, proprietary JavaScript
with no licence statement; it is **not** treated as open source. CountIT's detector
was designed and written independently (see `technical-decisions.md`); the only place
where the observed *approach* is re-created is the evaluation baseline in
`tests/eval/baseline.ts`, written from the OpenCV documentation to measure against.

Legend: **VERIFIED** = directly observed · **INFERRED** = reasoned from observations ·
**UNKNOWN** = not determinable from public behaviour.

## 1. Features

| Finding | Status | Evidence |
|---|---|---|
| Two modes: "Round, similar objects (fast)" and "Anything — mark 1–3 examples (AI, 35 MB once)" | VERIFIED | UI strings in `count-objects.*.js` |
| Image import via drop / file picker / camera | VERIFIED | page UI, page text |
| Photo downscaled so longest side ≤ 1600 px before analysis | VERIFIED | call with constant 1600 on load |
| "Object diameter (px)" slider, auto-initialised | VERIFIED | UI strings + init code |
| "Sensitivity" slider | VERIFIED | UI strings |
| Correction: tap empty spot = add, tap mark = remove | VERIFIED | `tapHint` string + click handler |
| Corrections are lost on "Count again" | VERIFIED | `tapHint`: "Counting again resets your corrections." |
| Counter breakdown "N detected automatically, +a added by you, −r removed by you" | VERIFIED | `breakdown` string |
| AI mode: draw 1–3 boxes, result shown as a heat map + total | VERIFIED | code draws density as an overlay |
| Export: one annotated PNG named `count-<n>.png` | VERIFIED | export handler |
| No CSV/JSON export, no project saving, no undo, no size classes, no calibration | INFERRED | absent from UI strings and handlers |
| Zoom/pan of the image | INFERRED absent | no wheel/pinch handlers in the tool module; canvas is CSS-scaled |
| 14 UI languages, no Czech | VERIFIED | `hreflang` alternates (en, ja, es, zh, pt, fr, de, ru, it, ko, id, tr, nl, eu) |

## 2. Technology

| Finding | Status | Evidence |
|---|---|---|
| Vite build (Rolldown runtime chunk) | VERIFIED | REA inference `bundle_runtime: Vite`; `rolldown-runtime.*.js` |
| No UI framework (plain DOM modules) | INFERRED | module graph: `dom`, `ui`, `image`, `exif`, `isobmff` helpers |
| OpenCV.js `@techstark/opencv-js@5.0.0-release.1` loaded from jsDelivr | VERIFIED | `opencv.*.js` contains the URL; resource timing 3.50 MB transferred |
| OpenCV runs on the **main thread** | VERIFIED | `cv.imread` on the page canvas inside UI handlers; no worker for CV |
| ONNX Runtime Web 1.27.0, WASM backend, `numThreads = 1`, `graphOptimizationLevel: all` | VERIFIED | `ort` import + `env.wasm` settings |
| Model `/models/loca-count-int8.onnx`, 34,911,003 bytes, served by LoreaTec | VERIFIED | code constant + HEAD response |
| Model is LOCA (int8-quantised) | INFERRED (strong) | file name; inputs `image` [1,3,512,512] + `bboxes` [1,3,4]; output `density` — matches LOCA's interface |
| No WebGPU | VERIFIED | only `wasm` execution provider configured |
| Offline: Workbox service worker (`/sw.js`, `precacheAndRoute`) for the site shell | VERIFIED | `sw.js` content |
| Whether OpenCV/model are cached for offline use | UNKNOWN | the SW precache list is long; CDN/runtime caching rules were not fully traced |
| HEIC/EXIF handling helpers (`exif`, `isobmff` modules) | VERIFIED | module names; behaviour not traced |
| Analytics ping to `loreatec.jp/a/c.php` (page/title, no image) | VERIFIED | network resource list, page text |

## 3. Algorithms (as observed)

**Size estimation.** Grayscale → Gaussian blur 5×5 → Otsu threshold (inverted if
>55 % foreground) → distance transform → *global maximum* × 0.95 = radius estimate.
VERIFIED. Consequence (INFERRED, then measured — see §5): on a tightly packed bundle
the foreground is one connected blob, so the maximum of the distance transform is
the radius of the *bundle*, not of one pipe.

**Detection.** `medianBlur` (5 if r > 10 else 3) → `HoughCircles(HOUGH_GRADIENT,
dp 1.2, minDist 1.5r, Canny 120, accumulator 67 − sensitivity, radius 0.6r…1.45r)`.
VERIFIED. One radius band ≈ 2.4 : 1, so mixed diameters beyond that ratio cannot be
found in one pass; no verification, refinement, or duplicate logic beyond `minDist`.

**AI mode.** Image squashed (aspect ratio not preserved) to 512×512, ImageNet
normalisation, up to 3 boxes (repeated to fill 3 slots), output density summed over
pixels > 5 % of the maximum → rounded count. VERIFIED. No instance localisation; the
UI correctly shows a heat map rather than per-object marks.

## 4. Licensing findings

| Component | Licence | Notes |
|---|---|---|
| LoreaTec page/bundle | proprietary (no licence) | not reused |
| OpenCV.js (`@techstark/opencv-js`) | Apache-2.0 | used by CountIT, bundled locally |
| ONNX Runtime Web | MIT | candidate for Phase 4 |
| LOCA code (`github.com/djukicn/loca`) | MIT | the URL in the brief (`valeoai/LOCA`) returns 404 |
| LOCA pretrained weights | **unclear** | published on Google Drive without a separate licence; the backbone is initialised from **SwAV ResNet-50, whose repository is CC BY-NC 4.0** (non-commercial); trained on FSC-147 (research dataset) |
| GeCo (`jerpelhan/GeCo`) | MIT (code) | detection-based few-shot counter, weights licence to be checked |
| GECO2 | no licence asserted (GitHub `NOASSERTION`) | not usable until clarified |

**Implication:** shipping LOCA weights in a product that may be used commercially is
a legal risk that must be resolved by the owner before Phase 4 (see roadmap).

## 5. Measured comparison (synthetic suite)

The observed size-estimation + Hough approach was re-implemented from the OpenCV
documentation (`tests/eval/baseline.ts`) and run on the same synthetic scenes as
CountIT. It is a re-creation of the approach, **not** the reference site itself; the
site cannot be driven reproducibly from tests, and its UI parameters (manual diameter
slider) would let a user rescue many of the failures below.

| scene | ground truth | baseline count | CountIT count |
|---|---|---|---|
| uniform rods (hex-packed) | 296 | 34 | 292 |
| uniform tubes | 217 | 0 | 216 |
| mixed 4 sizes | 260 | 156 | 260 |
| uneven light | 369 | 361 | 367 |
| clutter | 120 | 120 | 127 |

Main cause of baseline failures: the distance-transform estimate returns the size of
the packed bundle (capped), so the radius band misses the pipes entirely. With a
correct manual diameter the baseline would do much better on single-size scenes; it
still cannot handle mixed sizes beyond ~2.4 : 1 in one pass. Full table: `docs/testing.md`.

## 6. What to reproduce, what to improve

Reproduce: zero-upload processing; instant classical mode; tap-to-fix; numbered PNG;
auto size estimate; honest heat map for density-only AI output.

Improve (all implemented in the MVP unless noted):
- detection off the main thread (Web Worker) with progress and cancel;
- multi-band detection + per-candidate verification instead of a single radius band;
- corrections that survive re-analysis; undo/redo; move/resize;
- size classes, per-class counts, optional mm calibration;
- zoom/pan/pinch for dense bundles;
- CSV/JSON export, local project history;
- OpenCV bundled and precached instead of fetched from a CDN at first use;
- explicit offline readiness per component;
- Czech UI;
- AI mode (Phase 4) only after the weights' licence is clarified, preferably with a
  detection-based model so objects can be individually corrected.

## 7. Unknowns

- Exact offline behaviour of the OpenCV CDN file and the model on the reference site.
- How LoreaTec produced the int8 ONNX export (export script, opset, calibration data).
- Accuracy of the reference site on real photographs (no shared test set exists).
