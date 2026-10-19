# Testing and evaluation

## Commands

```bash
npm test          # unit + regression tests (Node, real OpenCV.js build)
npm run eval      # synthetic evaluation tables (prints metrics)
npm run typecheck
npm run build     # production build incl. service worker
```

Diagnostics for one scene (prints stage-by-stage recall, missed/false circles and
optionally writes an annotated PNG):

```bash
DEBUG_SCENE=clutter DEBUG_PNG=/tmp/clutter.png npx vitest run tests/eval/debug
```

## What is tested

| Area | File | Kind |
|---|---|---|
| Circle detector: precision/recall floors per scene, radius error, tube wall vs bore, monotone sensitivity, empty image | `tests/unit/detector.test.ts` | regression on synthetic scenes |
| Size clustering: 1 vs 4 groups, noise robustness, small distinct groups, fixed k | `tests/unit/clustering.test.ts` | unit |
| Corrections survive re-analysis; rejected detections stay rejected; counts after add/remove; numbering; grouping stability; merge/split; calibration | `tests/unit/project.test.ts` | unit |
| Undo/redo, drag coalescing, history bound | `tests/unit/history.test.ts` | unit |
| CSV (Czech decimals, BOM, mm only when calibrated); JSON project round-trip and validation | `tests/unit/serialize.test.ts` | unit |

Synthetic scenes (`tests/synthetic/scene.ts`) are deterministic (seeded) renders of
pipe/rod end faces with exact ground truth: hex packing or random packing, mixed
radii, hollow tubes, uneven light, noise, blur, line/rectangle clutter, vertical
squash (≈ perspective), and an occluding bar. Matching: one-to-one greedy by centre
distance < 0.5·r.

## Results (synthetic, 2026-10-19)

Sensitivity 50, analysis at scene resolution, Node 24 on an Intel i5-12500 (single
thread, OpenCV.js WASM). "Baseline" = single-band Hough with distance-transform size
estimate, i.e. the approach observed on the reference site, re-implemented for
comparison (`tests/eval/baseline.ts`) — not the reference site itself.

| scene | GT | CountIT | P | R | count err | r MAPE | ms | baseline | P | R | count err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform-rods | 296 | 292 | 1.000 | 0.986 | −4 | 2.5 % | 370 | 34 | 0.176 | 0.020 | −262 |
| uniform-tubes | 217 | 216 | 1.000 | 0.995 | −1 | 1.5 % | 501 | 0 | — | 0.000 | −217 |
| dense-small (r = 9 px) | 2496 | 1931 | 1.000 | 0.774 | −565 | 13.0 % | 631 | 264 | 0.443 | 0.047 | −2232 |
| mixed-4-sizes | 260 | 260 | 1.000 | 1.000 | 0 | 2.8 % | 350 | 156 | 1.000 | 0.600 | −104 |
| mixed-tubes | 220 | 220 | 1.000 | 1.000 | 0 | 3.5 % | 440 | 0 | — | 0.000 | −220 |
| uneven-light | 369 | 367 | 0.997 | 0.992 | −2 | 2.9 % | 295 | 361 | 1.000 | 0.978 | −8 |
| noisy-blurred | 369 | 399 | 0.847 | 0.916 | +30 | 8.4 % | 445 | 45 | 0.156 | 0.019 | −324 |
| clutter | 120 | 127 | 0.945 | 1.000 | +7 | 2.7 % | 275 | 120 | 1.000 | 1.000 | 0 |
| perspective-0.88 | 352 | 343 | 1.000 | 0.974 | −9 | 7.3 % | 348 | 100 | 1.000 | 0.284 | −252 |
| occluded | 280 | 263 | 1.000 | 0.939 | −17 | 2.6 % | 288 | 31 | 0.194 | 0.021 | −249 |

Size classes (detect → auto cluster):

| scene | true classes | found | class accuracy |
|---|---|---|---|
| mixed-4-sizes (20/26.9/33.7/42.4 mm ratios) | 4 | 4 | 99.2 % |
| mixed-tubes | 2 | 2 | 100 % |
| uniform-rods | 1 | 1 | 100 % |
| uniform-tubes | 1 | 1 | 100 % |

End-to-end in the browser (production build, Chromium): demo image with 162 pipes
(134 small + 28 large, ground truth from the generator) → 162 detected, 2 groups of
134 and 28, Ø 52 px / 71 px (true 52 / 72 px), analysis 0.5 s at 1400×1000. After
removing one, adding one and re-running: still 162 (161 automatic, 1 manual, 1
removed). Reload with the preview server **stopped**: app, saved project and
detection all worked offline.

## Not yet tested (honest gaps)

- **Real photographs.** No annotated real pipe photos are in the repo; synthetic
  scenes are cleaner than real ends (no rust, burrs, chamfers, specular highlights,
  labels, paint marks). This is the most important next step: collect 20–50 real
  photos, annotate centres + diameters, add them as an evaluation set.
- **Comparison with the reference site on equal inputs** — the site cannot be driven
  reproducibly from tests; only the re-created approach is compared.
- **Mobile devices** — layout checked in a 375×812 emulated viewport only; no
  measurements yet on real iOS/Android hardware (speed, memory, camera input).
- **Browser E2E automation** (Playwright) — not yet added; the flow above was
  verified manually through the app's browser pane.
- Memory use was not profiled.
