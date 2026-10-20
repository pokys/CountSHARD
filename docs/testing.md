# Testing and evaluation

## Commands

```bash
npm test          # unit + regression tests (Node, real OpenCV.js build)
npm run eval      # evaluation tables (prints metrics); the Mode B part takes ~15–30 min
node scripts/fetch-fsc147-sample.mjs   # once: 20 FSC-147 images for the Mode B eval (not committed)
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
| Multi-shot: registration recovers the true homography (< 2 px at the corners) and refuses a different scene; fused count = ground truth with zero duplicates on 4 synthetic series (2–5 photos); no duplicates on harder series; a different-scene photo is refused alone and left out among good ones without any effect; wrong-but-plausible registration (half a pitch off) and residual too large for the object size are refused; concentric bore/inner ring merges with its object; minority/uncertain policy | `tests/unit/multishot.test.ts` | regression on synthetic multi-view sets + unit |
| Fused objects in the project: corrections survive a fused merge; uncertain candidates are selectable but count only after confirmation; a removed candidate stays out on re-fusion | `tests/unit/project.test.ts` | unit |
| Mode B pre/post-processing: tile ownership partitions the image exactly (6 sizes), context margin, aspect ratio kept, exemplar-band scaling, tile limit, exemplar window + padding to 3 boxes, dropped far exemplars; ImageNet normalisation + letterbox, stitching preserves the density sum, resize; thresholded sum | `tests/unit/ai.test.ts` | unit |
| Mode B estimate correction: area-weighted region sums, corrected = raw − model(region) + user(region), count edit/remove, overlap/too-small/non-integer refused, never negative, ≤ 3 exemplars, stale result after exemplar change; heat-map ramp; CSV/JSON mark the estimate, round-trip, invalid density grid rejected, old projects default to round objects | `tests/unit/ai.test.ts` | unit |
| Shipped model: file sizes + SHA-256 equal the download manifest; the int8 model runs in onnxruntime-web (WASM, Node, app session options) and counts a 480 × 360 synthetic rod scene within 10 %; one tile grows RSS by < 250 MB (memory regression guard) | `tests/unit/aiModel.test.ts` | smoke |
| Crash detection: a leftover run marker is reported once; a WebGPU crash switches the device to WASM | `tests/unit/ai.test.ts` | unit |
| GeCo2: scale rule (longer side → S, exemplar cap 80·S/1024), ImageNet input + padding, exemplar boxes; detection post-processing (peaks > max/8, NMS, padding filter); edits (remove → rejected / manual deleted, re-run keeps edits, counting area, numbering, tap-added size); automatic model choice (desktop WebGPU → GeCo2 1024, phones → WASM 768, slow / low-memory / unknown → LOCA, measured times override, warnings); download manifest; CSV / JSON (model recorded, count not estimate, malformed result rejected) | `tests/unit/geco2.test.ts` | unit |
| GeCo2 int8 graphs end to end through the app's TS pipeline (onnxruntime-web WASM): a 480 × 360 synthetic rod scene counted exactly, every box on a true rod — only with `GECO2_MODELS=<folder with the 768 px graphs>` | `tests/unit/geco2.test.ts` | smoke (opt-in) |

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

| scene | GT | CountSHARD | P | R | count err | r MAPE | ms | baseline | P | R | count err |
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

## Multi-shot results (synthetic, 2026-10-19)

`npm run eval` → `tests/eval/multishot.eval.test.ts`. Each series renders one scene
of the suite above, padded with textured surroundings, as 2/3/5 photos with their own
small homography (shift ≤ 2.5 % of the longer side, rotation ≤ 2°, scale ≤ 3 %,
perspective ≤ 0.02), their own occluding bar (7 % of the width, objects ≥ 50 % covered
are "not visible" in that photo) and independent noise. GT = objects visible in at
least one photo, in reference coordinates. "dup" = counted objects landing on a GT
object another counted object already took. Sensitivity 50, same machine as above.

| scene | photos | seed | GT | reference photo alone: count (recall, FP) | fused count | R | FP | dup | flagged ▲ | uncertain ? (of them real) | registration inliers / RMS px per extra photo | detect ms/photo | register ms/photo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform-rods | 2 | 1 | 296 | 263 (R 0.889, FP 0) | 291 | 0.983 | 0 | 0 | 56 | 5 (5 real) | 691/1.00 | 413 | 646 |
| uniform-rods | 2 | 2 | 295 | 265 (R 0.898, FP 0) | 285 | 0.966 | 0 | 0 | 46 | 1 (1 real) | 625/1.07 | 370 | 530 |
| uniform-rods | 3 | 1 | 296 | 263 (R 0.889, FP 0) | 295 | 0.997 | 0 | 0 | 2 | 1 (1 real) | 691/1.00 504/1.06 | 328 | 515 |
| uniform-rods | 3 | 2 | 296 | 265 (R 0.895, FP 0) | 295 | 0.997 | 0 | 0 | 10 | 1 (1 real) | 625/1.07 492/1.05 | 378 | 517 |
| uniform-rods | 5 | 1 | 296 | 263 (R 0.889, FP 0) | 296 | 1.000 | 0 | 0 | 2 | 0 (0 real) | 691/1.00 504/1.06 589/1.04 626/1.07 | 357 | 520 |
| uniform-rods | 5 | 2 | 296 | 265 (R 0.895, FP 0) | 295 | 0.997 | 0 | 0 | 2 | 1 (1 real) | 625/1.07 492/1.05 576/1.00 622/1.03 | 362 | 559 |
| uniform-tubes | 2 | 1 | 217 | 204 (R 0.935, FP 1) | 216 | 0.995 | 0 | 0 | 34 | 3 (1 real) | 575/0.95 | 512 | 511 |
| uniform-tubes | 2 | 2 | 217 | 201 (R 0.917, FP 2) | 206 | 0.949 | 0 | 0 | 22 | 15 (11 real) | 521/0.93 | 515 | 520 |
| uniform-tubes | 3 | 1 | 217 | 204 (R 0.935, FP 1) | 217 | 1.000 | 0 | 0 | 0 | 5 (0 real) | 575/0.95 511/0.89 | 519 | 528 |
| uniform-tubes | 3 | 2 | 217 | 201 (R 0.917, FP 2) | 217 | 1.000 | 0 | 0 | 0 | 6 (0 real) | 521/0.93 507/1.01 | 524 | 536 |
| uniform-tubes | 5 | 1 | 217 | 204 (R 0.935, FP 1) | 217 | 1.000 | 0 | 0 | 0 | 6 (0 real) | 575/0.95 511/0.89 553/0.93 564/0.89 | 539 | 546 |
| uniform-tubes | 5 | 2 | 217 | 201 (R 0.917, FP 2) | 217 | 1.000 | 0 | 0 | 0 | 8 (0 real) | 521/0.93 507/1.01 497/0.88 490/0.93 | 532 | 529 |
| mixed-4-sizes | 2 | 1 | 260 | 241 (R 0.927, FP 0) | 260 | 1.000 | 0 | 0 | 36 | 0 (0 real) | 967/1.30 | 389 | 584 |
| mixed-4-sizes | 2 | 2 | 260 | 237 (R 0.912, FP 0) | 260 | 1.000 | 0 | 0 | 43 | 0 (0 real) | 683/1.42 | 367 | 501 |
| mixed-4-sizes | 3 | 1 | 260 | 241 (R 0.927, FP 0) | 260 | 1.000 | 0 | 0 | 0 | 0 (0 real) | 967/1.30 652/1.36 | 381 | 516 |
| mixed-4-sizes | 3 | 2 | 260 | 237 (R 0.912, FP 0) | 260 | 1.000 | 0 | 0 | 0 | 0 (0 real) | 683/1.42 775/1.32 | 363 | 493 |
| mixed-4-sizes | 5 | 1 | 260 | 241 (R 0.927, FP 0) | 260 | 1.000 | 0 | 0 | 0 | 0 (0 real) | 967/1.30 652/1.36 778/1.28 805/1.39 | 378 | 531 |
| mixed-4-sizes | 5 | 2 | 260 | 237 (R 0.912, FP 0) | 260 | 1.000 | 0 | 0 | 0 | 0 (0 real) | 683/1.42 775/1.32 745/1.34 673/1.35 | 374 | 531 |
| clutter | 2 | 1 | 120 | 114 (R 0.900, FP 6) | 121 | 1.000 | 1 | 0 | 22 | 11 (0 real) | 1135/1.21 | 330 | 558 |
| clutter | 2 | 2 | 120 | 113 (R 0.917, FP 3) | 121 | 1.000 | 1 | 0 | 18 | 6 (0 real) | 1202/1.25 | 328 | 518 |
| clutter | 3 | 1 | 120 | 114 (R 0.900, FP 6) | 123 | 1.000 | 3 | 0 | 1 | 15 (0 real) | 1135/1.21 1442/1.20 | 321 | 520 |
| clutter | 3 | 2 | 120 | 113 (R 0.917, FP 3) | 121 | 1.000 | 1 | 0 | 0 | 9 (0 real) | 1202/1.25 1221/1.21 | 333 | 563 |
| clutter | 5 | 1 | 120 | 114 (R 0.900, FP 6) | 122 | 1.000 | 2 | 0 | 1 | 19 (0 real) | 1135/1.21 1442/1.20 1154/1.22 982/1.19 | 321 | 536 |
| clutter | 5 | 2 | 120 | 113 (R 0.917, FP 3) | 121 | 1.000 | 1 | 0 | 0 | 16 (0 real) | 1202/1.25 1221/1.21 1273/1.23 1225/1.28 | 331 | 537 |
| noisy-blurred | 2 | 1 | 369 | 408 (R 0.856, FP 92) | 370 | 0.911 | 34 | 0 | 78 | 142 (20 real) | 431/1.36 | 553 | 556 |
| noisy-blurred | 2 | 2 | 369 | 434 (R 0.856, FP 118) | 369 | 0.870 | 48 | 0 | 71 | 186 (28 real) | 425/1.30 | 571 | 597 |
| noisy-blurred | 3 | 1 | 369 | 408 (R 0.856, FP 92) | 410 | 0.943 | 62 | 0 | 30 | 173 (11 real) | 431/1.36 382/1.33 | 567 | 576 |
| noisy-blurred | 3 | 2 | 369 | 434 (R 0.856, FP 118) | 417 | 0.932 | 73 | 0 | 41 | 205 (18 real) | 425/1.30 390/1.31 | 544 | 592 |
| noisy-blurred | 5 | 1 | 369 | 408 (R 0.856, FP 92) | 414 | 0.951 | 63 | 0 | 64 | 262 (12 real) | 431/1.36 382/1.33 445/1.38 441/1.31 | 561 | 553 |
| noisy-blurred | 5 | 2 | 369 | 434 (R 0.856, FP 118) | 434 | 0.938 | 88 | 0 | 67 | 287 (19 real) | 425/1.30 390/1.31 394/1.26 409/1.23 | 548 | 590 |

Reading the table:
- **Zero duplicates in all 30 series.** Before the concentric-merge rule (no radius
  gate in matching, plus "a centre inside a known object is the same object"), the
  bore of a tube or the inner shading ring of a rod seen only in one photo created
  up to 6 duplicates per series (first measurement).
- Rods, tubes, mixed sizes with ≥ 3 photos: fused count within 1 of GT (9 of 12 series
  exact, 0 false positives), against a reference-photo recall of 0.89–0.94 (its own
  occluder). With 2 photos, objects
  hidden in one photo cannot be settled by majority; up to 11 of 217 (tubes) stay
  uncertain candidates.
- `clutter`: 3–6 reference false positives drop to 1–3; the rest become uncertain
  candidates (not counted).
- `noisy-blurred` is detector-limited: fusion raises recall (0.856 → 0.87–0.95) and cuts
  false positives (92–118 → 34–88), but the count is still off by up to +65; many
  uncertain candidates remain. Systematic false hits that recur in several photos are
  counted.
- Registration: 382–1442 RANSAC inliers, RMS residual 0.9–1.4 px in every series; a
  photo of a different scene is refused (`tests/unit/multishot.test.ts`).

End-to-end in the browser (dev build, Chromium): reference = demo image with a grey
bar over part of the bundle (144 found); added a shifted/rotated unoccluded variant, a
variant occluded elsewhere, and a different bundle (demo with another seed). Result:
3 photos fused (inliers 602 / 491, error 1.2 / 1.1 px), **162** objects (true 162),
"18 found only in the other photos"; the different bundle was reported as "not used –
photos do not correspond". Undo returned to 144, redo to 162. With the unoccluded demo
as reference, the app reported that fusion brought no improvement (162 → 162).

## Mode B results (AI estimate, 2026-10-20)

`npm run eval` → `tests/eval/ai.eval.test.ts`. The shipped int8 model through
onnxruntime-web 1.30 (WASM, 1 thread, Node 24, Intel i5-12500). Three ways of running the
same model on the same exemplars:

- **squash 512: sum** — LOCA's own protocol: resize the whole photo to 512 × 512, sum the density.
- **squash: > 5 % max** — the reference site's post-processing (sum only density above 5 % of the maximum).
- **tiles: sum** — CountSHARD: aspect-preserving scale from the exemplar size + 512 px tiles (what the app does).

Model export verification against PyTorch is in `scripts/loca/README.md` (count within
1.9 % on six FSC-147 images, fp32 ONNX identical).

### Synthetic pipe/rod scenes (exemplars = the 3 true objects nearest the centre)

| scene | GT | size px | exemplar px | squash 512: sum | squash: > 5 % max | tiles | scale | tiles: sum | ms (tiles) |
|---|---|---|---|---|---|---|---|---|---|
| uniform-rods | 296 | 900×700 | 44 | 294.4 | 278.0 | 4 | 0.82 | 303.1 | 28795 |
| uniform-tubes | 217 | 900×700 | 52 | 212.8 | 204.8 | 2 | 0.69 | 216.9 | 17312 |
| dense-small | 2496 | 1000×800 | 18 | 2307.8 | 2256.4 | 16 (limit) | 1.85 | 2574.1 | 101228 |
| mixed-4-sizes | 260 | 1200×900 | 41 | 270.7 | 251.4 | 6 | 0.89 | 284.0 | 41820 |
| mixed-tubes | 220 | 1200×900 | 36 | 227.2 | 212.7 | 6 | 1.02 | 238.5 | 40811 |
| uneven-light | 369 | 900×700 | 39 | 367.4 | 346.0 | 4 | 0.91 | 382.5 | 29276 |
| noisy-blurred | 369 | 900×700 | 40 | 375.2 | 352.4 | 4 | 0.91 | 393.2 | 29511 |
| clutter | 120 | 1000×800 | 47 | 118.2 | 111.7 | 4 | 0.77 | 123.8 | 29944 |
| perspective-0.88 | 352 | 900×700 | 45 | 356.3 | 335.5 | 4 | 0.80 | 361.0 | 29186 |
| occluded | 280 | 900×700 | 43 | 285.2 | 268.0 | 4 | 0.84 | 297.8 | 29438 |
| **MAE** | | | | 23.1 | 36.2 | | | 19.6 | |
| **mean abs. % error** | | | | 2.4 % | 5.5 % | | | 4.6 % | |

### FSC-147 sample (20 test images, 3 annotated exemplars each)

12 "tree logs" + 8 other classes, picked with a fixed seed (`scripts/fetch-fsc147-sample.mjs`).
Images are 384 px tall.

| image | GT | size px | exemplar px | squash 512: sum | squash: > 5 % max | tiles | scale | tiles: sum | ms (tiles) |
|---|---|---|---|---|---|---|---|---|---|
| 4943 (tree logs) | 65 | 549×384 | 62 | 65.2 | 63.3 | 1 | 0.93 | 65.8 | 5670 |
| 4947 (tree logs) | 205 | 681×384 | 36 | 215.3 | 202.1 | 2 | 1.05 | 226.2 | 12309 |
| 4950 (tree logs) | 33 | 522×384 | 56 | 33.2 | 33.0 | 1 | 0.98 | 34.3 | 6441 |
| 4951 (tree logs) | 166 | 408×384 | 21 | 182.0 | 162.9 | 4 | 1.61 | 205.5 | 23844 |
| 4952 (tree logs) | 247 | 490×384 | 24 | 247.3 | 224.3 | 4 | 1.47 | 262.0 | 25420 |
| 4957 (tree logs) | 93 | 531×384 | 37 | 90.6 | 86.2 | 2 | 1.11 | 92.4 | 11974 |
| 4958 (tree logs) | 118 | 578×384 | 65 | 82.5 | 79.8 | 1 | 0.89 | 99.1 | 6057 |
| 4961 (tree logs) | 363 | 575×384 | 17 | 388.9 | 356.7 | 6 | 2.41 | 394.8 | 41129 |
| 4964 (tree logs) | 51 | 514×384 | 56 | 52.0 | 50.4 | 1 | 1.00 | 53.3 | 6403 |
| 4966 (tree logs) | 46 | 512×384 | 82 | 34.3 | 34.1 | 1 | 1.00 | 40.1 | 8961 |
| 4967 (tree logs) | 19 | 580×384 | 99 | 12.9 | 13.3 | 1 | 0.88 | 19.3 | 5995 |
| 4970 (tree logs) | 44 | 514×384 | 46 | 46.7 | 45.4 | 1 | 1.00 | 47.5 | 5751 |
| 2227 (apples) | 11 | 408×384 | 81 | 8.1 | 8.5 | 1 | 1.17 | 8.1 | 5866 |
| 2916 (stamps) | 202 | 512×384 | 20 | 205.6 | 193.2 | 6 | 1.95 | 199.6 | 46865 |
| 5839 (strawberries) | 45 | 576×384 | 89 | 55.5 | 54.0 | 1 | 0.89 | 53.8 | 5798 |
| 5852 (strawberries) | 42 | 576×384 | 63 | 47.1 | 46.3 | 1 | 0.89 | 46.9 | 8978 |
| 5917 (eggs) | 12 | 512×384 | 73 | 13.5 | 13.6 | 1 | 1.00 | 12.5 | 5930 |
| 5925 (eggs) | 22 | 408×384 | 122 | 20.4 | 19.8 | 1 | 0.80 | 23.6 | 5765 |
| 6847 (sheep) | 67 | 682×384 | 41 | 63.1 | 61.0 | 2 | 0.89 | 65.2 | 18318 |
| 7109 (keyboard keys) | 48 | 512×384 | 35 | 46.4 | 45.2 | 2 | 1.04 | 46.4 | 10868 |
| **MAE** | | | | 7.1 | 6.9 | | | 8.3 | |
| **mean abs. % error** | | | | 10.7 % | 10.6 % | | | 8.7 % | |

### FSC-147 "tree logs", 2 × 2 mosaic (a wider shot of a bigger pile)

Each of the 12 log images tiled 2 × 2: 4× the objects, each a quarter of the photo area.
GT = 4 × count; exemplars from the top-left copy.

| mosaic | GT | size px | exemplar px | squash 512: sum | squash: > 5 % max | tiles | scale | tiles: sum | ms (tiles) |
|---|---|---|---|---|---|---|---|---|---|
| 4943 ×4 | 260 | 1098×768 | 62 | 286.9 | 265.4 | 2 | 0.58 | 274.0 | 10755 |
| 4947 ×4 | 820 | 1362×768 | 36 | 907.0 | 816.7 | 8 | 1.05 | 899.2 | 46949 |
| 4950 ×4 | 132 | 1044×768 | 56 | 137.9 | 128.1 | 4 | 0.70 | 139.7 | 21272 |
| 4951 ×4 | 664 | 816×768 | 21 | 644.6 | 591.5 | 9 | 1.61 | 817.8 | 58102 |
| 4952 ×4 | 988 | 980×768 | 24 | 1023.8 | 930.6 | 12 | 1.47 | 1055.8 | 78033 |
| 4957 ×4 | 372 | 1062×768 | 37 | 390.7 | 350.6 | 6 | 1.11 | 379.1 | 38663 |
| 4958 ×4 | 472 | 1156×768 | 65 | 450.0 | 430.3 | 2 | 0.56 | 446.0 | 10973 |
| 4961 ×4 | 1452 | 1150×768 | 17 | 1439.9 | 1378.9 | 15 (limit) | 1.83 | 1477.7 | 90483 |
| 4964 ×4 | 204 | 1028×768 | 56 | 210.8 | 193.4 | 2 | 0.62 | 212.2 | 11191 |
| 4966 ×4 | 184 | 1024×768 | 82 | 163.6 | 157.1 | 1 | 0.50 | 172.2 | 5375 |
| 4967 ×4 | 76 | 1160×768 | 99 | 67.4 | 64.2 | 1 | 0.44 | 81.0 | 5275 |
| 4970 ×4 | 176 | 1028×768 | 46 | 194.4 | 174.1 | 4 | 0.79 | 188.6 | 22023 |
| **MAE** | | | | 23.5 | 27.5 | | | 34.9 | |
| **mean abs. % error** | | | | 6.6 % | 6.5 % | | | 7.0 % | |

### Reading the Mode B tables

- **Tiling did not beat squashing overall.** Mean absolute % error, squash vs tiles:
  synthetic 2.4 % vs 4.6 %, FSC-147 10.7 % vs 8.7 %, mosaics 6.6 % vs 7.0 % (MAE 23.5 vs 34.9).
  The expected gain on "many objects in a bigger photo" (mosaics) did not appear: LOCA was
  trained with a similar tiling augmentation and copes with 4× smaller objects better than
  assumed.
- **What tiling does well:** keeping the aspect ratio for photos with large objects
  (4958: 82 → 99 of 118; 4966: 34 → 40 of 46; 4967: 13 → 19 of 19; mosaic 4967: 67 → 81 of 76)
  and very small dense objects (synthetic dense-small, 18 px rods: −188 squashed vs +78 tiled
  of 2,496).
- **What it does badly:** when it *upscales* (small exemplars, scale > 1) it over-counts:
  4951: 182 → 206 of 166, mosaic 4951: 645 → 818 of 664; synthetic scenes +2 … +9 %
  (uniform-tubes exact). The lower end of the exemplar band (36 px) is the likely cause; it
  was taken from FSC-147 training statistics and deliberately not tuned on these images
  (follow-up: validate it on the FSC-147 validation split).
- The reference site's "> 5 % of max" threshold lowers every count; it was worse on the
  synthetic set and equal on FSC-147, so CountSHARD uses the plain sum.
- These are 10 synthetic scenes, 20 small dataset photos and 12 mosaics of them, not real
  photos of the target use cases.

### Memory (2026-10-20)

`node scripts/ai-mem-probe.mjs <optimisation level>`: Node RSS after creating both sessions,
then after one features + head run on a 512 × 512 tile (onnxruntime-web 1.30 WASM, 1 thread).

| features graph | level | after sessions | after run |
|---|---|---|---|
| v1 (full attention) | all | 259 MB | 832 MB |
| v2 (256-query chunks) | all | 248 MB | 346 MB |
| v2 | basic | 276 MB | 352 MB |
| v2 (shipped settings) | disabled | 199 MB | 309 MB |

`tests/unit/aiModel.test.ts` fails if a WASM run grows RSS by ≥ 250 MB (v1 grew it by
1,046 MB with level 'disabled', v2 passes). Phones and WebGPU memory not measured.

### In the browser (production build, Chromium, Intel Gen12 iGPU, WebGPU)

v2 model (2026-10-20), same demo photo: WebGPU ≈ 162 in 11.4 s (4 tiles), WASM ≈ 162 in
27.5 s; a simulated crash marker showed the Czech message after reload and switched the device
to WASM; the old `countshard-ai-v1` cache was deleted; after an AI run the OpenCV engine showed
as stopped and a circle re-count afterwards worked.

v1 model: Demo photo (1400 × 1000, 162 pipes of two sizes) → AI mode → model downloaded (63 MB, three
files, checksums verified, progress shown) → 3 exemplar boxes on small pipes → **≈ 162**
(true 162), 4 tiles, 16.8 s on the first run (incl. shader compilation), 12.6 s warm
(≈ 3.1 s per tile). Heat map has one blob on every pipe, small and large. A correction region
with model ≈ 27.7 and a user count of 30 gave ≈ 164 (= 162 − 27.7 + 30). With the preview
server **stopped**, reload → saved project with result and correction restored → estimate
re-run from the cached model: same result, 16.6 s. The status chip listed "AI model
downloaded" and "AI ready (WebGPU)" separately. Not tested in the browser: the WASM fallback
(tested in Node only), phones, PNG export of an AI result (shares `drawAiOverlay` with the
viewer; CSV/JSON covered by unit tests).

## GeCo2 feasibility (Mode B second model, spike, 2026-10-20)

Not part of the app. Scripts and how to reproduce: `scripts/geco2/README.md`; decision and
recommendation: `technical-decisions.md` ("Mode B — GeCo2 as a second model"). Same cases as
the LOCA tables above (exemplars, images, mosaics; dumped by `scripts/geco2/dump-cases.ts`).
GeCo2 preprocessing and counting as in its own `inference.py`: longer side → 1024 px, further
down so the mean exemplar side is ≤ 80 px, pad to 1024 × 1024; local maxima of the objectness
map above max/8, NMS (IoU 0.5), count = boxes left. GeCo2 never tiles.

Columns: **refine** = GeCo2's own pipeline in PyTorch incl. the SAM mask-decoder box
refinement; **no refine** = PyTorch without it (what the exported graphs compute); **int8** =
the exported weight-only int8 ONNX graphs (onnxruntime 1.31 CPU), the form a browser build would
ship. TP / FP / FN for the int8 detections against the true circles (match: centre within 0.5 r).

### Synthetic pipe/rod scenes

| scene | GT | LOCA squash | LOCA tiles (app) | GeCo2 refine | GeCo2 no refine | GeCo2 int8 | int8 TP / FP / FN |
|---|---|---|---|---|---|---|---|
| uniform-rods | 296 | 294.4 | 303.1 | 296 | 296 | 296 | 296 / 0 / 0 |
| uniform-tubes | 217 | 212.8 | 216.9 | 217 | 217 | 217 | 217 / 0 / 0 |
| dense-small | 2496 | 2307.8 | 2574.1 | 2510 | 2510 | 2509 | 2496 / 13 / 0 |
| mixed-4-sizes | 260 | 270.7 | 284.0 | 260 | 260 | 260 | 260 / 0 / 0 |
| mixed-tubes | 220 | 227.2 | 238.5 | 220 | 220 | 220 | 220 / 0 / 0 |
| uneven-light | 369 | 367.4 | 382.5 | 369 | 369 | 369 | 369 / 0 / 0 |
| noisy-blurred | 369 | 375.2 | 393.2 | 369 | 369 | 369 | 369 / 0 / 0 |
| clutter | 120 | 118.2 | 123.8 | 122 | 122 | 122 | 120 / 2 / 0 |
| perspective-0.88 | 352 | 356.3 | 361.0 | 352 | 352 | 352 | 352 / 0 / 0 |
| occluded | 280 | 285.2 | 297.8 | 282 | 282 | 283 | 280 / 3 / 0 |
| **MAE** | | 23.1 | 19.6 | 1.8 | 1.8 | 1.8 | |
| **mean abs. % error** | | 2.4 % | 4.6 % | 0.3 % | 0.3 % | 0.3 % | |

### FSC-147 sample

| image | GT | LOCA squash | LOCA tiles (app) | GeCo2 refine | GeCo2 no refine | GeCo2 int8 |
|---|---|---|---|---|---|---|
| 4943 (tree logs) | 65 | 65.2 | 65.8 | 71 | 71 | 71 |
| 4947 (tree logs) | 205 | 215.3 | 226.2 | 231 | 240 | 241 |
| 4950 (tree logs) | 33 | 33.2 | 34.3 | 33 | 33 | 33 |
| 4951 (tree logs) | 166 | 182.0 | 205.5 | 212 | 226 | 223 |
| 4952 (tree logs) | 247 | 247.3 | 262.0 | 273 | 289 | 293 |
| 4957 (tree logs) | 93 | 90.6 | 92.4 | 127 | 135 | 135 |
| 4958 (tree logs) | 118 | 82.5 | 99.1 | 117 | 122 | 123 |
| 4961 (tree logs) | 363 | 388.9 | 394.8 | 416 | 425 | 424 |
| 4964 (tree logs) | 51 | 52.0 | 53.3 | 56 | 59 | 58 |
| 4966 (tree logs) | 46 | 34.3 | 40.1 | 50 | 52 | 52 |
| 4967 (tree logs) | 19 | 12.9 | 19.3 | 25 | 25 | 25 |
| 4970 (tree logs) | 44 | 46.7 | 47.5 | 53 | 57 | 57 |
| 2227 (apples) | 11 | 8.1 | 8.1 | 10 | 10 | 10 |
| 2916 (stamps) | 202 | 205.6 | 199.6 | 204 | 204 | 204 |
| 5839 (strawberries) | 45 | 55.5 | 53.8 | 46 | 50 | 51 |
| 5852 (strawberries) | 42 | 47.1 | 46.9 | 49 | 51 | 50 |
| 5917 (eggs) | 12 | 13.5 | 12.5 | 16 | 16 | 16 |
| 5925 (eggs) | 22 | 20.4 | 23.6 | 23 | 24 | 24 |
| 6847 (sheep) | 67 | 63.1 | 65.2 | 67 | 68 | 68 |
| 7109 (keyboard keys) | 48 | 46.4 | 46.4 | 47 | 47 | 47 |
| **MAE** | | 7.2 | 8.3 | 11.7 | 15.4 | 15.5 |
| **mean abs. % error** | | 10.7 % | 8.7 % | 12.6 % | 16.2 % | 16.1 % |

### FSC-147 "tree logs", 2 × 2 mosaic

| mosaic | GT | LOCA squash | LOCA tiles (app) | GeCo2 refine | GeCo2 no refine | GeCo2 int8 |
|---|---|---|---|---|---|---|
| 4943 ×4 | 260 | 286.9 | 274.0 | 263 | 267 | 266 |
| 4947 ×4 | 820 | 907.0 | 899.2 | 936 | 936 | 930 |
| 4950 ×4 | 132 | 137.9 | 139.7 | 132 | 132 | 132 |
| 4951 ×4 | 664 | 644.6 | 817.8 | 939 | 939 | 940 |
| 4952 ×4 | 988 | 1023.8 | 1055.8 | 1164 | 1164 | 1163 |
| 4957 ×4 | 372 | 390.7 | 379.1 | 466 | 488 | 487 |
| 4958 ×4 | 472 | 450.0 | 446.0 | 448 | 456 | 455 |
| 4961 ×4 | 1452 | 1439.9 | 1477.7 | 1382 | 1382 | 1383 |
| 4964 ×4 | 204 | 210.8 | 212.2 | 221 | 225 | 225 |
| 4966 ×4 | 184 | 163.6 | 172.2 | 185 | 188 | 189 |
| 4967 ×4 | 76 | 67.4 | 81.0 | 98 | 100 | 100 |
| 4970 ×4 | 176 | 194.4 | 188.6 | 210 | 221 | 221 |
| **MAE** | | 23.5 | 34.9 | 69.3 | 72.5 | 71.9 |
| **mean abs. % error** | | 6.6 % | 7.0 % | 13.9 % | 15.4 % | 15.4 % |

(Mosaics with > 800 candidates skip the SAM refinement in GeCo2's own code, so refine =
no refine there.)

### On phones (bench page `https://pokys.github.io/geco2-bench/`, 2026-10-20)

Page source in `scripts/geco2/bench-repo/` (Czech UI, models cached, crash marker in
localStorage). Same int8 graphs; "PC" = onnxruntime CPU count for the identical input.

| device | input | backend | test photo | count (PC, truth) | time per photo (warm) | result |
|---|---|---|---|---|---|---|
| iPhone (Safari 27, iOS 26+) | 768 | WebGPU | uniform-tubes | — | — | **tab crashed** in the first backbone run (twice) |
| iPhone | 768 | WASM | uniform-tubes | 218 (218, 217) | 16.4 s (backbone 8.1 + head 8.3) | OK |
| iPhone | 768 | WASM | dense-small | 2502 (2502, 2496) | 16.8 s | OK |
| iPhone | 768 | WASM | 4951 tree logs | 256 (256, 166) | 16.5 s | OK |
| iPhone | 1024 | WASM | uniform-tubes | 217 (217, 217) | 30.7 s | OK |

| Galaxy A34 5G (Chrome, Mali "valhall", deviceMemory 4) | 768 | WASM | uniform-tubes | 218 (218, 217) | 72.5 s (first run 79.4 s) | OK |
| Galaxy A34 5G | 768 | WASM | dense-small | 2502 (2502, 2496) | 57.4 s | OK |
| Galaxy A34 5G | 768 | WASM | 4951 tree logs | 257 (256, 166) | 57.5 s | OK |
| Galaxy A34 5G | 1024 | WASM | uniform-tubes | 217 (217, 217) | 103.9 s | OK |

All counts on both phones equal the PC's (one ±1 on the Galaxy). On the iPhone WebGPU fails on
GPU memory, not compute (Safari reports `maxBufferSize` 1024 MB); its CPU path runs ≈ 3×
faster than the loaded test PC. The Galaxy A34 survives every WASM run but needs ≈ 1 min per
photo at 768 px (≈ 3.5× the iPhone). WebGPU on the Galaxy at 768 px crashed Chrome (reported
by the tester, not in the page log), so GeCo2 on WebGPU fails on both phones.

### In the app (dev build, Chromium, Intel Gen12 iGPU, 2026-10-20)

Demo photo (1400 × 1000, 162 pipes of two sizes), 3 exemplars on small pipes, AI model
"Automaticky" → GeCo2 (WebGPU, 1024 px). Files downloaded from the GeCo2 site (87 MB +
runtime, hashes verified). **162 of 162** in 5.1 s (8.5 s incl. session start). Tap on a box →
161 ("1 odebráno"), tap on empty background → 162 ("1 přidáno ručně"), undo works; counting
area over the left half → 78, numbering and card follow. Switched to LOCA on the same photo:
only LOCA's 35 MB downloaded (runtime shared), ≈ 83 inside the same area, 2.7 s. Offline menu
lists LOCA and GeCo2 1024 px with sizes; deleting LOCA keeps GeCo2 usable (runtime moved to
GeCo2's cache). A simulated GeCo2 crash on WebGPU shows the Czech message, sets WASM for GeCo2
and the panel's "Přepnout na LOCA" switches and remembers LOCA. GeCo2 on WASM 768 px in the
same browser: **162 of 162 in 20.5 s** (estimate before the run: 22 s). Not yet tried in the
app on the phones (the bench page measured them, see above).

Exemplar adjustment (dev build, same photo): tap an exemplar → handles; dragging a corner
resized 60 → 100 px, dragging inside moved it by 80 px; one undo reverts exactly one drag, redo
restores it; three tight boxes of neighbouring pipes can be drawn one after another (a drag over
an unselected exemplar draws a new box). Exemplars enlarged to ≈ 2× the pipes → the GeCo2 card
warns "Vzory jsou 1,8× větší než nalezené kusy…" (count still 162).

### Reading the GeCo2 tables

- **Pipes/rods (the app's use case): GeCo2 is near-perfect and object-exact.** Every scene
  within +3, all true objects found (one missed in PyTorch on "occluded"); false positives:
  13 of 2,509 on 18 px rods, 2 in clutter, 3 at the occluding bar. LOCA's best protocol has
  MAE 23 and gives no objects. Mixed sizes (4 diameters from 3 exemplars) are exact without
  tiling.
- **FSC-147 is worse than LOCA**, almost entirely on "tree logs": GeCo2 also boxes partly
  hidden logs at the back and sides of a pile, which the dataset's annotation leaves out
  (checked visually on 4951 and 4957; the boxes are on real log ends). Other classes are on
  par with or better than LOCA.
- **int8 costs nothing measurable** (no refine vs int8: MAE 1.8 / 1.8, 15.4 / 15.5,
  72.5 / 71.9). The **SAM box refinement** is worth ≈ 4 MAE on FSC (fewer duplicate boxes
  after NMS) and nothing on the synthetic scenes; it is not exported (see technical-decisions).
- Only 10 synthetic scenes, 20 dataset photos and 12 mosaics; still no real pipe photos.

### Input resolution (int8, same rules with 1024 replaced by S and the exemplar cap 80·S/1024)

| set | LOCA tiles (app) | GeCo2 1024 | GeCo2 768 | GeCo2 512 |
|---|---|---|---|---|
| synthetic, MAE | 19.6 | 1.8 | 1.1 | 190.4 |
| — dense-small (2,496 rods, 18 px) | 2574.1 | 2509 (13 FP, 0 FN) | 2501 (13 FP, 8 FN) | 641 (1,953 missed) |
| — mixed-4-sizes (260) | 284.0 | 260 | 260 | 272 (12 FP) |
| FSC-147 sample, MAE | 8.3 | 15.5 | 14.0 | 26.6 |
| tree-log mosaics, MAE | 34.9 | 71.9 | 82.8 | 113.8 |

768 px keeps GeCo2's accuracy on the synthetic scenes (objects ≥ 18 px in the photo become
≥ 13 px) and on FSC-147, and costs a little on the mosaics; 512 px loses small objects.

### Export check, size, speed and memory

- Rewritten export modules vs the original GeCo2 forward: objectness max abs. difference
  ≤ 7e-6, ltrb ≤ 2e-6 (3 images). fp32 ONNX (onnxruntime CPU) gives the same counts as
  PyTorch (217 = 217, 4951: 226 = 226).
- Weight-only int8 sizes: backbone 71.3 MB, head 16.0 MB, exemplars 0.1 MB = **87.4 MB**
  (LOCA 34.9 MB). 768 px: 82 MB, 512 px: 79 MB (the backbone weights dominate).
- **onnxruntime-web WASM, 1 thread, Node 24** (`node scripts/geco2/geco2-probe.mjs <dir> <size> w8 disabled`),
  RSS after one backbone + exemplars + head run. Times were measured while other CPU jobs ran
  (all of them; uncontended runs would be somewhat faster — estimate 20–30 %):

  | graphs | input | after sessions | after run | backbone | head |
  |---|---|---|---|---|---|
  | first export | 1024 | 338 MB | 990 MB | 41.6 s | 36.8 s |
  | first export, arena + mem pattern off | 1024 | 338 MB | 989 MB | 41.7 s | 37.4 s |
  | lean (Hiera in row bands, head in stripes) | 1024 | 346 MB | **835 MB** | 50.9 s | 50.3 s |
  | first export | 768 | 334 MB | 739 MB | 26.4 s | 26.2 s |
  | first export | 512 | 371 MB | 556 MB | 12.2 s | 10.0 s |

  For comparison LOCA: ≈ 310 MB after a run, ≈ 5 s per tile; the iPhone crashed at ≈ 830 MB.
- **Chromium-based browser, Intel Gen12 iGPU, onnxruntime-web 1.30**, 1024 px,
  `scripts/geco2/bench.html`: WebGPU first run 7.1–9.8 s (incl. shader compilation), warm
  **5.4–6.7 s** per photo (backbone 2.6–3.9 s, head 2.3–2.9 s), session creation 5–8 s;
  counts equal the CPU counts (217, 223). WASM in the same browser: 97.6 s (CPU busy with
  other jobs). LOCA on the same bench: 3.5 s for one 512 tile (WebGPU, first run).
- **GPU memory** (Windows "GPU Process Memory" counters of the browser's GPU process, dedicated
  + shared, while the page holds its sessions after one run; 45 MB after leaving the page):
  GeCo2 1024 ≈ **2.0 GB**, LOCA ≈ 0.85 GB.
- **WebGPU correctness.** An in-graph int64 `Clip` (row indices of the striped upsampling)
  returned wrong data on WebGPU (objectness off by 0.41); fixed by literal index constants.
  Remaining: the int8 backbone's deepest output differs from CPU locally (max 0.65 at values
  up to 2.1) only when its intermediates are not graph outputs — a buffer-reuse problem
  inside onnxruntime-web; final objectness within 0.04 of CPU and equal counts on the two
  photos tested. Phones were not tested.

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
- **Mode B on real photos** of the target objects, on phones, and with the WASM fallback in
  a browser (WASM measured in Node only); whether the exemplar band should be lowered to stop
  the over-count when tiles upscale.
- **GeCo2 on phones** (iPhone 17 Pro, Galaxy A): speed, GPU memory, whether the tab survives;
  GeCo2 on real pipe photos; the onnxruntime-web WebGPU backbone deviation (see above).
