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
| Multi-shot: registration recovers the true homography (< 2 px at the corners) and refuses a different scene; fused count = ground truth with zero duplicates on 4 synthetic series (2–5 photos); no duplicates on harder series; a different-scene photo is refused alone and left out among good ones without any effect; wrong-but-plausible registration (half a pitch off) and residual too large for the object size are refused; concentric bore/inner ring merges with its object; minority/uncertain policy | `tests/unit/multishot.test.ts` | regression on synthetic multi-view sets + unit |
| Fused objects in the project: corrections survive a fused merge; uncertain candidates are selectable but count only after confirmation; a removed candidate stays out on re-fusion | `tests/unit/project.test.ts` | unit |

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
