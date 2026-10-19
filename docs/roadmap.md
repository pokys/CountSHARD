# Roadmap

Status as of 2026-10-19.

## Done

- **Phase 0–2** — environment check, reference investigation with REA
  (`research/loreatec-analysis.md`), specification review (`technical-decisions.md`).
- **Phase 3 — MVP:** import (file, drag & drop, camera), worker-based circle
  detection with progress/cancel, tap corrections, move/resize, undo/redo, zoom/pan/
  pinch, original/annotated compare, numbered PNG, CSV, JSON project export/import,
  local project history, offline PWA with per-layer readiness, Czech + English UI.
- **Phase 5 — multi-diameter:** automatic size classes, fixed-k mode, merge/split,
  manual reassignment that survives re-grouping, calibration from an object or a
  group's nominal diameter, spread warnings.
- **Phase 6 — multi-shot (small viewpoint changes):** ORB + ratio test + RANSAC
  homography of every extra photo onto the project photo, with refusal when the
  registration is unreliable (too few inliers, low inlier ratio, poor coverage, high
  residual, implausibly large viewpoint change) or when the mapped detections
  disagree; one-to-one matching with a 0.5·r gate; per object "seen in k of n";
  majority vote with "minority" (counted, flagged) and "uncertain" (not counted until
  confirmed) objects. Acceptance test met on the synthetic set: fused count = ground
  truth with zero duplicates, and a photo of a different scene is refused
  (`tests/unit/multishot.test.ts`; results in `testing.md`). Real photos not tested.

- **Phase 4 — generic AI counting (Mode B):** LOCA few-shot (owner accepted the
  non-commercial weight licence) exported to two ONNX graphs, weight-only int8 (34.9 MB,
  count within 1.9 % of PyTorch); onnxruntime-web 1.30 in a worker, WebGPU with WASM
  fallback; lazy download with progress into Cache Storage, offline afterwards; aspect-
  preserving scale + 512 px tiling instead of squashing; mode switch, 1–3 exemplar boxes,
  heat map + total labelled as an estimate, region corrections, PNG/CSV/JSON marked as an
  estimate. Evaluated on synthetic scenes, a 20-image FSC-147 sample and 2 × 2 mosaics
  (`testing.md`): 7–9 % mean abs. error on the dataset images; the tiling was *not* more
  accurate than LOCA's squash-to-512 overall (it over-counts when it upscales).

## Next

1. **Real-photo evaluation set** (highest value). 20–50 photos of real bundles,
   annotated centres/diameters (a small annotation export from CountSHARD itself is
   enough: count with the app, fix every error by hand, export JSON). Re-tune the
   detector on it; add it to `npm run eval`.
2. **Playwright E2E** for import → detect → correct → export and offline reload.
3. **Multi-shot follow-ups:** evaluate on real photo series (same set as item 1,
   2–3 shots each); keep extra photos with the project (currently memory only);
   re-run fusion automatically when the sensitivity changes (today a single-photo
   re-analysis replaces the fused result).
4. **Mode B follow-ups:** evaluate on real photos of the target objects (same set as item 1);
   reduce the few-percent over-count when tiles upscale (validate the exemplar band on the
   FSC-147 *validation* split, not the test sample); profile WebGPU (≈ 3 s/tile on an Intel
   iGPU suggests CPU fallbacks) and consider fp16; a detection-capable model with clear
   weights licensing (individually correctable points) would replace the estimate workflow.
5. **Quality:** memory profiling on iOS/Android, tiling for > 2400 px analysis,
   blur/quality warning before analysis, guided photography tips after poor results.

## Later / optional

- Active reshooting hints ("this region disagrees between photos").
- PDF report, comparison of repeated counts.
- Detection presets (save sensitivity/size range per product type).
- **Live Scan** — only if multi-shot shows clear benefit; expected to cost battery
  and heat on phones for little gain over 2–5 deliberate photos. May be rejected.
