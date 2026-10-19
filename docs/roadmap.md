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

## Next

1. **Real-photo evaluation set** (highest value). 20–50 photos of real bundles,
   annotated centres/diameters (a small annotation export from CountIT itself is
   enough: count with the app, fix every error by hand, export JSON). Re-tune the
   detector on it; add it to `npm run eval`.
2. **Playwright E2E** for import → detect → correct → export and offline reload.
3. **Phase 6 — Multi-shot (small viewpoint changes)**
   - ORB features + RANSAC homography between each photo and a reference photo,
     using the whole frame (pipe faces alone are too self-similar to match reliably);
   - detections mapped to the reference, Hungarian matching with a distance gate
     relative to radius;
   - fused result: an object counts once; per object "seen in k of n photos";
     inconsistent regions highlighted for review;
   - acceptance test: synthetic scene rendered with 2–5 small homographies + noise;
     fused count must equal ground truth with **zero duplicates**; if registration
     residual is too high, the app must refuse to fuse and say so.
4. **Phase 4 — Generic AI counting** (blocked on a licence decision)
   - LOCA weights inherit a non-commercial SwAV backbone (CC BY-NC 4.0) — needs an
     owner decision or a model with clear weights licensing;
   - prefer a detection-capable model (individually correctable points/boxes) over a
     pure density map; if only density is available, present it as an *estimate*
     with a heat map and a separate correction workflow (count adjustment by region),
     never as individually identified objects;
   - ONNX export + int8 quantisation, ORT-Web with WebGPU where available, WASM
     fallback, lazy download with progress, cached for offline use.
5. **Quality:** memory profiling on iOS/Android, tiling for > 2400 px analysis,
   blur/quality warning before analysis, guided photography tips after poor results.

## Later / optional

- Active reshooting hints ("this region disagrees between photos").
- PDF report, comparison of repeated counts.
- Detection presets (save sensitivity/size range per product type).
- **Live Scan** — only if multi-shot shows clear benefit; expected to cost battery
  and heat on phones for little gain over 2–5 deliberate photos. May be rejected.
