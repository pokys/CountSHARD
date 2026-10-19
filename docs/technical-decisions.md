# Technical decisions and specification review

## Phase 2 review — answers

**1. Is the application feasible?** Yes for the core (classical detection of
circular ends, corrections, size classes, export, offline PWA) — now implemented and
measured. Generic AI counting is feasible technically (LOCA runs in ORT-Web, as the
reference proves) but blocked on weight licensing. Multi-shot with *small* viewpoint
changes is feasible; arbitrary viewpoints are research-grade.

**2. Straightforward:** import, offline shell, worker-based OpenCV, corrections,
undo, export, persistence, Czech localisation.

**3. Difficult:** robust detection on real stacks (specular ends, rust, chamfers,
packing seams); accurate diameters (needed for classes); tube bore vs. outer wall;
multi-shot correspondence in densely packed, self-similar scenes (every pipe looks
like its neighbour, so feature matching on the pipe faces is ambiguous).

**4. Speculative:** physically valid mm measurements in arbitrary 3-D scenes from a
single reference; multi-shot from substantially different angles; Live Scan.

**5. Algorithms chosen**
- Circles: multi-band Hough candidates + independent edge-support verification
  (details below).
- Size classes: KDE mode finding in log-radius space (auto) or 1-D k-means (fixed k).
- Generic AI (planned): detection-capable few-shot counter (GeCo-class) if its
  weights' licence allows, else LOCA density with an "estimate" workflow.
- Multi-shot (planned): homography (ORB/RANSAC on the *whole scene* including the
  bundle outline and surroundings) + Hungarian matching of detections + vote fusion.

**6. Stack.** The proposed TypeScript + Vite + React + OpenCV.js + Workers + Workbox
stack is appropriate and was kept. Changes: OpenCV is **bundled** into the worker
(no CDN at runtime, guaranteed offline after install); no TensorFlow.js; ONNX Runtime
is deferred until there is a licensable model.

**7. Reusable open source.** OpenCV.js (Apache-2.0), vite-plugin-pwa/Workbox (MIT),
ONNX Runtime Web (MIT, later). No suitable ready-made pipe counter library exists;
the reference site is proprietary.

**8. Order.** MVP + size classes first (they share the detector and are the core
use case), then multi-shot (small viewpoint changes), then generic AI (blocked on
licence), Live Scan last or never.

**9. Changes to the specification**
- Size classes (Mode C) are not a separate mode: every circle detection produces
  diameters, so grouping is always available in the same screen.
- Mode B (AI) is moved after multi-shot, because the most valuable use case (pipes)
  does not need it and its model licensing is unresolved.
- "Confidence" is reported as *edge match* (fraction of the rim with a strong radial
  edge) — a heuristic, explicitly not a probability.
- Calibration from a whole **group's nominal size** was added; it is more robust
  than one reference object.
- Live Scan: recommended to drop unless multi-shot shows clear gains (thermal/battery
  cost, low marginal value over 2–5 still photos).

**10. Main risks.** Real-photo accuracy is not yet measured (only synthetic data);
OpenCV worker is 14.5 MB (first load on mobile data); iOS memory limits with very
large photos (mitigated by 3072 px project cap); calibration misuse (mitigated by
warnings).

## Circle detection

Reference behaviour (single radius band from one size estimate) fails on packed
bundles and mixed sizes (see `research/loreatec-analysis.md` §5). CountIT's detector
(`src/analysis/circles.ts`):

1. **CLAHE** normalises uneven lighting.
2. **Multi-band Hough.** Radii from ~0.6 % to 22 % of the short side are covered by
   bands with ratio 1.5. Each band runs on a pyramid level where its smallest radius
   is ~14 px (small bands upsampled ≤ 2×, large bands downsampled), keeping cost
   roughly constant per band. The median pre-filter is 3×3 for small radii — a 5×5
   median erased the 1-px seams between tightly packed small rods (measured: recall
   loss on `dense-small`).
3. **Verification by edge support.** Each candidate is re-scored on the full-res
   gradient field: fraction of perimeter samples with a strong (adaptive threshold)
   gradient within ~37° of radial. Candidates with near-balanced edge *polarity*
   (≥ 40 % minority) are rejected — circles fitted through line clutter have mixed
   polarity; real end faces do not. Polarity is read slightly inside the rim, because
   at the seam centre the sign follows whichever neighbour is brighter (measured:
   strict polarity at the seam lost darker faces between bright neighbours).
4. **Refinement.** Coordinate descent maximising coverage (strength only as a tie
   breaker — pure gradient strength drifted onto neighbours' edges), then a ±7 %
   radius polish on edge strength with coverage held, which sets the diameter used for
   classification.
5. **Tubes.** For a bore candidate (positive polarity: dark inside), a concentric wall
   with negative polarity is searched at 1.15–1.7× and preferred in suppression even
   with lower coverage (the wall touches neighbours). Using polarity rather than a
   generic "prefer larger" rule avoided false outer circles on solid rods (measured).
6. **Suppression.** Greedy by score; conflict when centre distance < 0.7·(r₁+r₂)
   (touching neighbours sit at 1.0). Circles enclosing ≥ 2 accepted smaller centres
   (bundle outline) are removed.

Sensitivity maps linearly to the minimum edge coverage (0.78 → 0.32).

## Size classes

1-D problem, so DBSCAN/HDBSCAN add nothing over density-mode finding. Implementation
(`src/analysis/clustering.ts`): log-radius KDE (bandwidth = min(Silverman,
separation/2.5)), merge modes closer than the minimum separation (default 10 %) or
with a shallow valley (> 75 % of the lower peak), cut at valleys, require ≥ 3 members
per auto-discovered group. Measured: EN 10255 sizes 20/26.9/33.7/42.4 mm (ratios
≈ 1.25–1.35) are separated with 99–100 % class accuracy on synthetic data with
≈ 3 % radius error. Groups closer than ~10 % in diameter need the fixed-k mode or
manual merge/split.

Group identities (id, colour, nominal mm) are matched by size across re-grouping, and
manually assigned objects are locked to their group.

## Corrections and data integrity

Removing an automatic detection marks it `rejected` rather than deleting it; manual,
moved, resized and confirmed objects are never replaced by re-analysis
(`mergeDetections`). All coordinates are in project-image pixels; the viewer
transform and the analysis downscale are the only conversions, both tested via the
merge/round-trip tests and the end-to-end demo check.

## Offline

Workbox `generateSW` precaches the shell and the analysis worker (OpenCV embedded).
The status chip checks each layer (shell in cache + SW controlling, worker in cache,
engine actually started) rather than inferring readiness from registration.
`navigator.storage.persist()` is requested after the first saved project. Updates are
applied only on user confirmation so a running analysis is never interrupted.
