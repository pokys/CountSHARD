# Technical decisions and specification review

## Phase 2 review — answers

**1. Is the application feasible?** Yes for the core (classical detection of
circular ends, corrections, size classes, export, offline PWA) — now implemented and
measured. Generic AI counting (LOCA in ORT-Web) is implemented as Mode B; its weights are
non-commercial and the owner accepted that risk (see "Mode B" below). Multi-shot with *small* viewpoint
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
- Generic AI (implemented, Mode B): LOCA density with an "estimate" workflow — the
  weights decision is below; a detection-capable model (GeCo-class) remains preferable.
- Multi-shot (implemented): homography (ORB/RANSAC on the *whole scene* including the
  bundle outline and surroundings) + one-to-one matching of detections (greedy with a
  0.5·r gate — equivalent to Hungarian for non-overlapping objects) + majority vote;
  see `testing.md` for the measured results.

**6. Stack.** The proposed TypeScript + Vite + React + OpenCV.js + Workers + Workbox
stack is appropriate and was kept. Changes: OpenCV is **bundled** into the worker
(no CDN at runtime, guaranteed offline after install); no TensorFlow.js; ONNX Runtime
Web runs Mode B in its own worker, loaded only when AI counting is used.

**7. Reusable open source.** OpenCV.js (Apache-2.0), vite-plugin-pwa/Workbox (MIT),
ONNX Runtime Web (MIT, Mode B). No suitable ready-made pipe counter library exists;
the reference site is proprietary.

**8. Order.** MVP + size classes first (they share the detector and are the core
use case), then multi-shot (small viewpoint changes), then generic AI (Mode B, done after the
licence decision), Live Scan last or never.

**9. Changes to the specification**
- Size classes (Mode C) are not a separate mode: every circle detection produces
  diameters, so grouping is always available in the same screen.
- Mode B (AI) was moved after multi-shot, because the most valuable use case (pipes)
  does not need it and its model licensing needed an owner decision (taken 2026-10-20).
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
bundles and mixed sizes (see `research/loreatec-analysis.md` §5). CountSHARD's detector
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

## Mode B — generic objects (AI density estimate)

**Licence decision (2026-10-20).** The only candidate with published few-shot weights that
runs well in a browser is LOCA (ICCV 2023). Its code is MIT, but its weights were trained
from a SwAV ResNet-50 (CC BY-NC 4.0), so they are treated as non-commercial. The project
owner explicitly accepted this licence risk and asked for LOCA to be shipped; the notice is
in `THIRD_PARTY_NOTICES.md`, the UI shows a one-line licence note in the AI panel. Mode A
does not depend on the weights; replacing the model later only touches `public/models/`,
`src/ai/modelStore.ts` and the worker.

**Estimate, never objects.** LOCA outputs a density map; its sum is the count. Nothing in
the map identifies individual objects (peaks merge for touching objects, one object can
spread over several cells), so the app never draws per-object marks for Mode B. The UI and
every export label the number "ODHAD / ESTIMATE"; the heat map is drawn with a relative
colour scale and no numbers. The user corrects the estimate by **regions**: draw a box,
enter its true count, and that count replaces the model's density sum inside the box
(`src/ai/estimate.ts`). Regions may not overlap, so each pixel is counted by exactly one
source and the corrected total is `raw − Σ model(region) + Σ user(region)`.

**Export format.** The model is split into two ONNX graphs (`features`, `head`;
`scripts/loca/README.md`) so exemplar prototypes are computed once per photo and applied
to every tile. Weight-only int8 (int8 storage + `DequantizeLinear`, fp32 compute): 34.9 MB,
runs unchanged on WebGPU and WASM, count within 1.9 % of PyTorch on the verification images.
Dynamic int8 (ConvInteger) had about twice the error and is not supported by every
execution provider.

**Aspect ratio and tiling instead of squashing.** LOCA's protocol (and the reference site)
resizes every photo to 512 × 512. For a phone photo of many small objects that shrinks each
object to a few pixels and distorts shapes. CountSHARD scales the photo uniformly so that the
exemplars' size falls into the band LOCA saw in training (p25–p75 of FSC-147 training
exemplars at the 512 input: 36–92 px; inside the band the photo is just fitted into one
tile), then covers it with overlapping 512 × 512 tiles (32 px context, ownership split
halfway through each overlap, letterbox with the ImageNet mean). Tiles are capped at 16
(≈ inference time); the result says when the cap reduced the scale. The exemplar window is
the 512 window holding most exemplars; exemplars outside it are reported as unused. Measured
effect (`testing.md`): **no overall gain over squashing** (mean abs. error 4.6 % vs 2.4 %
synthetic, 8.7 % vs 10.7 % FSC-147, 7.0 % vs 6.6 % on 2 × 2 mosaics). It helps photos with
large objects (aspect ratio) and very small dense objects, and over-counts by a few percent
when it upscales. Kept because squashing makes an arbitrarily large phone photo unusable in
principle (objects shrink without bound) while tiling keeps the model's input at training
scale; the band was taken from training statistics, not tuned on the evaluation set, and is
the first thing to revisit with real photos.

**Post-processing.** Plain density sum (LOCA's own evaluation). The reference site's "sum
only density above 5 % of the max" was measured as well and was worse on the synthetic set
(MAE 36.2 vs 23.1 squashed) and equal on FSC-147 (6.9 vs 7.1), so it is not used.

**Runtime.** onnxruntime-web 1.30 (MIT) in its own worker, default bundle (WebGPU via JSEP +
WASM in one `.wasm`). WebGPU when an adapter exists, otherwise WASM. Multi-threaded WASM
needs cross-origin isolation, which GitHub Pages cannot provide, so WASM runs single-threaded.
Model files and the 28 MB WASM are **not** precached by the service worker (≈ 63 MB that only
AI users need): the AI panel downloads them on request with byte progress, verifies size and
SHA-256, and keeps them in Cache Storage (`countshard-ai-v2`; older versions are deleted);
the worker reads them from there, so Mode B works offline after that one download. The
status chip lists "AI model downloaded" and "AI ready" (sessions created, with the backend)
separately.

**Memory (v2, 2026-10-20).** The first release crashed iOS Safari: full self-attention over
4096 tokens allocates 512 MB per layer, and WASM memory never shrinks. The encoder's attention
is now exported in 256-query chunks (same weights, same results to 1e-5; ≈ 310 MB instead
of ≈ 830 MB after a run). Sessions use graph optimisation 'disabled', so int8 weights are not
expanded to fp32 copies. App guards: a "running" marker in localStorage reveals a run that
killed the page (message on the next start; after a WebGPU crash the device uses WASM); a WebGPU
error during a run is retried on WASM; the OpenCV worker is stopped in AI mode.

## Mode B — GeCo2 as a second model (feasibility spike, 2026-10-20)

**Question.** Can GeCo2 (AAAI 2026; code MIT, SAM2 Hiera B+ backbone Apache 2.0, official
weights `CNTQG_multitrain_ca44.pth`) run in the browser next to LOCA, and is it worth it?
Unlike LOCA it is a *detector*: one box per object, so its results could be shown and
corrected like circle-mode objects. Scripts and exact numbers: `scripts/geco2/README.md`,
`docs/testing.md` ("GeCo2 feasibility"). Nothing in the app was changed.

**It can be exported.** Three ONNX graphs (backbone / exemplar prototypes / detection head),
opset 17, fp32 ONNX gives the same counts as PyTorch. Deformable attention exports as
plain `GridSample` (the pure-PyTorch path of Deformable-DETR); every op runs on onnxruntime-web
1.30 WebGPU and WASM (RoIAlign only in the tiny exemplar graph, run on WASM). Weight-only int8
as for LOCA: **87.4 MB** (backbone 71.3 MB, head 16.0 MB) vs LOCA's 34.9 MB; the backbone
does not shrink at lower input sizes. GeCo2's optional SAM mask-decoder box refinement (needs
SAM2 weights, runs per box) is not exported; it changes counts on dense FSC-147 photos by up
to ≈ 6 % (fewer duplicates after NMS) and not at all on the synthetic scenes.
Instead the app drops a box that lies mostly inside a stronger one (overlap > 70 % of the
smaller box's area), which IoU-based NMS misses for a small box nested in a big one. Measured
by `scripts/geco2/containment_eval.py` on the saved objectness maps: FSC-147 MAE 15.5 → 10.6
(> 80 %: 11.1, > 60 %: 10.2), synthetic 1.8 → 1.7. 70 % was chosen as the safer of the two
best values, since touching items rarely overlap that much.

**Accuracy (same inputs as the LOCA eval).** Synthetic pipe/rod scenes — CountSHARD's actual
use case — are GeCo2's strength: MAE 1.8 vs LOCA 19.6 (app tiling) / 23.1 (squash), and
the detections are individually correct (e.g. 260/260 on four mixed sizes from three
exemplars; 2,496 true + 14 false on 18 px rods). On the FSC-147 sample it is worse than LOCA,
almost entirely on "tree logs", where it also boxes partly hidden logs that the dataset's
annotation does not count (MAE 11.7 vs 8.3; on the 2 × 2 log mosaics 69 vs 35). Other FSC
classes are on par or better.

**Cost — the blocker for phones.**
- *WASM (1 thread, as on GitHub Pages):* ≈ 100 s per photo at 1024 px and ≈ 835 MB RSS after
  one run even after making the graphs leaner (Hiera blocks in row bands, head in row
  stripes); the iPhone already crashed at ≈ 830 MB with LOCA v1. Not usable.
- *WebGPU (Intel Gen12 iGPU, Chromium):* ≈ 5.5–7 s per photo at 1024 px (warm), which is
  fine — but the tab then holds ≈ 2.0 GB of GPU memory (LOCA: ≈ 0.8 GB), because
  onnxruntime-web keeps its buffer pool. On phones GPU memory is the same RAM as the tab's.
- Two onnxruntime-web WebGPU problems were found: an int64 `Clip` gave wrong row indices
  (fixed in the export by literal index constants), and the int8 backbone's deepest output
  deviates locally (up to 0.65) unless intermediates are kept alive — counts were unaffected
  on the tested photos (217 = 217, 223 = 223), but this must be understood before shipping.
- 768 / 512 px inputs cut memory and time roughly with the area (WASM, first export: 739 / 556 MB,
  52 / 22 s). **768 px keeps the accuracy** (synthetic MAE 1.1, FSC-147 14.0); 512 px loses
  small objects (18 px rods: 641 of 2,496). 768 px is the candidate for phones; its WebGPU
  footprint was not measured (estimate ≈ 1.1–1.3 GB, scaling the 1024 px figure by area).

**Licence.** Code MIT, backbone Apache 2.0, but the checkpoint's host (Hugging Face dataset
`jerpelhan/geco2-assets`) carries no licence tag — unclear, to be asked of the authors before
shipping (not obviously better or worse than LOCA's CC BY-NC backbone).

**Phone test (2026-10-20, iPhone, bench page).** WebGPU at 768 px crashes the Safari tab in
the first backbone run; **WASM works**: 768 px ≈ 16–17 s, 1024 px ≈ 31 s per photo, counts
identical to the PC. So on the iPhone the right path is GeCo2 on WASM (768 px by default),
which overturns "never on WASM" below for this class of CPU. Galaxy A34 5G: WASM works with
equal counts but takes ≈ 1 min per photo at 768 px (≈ 1.7 min at 1024 px) — usable only as an
explicit "slow but precise" choice; LOCA stays the default there. Updated plan for stage 2:
"Automaticky" picks GeCo2 only where a short benchmark predicts ≲ 20–30 s per photo (PC with
WebGPU: GeCo2 1024 on WebGPU; iPhone-class CPU: GeCo2 768 on WASM), never GeCo2 on WebGPU on
phones, LOCA otherwise; manual choice of GeCo2 on a slow device shows the expected time.

**Integrated (stage 2, 2026-10-20)** as a second Mode B model next to LOCA (LOCA unchanged):
- *Choice*: AI panel select "Automaticky" (default) / "LOCA – rychlý odhad" / "GeCo2 – přesnější,
  jednotlivé kusy", remembered per device. "Automaticky" = GeCo2 when the expected time per photo
  is ≤ 25 s and the device does not report < 3 GB memory, else LOCA; the panel says which model
  it chose and why. Expected time: run times measured on the device (averaged), before the
  first run WebGPU ≈ 8 s or a 180³ float matrix-product benchmark × 0.7 s/ms (calibrated on the
  test PC: 29.7 ms ↔ 20.8 s; an estimate elsewhere). A manual GeCo2 choice on a slow or
  memory-poor device shows a warning with the expected time.
- *Backend*: GeCo2 on WebGPU (1024 px) only on non-mobile devices; phones always WASM at 768 px
  (WebGPU crashed both phones; WASM works). A WebGPU failure is retried once on WASM and the
  device then stays on WASM; a crash (run marker with model + backend) shows a message and an
  "Přepnout na LOCA" button.
- *Hosting*: the GeCo2 files (≈ 83–87 MB per input size) are plain files in
  `public/models/geco2/` (not Git LFS: Pages does not serve LFS content), so the app does not
  depend on another site. Until 2026-10-26 they came from `pokys.github.io/geco2-bench`; files
  cached from there are re-keyed to the new URLs without a download. The app checks size +
  SHA-256 and caches them separately.
- *Results are objects, not an estimate*: numbered boxes, tap removes/adds, counting area,
  undo; re-runs keep the user's edits; exports record the model (CSV every row, JSON
  `summary.detections`, PNG footer). LOCA results stay density estimates; switching models
  replaces the AI result of the photo.
- Measured in the browser (Chromium, Intel iGPU): demo photo 162 of 162 on WebGPU 1024 px in
  5.1 s, and 162 of 162 on WASM 768 px in 20.5 s (benchmark estimate before the run: 22 s).

**Exemplars after the first real-photo test (2026-10-20).** Exemplars stay axis-aligned
rectangles: GeCo2 (like LOCA) takes only a box — the mean appearance inside it and its width
and height — and was trained with axis-aligned boxes around objects in any orientation, so
lassos or rotated boxes would be reduced to their bounding box anyway. What helps is precision:
a tapped exemplar shows corner handles (drag inside = move, corner = resize; only the selected
one moves, a drag over another one draws a new box because tight boxes of neighbours touch),
and the GeCo2 result warns when the exemplars are > 1.5× larger or < 0.6× the median size of
the items found. The same change made every drag one undo step (`dragMerge`); before, undo
after moving or resizing a circle returned only to the last intermediate position.

**Recommendation (before the phone test): GeCo2 only on strong devices with WebGPU, LOCA stays the default and the
fallback; never GeCo2 on WASM.** Concretely: desktop/laptop with WebGPU → GeCo2 offered and
"Automaticky" may pick it; iPhone 17 Pro class → the 768 px variant, and only after an
on-device test shows its GPU footprint survives (estimate: plausible but unmeasured; 1024 px
at ≈ 2 GB is likely too much); Galaxy A class and any device without WebGPU → LOCA only. Before stage 2: test on the real devices, look for a way to
cap onnxruntime-web's WebGPU buffer pool (or run the head in two sessions), and track down the
backbone deviation.
