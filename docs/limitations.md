# Known limitations

**Accuracy is measured only on synthetic images.** Real-world performance is
unknown until a real annotated set exists (see `testing.md`). Always check the
result visually; the app says so in the UI.

**Very small or very dense objects.** Below ≈ 20 px diameter in the analysed image
recall drops (synthetic `dense-small`, 18 px diameter, 2,496 objects: 77 % recall).
Use "High resolution" (2400 px), photograph closer, or split the bundle into two
photos.

**Noise and blur** add false positives (synthetic `noisy-blurred`: precision 0.85).
Lower the sensitivity or remove false hits by tapping.

**Non-circular ends.** Square/rectangular profiles, flattened or strongly oval
tubes, and objects photographed at an angle (ellipses) are detected less reliably;
perspective inflates diameter error (7 % at 0.88 squash). For such objects the AI mode
("Obecné objekty (AI)") gives an estimate instead (see below).

**Size classes** need about ≥ 10 % diameter difference to be separated
automatically; closer sizes need "fixed number" mode or manual merge/split. Groups
with fewer than 3 objects are not created automatically.

**Millimetres are approximate.** Calibration assumes all end faces lie in one
plane perpendicular to the camera axis; perspective, tilt, staggered ends and lens
distortion cause errors. Calibrate from a group's nominal size rather than a single
object where possible.

**Image size.** Photos are normalised to ≤ 3072 px (longest side) and analysed at
≤ 1600 px (or 2400 px). Very large photos (> 50 MP) may fail to decode on phones
with little memory. HEIC opens only in Safari.

**First load** (Mode A) downloads ≈ 15 MB (mostly the OpenCV engine, ≈ 3.5–4 MB compressed
over the network depending on server compression). Offline use requires that first
load to finish; the status chip shows when it has.

**Storage.** Projects live in the browser's IndexedDB on one device; clearing site
data deletes them. Export JSON to keep a copy. Persistent storage is requested but
browsers may refuse it.

**Multi-shot fusion** is measured only on synthetic photo series (small shift,
rotation ≤ 2°, scale ≤ 3 %, slight perspective, one occluding bar per photo,
independent noise; see `testing.md`). Known limits:
- Photos must be taken from *nearly the same viewpoint*. Corner displacement above
  20 % of the diagonal, or any registration with < 40 RANSAC inliers, inlier ratio
  < 15 %, inliers covering < 20 % of the frame or RMS residual > 2.5 px, is refused.
  Larger viewpoint changes (walking around the stack, strong tilt) are out of scope.
- After alignment, a photo is also refused when fewer than 50 % of its *confident*
  detections (the better-scoring half) have a counterpart in the reference photo and
  vice versa. Until 2026-10-20 all detections were used for this check, which wrongly
  refused correctly aligned real photos whose borderline detections differ from shot
  to shot (reported by a user: "detekce po zarovnání nesouhlasí"). The refusal message
  now shows the measured agreement.
- The homography assumes the end faces lie roughly in one plane. Strongly staggered
  ends or a deep stack seen at an angle violate this and are likely to be refused or
  to fuse less accurately.
- With **two** photos a disagreement cannot be settled by majority: an object seen in
  only one of them counts only if its edge support is strong (≥ 0.8) and it does not
  overlap an object both photos agree on; otherwise it is an *uncertain candidate*
  ("?") that does not count until confirmed. Hidden-in-one-photo objects can
  therefore end up as candidates (synthetic: up to 11 of 217 tubes with 2 photos).
- Noisy, blurred photos: fusion improves recall and removes most single-photo false
  positives, but systematic false hits that recur in several photos are counted
  (synthetic `noisy-blurred`: 34–88 false positives depending on the series, vs
  92–118 for the reference photo alone).
- The extra photos are kept in memory only; the project saves the fused result and a
  per-photo summary, not the extra images. Moving the sensitivity slider or "Count
  again" re-analyses the project photo alone and replaces the (unreviewed) fused
  objects; corrections are kept as usual.
- Objects visible only in the extra photos but outside the reference frame cannot be
  shown and are not counted.
- Fusion takes ≈ 0.4–0.6 s detection + ≈ 0.5 s registration per photo at
  ≈ 1000×800 px in Node (WASM, single thread); phones will be slower.

**Generic objects (AI, Mode B)** — measured only on 10 synthetic scenes, 20 small
FSC-147 images and 12 mosaics of them (`testing.md`); no real photos of the target use cases.
- The result is an **estimate**: the sum of a density map. Individual objects are not
  found, numbered or measured; there are no size groups or millimetres in this mode.
  Mean absolute error on the 20 FSC-147 images was 8.7 % with outliers of 20–26 %
  (+24 % on a dense stack of small log ends, −26 % on a photo of 11 apples).
- With the default tiling the model tends to **over-count when the photo is upscaled**
  (small exemplars, scale > 1): +2 … +9 % on synthetic scenes, up to +24 % on one FSC-147
  image. Overall the tiling was not more accurate than LOCA's own squash-to-512 on the
  evaluation images (`testing.md`).
- Exemplars must be **close together** (inside one 512 px window at model scale); the
  others are ignored and the result says so. 1–2 exemplars are repeated to the 3 LOCA
  needs, which is less informative than 3 different ones.
- Very small objects relative to the photo are processed at a reduced scale once the
  16-tile limit is reached; the result then warns that the estimate may be low.
- **Speed:** ≈ 3 s per 512 px tile with WebGPU on an Intel Gen12 laptop iGPU, ≈ 5 s per tile
  on single-thread WASM (desktop CPU, measured in Node); phones without WebGPU will be
  several times slower. A 16-tile photo can take minutes on WASM. Multi-threaded WASM is
  not available on GitHub Pages (no cross-origin isolation).
- **Memory:** the int8 model is expanded to fp32 when loaded (≈ 140 MB plus activations);
  low-memory phones may fail to create the session.
- **Download:** ≈ 63 MB on first use (35 MB model + 28 MB ONNX Runtime WASM). Clearing site
  data removes it; the panel offers re-download and deletion.
- **Licence:** the weights derive from CC BY-NC 4.0 SwAV weights — non-commercial use only.
- Corrections are rectangles that must not overlap; a region's model value is the
  area-weighted density sum (cells are 4 × 4 model pixels), so objects cut by the region
  border count fractionally.
- Changing the exemplars marks the result as stale; it is not re-run automatically.

**Not implemented yet:** Live Scan, PDF reports, automated browser E2E tests.
