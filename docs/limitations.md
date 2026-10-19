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
perspective inflates diameter error (7 % at 0.88 squash). Generic AI counting (Mode B)
is not implemented yet.

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

**First load** downloads ≈ 15 MB (mostly the OpenCV engine, ≈ 3.5–4 MB compressed
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

**Not implemented yet:** generic AI counting, Live Scan, PDF reports, automated
browser E2E tests.
