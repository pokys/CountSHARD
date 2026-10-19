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

**Not implemented yet:** generic AI counting, multi-shot fusion, Live Scan, PDF
reports, automated browser E2E tests.
