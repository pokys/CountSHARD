# CountSHARD

**Offline počítání trubek, tyčí a dalších kulatých objektů na fotografii.**
Vyfoťte svazek z čela, aplikace objekty spočítá, rozdělí podle průměru a vy chyby
opravíte klepnutím. Vše běží v prohlížeči – fotografie zařízení neopouštějí.

*Offline-first PWA that counts pipes, rods and other round objects seen end-on,
groups them by diameter and lets you correct the result. Everything runs locally in
the browser.*

## Features (MVP)

- Photo from file, drag & drop or the phone camera; EXIF orientation handled.
- Automatic detection of circular ends in a Web Worker (OpenCV.js), with progress
  and cancel. Works for solid rods and tubes (measures the outer wall, not the bore)
  and for mixed sizes in one photo.
- Corrections: tap to add / tap to remove, select to move or resize, undo/redo,
  "hold for original" compare, zoom/pan/pinch. Re-running detection keeps every
  correction (removed false hits stay removed).
- Automatic size groups (no fixed number assumed), fixed-k mode, merge/split,
  manual reassignment, per-group counts and diameters; optional mm calibration from
  one object or a group's nominal size, with uncertainty warnings.
- Clear result breakdown: automatic / added / removed / manually confirmed.
- Export: numbered PNG with legend, CSV (Czech Excel friendly), JSON project
  (re-importable, fully editable). Projects are auto-saved locally (IndexedDB).
- Installable PWA with an honest offline status (shell cached / engine cached /
  engine running).
- Czech UI, English included.

- Multi-shot: fuse 2–5 photos of the same bundle taken from nearly the same spot;
  every object counts once and shows in how many photos it was seen.

- **Generic objects (AI), mode "Obecné objekty (AI)":** mark 1–3 example objects and a
  LOCA model running in the browser (WebGPU or WASM) estimates how many there are. The
  result is shown as a density heat map and a total labelled as an **estimate** — objects
  are not individually identified — and can be corrected region by region. The model
  (≈ 63 MB with its runtime) is downloaded only when first used, then works offline.
  **The AI model's weights are for non-commercial use only** (CC BY-NC 4.0 via SwAV) —
  see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Not yet: Live Scan, PDF reports — see [docs/roadmap.md](docs/roadmap.md).

## Run

Requires Node 20+.

```bash
npm install
```

```bash
npm run dev
```

```bash
npm test
```

Production build and local preview (service worker active only here):

```bash
npm run build
```

```bash
npx vite preview --port 4173
```

The `dist/` folder is static and can be served from any HTTPS host (service
workers require HTTPS or localhost). The app uses relative paths, so it also works
from a sub-directory. Include OpenCV's Apache-2.0 licence when deploying (see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Documentation

- [Reference investigation (LoreaTec, REA)](docs/research/loreatec-analysis.md)
- [Technical decisions & spec review](docs/technical-decisions.md)
- [Architecture](docs/architecture.md)
- [Testing & measured results](docs/testing.md)
- [Limitations](docs/limitations.md)
- [Roadmap](docs/roadmap.md)

## Accuracy

On the synthetic evaluation suite the detector reaches 0.95–1.00 precision and
0.92–1.00 recall on most scenes and recovers four mixed pipe sizes with 99 % class
accuracy; dense tiny objects and heavy noise are weaker. **It has not yet been
evaluated on real photographs** — see [docs/testing.md](docs/testing.md). Always
check the marked image before relying on a count.

## Deployment

Every push to `main` runs the tests, builds the app and publishes it to GitHub Pages
(`.github/workflows/pages.yml`). In the repository settings, *Pages → Source* must be
set to **GitHub Actions**. Pages serves over HTTPS, so the service worker and offline
mode work from the published link.

## Licence

Code: [MIT](LICENSE). Third-party components: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The AI model files in `public/models/loca/` are derived from non-commercially licensed
weights (CC BY-NC 4.0) and are **not** covered by the MIT licence.
