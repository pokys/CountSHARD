# Third-party software and models shipped with the application

| Component | Version | Licence | Where |
|---|---|---|---|
| OpenCV (via `@techstark/opencv-js`) | 5.0.0-release.1 | Apache-2.0 | embedded in `assets/analysis.worker-*.js` |
| React, React DOM | 18.3.1 | MIT | main bundle |
| Workbox (window, service worker runtime via `vite-plugin-pwa`) | 7.4.1 | MIT | `sw.js`, `workbox-*.js` |
| ONNX Runtime Web (`onnxruntime-web`), © Microsoft Corporation | 1.30.0 | MIT | `assets/ai.worker-*.js`, `assets/ort-wasm-simd-threaded.jsep-*.wasm` (WASM downloaded only when AI counting is first used) |
| LOCA model code (architecture, re-exported to ONNX), © 2023 Nikola Đukić | `main` (2023) | MIT | graph structure of `models/loca/*.onnx` |
| **LOCA few-shot weights** (int8 re-quantised) | checkpoint `loca_few_shot.pt` | **derived from SwAV ResNet-50 weights, CC BY-NC 4.0 — non-commercial use only** | `models/loca/loca-features-w8.onnx`, `models/loca/loca-head-w8.onnx` (downloaded only when AI counting is first used) |
| GeCo2 model code (architecture, re-exported to ONNX), © 2025 jerpelhan | `main` (commit `67d27e8`, 2026) | MIT | graph structure of `geco2-*-w8.onnx` (served from <https://pokys.github.io/geco2-bench/models/>, downloaded only when GeCo2 is first used) |
| SAM2 Hiera B+ image encoder (backbone of GeCo2), © Meta Platforms, Inc. | `sam2_hiera_base_plus` (2024) | Apache-2.0 | `geco2-backbone-*-w8.onnx` |
| **GeCo2 weights** (int8 re-quantised) | checkpoint `CNTQG_multitrain_ca44.pth` | **not stated by the authors** (see below) | `geco2-{backbone,exemplars,head}-{768,1024}-w8.onnx` |

Build/test-only tools (not shipped): Vite 5.4 (MIT), Vitest 2.1 (MIT), TypeScript 5.6
(Apache-2.0), `@vitejs/plugin-react` (MIT), `vite-plugin-pwa` (MIT), `jpeg-js` 0.4.4
(BSD-3-Clause, evaluation only). The FSC-147 images used by `npm run eval` are downloaded on
demand by `scripts/fetch-fsc147-sample.mjs` and are not part of the repository or the app.

## AI model weights — licence notice

Mode B ("Obecné objekty (AI)") uses LOCA — N. Đukić, A. Lukežič, V. Zavrtanik, M. Kristan,
"A Low-Shot Object Counting Network With Iterative Prototype Adaptation", ICCV 2023,
<https://github.com/djukicn/loca>. The code is MIT-licensed. The published weights were
trained starting from a ResNet-50 pre-trained with SwAV (Caron et al., 2020,
<https://github.com/facebookresearch/swav>), whose weights are licensed under
**CC BY-NC 4.0** (<https://creativecommons.org/licenses/by-nc/4.0/>). The LOCA weights are
therefore treated as a derivative of a non-commercial work: **the AI model may be used for
non-commercial purposes only**.

The project owner reviewed this and explicitly accepted the licence risk of shipping these
weights (decision recorded 2026-10-20, see `docs/technical-decisions.md`). Changes made to
the weights: conversion to ONNX and int8 weight-only quantisation (`scripts/loca/`). No
retraining. The rest of CountSHARD (Mode A, round objects) does not depend on these weights.

GeCo2 (optional second model) — J. Pelhan, A. Lukežič, M. Kristan, "Generalized-Scale Object
Counting with Gradual Query Aggregation", AAAI 2026, <https://github.com/jerpelhan/GECO2>. The
code is MIT-licensed; its backbone is Meta's SAM2 Hiera B+ (Apache-2.0). The weights are
published at huggingface.co/datasets/jerpelhan/geco2-assets without a stated licence (checked
2026-10-20); they were trained on counting datasets whose images come from the web. Changes:
conversion to ONNX (three graphs) and int8 weight-only quantisation (`scripts/geco2/`). The
files are not in this repository; the app downloads them from a separate site on first use.
Ask the authors before any commercial use.

## Licence texts

### Apache-2.0 (OpenCV)

The Apache-2.0 licence requires that redistributions include the licence text and any
NOTICE file. The OpenCV licence is available at <https://github.com/opencv/opencv/blob/4.x/LICENSE>
and in `node_modules/@techstark/opencv-js/LICENSE` after `npm install`; the deployment
workflow ships it as `OPENCV-LICENSE.txt`.

SAM2 (GeCo2 backbone) is Apache-2.0 as well: <https://github.com/facebookresearch/sam2/blob/main/LICENSE>.

### MIT (ONNX Runtime Web; LOCA code; GeCo2 code)

Copyright (c) Microsoft Corporation (ONNX Runtime).
Copyright (c) 2023 Nikola Đukić (LOCA).
Copyright (c) 2025 jerpelhan (GeCo2).

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
