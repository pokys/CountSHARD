# GeCo2 phone bench

A single test page: does the GeCo2 few-shot counter (converted to ONNX int8) run in a phone
browser, how fast, and does the tab survive its memory use? Used to decide whether CountSHARD
gets GeCo2 as a second AI model next to LOCA. Not part of the app.

Open the Pages URL on the phone, pick input size (768 / 1024 px) and backend (WebGPU / WASM),
press **Spustit test**. Models download once (≈ 83–88 MB) and are kept in Cache Storage. If the
tab crashes, reopening the page reports the step it reached. **Kopírovat výsledky** copies the
device info and all results as JSON.

## Contents

- `index.html` — the page (onnxruntime-web 1.30, default WebGPU + WASM bundle in `ort/`).
- `models/geco2-{backbone,exemplars,head}-{768,1024}-w8.onnx` — GeCo2 split into three graphs,
  weight-only int8 (int8 weights + DequantizeLinear, fp32 compute).
- `inputs/` — test photos already scaled for the model, exemplar boxes, and the count
  onnxruntime (CPU) gets from the same input.

The export scripts live in the CountSHARD repository (`scripts/geco2/`).

## Credits and licences

- GeCo2: J. Pelhan, A. Lukežič, M. Kristan, "Generalized-Scale Object Counting with Gradual
  Query Aggregation", AAAI 2026. Code <https://github.com/jerpelhan/GECO2> (MIT). Weights
  `CNTQG_multitrain_ca44.pth` from huggingface.co/datasets/jerpelhan/geco2-assets (no licence
  stated there); converted here for testing only.
- Backbone: SAM2 Hiera B+ (Meta, Apache 2.0).
- onnxruntime-web (Microsoft, MIT).
- Test photos: synthetic renders, and one image from FSC-147 (Ranjan et al., CVPR 2021; images
  collected from the web) used for evaluation only.
